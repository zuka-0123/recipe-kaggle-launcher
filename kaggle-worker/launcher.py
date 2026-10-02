"""GitHub's standard public-repository runner launches; it never fetches recipes."""
import base64
import datetime as dt
import io
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import urllib.error
import urllib.request
from urllib.parse import urlsplit
import zipfile

HERE = Path(__file__).resolve().parent
ACTIVE_STATUSES = {'running', 'queued', 'starting', 'pending', 'initializing'}


class LaunchError(Exception):
    def __init__(self, code):
        self.code = code
        super().__init__(code)


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


def request_json(url, token, payload=None):
    request = urllib.request.Request(url, headers={'Authorization': 'Bearer ' + token,
        'Content-Type': 'application/json', 'Accept': 'application/json',
        'User-Agent': 'PersonalRecipeKB-launcher'},
        data=None if payload is None else json.dumps(payload).encode())
    try:
        with urllib.request.build_opener(NoRedirect()).open(request, timeout=35) as response:
            body = response.read(128 * 1024 + 1)
            if len(body) > 128 * 1024:
                raise LaunchError('response_too_large')
            return json.loads(body)
    except (urllib.error.URLError, ValueError):
        raise LaunchError('launcher_api_failed') from None


def base_url(value):
    parsed = urlsplit(value or '')
    if parsed.scheme != 'https' or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment or parsed.path not in ['', '/']:
        raise LaunchError('invalid_worker_url')
    return value.rstrip('/')


def public_repository(repository, github_token):
    if not re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+', repository or ''):
        raise LaunchError('invalid_repository')
    info = request_json('https://api.github.com/repos/' + repository, github_token)
    if info.get('private') is not False or info.get('visibility') != 'public':
        raise LaunchError('public_repository_required')


def status_failure_code(diagnostic):
    """Return only an HTTP status; never expose Kaggle stderr or credentials."""
    statuses = r'(401|403|404|429|5\d\d)'
    patterns = [
        r'\b' + statuses + r'\s+(?:Client|Server)\s+Error\b',
        r'\bHTTP(?:\s+(?:status|error))?\s*[:=(]?\s*' + statuses + r'\b',
        r'\b(?:status|response)(?:\s+code)?\s*[:=(]?\s*' + statuses + r'\b',
        r'\bApiException\s*:\s*\(' + statuses + r'\)',
        r'\b' + statuses + r'\s*[-:]\s*(?:Unauthorized|Forbidden|Not\s*Found|Too\s+Many\s+Requests|Internal\s+Server\s+Error|Bad\s+Gateway|Service\s+Unavailable|Gateway\s+Timeout)\b',
    ]
    for pattern in patterns:
        match = re.search(pattern, diagnostic, re.I)
        if match:
            return 'kaggle_status_http_' + match.group(1)
    return 'kaggle_status_unavailable'


def kernel_status(reference):
    """Unknown/auth errors fail closed; a clear 404 permits first private push."""
    try:
        result = subprocess.run(['kaggle', 'kernels', 'status', reference],
            capture_output=True, text=True, timeout=60, check=False)
    except OSError:
        raise LaunchError('kaggle_status_oserror') from None
    except subprocess.TimeoutExpired:
        raise LaunchError('kaggle_status_timeout') from None
    if result.returncode != 0:
        diagnostic = (result.stdout or '') + '\n' + (result.stderr or '')
        if re.search(r'\b404\b[^\n]{0,60}\bnot\s*found\b', diagnostic, re.I):
            return 'missing'
        raise LaunchError(status_failure_code(diagnostic))
    match = re.search(r'(?:status\s*[:=]\s*|status\s+)["\']?([a-z_]+)', result.stdout, re.I)
    status = match.group(1).lower() if match else ''
    if status in ACTIVE_STATUSES:
        raise LaunchError('kaggle_busy')
    if status not in {'complete', 'completed', 'error', 'failed', 'cancelled', 'canceled'}:
        raise LaunchError('kaggle_status_unknown')
    return status


def generic_bundle():
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as archive:
        archive.write(HERE / 'worker.py', 'worker.py')
        archive.write(HERE / 'requirements-worker.txt', 'requirements-worker.txt')
        for path in sorted((HERE / 'workerlib').rglob('*.py')):
            archive.write(path, path.relative_to(HERE).as_posix())
    return base64.b64encode(buffer.getvalue()).decode('ascii')


def notebook(claim):
    expires = dt.datetime.fromisoformat(claim['expires_at'].replace('Z', '+00:00'))
    now = dt.datetime.now(dt.timezone.utc)
    remaining = (expires - now).total_seconds()
    if not 120 <= remaining <= 7200 or not claim.get('worker_token'):
        raise LaunchError('invalid_token_expiry')
    # Only short-lived, batch-scoped access is embedded. No long-lived credential.
    config = {k: claim[k] for k in ['batch_id', 'worker_token', 'worker_base_url', 'expires_at']}
    config['worker_base_url'] = base_url(config['worker_base_url'])
    code = '''import base64, json, os, pathlib, shutil, subprocess, sys, tempfile, zipfile, io
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
root = pathlib.Path(tempfile.mkdtemp(prefix='recipe-batch-', dir='/tmp'))
try:
    zipfile.ZipFile(io.BytesIO(base64.b64decode(BUNDLE))).extractall(root)
    sys.path.insert(0, str(root))
    install = subprocess.run([sys.executable, '-m', 'pip', 'install', '--disable-pip-version-check', '-q', '-r', str(root / 'requirements-worker.txt')], capture_output=True, timeout=600)
    if install.returncode:
        raise RuntimeError('dependency_install_failed')
    from worker import run
    run(CONFIG, root)
except BaseException:
    try:
        from worker import fail_batch
        fail_batch(CONFIG, 'notebook_failed')
    except BaseException:
        pass
    print('Recipe batch stopped. Check the Web UI job status; no input or credentials are logged.')
finally:
    shutil.rmtree(root, ignore_errors=True)
'''
    code = 'BUNDLE = ' + repr(generic_bundle()) + '\nCONFIG = ' + repr(config) + '\n' + code
    return {'nbformat': 4, 'nbformat_minor': 5,
        'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}},
        'cells': [{'id': 'recipe-batch', 'cell_type': 'code', 'execution_count': None,
            'metadata': {}, 'outputs': [], 'source': code.splitlines(keepends=True)}]}


def launch(claim, owner, slug):
    reference = owner + '/' + slug
    kernel_status(reference)
    # This folder is runner temporary storage, never committed/cached/uploaded as an artifact.
    with tempfile.TemporaryDirectory(prefix='recipe-launch-') as tmp:
        directory = Path(tmp)
        source = directory / 'worker.ipynb'
        source.write_text(json.dumps(notebook(claim)), encoding='utf-8')
        source.chmod(0o600)
        metadata = {'id': reference, 'title': slug.replace('-', ' '),
            'code_file': 'worker.ipynb', 'language': 'python', 'kernel_type': 'notebook',
            'is_private': True, 'enable_gpu': True, 'enable_internet': True,
            'machine_shape': 'NvidiaTeslaT4',
            'dataset_sources': [], 'competition_sources': [], 'kernel_sources': []}
        (directory / 'kernel-metadata.json').write_text(json.dumps(metadata), encoding='utf-8')
        try:
            result = subprocess.run(['kaggle', 'kernels', 'push', '-p', str(directory),
                '--accelerator', 'NvidiaTeslaT4', '--timeout', '6600'],
                capture_output=True, text=True, timeout=180, check=False)
        except (OSError, subprocess.TimeoutExpired):
            raise LaunchError('kaggle_launch_failed') from None
        if result.returncode or not re.search(r'Kernel version(?: \d+)? successfully pushed\.', result.stdout or ''):
            # Do not print Kaggle stderr: it can contain private URLs or source fragments.
            raise LaunchError('kaggle_launch_failed')
    return reference


def main():
    batch = os.environ.get('BATCH_ID', '')
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,100}', batch):
        raise LaunchError('invalid_batch_id')
    origin = base_url(os.environ.get('CF_WORKER_URL'))
    launcher_token = os.environ.get('CF_LAUNCHER_TOKEN', '')
    if not launcher_token:
        raise LaunchError('missing_launcher_token')
    endpoint = origin + '/api/launcher/batches/' + batch
    try:
        public_repository(os.environ.get('GITHUB_REPOSITORY'), os.environ.get('GITHUB_TOKEN', ''))
        if not os.environ.get('KAGGLE_API_TOKEN') and not (os.environ.get('KAGGLE_USERNAME') and os.environ.get('KAGGLE_KEY')):
            raise LaunchError('missing_kaggle_auth')
        owner, slug = os.environ.get('KAGGLE_KERNEL_OWNER', ''), os.environ.get('KAGGLE_KERNEL_SLUG', '')
        if not re.fullmatch(r'[A-Za-z0-9_-]+', owner) or not re.fullmatch(r'[a-z0-9]+(?:-[a-z0-9]+)*', slug):
            raise LaunchError('invalid_kernel_ref')
        claim = request_json(endpoint + '/claim', launcher_token, {})
        if claim.get('batch_id') != batch or base_url(claim.get('worker_base_url')) != origin:
            raise LaunchError('invalid_claim')
        reference = launch(claim, owner, slug)
        request_json(endpoint + '/report', launcher_token,
            {'ok': True, 'notebook_ref': reference})
        print('Private Kaggle batch submitted.')
    except LaunchError as error:
        # Reporting also before claim allows the API to expire a dispatched batch.
        try:
            request_json(endpoint + '/report', launcher_token,
                {'ok': False, 'code': error.code, 'message': '無料Kaggle batchを起動できませんでした。Web UIで状態を確認してください。'})
        except LaunchError:
            pass
        print('Launcher stopped: ' + error.code)
        return 1
    except Exception:
        try:
            request_json(endpoint + '/report', launcher_token,
                {'ok': False, 'code': 'launcher_failed', 'message': 'batch起動処理に失敗しました。'})
        except Exception:
            pass
        print('Launcher stopped: launcher_failed')
        return 1
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (KeyError, ValueError, LaunchError):
        print('Launcher configuration invalid.')
        raise SystemExit(1) from None
