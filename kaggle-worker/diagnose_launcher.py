"""Temporary CPU-only Kaggle API compatibility diagnosis. No recipe access."""
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
from launcher import public_repository, kernel_status, LaunchError, status_failure_code

HERE = Path(__file__).resolve().parent

def main():
    public_repository(os.environ.get('GITHUB_REPOSITORY'), os.environ.get('GITHUB_TOKEN', ''))
    owner = os.environ.get('KAGGLE_KERNEL_OWNER', '')
    if not re.fullmatch(r'[A-Za-z0-9_-]+', owner):
        raise LaunchError('invalid_kernel_ref')
    reference = owner + '/recipe-asr-cpu-diagnosis'
    # It now exists and the first diagnosis is complete; reject an active run.
    kernel_status(reference)
    requirements = (HERE / 'requirements-worker.txt').read_text(encoding='utf-8')
    code = '''import importlib.metadata, inspect, os, pathlib, subprocess, sys, tempfile, traceback, wave
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
root = pathlib.Path(tempfile.mkdtemp(prefix='recipe-cpu-diagnosis-'))
stage = 'install'
try:
    (root / 'requirements.txt').write_text(REQUIREMENTS)
    installed = subprocess.run([sys.executable, '-m', 'pip', 'install', '--disable-pip-version-check', '-q', '-r', str(root / 'requirements.txt')], capture_output=True, text=True, timeout=600)
    if installed.returncode:
        raise RuntimeError('dependency_install_failed')
    for package in ['torch','transformers','huggingface-hub','faster-whisper','ctranslate2','onnxruntime','av','autoawq']:
        try:
            version = importlib.metadata.version(package)
            print('recipe_cpu_diag package=' + package + ' version=' + version, flush=True)
        except importlib.metadata.PackageNotFoundError:
            print('recipe_cpu_diag package=' + package + ' version=absent', flush=True)
    stage = 'awq_import'
    from transformers import AwqConfig
    from awq import AutoAWQForCausalLM
    from awq.modules.linear.gemm import WQLinear_GEMM
    print('recipe_cpu_diag stage=awq_import result=ok', flush=True)
except Exception as error:
    frames = traceback.extract_tb(error.__traceback__)
    # No inputs, URLs, paths, credentials or exception strings are emitted.
    names = [frame.name for frame in frames if frame.name.isidentifier()]
    print('recipe_cpu_diag stage=' + stage + ' exception=' + type(error).__name__ + ' frames=' + ','.join(names[-8:]), flush=True)
    message = str(error)
    import re
    missing = re.search(r"cannot import name ['\\\"]([A-Za-z_]+)['\\\"] from ['\\\"]([A-Za-z_.]+)['\\\"]", message)
    if missing:
        print('recipe_cpu_diag missing_name=' + missing.group(1) + ' module=' + missing.group(2), flush=True)
    import re
    keyword = re.search(r"unexpected keyword argument ['\\\"]([A-Za-z_]+)['\\\"]", message)
    if keyword:
        print('recipe_cpu_diag unexpected_keyword=' + keyword.group(1), flush=True)
finally:
    import shutil
    shutil.rmtree(root, ignore_errors=True)
'''
    code = 'REQUIREMENTS = ' + repr(requirements) + '\n' + code
    notebook = {'nbformat': 4, 'nbformat_minor': 5, 'metadata': {'kernelspec': {'display_name':'Python 3','language':'python','name':'python3'}}, 'cells':[{'id':'cpu-diagnosis','cell_type':'code','metadata':{},'execution_count':None,'outputs':[],'source':code.splitlines(keepends=True)}]}
    with tempfile.TemporaryDirectory(prefix='recipe-cpu-launch-') as temporary:
        directory = Path(temporary)
        (directory / 'worker.ipynb').write_text(json.dumps(notebook), encoding='utf-8')
        metadata = {'id':reference,'title':'Recipe ASR CPU diagnosis','code_file':'worker.ipynb','language':'python','kernel_type':'notebook','is_private':True,'enable_gpu':False,'enable_internet':True,'dataset_sources':[],'competition_sources':[],'kernel_sources':[]}
        (directory / 'kernel-metadata.json').write_text(json.dumps(metadata), encoding='utf-8')
        result = subprocess.run(['kaggle','kernels','push','-p',str(directory),'--timeout','1200'], capture_output=True,text=True,timeout=180)
        if result.returncode or not re.search(r'Kernel version(?: \d+)? successfully pushed\.', result.stdout or ''):
            print('CPU push status: ' + status_failure_code((result.stdout or '') + '\n' + (result.stderr or '')))
            raise LaunchError('cpu_diagnosis_launch_failed')
    print('Private CPU diagnosis submitted; no GPU, batch or recipe access.')

if __name__ == '__main__':
    try:
        main()
    except LaunchError as error:
        print('CPU diagnosis launcher stopped: ' + error.code)
        raise SystemExit(1) from None
    except Exception as error:
        print('CPU diagnosis exception type: ' + type(error).__name__)
        print('CPU diagnosis launcher stopped.')
        raise SystemExit(1) from None
