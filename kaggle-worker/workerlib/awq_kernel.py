"""The native AWQ kernel verified on Kaggle's free T4; no alternative backend."""
import gc
from contextlib import redirect_stdout, redirect_stderr
import importlib
import importlib.metadata
import io
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import sysconfig
import time
import urllib.request
import zipfile

from .errors import AppError

SOURCE_COMMIT = '3d1413bf19a72451632f8dc2bb44c88bf2faed3e'
BUILD_SECONDS = 420
RESERVE_SECONDS = 120  # Model load and result delivery; batch already reserves token expiry.
_verified_native = None


def _timeout(limit, deadline):
    remaining = int(deadline - time.time() - RESERVE_SECONDS)
    if remaining <= 0:
        raise AppError('batch_timeout', 'カーネル準備に使えるbatch時間が不足しています。', 503)
    return min(limit, remaining)


def _run(command, limit, deadline, env=None):
    timeout = _timeout(limit, deadline)
    process = subprocess.Popen(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, start_new_session=True)
    try:
        stdout, _ = process.communicate(timeout=timeout)
    except BaseException:
        # Also terminate pip's ninja/nvcc children on timeout or the batch alarm.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.communicate()
        raise
    if process.returncode:
        raise RuntimeError('native_command_failed')
    return stdout


def _versions(nvcc_output, torch_cuda):
    actual = re.search(r'release (\d{1,2})\.(\d{1,2})(?!\d)', nvcc_output)
    expected = re.fullmatch(r'(\d{1,2})\.(\d{1,2})', torch_cuda or '')
    if not actual or not expected:
        raise RuntimeError('native_toolkit_unknown')
    actual_version, expected_version = tuple(map(int, actual.groups())), tuple(map(int, expected.groups()))
    print('recipe_awq_kernel stage=toolkit nvcc_major=' + str(actual_version[0])
        + ' nvcc_minor=' + str(actual_version[1]) + ' torch_cuda_major=' + str(expected_version[0])
        + ' torch_cuda_minor=' + str(expected_version[1]), flush=True)
    if actual_version[0] != expected_version[0]:
        raise RuntimeError('native_toolkit_mismatch')
    if actual_version[1] != expected_version[1]:
        print('recipe_awq_kernel stage=toolkit warning_cuda_minor_mismatch=1', flush=True)


def _select_extension(source):
    marker = 'additional_setup_kwargs = {'
    if source.count(marker) != 1:
        raise RuntimeError('native_source_shape_unknown')
    return source.replace(marker,
        'extensions = [extension for extension in extensions if extension.name == "awq_ext"]\n' + marker)


def _check_kernel(native):
    if not callable(getattr(native, 'gemm_forward_cuda', None)) or not callable(getattr(native, 'dequantize_weights_cuda', None)):
        raise RuntimeError('native_symbols_missing')
    import torch
    from awq.utils.packing_utils import dequantize_gemm
    torch.manual_seed(42)
    qweight = torch.randint(-(2**31), 2**31, (512, 32), device='cuda:0', dtype=torch.int32)
    qzeros = torch.randint(-(2**31), 2**31, (4, 32), device='cuda:0', dtype=torch.int32)
    scales = (torch.rand((4, 256), device='cuda:0', dtype=torch.float16) + 0.5) * 0.05
    with torch.inference_mode():
        reference = dequantize_gemm(qweight, qzeros, scales, 4, 128)
        restored = native.dequantize_weights_cuda(qweight, scales, qzeros, 0, 0, 0, False)
        torch.testing.assert_close(restored, reference, rtol=0.001, atol=0.001)
        maximum_error = 0.0
        for rows in [1, 16]:
            matrix = torch.randn((rows, 512), device='cuda:0', dtype=torch.float16)
            result = native.gemm_forward_cuda(matrix, qweight, scales, qzeros, 8)
            expected = torch.matmul(matrix, reference)
            torch.testing.assert_close(result, expected, rtol=0.02, atol=0.1)
            maximum_error = max(maximum_error, float((result - expected).abs().max().item()))
        torch.cuda.synchronize(0)
    print('recipe_awq_kernel stage=check result=ok max_abs_error=' + str(round(maximum_error, 6)), flush=True)
    del qweight, qzeros, scales, reference, restored, matrix, result, expected
    gc.collect()
    torch.cuda.empty_cache()


def ensure_native_awq(root, deadline):
    """Build only when absent, then verify once before any 14B model load."""
    global _verified_native
    if _verified_native is not None:
        return _verified_native
    stage = 'import'
    try:
        import torch
        if not torch.cuda.is_available() or torch.cuda.get_device_capability(0) != (7, 5):
            raise RuntimeError('free_t4_required')
        try:
            native = importlib.import_module('awq_ext')
        except ModuleNotFoundError as error:
            if error.name != 'awq_ext':
                raise
            # Reserve a full bounded build plus model-load/result time before starting.
            if deadline - time.time() < BUILD_SECONDS + RESERVE_SECONDS:
                raise AppError('batch_timeout', 'カーネル準備に使えるbatch時間が不足しています。', 503)
            stage = 'toolkit'
            from torch.utils.cpp_extension import CUDA_HOME, get_compiler_abi_compatibility_and_version
            if not CUDA_HOME:
                raise RuntimeError('native_toolkit_missing')
            nvcc = Path(CUDA_HOME) / 'bin' / 'nvcc'
            if not nvcc.is_file() or not (Path(CUDA_HOME) / 'include' / 'cuda_runtime.h').is_file():
                raise RuntimeError('native_toolkit_missing')
            if not (Path(sysconfig.get_path('include')) / 'Python.h').is_file():
                raise RuntimeError('python_headers_missing')
            _versions(_run([str(nvcc), '--version'], 20, deadline), torch.version.cuda)
            compiler = shutil.which('g++')
            if not compiler:
                raise RuntimeError('native_compiler_incompatible')
            # Run the real ABI check, but keep any compiler path in its warning private.
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                compatible = get_compiler_abi_compatibility_and_version(compiler)[0]
            if not compatible:
                raise RuntimeError('native_compiler_incompatible')
            original_torch = importlib.metadata.version('torch')
            stage = 'fetch'
            source_root = Path(root) / 'native-awq'
            source_root.mkdir(parents=True, exist_ok=True)
            url = 'https://codeload.github.com/casper-hansen/AutoAWQ_kernels/zip/' + SOURCE_COMMIT
            with urllib.request.urlopen(url, timeout=_timeout(30, deadline)) as response:
                archive = response.read(20 * 1024 * 1024 + 1)
            if len(archive) > 20 * 1024 * 1024:
                raise RuntimeError('native_source_too_large')
            with zipfile.ZipFile(io.BytesIO(archive)) as source_zip:
                for entry in source_zip.infolist():
                    if not (source_root / entry.filename).resolve().is_relative_to(source_root.resolve()):
                        raise RuntimeError('native_source_path_invalid')
                source_zip.extractall(source_root)
            source = source_root / ('AutoAWQ_kernels-' + SOURCE_COMMIT)
            setup = source / 'setup.py'
            setup.write_text(_select_extension(setup.read_text(encoding='utf-8')), encoding='utf-8')
            stage = 'dependencies'
            _run([sys.executable, '-m', 'pip', 'install', '--disable-pip-version-check', '-q',
                '--no-deps', 'setuptools', 'wheel', 'numpy', 'ninja'], 90, deadline)
            if importlib.metadata.version('torch') != original_torch:
                raise RuntimeError('torch_changed')
            stage = 'build'
            env = dict(os.environ, MAX_JOBS='2', COMPUTE_CAPABILITIES='75', CC=compiler, CXX=compiler)
            for key in ['CUDA_VERSION', 'TORCH_VERSION', 'PYPI_BUILD', 'TORCH_DONT_CHECK_COMPILER_ABI']:
                env.pop(key, None)
            wheels = source_root / 'wheels'
            wheels.mkdir(exist_ok=True)
            started = time.monotonic()
            _run([sys.executable, '-m', 'pip', 'wheel', '--disable-pip-version-check', '--no-deps',
                '--no-build-isolation', '--wheel-dir', str(wheels), str(source)], BUILD_SECONDS, deadline, env)
            built = list(wheels.glob('autoawq_kernels-*.whl'))
            if len(built) != 1 or importlib.metadata.version('torch') != original_torch:
                raise RuntimeError('native_build_invalid')
            print('recipe_awq_kernel stage=build result=ok seconds=' + str(round(time.monotonic() - started, 3)), flush=True)
            stage = 'install'
            _run([sys.executable, '-m', 'pip', 'install', '--disable-pip-version-check', '-q',
                '--no-deps', str(built[0])], 60, deadline)
            if importlib.metadata.version('torch') != original_torch:
                raise RuntimeError('torch_changed')
            importlib.invalidate_caches()
            native = importlib.import_module('awq_ext')
        stage = 'check'
        _check_kernel(native)
        _verified_native = native
        return native
    except AppError:
        raise
    except Exception:
        print('recipe_awq_kernel stage=' + stage + ' result=failed', flush=True)
        raise AppError('model_api_incompatible', '無料GPUのnative AWQ環境を準備できません。時間を置いて再試行します。', 503) from None
