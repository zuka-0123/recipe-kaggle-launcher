"""Bounded native-kernel setup tests. No CUDA packages/models are loaded locally."""
from contextlib import redirect_stdout
import io
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'kaggle-worker'))
from workerlib import awq_kernel as kernel
from workerlib.errors import AppError


class AwqKernelTests(unittest.TestCase):
    def setUp(self):
        self.cache = patch.object(kernel, '_verified_native', None)
        self.cache.start()
        self.addCleanup(self.cache.stop)

    def torch(self, capability=(7, 5)):
        return types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: True,
            get_device_capability=lambda _: capability), version=types.SimpleNamespace(cuda='12.8'))

    def test_imported_kernel_is_verified_once_and_reused_without_install(self):
        native = object()
        with (patch.dict(sys.modules, {'torch': self.torch()}), patch.object(kernel.importlib,
                'import_module', return_value=native) as import_, patch.object(kernel, '_check_kernel') as check,
                patch.object(kernel, '_run', side_effect=AssertionError('must not install'))):
            self.assertIs(kernel.ensure_native_awq(Path('.'), 10000), native)
            self.assertIs(kernel.ensure_native_awq(Path('.'), 10000), native)
        check.assert_called_once_with(native)
        import_.assert_called_once_with('awq_ext')

    def test_broken_import_or_numeric_check_never_builds_or_falls_back(self):
        for error in [ImportError('private-token'), ModuleNotFoundError('private-token', name='other_dependency')]:
            with (self.subTest(error=type(error)), patch.dict(sys.modules, {'torch': self.torch()}), patch.object(
                    kernel.importlib, 'import_module', side_effect=error), patch.object(kernel, '_run') as run,
                    redirect_stdout(io.StringIO()) as output, self.assertRaises(AppError) as caught):
                kernel.ensure_native_awq(Path('.'), 10000)
            self.assertEqual(caught.exception.code, 'model_api_incompatible')
            self.assertNotIn('private-token', caught.exception.message + output.getvalue())
            run.assert_not_called()
        with patch.dict(sys.modules, {'torch': self.torch()}), patch.object(kernel.importlib,
                'import_module', return_value=object()), patch.object(kernel, '_check_kernel',
                side_effect=AssertionError('private numeric tensor')), redirect_stdout(io.StringIO()), self.assertRaises(AppError):
            kernel.ensure_native_awq(Path('.'), 10000)
        self.assertIsNone(kernel._verified_native)

    def test_absent_kernel_does_not_start_build_when_deadline_reserve_is_short(self):
        with patch.dict(sys.modules, {'torch': self.torch()}), patch.object(kernel.importlib,
                'import_module', side_effect=ModuleNotFoundError(name='awq_ext')), patch.object(kernel.time,
                'time', return_value=1000), patch.object(kernel, '_run') as run, self.assertRaises(AppError) as caught:
            kernel.ensure_native_awq(Path('.'), 1539)
        self.assertEqual(caught.exception.code, 'batch_timeout')
        run.assert_not_called()

    def test_requires_free_t4_without_other_device_fallback(self):
        with patch.dict(sys.modules, {'torch': self.torch((8, 0))}), patch.object(kernel.importlib,
                'import_module') as import_, redirect_stdout(io.StringIO()), self.assertRaises(AppError):
            kernel.ensure_native_awq(Path('.'), 10000)
        import_.assert_not_called()

    def test_build_uses_pinned_official_source_same_torch_and_bounded_sm75_only(self):
        self.simulate_build()

    def test_torch_replacement_stops_before_build_install_or_model_load(self):
        self.simulate_build(changed_torch=True)

    def simulate_build(self, changed_torch=False):
        archive = io.BytesIO()
        with zipfile.ZipFile(archive, 'w') as zipped:
            zipped.writestr('AutoAWQ_kernels-' + kernel.SOURCE_COMMIT + '/setup.py',
                'extensions = official_extensions\nadditional_setup_kwargs = {"ext_modules": extensions}\n')
        commands = []
        native = object()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            toolkit = root / 'toolkit'
            (toolkit / 'bin').mkdir(parents=True)
            (toolkit / 'include').mkdir()
            (toolkit / 'bin' / 'nvcc').touch()
            (toolkit / 'include' / 'cuda_runtime.h').touch()
            (toolkit / 'include' / 'Python.h').touch()
            def abi_check(_):
                print('private-compiler-path')
                print('private-compiler-path', file=sys.stderr)
                return True, '13.2'
            extension = types.SimpleNamespace(CUDA_HOME=str(toolkit),
                get_compiler_abi_compatibility_and_version=abi_check)
            def run(command, limit, deadline, env=None):
                commands.append((command, limit, env))
                if '--version' in command:
                    return 'nvcc private-path release 12.8 private-token'
                if 'wheel' in command and '--wheel-dir' in command:
                    folder = Path(command[command.index('--wheel-dir') + 1])
                    (folder / 'autoawq_kernels-0.0.9+cu128-cp313-cp313-linux_x86_64.whl').touch()
                return ''
            versions = ['2.10.0+cu128', '2.11.0'] if changed_torch else ['2.10.0+cu128'] * 4
            with patch.dict(sys.modules, {'torch': self.torch(), 'torch.utils.cpp_extension': extension}), patch.object(
                    kernel.importlib, 'import_module', side_effect=[ModuleNotFoundError(name='awq_ext'), native]), patch.object(
                    kernel.importlib.metadata, 'version', side_effect=versions), patch.object(kernel.sysconfig,
                    'get_path', return_value=str(toolkit / 'include')), patch.object(kernel.shutil,
                    'which', return_value='/usr/bin/g++'), patch.object(kernel.urllib.request,
                    'urlopen', return_value=io.BytesIO(archive.getvalue())) as fetch, patch.object(kernel,
                    '_run', side_effect=run), patch.object(kernel, '_check_kernel') as check, patch.object(
                    kernel.time, 'time', return_value=1000), patch.dict(kernel.os.environ,
                    {'CUDA_VERSION': 'spoof', 'TORCH_VERSION': 'spoof', 'PYPI_BUILD': '1',
                     'TORCH_DONT_CHECK_COMPILER_ABI': '1'}), redirect_stdout(io.StringIO()) as output:
                if changed_torch:
                    with self.assertRaises(AppError) as caught:
                        kernel.ensure_native_awq(root, 6000)
                    self.assertEqual(caught.exception.code, 'model_api_incompatible')
                    self.assertFalse(any('--wheel-dir' in c for c, _, _ in commands))
                    check.assert_not_called()
                else:
                    self.assertIs(kernel.ensure_native_awq(root, 6000), native)
                    check.assert_called_once_with(native)
            self.assertNotIn('private-token', output.getvalue())
            self.assertNotIn('private-compiler-path', output.getvalue())
            self.assertNotIn(str(root), output.getvalue())
            self.assertEqual(fetch.call_args.args[0],
                'https://codeload.github.com/casper-hansen/AutoAWQ_kernels/zip/' + kernel.SOURCE_COMMIT)
            if changed_torch:
                return
            build, timeout, env = next(item for item in commands if '--wheel-dir' in item[0])
            self.assertEqual(timeout, 420)
            self.assertIn('--no-deps', build)
            self.assertIn('--no-build-isolation', build)
            self.assertEqual((env['MAX_JOBS'], env['COMPUTE_CAPABILITIES']), ('2', '75'))
            for name in ['CUDA_VERSION', 'TORCH_VERSION', 'PYPI_BUILD', 'TORCH_DONT_CHECK_COMPILER_ABI']:
                self.assertNotIn(name, env)
            for command, _, _ in commands:
                if 'pip' in command:
                    self.assertIn('--no-deps', command)
            changed_setup = (root / 'native-awq' / ('AutoAWQ_kernels-' + kernel.SOURCE_COMMIT) / 'setup.py').read_text()
            self.assertIn('extension.name == "awq_ext"', changed_setup)

    def test_timeout_stops_entire_compiler_group_without_leaking_output(self):
        process = Mock(pid=123)
        process.communicate.side_effect = [subprocess.TimeoutExpired('private-path', 420,
            output='private-token', stderr='private-source'), ('private-token', 'private-source')]
        with patch.object(kernel.subprocess, 'Popen', return_value=process) as popen, patch.object(
                kernel.os, 'killpg', create=True) as kill, patch.object(kernel.signal, 'SIGKILL', 9, create=True), patch.object(kernel.time, 'time', return_value=1000), self.assertRaises(subprocess.TimeoutExpired):
            kernel._run(['build'], 420, 1600)
        self.assertTrue(popen.call_args.kwargs['start_new_session'])
        kill.assert_called_once_with(123, 9)
        self.assertEqual(process.communicate.call_args_list[0].kwargs, {'timeout': 420})

    def test_per_command_timeout_keeps_load_result_reservation(self):
        process = Mock(returncode=0)
        process.communicate.return_value = ('output', '')
        with patch.object(kernel.subprocess, 'Popen', return_value=process), patch.object(kernel.time,
                'time', return_value=1000):
            self.assertEqual(kernel._run(['build'], 420, 1400), 'output')
        process.communicate.assert_called_once_with(timeout=280)
        with patch.object(kernel.subprocess, 'Popen') as popen, patch.object(kernel.time,
                'time', return_value=1000), self.assertRaises(AppError):
            kernel._run(['build'], 420, 1119)
        popen.assert_not_called()

    def test_toolkit_major_mismatch_fails_minor_difference_warns_unknown_fails(self):
        with redirect_stdout(io.StringIO()) as output:
            kernel._versions('release 12.4 private-token', '12.8')
        self.assertIn('warning_cuda_minor_mismatch=1', output.getvalue())
        self.assertNotIn('private-token', output.getvalue())
        for actual, expected in [('release 13.0','12.8'), ('private-token','12.8'), ('release 12.8','unknown')]:
            with self.subTest(actual=actual), redirect_stdout(io.StringIO()), self.assertRaises(RuntimeError):
                kernel._versions(actual, expected)


if __name__ == '__main__':
    unittest.main()
