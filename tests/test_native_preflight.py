"""CPU checks of the disposable notebook, without importing model packages."""
import ast
import contextlib
import io
from pathlib import Path
import re
import subprocess
import urllib.error
import urllib.request
import unittest
from unittest.mock import Mock


SCRIPT = Path(__file__).resolve().parents[1] / 'kaggle-worker' / 'qwen_preflight_launcher.py'


def notebook_code():
    tree = ast.parse(SCRIPT.read_text(encoding='utf-8'))
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'main')
    return next(node.value.value for node in main.body if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == 'code' for target in node.targets)
        and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str))


def helpers(**globals_):
    tree = ast.parse(notebook_code())
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    namespace = {'re': re, 'urllib': __import__('urllib'), **globals_}
    exec(compile(ast.Module(body=selected, type_ignores=[]), '<preflight-helpers>', 'exec'), namespace)
    return namespace


class NativePreflightTests(unittest.TestCase):
    def test_launcher_and_generated_notebook_compile_without_execution(self):
        compile(SCRIPT.read_text(encoding='utf-8'), str(SCRIPT), 'exec')
        compile(notebook_code(), '<native-preflight>', 'exec')

    def test_official_setup_only_filters_extensions(self):
        source = 'extensions = official_extensions\nadditional_setup_kwargs = {"ext_modules": extensions}\n'
        changed = helpers()['select_awq_extension'](source)
        self.assertEqual(changed, source.replace('additional_setup_kwargs = {',
            'extensions = [extension for extension in extensions if extension.name == "awq_ext"]\nadditional_setup_kwargs = {'))
        for wrong in ['', source + source]:
            with self.assertRaises(RuntimeError):
                helpers()['select_awq_extension'](wrong)

    def test_nvcc_release_must_match_actual_torch_cuda(self):
        release = helpers()['toolkit_release']
        self.assertEqual(release('Cuda compilation tools, release 12.8, V12.8.93', '12.8'), (12, 8))
        for output, torch_cuda in [('release 12.4', '12.8'), ('release 13.0', '12.8'),
                ('private-token', '12.8'), ('release 12.8', None)]:
            with self.assertRaises(RuntimeError) as error:
                release(output, torch_cuda)
            self.assertEqual(str(error.exception), 'native_toolkit_mismatch')

    def test_bounded_build_kills_process_group_and_hides_output(self):
        process = Mock(pid=123)
        process.communicate.side_effect = [subprocess.TimeoutExpired('private-command', 420,
            output='private-token', stderr='private-source'), ('private-token', 'private-source')]
        popen = Mock(return_value=process)
        killpg = Mock()
        namespace = helpers(subprocess=type('Subprocess', (), {'Popen': popen, 'PIPE': -1,
            'TimeoutExpired': subprocess.TimeoutExpired}),
            os=type('OS', (), {'killpg': killpg}), signal=type('Signal', (), {'SIGKILL': 9}))
        output = io.StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output), self.assertRaises(RuntimeError) as error:
            namespace['run_captured'](['build'], 420)
        self.assertEqual(str(error.exception), 'bounded_command_timeout')
        self.assertEqual(output.getvalue(), '')
        killpg.assert_called_once_with(123, 9)
        self.assertTrue(popen.call_args.kwargs['start_new_session'])
        self.assertEqual(process.communicate.call_args_list[0].kwargs, {'timeout': 420})

    def test_failed_command_does_not_expose_compiler_output(self):
        process = Mock(returncode=1)
        process.communicate.return_value = ('private-source', 'private-token')
        namespace = helpers(subprocess=type('Subprocess', (), {'Popen': Mock(return_value=process),
            'PIPE': -1, 'TimeoutExpired': subprocess.TimeoutExpired}))
        with self.assertRaises(RuntimeError) as error:
            namespace['run_captured'](['build'], 420)
        self.assertEqual(str(error.exception), 'bounded_command_failed')

    def test_fixed_source_bounds_gpu_route_and_no_service_credentials(self):
        code = notebook_code()
        self.assertIn('3d1413bf19a72451632f8dc2bb44c88bf2faed3e', code)
        self.assertIn("MAX_JOBS='2',COMPUTE_CAPABILITIES='75'", code)
        self.assertIn("'--no-build-isolation','--wheel-dir',str(wheels),str(source)],420", code)
        self.assertIn("gemm.awq_ext, gemm.TRITON_AVAILABLE, gemm.user_has_been_warned = native, False, True", code)
        self.assertIn('gemm.awq_ext, gemm.TRITON_AVAILABLE, gemm.user_has_been_warned = original', code)
        self.assertIn("torch.cuda.get_device_capability(0) != (7,5)", code)
        for secret in ['worker_token', 'CF_LAUNCHER_TOKEN', 'CF_WORKER_URL', 'drive_file_id', 'device_map="auto"']:
            self.assertNotIn(secret, code)
        launcher = SCRIPT.read_text(encoding='utf-8')
        self.assertIn("kernel_status(owner + '/' + slug)", launcher)
        self.assertIn('kernel_status(reference)', launcher)

    def test_failure_reason_only_emits_fixed_labels(self):
        reason = helpers()['safe_failure_reason']
        for text in ['native_toolkit_missing','native_toolkit_mismatch','bounded_command_failed',
                'bounded_command_timeout','native_compiler_incompatible']:
            self.assertEqual(reason(RuntimeError(text)),text)
        private = 'secret-token https://private.example /private/path'
        for error,expected in [(RuntimeError(private),'preflight_failed'),
                (RuntimeError('native_toolkit_missing ' + private),'preflight_failed'),
                (ImportError(private),'native_import_failed'),
                (AssertionError(private),'kernel_numeric_mismatch'),
                (urllib.error.URLError(private),'native_fetch_failed'),
                (OSError(private),'native_environment_failed'),
                (ValueError(private),'preflight_failed')]:
            with self.subTest(expected=expected):
                self.assertEqual(reason(error),expected)

    def test_toolkit_diagnostic_exposes_only_version_numbers_even_on_mismatch(self):
        diagnostic = helpers()['safe_toolkit_diagnostic']
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            diagnostic('private-token /private/nvcc release 12.4, build secret-url','12.8')
            diagnostic('private-token', 'private-path')
            diagnostic('release 12.999','12.8 private-token')
            diagnostic(None,None)
        self.assertEqual(output.getvalue(),
            'recipe_preflight stage=toolkit_check nvcc_major=12 nvcc_minor=4 torch_cuda_major=12 torch_cuda_minor=8\n')


if __name__ == '__main__':
    unittest.main()
