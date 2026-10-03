"""CPU-only contract tests. Models, network and Kaggle submission are mocked."""
import datetime as dt
import copy
from contextlib import contextmanager, nullcontext, redirect_stdout
import enum
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'kaggle-worker'))
import launcher
import worker
from workerlib.errors import AppError


def claim():
    return {'batch_id': 'batch-1', 'worker_token': 'short-lived-test-token',
        'worker_base_url': 'https://recipe.example',
        'expires_at': (dt.datetime.now(dt.timezone.utc) + dt.timedelta(minutes=100)).isoformat()}


class LauncherTests(unittest.TestCase):
    def test_notebook_bundle_pins_pyav_compatible_with_faster_whisper_decode(self):
        import base64
        import zipfile
        with zipfile.ZipFile(io.BytesIO(base64.b64decode(launcher.generic_bundle()))) as bundle:
            requirements = bundle.read('requirements-worker.txt').decode('utf-8')
        av_requirements = [line.strip() for line in requirements.splitlines()
            if line.strip().startswith(('av==', 'av>=', 'av<', 'av~=', 'av!='))]
        self.assertEqual(av_requirements, ['av==18.1.0'])

    def test_public_repository_only(self):
        with patch.object(launcher, 'request_json', return_value={'private': True, 'visibility': 'private'}):
            with self.assertRaisesRegex(launcher.LaunchError, 'public_repository_required'):
                launcher.public_repository('owner/repo', 'token')

    def test_running_and_queued_notebooks_block_launch(self):
        for status in ['running', 'queued', 'mystery']:
            with self.subTest(status=status), patch.object(launcher.subprocess, 'run', return_value=types.SimpleNamespace(returncode=0, stdout=f'owner/recipe-worker has status "{status}"')) as call:
                with self.assertRaises(launcher.LaunchError):
                    launcher.launch(claim(), 'owner', 'recipe-worker')
                self.assertEqual(call.call_count, 1)

    def test_private_t4_and_temporary_source_only(self):
        inspected = []
        def cli(command, **kwargs):
            if 'status' in command:
                return types.SimpleNamespace(returncode=0, stdout='owner/recipe-worker has status "complete"')
            self.assertIn('NvidiaTeslaT4', command)
            self.assertIn('--timeout', command)
            temporary = Path(command[command.index('-p') + 1])
            metadata = json.loads((temporary / 'kernel-metadata.json').read_text())
            self.assertIs(metadata['is_private'], True)
            self.assertIs(metadata['enable_gpu'], True)
            self.assertEqual(metadata['machine_shape'], 'NvidiaTeslaT4')
            self.assertEqual(metadata['title'], 'recipe worker')
            document = json.loads((temporary / 'worker.ipynb').read_text())
            source = ''.join(document['cells'][0]['source'])
            self.assertIn('short-lived-test-token', source)
            self.assertNotIn('CF_LAUNCHER_TOKEN', source)
            compile(source, 'notebook-cell', 'exec')
            inspected.append(temporary)
            return types.SimpleNamespace(returncode=0, stdout='Kernel version 42 successfully pushed.  Please check progress at https://www.kaggle.com/code/owner/recipe-worker')
        with patch.object(launcher.subprocess, 'run', side_effect=cli):
            self.assertEqual(launcher.launch(claim(), 'owner', 'recipe-worker'), 'owner/recipe-worker')
        self.assertFalse(inspected[0].exists())

    def test_expired_token_rejected(self):
        data = claim()
        data['expires_at'] = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(seconds=1)).isoformat()
        with self.assertRaisesRegex(launcher.LaunchError, 'invalid_token_expiry'):
            launcher.notebook(data)

    def test_zero_exit_code_does_not_hide_kaggle_push_error(self):
        with patch.object(launcher, 'kernel_status'), patch.object(launcher.subprocess, 'run', return_value=types.SimpleNamespace(returncode=0, stdout='Kernel push error: quota unavailable')):
            with self.assertRaisesRegex(launcher.LaunchError, 'kaggle_launch_failed'):
                launcher.launch(claim(), 'owner', 'recipe-worker')

    def test_first_launch_permits_explicit_404_only(self):
        with patch.object(launcher.subprocess, 'run', return_value=types.SimpleNamespace(returncode=1, stdout='', stderr='404 Client Error: Not Found')):
            self.assertEqual(launcher.kernel_status('owner/recipe-worker'), 'missing')
        for diagnostic in ['403 Client Error: Forbidden', '401 Client Error: Unauthorized', 'Cannot access private kernel', 'Connection timed out']:
            with self.subTest(diagnostic=diagnostic), patch.object(launcher.subprocess, 'run', return_value=types.SimpleNamespace(returncode=1, stdout='', stderr=diagnostic)), self.assertRaises(launcher.LaunchError):
                launcher.kernel_status('owner/recipe-worker')

    def test_status_error_reports_only_safe_http_code(self):
        for status, message in [(401, '401 Client Error: Unauthorized'), (403, '403 - Forbidden'),
                (429, 'ApiException: (429)'), (500, 'HTTP status: 500'), (503, 'response code: 503')]:
            diagnostic = message + ' at https://private.example/token-secret'
            with self.subTest(status=status), patch.object(launcher.subprocess, 'run', return_value=types.SimpleNamespace(returncode=1, stdout='', stderr=diagnostic)):
                with self.assertRaises(launcher.LaunchError) as captured:
                    launcher.kernel_status('owner/recipe-worker')
                self.assertEqual(captured.exception.code, f'kaggle_status_http_{status}')
                self.assertNotIn('token-secret', str(captured.exception))
        self.assertEqual(launcher.status_failure_code('credential contains 403 but no HTTP status'), 'kaggle_status_unavailable')

    def test_status_timeout_and_oserror_have_safe_distinct_codes(self):
        for error, code in [(OSError('private path'), 'kaggle_status_oserror'),
                (launcher.subprocess.TimeoutExpired('private cmd', 60), 'kaggle_status_timeout')]:
            with self.subTest(code=code), patch.object(launcher.subprocess, 'run', side_effect=error):
                with self.assertRaises(launcher.LaunchError) as captured:
                    launcher.kernel_status('owner/recipe-worker')
                self.assertEqual(captured.exception.code, code)

    def test_v222_sdk_enum_status_output(self):
        # Definitions match Kaggle's official kagglesdk kernels_enums.py.
        class KernelWorkerStatus(enum.Enum):
            QUEUED = 0
            RUNNING = 1
            COMPLETE = 2
            ERROR = 3
            CANCEL_REQUESTED = 4
            CANCEL_ACKNOWLEDGED = 5
            NEW_SCRIPT = 6
        for status in KernelWorkerStatus:
            stdout = 'owner/recipe-worker has status "%s"\n' % status
            with self.subTest(status=status.name), patch.object(launcher.subprocess, 'run', return_value=types.SimpleNamespace(returncode=0, stdout=stdout)):
                if status.name in ['COMPLETE', 'ERROR', 'CANCEL_ACKNOWLEDGED']:
                    self.assertEqual(launcher.kernel_status('owner/recipe-worker'), status.name.lower())
                else:
                    with self.assertRaisesRegex(launcher.LaunchError, 'kaggle_busy'):
                        launcher.kernel_status('owner/recipe-worker')

    def test_unknown_enum_and_failure_message_cannot_override_status(self):
        for stdout in ['owner/recipe-worker has status "KernelWorkerStatus.UNKNOWN"\n',
                'owner/recipe-worker has status "KernelWorkerStatus.RUNNING"\nFailure message: "has status complete"\n',
                'another/private-kernel has status "KernelWorkerStatus.COMPLETE"\n',
                'owner/recipe-worker has status "AnotherEnum.COMPLETE"\n']:
            with self.subTest(stdout=stdout), patch.object(launcher.subprocess, 'run', return_value=types.SimpleNamespace(returncode=0, stdout=stdout)), self.assertRaises(launcher.LaunchError):
                launcher.kernel_status('owner/recipe-worker')

    def test_urls_do_not_redirect_credentials(self):
        self.assertIsNone(launcher.NoRedirect().redirect_request(None, None, None, None, None, None))
        for url in ['http://recipe.example', 'https://token@recipe.example', 'https://recipe.example/api', 'https://recipe.example/?x=1']:
            with self.subTest(url=url), self.assertRaises(launcher.LaunchError):
                launcher.base_url(url)


class WorkerTests(unittest.TestCase):
    def fake_sdpa_modules(self, capability=(7, 5)):
        entered = []
        @contextmanager
        def kernel(*, backends):
            entered.append(backends)
            yield
        hook = lambda *args: True
        integration = types.SimpleNamespace(use_gqa_in_sdpa=hook)
        torch = types.SimpleNamespace(__version__='2.8.0+private-build',
            cuda=types.SimpleNamespace(get_device_capability=lambda _: capability))
        attention = types.SimpleNamespace(sdpa_kernel=kernel,
            SDPBackend=types.SimpleNamespace(EFFICIENT_ATTENTION='efficient-only'))
        modules = {'torch': torch, 'torch.nn.attention': attention,
            'transformers.integrations': types.SimpleNamespace(sdpa_attention=integration)}
        return modules, integration, attention, entered, hook

    def test_t4_context_uses_only_efficient_and_restores_gqa_hook(self):
        modules, integration, _, entered, hook = self.fake_sdpa_modules()
        with patch.dict(sys.modules, modules):
            with worker.t4_efficient_sdpa():
                self.assertFalse(integration.use_gqa_in_sdpa(None, None))
            self.assertIs(integration.use_gqa_in_sdpa, hook)
        self.assertEqual(entered, [['efficient-only']])

    def test_t4_context_rejects_unsupported_runtime_without_other_backend(self):
        for capability, missing_hook in [((8, 0), False), ((7, 5), True)]:
            modules, integration, _, entered, hook = self.fake_sdpa_modules(capability)
            if missing_hook:
                integration.use_gqa_in_sdpa = None
            with self.subTest(capability=capability, missing_hook=missing_hook), patch.dict(sys.modules, modules):
                with self.assertRaises(AppError) as caught:
                    with worker.t4_efficient_sdpa():
                        self.fail('An incompatible runtime cannot enter generation')
            self.assertEqual(caught.exception.code, 'model_api_incompatible')
            self.assertEqual(entered, [])

    def test_t4_context_restores_hook_on_backend_failure_and_oom(self):
        for message, code in [('No available kernel private-token https://private.example', 'model_api_incompatible'),
                ('CUDA out of memory private-token', 'gpu_memory')]:
            modules, integration, _, _, hook = self.fake_sdpa_modules()
            with self.subTest(code=code), patch.dict(sys.modules, modules):
                try:
                    with worker.t4_efficient_sdpa():
                        raise RuntimeError(message)
                except Exception as error:
                    result = worker.error_result({'job_id': 'job-1'}, error)
                self.assertIs(integration.use_gqa_in_sdpa, hook)
            self.assertEqual(result['error']['code'], code)
            self.assertTrue(result['error']['retryable'])
            self.assertNotIn('private-token', json.dumps(result))
            self.assertNotIn('https://', json.dumps(result))

    def test_runtime_diagnostics_emit_only_sanitized_version_capability_and_counts(self):
        modules, _, _, _, _ = self.fake_sdpa_modules()
        output = io.StringIO()
        with redirect_stdout(output):
            worker.safe_runtime_diagnostic(modules['torch'], 1234)
            worker.safe_runtime_diagnostic(modules['torch'], 1234, 567)
        self.assertEqual(output.getvalue().splitlines(), [
            'recipe_runtime torch=2.8 capability=7.5 input_tokens=1234',
            'recipe_runtime torch=2.8 capability=7.5 input_tokens=1234 output_tokens=567'])
        self.assertNotIn('private', output.getvalue())
        modules['torch'].__version__ = 'private-token'
        modules['torch'].cuda.get_device_capability = lambda _: ('private-token', 5)
        output = io.StringIO()
        with redirect_stdout(output):
            worker.safe_runtime_diagnostic(modules['torch'], 1, 'private-token')
            worker.safe_runtime_diagnostic(modules['torch'], 'private-token')
        self.assertEqual(output.getvalue().strip(), 'recipe_runtime torch=unknown capability=unknown input_tokens=1')

    def test_generation_context_and_youtube_time_limit_respect_deadline(self):
        class Inputs(dict):
            def to(self, device): return self
        class Output:
            def __getitem__(self, key): return types.SimpleNamespace(shape=(10,))
        @contextmanager
        def inference(): yield
        entries = []
        @contextmanager
        def efficient():
            entries.append('entered')
            yield
        torch = types.SimpleNamespace(inference_mode=inference, __version__='2.8.0',
            manual_seed=Mock(), cuda=types.SimpleNamespace(get_device_capability=lambda _: (7, 5)))
        chat_calls = []
        class Processor:
            @property
            def tokenizer(self): return self
            def apply_chat_template(self, *args, **kwargs):
                chat_calls.append(kwargs)
                return 'mock input'
            def __call__(self, *args, **kwargs): return Inputs(input_ids=types.SimpleNamespace(shape=(1, token_count)))
            def decode(self, *args, **kwargs): return '{"title":{"original":"料理"},"ingredients":[],"steps":[]}'
        cases = [(worker.NF4_MODEL, 'youtube', 3999, 1000, [], 360),
            (worker.NF4_MODEL, 'youtube', 4000, 1000, [], 600),
            (worker.NF4_MODEL, 'youtube', 6746, 500, [], 480),
            (worker.NF4_MODEL, 'web', 6746, 1000, [], 360),
            (worker.NF4_MODEL, 'text', 24000, 1000, [], 360),
            (worker.QWEN3_AWQ_MODEL, 'image', 4996, 1000,
                [{'path': 'unused.jpg', 'ref_id': 'src_001'}], 360),
            (worker.QWEN3_AWQ_MODEL, 'text', 12000, 1000, [], 900),
            (worker.QWEN3_AWQ_MODEL, 'text', 2379, 500, [], 480),
            (worker.QWEN3_AWQ_MODEL, 'youtube', 3999, 1000, [], 900),
            (worker.QWEN3_AWQ_MODEL, 'youtube', 4000, 1500, [], 1200),
            (worker.QWEN3_AWQ_MODEL, 'youtube', 4000, 1000, [], 980),
            (worker.QWEN3_AWQ_MODEL, 'youtube', 6746, 700, [], 680),
            (worker.QWEN3_AWQ_MODEL, 'text', 12001, 1000, [], None),
            (worker.NF4_MODEL, 'text', 24001, 1000, [], None)]
        for model_name, source_type, token_count, remaining, images, expected_seconds in cases:
            with self.subTest(source=source_type, tokens=token_count, remaining=remaining):
                models = worker.Models({'llm_model': model_name}, {}, 'rules', Path('.'), 1000 + remaining)
                generate = Mock(return_value=Output())
                models.model = types.SimpleNamespace(generate=generate)
                models.processor = Processor()
                template = {key: 'fixed' for key in ['recipe_id', 'schema_version', 'created_at', 'updated_at']}
                template['source'] = {'type': source_type}
                before = len(entries)
                with patch.dict(sys.modules, {'torch': torch, 'qwen_vl_utils': types.SimpleNamespace(process_vision_info=lambda _: ([], None))}), patch.object(models, 'load'), patch.object(worker, 'build_structure_prompts', return_value=('rules', 'prompt')), patch.object(worker, 't4_efficient_sdpa', efficient), patch.object(worker, 'awq_gpu_gemm', nullcontext), patch.object(worker.time, 'time', return_value=1000), redirect_stdout(io.StringIO()):
                    if expected_seconds is None:
                        with self.assertRaises(AppError) as error:
                            models.structure({'images': images, 'evidence': []}, template)
                        self.assertEqual(error.exception.code, 'input_too_large')
                        generate.assert_not_called()
                        continue
                    models.structure({'images': images, 'evidence': []}, template)
                self.assertEqual(len(entries) - before, 0 if images else 1)
                self.assertEqual(generate.call_args.kwargs['max_time'], expected_seconds)
                self.assertEqual(generate.call_args.kwargs['max_new_tokens'], 6144)
                awq = model_name == worker.QWEN3_AWQ_MODEL and not images
                self.assertEqual(generate.call_args.kwargs['do_sample'], awq)
                if awq:
                    self.assertIs(chat_calls[-1]['enable_thinking'], False)
                    self.assertEqual({k: generate.call_args.kwargs[k] for k in
                        ['temperature', 'top_p', 'top_k', 'min_p']},
                        {'temperature': 0.7, 'top_p': 0.8, 'top_k': 20, 'min_p': 0.0})
                    torch.manual_seed.assert_called_with(42)
                else:
                    self.assertNotIn('enable_thinking', chat_calls[-1])

    def test_generation_cutoff_keeps_raw_and_distinguishes_invalid_json(self):
        class Inputs(dict):
            def to(self, _): return self
        class Output:
            def __getitem__(self, _): return types.SimpleNamespace(shape=(output_count,))
        torch = types.SimpleNamespace(inference_mode=nullcontext, manual_seed=Mock(),
            cuda=types.SimpleNamespace(), __version__='2.10')
        for elapsed, output_count, limited in [(1200.1, 2242, True), (50, 6144, True), (50, 32, False)]:
            with self.subTest(elapsed=elapsed, count=output_count):
                models = worker.Models({'llm_model': worker.QWEN3_AWQ_MODEL}, {}, 'rules', Path('.'), 4000)
                models.model = types.SimpleNamespace(generate=Mock(return_value=Output()))
                models.processor = Mock(return_value=Inputs(input_ids=types.SimpleNamespace(shape=(1, 6889))))
                models.processor.decode.return_value = '{"title":'
                with patch.dict(sys.modules, {'torch': torch}), patch.object(models, 'load'), patch.object(worker, 'build_structure_prompts', return_value=('rules', 'prompt')), patch.object(worker, 't4_efficient_sdpa', nullcontext), patch.object(worker, 'awq_gpu_gemm', nullcontext), patch.object(worker.time, 'time', return_value=1000), patch.object(worker.time, 'monotonic', side_effect=[0, elapsed]), redirect_stdout(io.StringIO()):
                    with self.assertRaises(AppError) as caught:
                        models.structure({'evidence': []}, {'source': {'type': 'youtube'}})
                self.assertEqual(caught.exception.code, 'llm_not_json')
                self.assertEqual(caught.exception.details['raw_output'], '{"title":')
                self.assertEqual(bool(caught.exception.details.get('generation_limit_reached')), limited)

    def fake_model_modules(self):
        torch = types.SimpleNamespace(float16='mock-fp16', cuda=types.SimpleNamespace(
            is_available=lambda: True, empty_cache=Mock()))
        transformer = types.SimpleNamespace(BitsAndBytesConfig=Mock(return_value='mock-nf4'),
            AwqConfig=Mock(return_value='mock-awq'),
            activations=types.SimpleNamespace(GELUTanh=Mock()),
            AutoTokenizer=types.SimpleNamespace(from_pretrained=Mock(return_value='mock-tokenizer')),
            AutoProcessor=types.SimpleNamespace(from_pretrained=Mock(return_value='mock-processor')),
            AutoModelForCausalLM=types.SimpleNamespace(from_pretrained=Mock(return_value=types.SimpleNamespace(eval=Mock()))),
            Qwen2_5_VLForConditionalGeneration=types.SimpleNamespace(from_pretrained=Mock(return_value=types.SimpleNamespace(eval=Mock()))))
        return torch, transformer

    def test_rollback_llm_nf4_and_vlm_fp16_release_previous_model(self):
        torch, transformers = self.fake_model_modules()
        models = worker.Models({'llm_model': worker.NF4_MODEL}, {}, 'rules', Path('.'), 10000)
        with patch.dict(sys.modules, {'torch': torch, 'transformers': transformers}):
            models.load('llm')
            transformers.BitsAndBytesConfig.assert_called_once_with(load_in_4bit=True,
                bnb_4bit_quant_type='nf4', bnb_4bit_compute_dtype='mock-fp16')
            llm_call = transformers.AutoModelForCausalLM.from_pretrained.call_args
            self.assertEqual(llm_call.args, ('Qwen/Qwen2.5-7B-Instruct',))
            self.assertEqual(llm_call.kwargs['quantization_config'], 'mock-nf4')
            self.assertEqual(llm_call.kwargs['device_map'], {'': 'cuda:0'})
            self.assertFalse(llm_call.kwargs['trust_remote_code'])
            self.assertTrue(llm_call.kwargs['use_safetensors'])
            models.load('llm')
            self.assertEqual(transformers.AutoModelForCausalLM.from_pretrained.call_count, 1)
            def vlm_load(*args, **kwargs):
                self.assertIsNone(models.model)
                self.assertIsNone(models.kind)
                self.assertEqual(args, ('Qwen/Qwen2.5-VL-3B-Instruct',))
                self.assertNotIn('quantization_config', kwargs)
                self.assertEqual(kwargs['torch_dtype'], 'mock-fp16')
                self.assertEqual(kwargs['device_map'], {'': 'cuda:0'})
                return types.SimpleNamespace(eval=Mock())
            transformers.Qwen2_5_VLForConditionalGeneration.from_pretrained.side_effect = vlm_load
            models.load('vlm')
            self.assertEqual(models.kind, 'vlm')
            self.assertEqual(torch.cuda.empty_cache.call_count, 2)

    def test_default_awq_load_keeps_quantized_weights_on_cuda_without_bnb(self):
        torch, transformers = self.fake_model_modules()
        models = worker.Models({'llm_load_in_4bit': True}, {}, 'rules', Path('.'), 10000)
        with patch.dict(sys.modules, {'torch': torch, 'transformers': transformers}), redirect_stdout(io.StringIO()):
            models.load('llm')
        transformers.AwqConfig.assert_called_once_with(bits=4, group_size=128,
            zero_point=True, version='gemm', backend='autoawq', do_fuse=False)
        transformers.BitsAndBytesConfig.assert_not_called()
        loaded = transformers.AutoModelForCausalLM.from_pretrained.call_args
        self.assertEqual(loaded.args, ('Qwen/Qwen3-14B-AWQ',))
        self.assertEqual(loaded.kwargs['quantization_config'], 'mock-awq')
        self.assertEqual(loaded.kwargs['device_map'], {'': 'cuda:0'})
        self.assertFalse(loaded.kwargs['trust_remote_code'])
        self.assertTrue(loaded.kwargs['use_safetensors'])
        self.assertEqual(loaded.kwargs['attn_implementation'], 'sdpa')
        self.assertEqual(loaded.kwargs['torch_dtype'], 'mock-fp16')

    def test_awq_load_failure_has_no_nf4_or_cpu_retry(self):
        torch, transformers = self.fake_model_modules()
        transformers.AutoModelForCausalLM.from_pretrained.side_effect = RuntimeError('private-token')
        models = worker.Models({}, {}, 'rules', Path('.'), 10000)
        output = io.StringIO()
        with patch.dict(sys.modules, {'torch': torch, 'transformers': transformers}), redirect_stdout(output), self.assertRaises(RuntimeError):
            models.load('llm')
        self.assertEqual(transformers.AutoModelForCausalLM.from_pretrained.call_count, 1)
        transformers.BitsAndBytesConfig.assert_not_called()
        self.assertIsNone(models.model)
        self.assertIsNone(models.kind)
        self.assertNotIn('private-token', output.getvalue())

    def test_awq_import_alias_is_added_once_and_existing_alias_is_preserved(self):
        gelu = Mock()
        activations = types.SimpleNamespace(GELUTanh=gelu)
        transformers = types.SimpleNamespace(activations=activations)
        with patch.dict(sys.modules, {'transformers': transformers}):
            worker.ensure_awq_import_compat()
            self.assertIs(activations.PytorchGELUTanh, gelu)
            worker.ensure_awq_import_compat()
            self.assertIs(activations.PytorchGELUTanh, gelu)
            existing = object()
            activations.PytorchGELUTanh = existing
            worker.ensure_awq_import_compat()
            self.assertIs(activations.PytorchGELUTanh, existing)
        gelu.assert_not_called()

    def test_awq_missing_gelu_class_stops_before_loading_model(self):
        for missing in [types.SimpleNamespace(), types.SimpleNamespace(GELUTanh=None),
                types.SimpleNamespace(GELUTanh='private-token')]:
            torch, transformers = self.fake_model_modules()
            transformers.activations = missing
            models = worker.Models({}, {}, 'rules', Path('.'), 10000)
            output = io.StringIO()
            with self.subTest(activations=missing), patch.dict(sys.modules,
                    {'torch': torch, 'transformers': transformers}), redirect_stdout(output), self.assertRaises(AppError) as error:
                models.load('llm')
            self.assertEqual(error.exception.code, 'model_api_incompatible')
            self.assertNotIn('private-token', error.exception.message + output.getvalue())
            transformers.AutoModelForCausalLM.from_pretrained.assert_not_called()

    def test_nf4_rollback_does_not_import_or_patch_awq_activation_alias(self):
        torch, transformers = self.fake_model_modules()
        del transformers.activations
        models = worker.Models({'llm_model': worker.NF4_MODEL}, {}, 'rules', Path('.'), 10000)
        with patch.dict(sys.modules, {'torch': torch, 'transformers': transformers}), patch.object(worker,
                'ensure_awq_import_compat', side_effect=AssertionError('AWQ helper must not run for NF4')), redirect_stdout(io.StringIO()):
            models.load('llm')
        transformers.BitsAndBytesConfig.assert_called_once()
        transformers.AwqConfig.assert_not_called()

    def test_awq_triton_path_is_explicit_and_restored_after_success_or_failure(self):
        extension = object()
        module = types.SimpleNamespace(awq_ext=extension, TRITON_AVAILABLE=True,
            user_has_been_warned=False, awq_gemm_triton=Mock(return_value='GPU result'),
            awq_dequantize_triton=Mock(return_value='GPU weight'),
            dequantize_gemm=Mock(side_effect=AssertionError('torch fallback is forbidden')))
        for fail in [False, True]:
            with self.subTest(fail=fail), patch.dict(sys.modules, {'awq.modules.linear.gemm': module}):
                try:
                    with worker.awq_gpu_gemm():
                        self.assertIsNone(module.awq_ext)
                        self.assertTrue(module.TRITON_AVAILABLE)
                        self.assertTrue(module.user_has_been_warned)
                        self.assertEqual(module.awq_gemm_triton(), 'GPU result')
                        self.assertEqual(module.awq_dequantize_triton(), 'GPU weight')
                        module.dequantize_gemm.assert_not_called()
                        if fail:
                            raise RuntimeError('private-token')
                except RuntimeError:
                    self.assertTrue(fail)
                self.assertIs(module.awq_ext, extension)
                self.assertTrue(module.TRITON_AVAILABLE)
                self.assertFalse(module.user_has_been_warned)
        for invalid in [types.SimpleNamespace(),
                types.SimpleNamespace(awq_ext=extension, TRITON_AVAILABLE=False,
                    user_has_been_warned=False, awq_gemm_triton=Mock(), awq_dequantize_triton=Mock()),
                types.SimpleNamespace(awq_ext=extension, TRITON_AVAILABLE=True,
                    user_has_been_warned=False, awq_gemm_triton=None, awq_dequantize_triton=Mock()),
                types.SimpleNamespace(awq_ext=extension, TRITON_AVAILABLE=True,
                    user_has_been_warned=False, awq_gemm_triton=Mock(), awq_dequantize_triton=None)]:
            with self.subTest(invalid=invalid), patch.dict(sys.modules,
                    {'awq.modules.linear.gemm': invalid}), self.assertRaises(AppError) as error:
                with worker.awq_gpu_gemm():
                    self.fail('Unavailable Triton must fail before generation')
            self.assertEqual(error.exception.code, 'model_api_incompatible')
            if hasattr(invalid, 'awq_ext'):
                self.assertIs(invalid.awq_ext, extension)
                self.assertFalse(invalid.user_has_been_warned)

    def test_gpu_diagnostic_emits_only_bounded_numbers_and_fixed_stages(self):
        mib = 1024**2
        cuda = types.SimpleNamespace(memory_allocated=lambda _: 123*mib,
            memory_reserved=lambda _: 234*mib, max_memory_allocated=lambda _: 345*mib,
            get_device_properties=lambda _: types.SimpleNamespace(total_memory=15360*mib))
        torch = types.SimpleNamespace(cuda=cuda)
        output = io.StringIO()
        with redirect_stdout(output):
            worker.safe_gpu_diagnostic(torch, 'load', 2.12345)
        self.assertEqual(output.getvalue().strip(),
            'recipe_gpu stage=load seconds=2.123 allocated_mib=123.0 reserved_mib=234.0 peak_mib=345.0 total_mib=15360.0')
        cuda.memory_allocated = lambda _: 'private-token'
        cuda.memory_reserved = lambda _: (_ for _ in ()).throw(RuntimeError('https://private.example'))
        cuda.max_memory_allocated = lambda _: -1
        cuda.get_device_properties = lambda _: types.SimpleNamespace(total_memory=129*1024**3)
        output = io.StringIO()
        with redirect_stdout(output):
            worker.safe_gpu_diagnostic(torch, 'generate', float('nan'))
            worker.safe_gpu_diagnostic(torch, 'private-token', 1)
            worker.safe_gpu_diagnostic(torch, 'generate', 'https://private.example')
        self.assertEqual(output.getvalue(), '')

    def test_explicit_false_disables_quantization_without_a_fallback(self):
        torch, transformers = self.fake_model_modules()
        models = worker.Models({'llm_model': worker.NF4_MODEL, 'llm_load_in_4bit': False}, {}, 'rules', Path('.'), 10000)
        with patch.dict(sys.modules, {'torch': torch, 'transformers': transformers}):
            models.load('llm')
        transformers.BitsAndBytesConfig.assert_not_called()
        self.assertNotIn('quantization_config', transformers.AutoModelForCausalLM.from_pretrained.call_args.kwargs)

    def test_non_boolean_quantization_setting_fails_before_model_download(self):
        torch, transformers = self.fake_model_modules()
        for value in ['true', 1, 0, None]:
            models = worker.Models({'llm_model': worker.NF4_MODEL, 'llm_load_in_4bit': value}, {}, 'rules', Path('.'), 10000)
            with self.subTest(value=value), patch.dict(sys.modules, {'torch': torch, 'transformers': transformers}), self.assertRaises(AppError):
                models.load('llm')
        transformers.AutoTokenizer.from_pretrained.assert_not_called()
        transformers.AutoModelForCausalLM.from_pretrained.assert_not_called()

    def test_quantized_load_failure_stops_without_retrying_unquantized(self):
        torch, transformers = self.fake_model_modules()
        failure = RuntimeError('private-token quantization CUDA failure')
        transformers.AutoModelForCausalLM.from_pretrained.side_effect = failure
        models = worker.Models({'llm_model': worker.NF4_MODEL}, {}, 'rules', Path('.'), 10000)
        with patch.dict(sys.modules, {'torch': torch, 'transformers': transformers}), self.assertRaises(RuntimeError):
            models.load('llm')
        transformers.AutoModelForCausalLM.from_pretrained.assert_called_once()
        self.assertIsNone(models.kind)
        result = worker.error_result({'job_id': 'job-1'}, failure)
        self.assertEqual(result['error']['code'], 'model_load_failed')
        self.assertTrue(result['error']['retryable'])
        self.assertNotIn('private-token', json.dumps(result))

    def test_images_use_full_schema_and_text_uses_compact_prompt(self):
        for images, mode in [([], 'compact'), ([{'path': 'unused.jpg'}], 'full')]:
            with self.subTest(mode=mode):
                models = worker.Models({}, {}, 'rules', Path('.'), 10000)
                with patch.dict(sys.modules, {'torch': types.SimpleNamespace()}), patch.object(models, 'load'), patch.object(worker, 'build_structure_prompts', side_effect=RuntimeError('stop before inference')) as build:
                    with self.assertRaises(RuntimeError):
                        models.structure({'images': images, 'evidence': []}, {'source': {}})
                self.assertEqual(build.call_args.args[-1], mode)

    @unittest.skipUnless((ROOT.parent / 'canonical.schema.json').exists(), 'Main workspace Canonical Schema is required for this contract test')
    def test_compact_prompt_keeps_complete_array_shapes_and_enums(self):
        schema = json.loads((ROOT.parent / 'canonical.schema.json').read_text(encoding='utf-8'))
        template = {'source': {'type': 'text'}, 'ingredients': [], 'steps': []}
        text = '料理（2人分）\n材料\n鶏肉1/2枚\nしょうゆ大さじ1\nみりん大さじ1/2\n砂糖小さじ1\n塩少々\n油適量\n作り方\n1.切る。\n2.中火で片面3分ずつ焼く。\n3.1分煮詰める。'
        extraction = {'evidence': [{'ref_id': 'src_001', 'type': 'manual_input', 'text': text}]}
        system, prompt = worker.build_structure_prompts('原典以外を推測しない。', schema, template, extraction)
        full, _ = worker.build_structure_prompts('原典以外を推測しない。', schema, template, extraction, 'full')
        root = json.loads(prompt.split('\n')[1])
        shapes = {line.split('[]: ', 1)[0]: json.loads(line.split('[]: ', 1)[1])
            for line in system.split('\n') if '[]: ' in line}
        self.assertEqual(root, template)
        self.assertEqual(prompt.count(text), 1)
        self.assertNotIn(text, system)
        self.assertNotIn('"template":', prompt)
        self.assertNotIn('"item_shapes":', prompt)
        self.assertNotIn('"evidence":', prompt)
        self.assertEqual(template['ingredients'], [])
        self.assertEqual(shapes['steps']['duration'], {'value': None, 'unit': None, 'raw_text': None})
        self.assertEqual(shapes['steps']['temperature'], {'value': None, 'unit': None, 'raw_text': None})
        self.assertIsNone(shapes['steps']['heat'])
        self.assertEqual(shapes['ingredients']['amount'], {'value': None, 'unit': None, 'raw_text': None})
        self.assertEqual(shapes['source_refs']['type'], 'manual_input')
        self.assertEqual(shapes['source_refs']['ref_id'], 'src_001')
        self.assertIn('すべてingredients', system)
        self.assertIn('前の工程の火加減を引き継がず', system)
        self.assertIn('raw_text="2人分"', system)
        self.assertIn('"ingredients[].amount.unit"', system)
        self.assertIn('value・unit・raw_textの3キー', system)
        self.assertIn('大さじはtbsp', system)
        self.assertIn('amount.value=null、amount.unit=null', system)
        self.assertIn('pinch等のenumにない単位', system)
        self.assertIn('原典の言語・表記を保持し、翻訳しません', system)
        self.assertIn('ingredient_id文字列だけの配列', system)
        self.assertIn('preparationは材料欄に明示された前処理だけ', system)
        self.assertIn('nutrition_statedは原典の栄養値だけ', system)
        self.assertLess(len(system), len(full))
        self.assertIn('JSON Schema:', full)

    @unittest.skipUnless((ROOT.parent / 'canonical.schema.json').exists(), 'Canonical Schema is required')
    def test_prompt_preserves_each_original_body_and_timestamp_once(self):
        schema = json.loads((ROOT.parent / 'canonical.schema.json').read_text(encoding='utf-8'))
        template = {'source': {'type': 'youtube', 'url': 'https://example.com/origin'},
            'ingredients': [], 'steps': []}
        evidence = [{'ref_id': 'src_001', 'type': 'youtube_description',
            'text': '  塩「少々」\nマヨネーズ\t適量\n', 'start_seconds': None, 'end_seconds': None},
            {'ref_id': 'src_002', 'type': 'youtube_transcript',
            'text': '原文の{記号}と"引用"を保持する。', 'start_seconds': 1.25, 'end_seconds': 3.5}]
        before = worker.copy.deepcopy(evidence)
        system, prompt = worker.build_structure_prompts('原典以外を推測しない。', schema, template, {'evidence': evidence})
        for row in evidence:
            self.assertEqual(prompt.count(row['text']), 1)
            self.assertNotIn(row['text'], system)
        metadata = [json.loads(line.removeprefix('根拠情報: ')) for line in prompt.splitlines() if line.startswith('根拠情報: ')]
        self.assertEqual(metadata, [{key: value for key, value in row.items() if key != 'text'} for row in evidence])
        self.assertEqual(prompt.count('https://example.com/origin'), 1)
        self.assertEqual(json.loads(prompt.split('\n')[1]), template)
        self.assertEqual(evidence, before)

    def test_absence_repairs_preserve_original_raw_output_and_values(self):
        raw = json.dumps({'title': {'original': '原典', 'normalized': ''},
            'ingredients': [{'name': {'raw': '塩', 'normalized': ''}, 'amount': {'value': 1, 'unit': 'pinch', 'raw_text': 'ひとつまみ'}}],
            'steps': [{'step': 1, 'duration': None, 'temperature': None, 'heat': 'medium'},
                {'step': 2, 'duration': {'value': 3, 'unit': 'minute', 'raw_text': '片面3分ずつ'}, 'temperature': 'unknown'},
                {'step': 3, 'duration': [], 'temperature': 0}]}, ensure_ascii=False)
        candidate = worker.parse_json(raw)
        repaired = worker.repair_literal_absence(candidate)
        self.assertEqual(repaired['steps'][0]['duration'], {'value': None, 'unit': None, 'raw_text': None})
        self.assertEqual(repaired['steps'][0]['temperature'], {'value': None, 'unit': None, 'raw_text': None})
        self.assertEqual(repaired['steps'][0]['heat'], 'medium')
        self.assertEqual(repaired['steps'][1]['duration']['value'], 3)
        self.assertEqual(repaired['steps'][1]['temperature'], 'unknown')
        self.assertEqual(repaired['steps'][2]['duration'], [])
        self.assertEqual(repaired['steps'][2]['temperature'], 0)
        self.assertEqual(repaired['ingredients'][0]['amount']['unit'], 'pinch')
        self.assertIsNone(repaired['title']['normalized'])
        self.assertIsNone(repaired['ingredients'][0]['name']['normalized'])
        self.assertIsNone(json.loads(raw)['steps'][0]['duration'])

    @unittest.skipUnless((ROOT.parent / 'canonical.schema.json').exists() and importlib.util.find_spec('jsonschema'), 'Canonical Schema and validator are required')
    def test_sparse_recipe_shape_is_complete_without_inventing_content(self):
        from jsonschema import Draft202012Validator
        schema = json.loads((ROOT.parent / 'canonical.schema.json').read_text(encoding='utf-8'))
        sparse = {'schema_version': '1.0', 'recipe_id': 'rec_test',
            'title': {'original': '原典料理'},
            'source': {'type': 'text', 'url': None, 'source_id': None, 'title': None,
                'creator': None, 'retrieved_at': '2026-10-03T00:00:00Z'},
            'ingredients': [{'name': {'raw': f'材料{i}'},
                'amount': {'raw_text': '適量'}} for i in range(6)],
            'steps': [{'instruction': f'原典の工程{i}'} for i in range(3)],
            'created_at': '2026-10-03T00:00:00Z', 'updated_at': '2026-10-03T00:00:00Z'}
        raw = json.dumps(sparse, ensure_ascii=False)
        candidate = worker.parse_json(raw)
        worker.complete_required_shape(candidate, schema)
        self.assertEqual(list(Draft202012Validator(schema).iter_errors(candidate)), [])
        self.assertEqual(len(candidate['ingredients']), 6)
        self.assertEqual(len(candidate['steps']), 3)
        self.assertEqual([i['ingredient_id'] for i in candidate['ingredients']], [f'ing_{i:03d}' for i in range(1, 7)])
        self.assertEqual([i['step'] for i in candidate['steps']], [1, 2, 3])
        for item in candidate['ingredients']:
            self.assertFalse(item['optional'])
            self.assertIsNone(item['preparation'])
            self.assertIsNone(item['note'])
            self.assertIsNone(item['source_ref'])
            self.assertEqual(item['amount'], {'raw_text': '適量', 'value': None, 'unit': None})
        self.assertEqual(candidate['steps'][0]['duration'], {'value': None, 'unit': None, 'raw_text': None})
        self.assertEqual(candidate['steps'][0]['temperature'], {'value': None, 'unit': None, 'raw_text': None})
        self.assertIsNone(candidate['steps'][0]['heat'])
        self.assertEqual(candidate['source_refs'], [])
        self.assertEqual(candidate['storage'], {'refrigerated': None, 'frozen': None, 'raw_text': None})
        self.assertEqual(json.loads(raw), sparse)
        before = worker.copy.deepcopy(candidate)
        worker.complete_required_shape(candidate, schema)
        self.assertEqual(candidate, before)

    @unittest.skipUnless((ROOT.parent / 'canonical.schema.json').exists(), 'Canonical Schema is required')
    def test_shape_completion_preserves_bad_types_extra_keys_and_existing_ids(self):
        schema = json.loads((ROOT.parent / 'canonical.schema.json').read_text(encoding='utf-8'))
        candidate = {'title': 'wrong type', 'ingredients': [
            {'ingredient_id': 'ing_001', 'optional': True, 'name': {'raw': '塩'},
                'amount': {'value': '1', 'unit': 'pinch'}, 'extra': 'keep'},
            {'name': {'raw': '油'}, 'optional': None},
            {'ingredient_id': 'ing_002', 'name': {'raw': '卵'}, 'preparation': 7},
            {'ingredient_id': None, 'name': []}, {'name': {'raw': '水'}}],
            'steps': [{'step': 9, 'instruction': '原典', 'duration': 'invalid'},
                {'instruction': '原典', 'temperature': []}], 'extra': {'keep': True}}
        worker.complete_required_shape(candidate, schema)
        self.assertEqual(candidate['title'], 'wrong type')
        self.assertEqual([i['ingredient_id'] for i in candidate['ingredients']], ['ing_001', 'ing_003', 'ing_002', None, 'ing_004'])
        self.assertTrue(candidate['ingredients'][0]['optional'])
        self.assertIsNone(candidate['ingredients'][1]['optional'])
        self.assertEqual(candidate['ingredients'][0]['amount'], {'value': '1', 'unit': 'pinch', 'raw_text': None})
        self.assertEqual(candidate['ingredients'][0]['extra'], 'keep')
        self.assertEqual(candidate['ingredients'][2]['preparation'], 7)
        self.assertEqual(candidate['ingredients'][3]['name'], [])
        self.assertEqual([i['step'] for i in candidate['steps']], [9, 2])
        self.assertEqual(candidate['steps'][0]['duration'], 'invalid')
        self.assertEqual(candidate['steps'][1]['temperature'], [])
        self.assertEqual(candidate['extra'], {'keep': True})

    @unittest.skipUnless((ROOT.parent / 'canonical.schema.json').exists() and importlib.util.find_spec('jsonschema'), 'Canonical Schema and validator are required')
    def test_missing_original_text_remains_blank_and_fails_schema_validation(self):
        from jsonschema import Draft202012Validator
        schema = json.loads((ROOT.parent / 'canonical.schema.json').read_text(encoding='utf-8'))
        candidate = {'ingredients': [{}], 'steps': [{}]}
        worker.complete_required_shape(candidate, schema)
        self.assertEqual(candidate['title']['original'], '')
        self.assertEqual(candidate['ingredients'][0]['name']['raw'], '')
        self.assertEqual(candidate['steps'][0]['instruction'], '')
        paths = {tuple(error.path) for error in Draft202012Validator(schema).iter_errors(candidate)}
        self.assertIn(('title', 'original'), paths)
        self.assertIn(('ingredients', 0, 'name', 'raw'), paths)
        self.assertIn(('steps', 0, 'instruction'), paths)

    def test_bootstrap_and_failure_report_import_without_site_packages(self):
        script = 'import sys; sys.path.insert(0, ' + repr(str(ROOT / 'kaggle-worker')) + '); from worker import bootstrap, fail_batch, BatchAPI; print("stdlib-bootstrap-import-ok")'
        result = worker.subprocess.run([sys.executable, '-S', '-c', script], capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0)
        self.assertEqual(result.stdout.strip(), 'stdlib-bootstrap-import-ok')

    def test_pip_failure_reports_batch_with_safe_stage_diagnostic(self):
        output = io.StringIO()
        result = types.SimpleNamespace(returncode=1, stderr='No matching distribution found for secret-token at https://private.example')
        with patch.object(worker.subprocess, 'run', return_value=result), patch.object(worker, 'fail_batch') as report, patch.object(worker, 'run') as run, redirect_stdout(output):
            worker.bootstrap(claim(), Path('.'))
        report.assert_called_once()
        run.assert_not_called()
        self.assertIn('stage=dependency_install code=pip_no_matching_distribution exception=RuntimeError', output.getvalue())
        self.assertNotIn('secret-token', output.getvalue())
        self.assertNotIn('private.example', output.getvalue())

    def test_worker_api_error_and_failure_report_keep_only_http_status(self):
        error = AppError('worker_api_failed', 'secret-body', details={'http_status': 403, 'cause_class': 'HTTPError'})
        output = io.StringIO()
        with patch.object(worker.subprocess, 'run', return_value=types.SimpleNamespace(returncode=0)), patch.object(worker, 'run', side_effect=error), patch.object(worker, 'fail_batch', side_effect=error), redirect_stdout(output):
            worker.bootstrap(claim(), Path('.'))
        text = output.getvalue()
        self.assertIn('stage=worker_run code=worker_api_failed exception=HTTPError http_status=403', text)
        self.assertIn('stage=failure_report code=report_failed exception=HTTPError http_status=403', text)
        self.assertNotIn('secret-body', text)

    def test_api_http_error_has_safe_stage_and_preserved_numeric_status(self):
        output = io.StringIO()
        opener = types.SimpleNamespace(open=lambda *a, **k: (_ for _ in ()).throw(worker.urllib.error.HTTPError('https://private.example/secret', 401, 'private-secret-body', {}, None)))
        with patch.object(worker.urllib.request, 'build_opener', return_value=opener), redirect_stdout(output):
            with self.assertRaises(AppError) as caught:
                worker.BatchAPI(claim()).request('/start', {})
        self.assertEqual(caught.exception.details['http_status'], 401)
        self.assertIn('stage=api_start code=worker_api_failed exception=HTTPError http_status=401', output.getvalue())
        self.assertNotIn('private', output.getvalue())

    def test_api_network_error_keeps_safe_reason_class(self):
        import ssl
        output = io.StringIO()
        error = worker.urllib.error.URLError(ssl.SSLCertVerificationError('private-source-token'))
        opener = types.SimpleNamespace(open=lambda *a, **k: (_ for _ in ()).throw(error))
        with patch.object(worker.urllib.request, 'build_opener', return_value=opener), patch.object(worker.time, 'sleep'), redirect_stdout(output):
            with self.assertRaises(AppError) as caught:
                worker.BatchAPI(claim()).request()
            worker.safe_diagnostic('failure_report', caught.exception, 'report_failed')
        self.assertEqual(caught.exception.details['reason_class'], 'SSLCertVerificationError')
        self.assertIn('stage=api_get code=worker_api_failed exception=URLError reason_class=SSLCertVerificationError', output.getvalue())
        self.assertIn('stage=failure_report code=report_failed exception=URLError reason_class=SSLCertVerificationError', output.getvalue())
        self.assertNotIn('private-source-token', output.getvalue())

    def test_api_has_dedicated_user_agent_and_accept_headers(self):
        captured = []
        def open_request(request, **kwargs):
            captured.append(request)
            return io.BytesIO(b'{"ok":true}')
        opener = types.SimpleNamespace(open=open_request)
        with patch.object(worker.urllib.request, 'build_opener', return_value=opener):
            worker.BatchAPI(claim()).request()
            worker.BatchAPI(claim()).request('/inputs/job-1', binary=True)
        self.assertTrue(all(request.get_header('User-agent') == 'PersonalRecipeKB-worker/1.0' for request in captured))
        self.assertEqual(captured[0].get_header('Accept'), 'application/json')
        self.assertEqual(captured[1].get_header('Accept'), 'application/octet-stream')

    def test_http_body_classification_emits_only_fixed_labels(self):
        cases = [({'cf-mitigated': 'challenge'}, b'private-token', 'edge_challenge', None),
            ({'content-type': 'application/json'}, b'{"error":{"code":"unauthorized","message":"private-token"}}', 'application_json', 'unauthorized'),
            ({'content-type': 'application/json'}, b'{"error":{"code":"private-token"}}', 'application_json', None),
            ({'content-type': 'text/html'}, b'<title>Just a moment</title>cloudflare private-token', 'edge_challenge', None),
            ({'content-type': 'text/html'}, b'Cloudflare Ray ID private-token', 'cloudflare_html', None)]
        for headers, body, kind, application_code in cases:
            with self.subTest(kind=kind, application_code=application_code):
                error = worker.urllib.error.HTTPError('https://private.example/token', 403, 'private-token', headers, io.BytesIO(body))
                details = worker.classify_http_error(error)
                self.assertEqual(details.get('response_kind'), kind)
                self.assertEqual(details.get('application_error_code'), application_code)
                output = io.StringIO()
                with redirect_stdout(output):
                    worker.safe_diagnostic('api_get', error, 'worker_api_failed', 403, details)
                self.assertNotIn('private', output.getvalue())

    def test_strict_json_and_markdown_json(self):
        self.assertEqual(worker.parse_json('```json\n{"title":null}\n```'), {'title': None})
        for raw in ['[1]', '{"quantity":NaN}', 'Here is your recipe: {}']:
            with self.subTest(raw=raw), self.assertRaises(AppError):
                worker.parse_json(raw)

    def test_thinking_tags_are_rejected_without_stripping_raw_output(self):
        for raw in ['<think>推論</think>{"title":null}', '<think></think>{"title":null}',
                '{"title":null}</think>', '```json\n<think>推論</think>{}\n```']:
            with self.subTest(raw=raw), self.assertRaises(AppError) as error:
                worker.parse_json(raw)
            self.assertEqual(error.exception.code, 'llm_not_json')
            self.assertEqual(error.exception.details['raw_output'], raw)

    def test_gpu_absent_never_runs_cpu(self):
        fake_torch = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False))
        with patch.dict(sys.modules, {'torch': fake_torch}), self.assertRaisesRegex(AppError, '無料T4'):
            worker.check_gpu()

    def test_non_t4_gpu_rejected(self):
        fake_torch = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: True, get_device_name=lambda _: 'A100'))
        with patch.dict(sys.modules, {'torch': fake_torch}), self.assertRaises(AppError):
            worker.check_gpu()

    def test_evidence_timestamps_are_original(self):
        evidence = [{'ref_id': 'src_001', 'type': 'youtube_transcript', 'text': '塩を少々入れる', 'start_seconds': 5, 'end_seconds': 8}]
        candidate = {'source_refs': [{'ref_id': 'src_001', 'type': 'other', 'text': '塩を少々', 'start_seconds': 99, 'end_seconds': 100}],
            'ingredients': [{'source_ref': 'src_001'}], 'steps': []}
        worker.enforce_evidence(candidate, evidence)
        self.assertEqual(candidate['source_refs'][0]['start_seconds'], 5)
        candidate['source_refs'][0]['text'] = '砂糖を100g入れる'
        with self.assertRaises(AppError):
            worker.enforce_evidence(candidate, evidence)
        candidate['source_refs'][0]['ref_id'] = 'invented'
        with self.assertRaises(AppError):
            worker.enforce_evidence(candidate, evidence)

    def test_malformed_evidence_values_are_preserved_for_schema_review(self):
        evidence = [{'ref_id': 'src_001', 'type': 'text_span', 'text': '塩を少々入れる',
            'start_seconds': None, 'end_seconds': None}]
        for invalid in [{'private': 'value'}, ['src_001'], 1, None]:
            for field in ['ref_id', 'text']:
                ref = {'ref_id': 'src_001', 'text': '塩を少々', 'type': 'unchanged',
                    'start_seconds': 99, 'end_seconds': 100}
                ref[field] = invalid
                candidate = {'source_refs': [ref], 'ingredients': [{'source_ref': invalid}],
                    'steps': [{'source_ref': invalid}]}
                # null text is a valid unknown quote; malformed values must not be rewritten.
                if field == 'text' and invalid is None:
                    continue
                before = copy.deepcopy(candidate)
                with self.subTest(field=field, value=invalid):
                    worker.enforce_evidence(candidate, evidence)
                    self.assertEqual(candidate, before)
        for invalid in [{'ref_id': 'src_001'}, 'src_001', None, 1]:
            candidate = {'source_refs': invalid, 'ingredients': [], 'steps': []}
            before = copy.deepcopy(candidate)
            with self.subTest(source_refs=invalid):
                worker.enforce_evidence(candidate, evidence)
                self.assertEqual(candidate, before)

    def test_malformed_reference_does_not_hide_unknown_string_reference(self):
        evidence = [{'ref_id': 'src_001', 'type': 'text_span', 'text': '塩を少々',
            'start_seconds': None, 'end_seconds': None}]
        cases = [{'source_refs': [{'ref_id': {}, 'text': []}, {'ref_id': 'unknown'}]},
            {'source_refs': {}, 'ingredients': [{'source_ref': []}, {'source_ref': 'unknown'}]},
            {'source_refs': [], 'steps': [{'source_ref': {}}, {'source_ref': 'unknown'}]},
            {'source_refs': [{'ref_id': 'src_001', 'text': '砂糖を100g'}]}]
        for candidate in cases:
            with self.subTest(candidate=candidate), self.assertRaises(AppError) as error:
                worker.enforce_evidence(candidate, evidence)
            self.assertEqual(error.exception.code, 'invalid_evidence')

    @unittest.skipUnless((ROOT.parent / 'canonical.schema.json').exists() and importlib.util.find_spec('jsonschema'), 'Canonical Schema and validator are required')
    def test_malformed_references_reach_review_but_fail_canonical_validation(self):
        from jsonschema import Draft202012Validator
        schema = json.loads((ROOT.parent / 'canonical.schema.json').read_text(encoding='utf-8'))
        evidence = [{'ref_id': 'src_001', 'type': 'text_span', 'text': '塩を少々',
            'start_seconds': None, 'end_seconds': None}]
        candidate = {'title': {'original': '料理'}, 'ingredients': [
                {'name': {'raw': '塩'}, 'source_ref': {'ref_id': 'src_001'}}],
            'steps': [{'instruction': '塩を少々', 'source_ref': ['src_001']}],
            'source_refs': [{'ref_id': {'id': 'src_001'}, 'text': '塩'},
                {'ref_id': 'src_001', 'text': {'quote': '塩'}}]}
        raw = json.dumps(candidate, ensure_ascii=False)
        worker.complete_required_shape(candidate, schema)
        before = copy.deepcopy(candidate)
        worker.enforce_evidence(candidate, evidence)
        self.assertEqual(candidate, before)
        paths = {tuple(error.path) for error in Draft202012Validator(schema).iter_errors(candidate)}
        for path in [('ingredients', 0, 'source_ref'), ('steps', 0, 'source_ref'),
                ('source_refs', 0, 'ref_id'), ('source_refs', 1, 'text')]:
            self.assertIn(path, paths)
        self.assertEqual(json.loads(raw)['source_refs'][1]['text'], {'quote': '塩'})

    def test_unknown_models_cannot_be_external_endpoints(self):
        with self.assertRaises(AppError):
            worker.model_name('https://paid.example/inference', 'Qwen/Qwen2.5-3B-Instruct')

    def test_batch_api_is_scoped_and_expiring(self):
        api = worker.BatchAPI(claim())
        self.assertEqual(api.url, 'https://recipe.example/api/worker/batches/batch-1')
        data = claim()
        data['batch_id'] = '../other'
        with self.assertRaises(AppError):
            worker.BatchAPI(data)

    def test_six_large_japanese_inputs_fit_batch_response_limit(self):
        payload = json.dumps({'jobs': [{'input': {'value': '材' * 120000}} for _ in range(6)]}, ensure_ascii=False).encode()
        self.assertGreater(len(payload), 2 * 1024 * 1024)
        opener = types.SimpleNamespace(open=lambda *args, **kwargs: io.BytesIO(payload))
        with patch.object(worker.urllib.request, 'build_opener', return_value=opener):
            batch = worker.BatchAPI(claim()).request()
        self.assertEqual(len(batch['jobs']), 6)

    def test_runtime_reserves_three_minutes_before_token_expiry(self):
        class FakeAPI:
            expires = 2000
            def __init__(self, config): pass
            def request(self, suffix='', payload=None):
                return {'batch_id': 'batch-1', 'jobs': [], 'schema': {},
                    'config': batch_config, 'extraction_prompt': 'rules'} if suffix == '' else {}
        cases = [({'max_batch_seconds': 3600}, 2000, 1820),
            ({'llm_model': worker.QWEN3_AWQ_MODEL, 'max_batch_seconds': 6600}, 9000, 7300),
            ({'llm_model': worker.QWEN3_AWQ_MODEL, 'max_batch_seconds': 3600}, 9000, 4600),
            ({'llm_model': worker.NF4_MODEL, 'max_batch_seconds': 6600}, 9000, 6400),
            ({'llm_model': worker.NF4_MODEL, 'max_batch_seconds': 3600}, 9000, 4600)]
        for batch_config, expiry, expected_deadline in cases:
            FakeAPI.expires = expiry
            with self.subTest(config=batch_config, expiry=expiry), tempfile.TemporaryDirectory() as temporary, patch.object(worker, 'BatchAPI', FakeAPI), patch.object(worker, 'Models') as models, patch.object(worker.time, 'time', return_value=1000), patch.object(worker, 'check_gpu', side_effect=AppError('gpu_unavailable', 'GPUなし')):
                worker.run(claim(), Path(temporary))
            self.assertEqual(models.call_args.args[-1], expected_deadline)

    def test_invalid_candidate_title_does_not_crash_fallback_decision(self):
        self.assertTrue(worker.needs_more({'title': 'wrong type', 'ingredients': [{}], 'steps': [{}]}))

    def test_cloud_worker_size_limits_match(self):
        settings = worker.settings_from({'max_input_chars': 120000, 'max_video_seconds': 2700}, Path('.'))
        self.assertEqual(settings.max_input_chars, 120000)
        self.assertEqual(settings.max_video_seconds, 2700)
        self.assertEqual(worker.MAX_IMAGE_BYTES, 15 * 1024 * 1024)
        low = worker.settings_from({'max_input_chars': 1234, 'max_video_seconds': 123}, Path('.'))
        self.assertEqual((low.max_input_chars, low.max_video_seconds), (1234, 123))

    def test_frame_seconds_are_at_most_six_finite_numbers(self):
        self.assertEqual(worker.frame_seconds([0, 1.5, 2, 3, 4, 2700], 2700), [0, 1.5, 2, 3, 4, 2700])
        for value in ['1,2', [True], [float('nan')], [float('inf')], [-1], [2701], [0]*7, ['1']]:
            with self.subTest(value=value), self.assertRaises(AppError):
                worker.frame_seconds(value, 2700)

    def test_error_does_not_expose_exception_data(self):
        result = worker.error_result({'job_id': 'job-1'}, RuntimeError('secret-source-and-token'))
        self.assertNotIn('secret-source-and-token', json.dumps(result))
        self.assertTrue(result['error']['retryable'])

    def test_runtime_failures_have_distinct_retry_codes_without_private_data(self):
        cases = [(RuntimeError('CUDA out of memory private-token'), 'gpu_memory'),
            (RuntimeError('libcudnn.so not found private-token'), 'cuda_library'),
            (RuntimeError('model download failed https://private.example'), 'model_download_failed'),
            (TypeError("unexpected keyword argument 'private-token'"), 'model_api_incompatible'),
            (TypeError("'NoneType' object is not iterable private-token"), 'model_input_missing'),
            (RuntimeError('private-token https://private.example'), 'model_load_failed')]
        for error, code in cases:
            with self.subTest(code=code):
                result = worker.error_result({'job_id': 'job-1'}, error)
                self.assertEqual(result['error']['code'], code)
                self.assertTrue(result['error']['retryable'])
                self.assertEqual(result['error']['message'], '無料の解析環境で処理に失敗しました。時間を置いて再試行します。')
                self.assertNotIn('private-token', json.dumps(result))
                self.assertNotIn('https://', json.dumps(result))

    @unittest.skipUnless(importlib.util.find_spec('jsonschema') and importlib.util.find_spec('trafilatura') and importlib.util.find_spec('youtube_transcript_api'), 'Extractor test dependencies are unavailable')
    def test_job_failure_stage_and_exception_class_only(self):
        from workerlib.extractors import text, youtube
        extraction = {'evidence': [{'ref_id': 'src_001', 'type': 'youtube_description', 'text': '一部だけ', 'start_seconds': None, 'end_seconds': None}],
            'images': [], 'warnings': [], 'video_url': 'https://www.youtube.com/watch?v=abcdefghijk'}
        incomplete = {'title': {'original': '料理'}, 'ingredients': [], 'steps': []}
        for stage in ['extract', 'structure', 'asr', 'frames']:
            with self.subTest(stage=stage), tempfile.TemporaryDirectory() as temporary:
                private_error = RuntimeError('private-source-token https://private.example')
                models = types.SimpleNamespace(schema={}, structure=lambda *_: (incomplete, '{}'),
                    transcribe=lambda *_: (_ for _ in ()).throw(private_error))
                kind = 'text' if stage in ['extract', 'structure'] else 'youtube'
                job = {'job_id': 'job-1', 'input_type': kind, 'input': {'value': '原典',
                    'allow_asr': stage == 'asr', 'frame_seconds': [1] if stage == 'frames' else []}, 'template': {'source': {}}}
                if stage == 'structure':
                    models.structure = lambda *_: (_ for _ in ()).throw(private_error)
                output = io.StringIO()
                with patch.object(text, 'extract_text', side_effect=private_error if stage == 'extract' else None, return_value=extraction), patch.object(youtube, 'extract_youtube', return_value=extraction), patch.object(youtube, 'download_audio', return_value=Path('unused.mp3')), patch.object(youtube, 'frames', side_effect=private_error), redirect_stdout(output):
                    with self.assertRaises(RuntimeError):
                        worker.process_job(job, None, models, worker.settings_from({}, Path(temporary)), Path(temporary))
                self.assertEqual(output.getvalue().strip(), f'recipe_job_diag stage={stage} exception=RuntimeError reason=runtime_other last_frame=other')

    def test_job_diagnostic_keeps_cuda_exception_class_without_message(self):
        OutOfMemoryError = type('OutOfMemoryError', (RuntimeError,), {})
        output = io.StringIO()
        with redirect_stdout(output):
            worker.safe_job_diagnostic('structure', OutOfMemoryError('secret tensor/source details'))
        self.assertEqual(output.getvalue().strip(), 'recipe_job_diag stage=structure exception=OutOfMemoryError reason=gpu_memory last_frame=no_traceback')

    def test_job_reason_labels_never_expose_error_text(self):
        cases = [('CUDA out of memory: private-token https://private.example', 'gpu_memory'),
            ('CUDA failed with error out of memory private-token', 'gpu_memory'),
            ('Library libcudnn_ops.so.9 not found private-token', 'cuda_library'),
            ('Could not load libcublas.so private-token', 'cuda_library'),
            ("Couldn't connect to 'https://huggingface.co' private-token", 'model_download'),
            ('private-token https://private.example arbitrary failure', 'runtime_other')]
        for message, reason in cases:
            with self.subTest(reason=reason):
                output = io.StringIO()
                with redirect_stdout(output):
                    worker.safe_job_diagnostic('asr', RuntimeError(message))
                self.assertEqual(output.getvalue().strip(), f'recipe_job_diag stage=asr exception=RuntimeError reason={reason} last_frame=no_traceback')
                self.assertNotIn('private-token', output.getvalue())
                self.assertNotIn('https://', output.getvalue())

    def test_typeerror_reason_and_last_frame_only_emit_fixed_labels(self):
        cases = [("got an unexpected keyword argument 'private-token'", 'api_signature'),
            ('incompatible function arguments: https://private.example', 'api_signature'),
            ("'NoneType' object is not iterable private-token", 'missing_value'),
            ("argument must be str, not NoneType private-token", 'missing_value'),
            ('private-token https://private.example arbitrary failure', 'runtime_other')]
        for message, reason in cases:
            def transcribe():
                raise TypeError(message)
            output = io.StringIO()
            with self.subTest(reason=reason), redirect_stdout(output):
                try:
                    transcribe()
                except TypeError as error:
                    worker.safe_job_diagnostic('asr', error)
            self.assertEqual(output.getvalue().strip(), f'recipe_job_diag stage=asr exception=TypeError reason={reason} last_frame=transcribe')
            self.assertNotIn('private-token', output.getvalue())
            self.assertNotIn('https://', output.getvalue())
        scope = {}
        exec(compile('def private_token_function():\n raise TypeError("private-token")',
            'https://private.example/secret-file', 'exec'), scope)
        try:
            scope['private_token_function']()
        except TypeError as error:
            output = io.StringIO()
            with redirect_stdout(output):
                worker.safe_job_diagnostic('asr', error)
            self.assertTrue(output.getvalue().strip().endswith('last_frame=other'))
            self.assertNotIn('private', output.getvalue())

    def test_evidence_and_hub_frame_labels_never_expose_paths_or_error_values(self):
        labels = {'enforce_evidence': 'enforce_evidence', 'snapshot_download': 'snapshot_download',
            'hf_hub_download': 'hf_hub_download', '_inner_fn': 'hub_wrapper',
            'download_model': 'download_model', 'snapshot_download_private_token': 'other'}
        for function, label in labels.items():
            scope = {}
            source = 'def ' + function + '():\n raise TypeError("unexpected keyword argument private-token https://private.example/input")'
            exec(compile(source, 'C:/private-token/private-input-file.py', 'exec'), scope)
            output = io.StringIO()
            with self.subTest(function=function), redirect_stdout(output):
                try:
                    scope[function]()
                except TypeError as error:
                    worker.safe_job_diagnostic('asr', error)
            self.assertEqual(output.getvalue().strip(),
                'recipe_job_diag stage=asr exception=TypeError reason=api_signature last_frame=' + label)
            for forbidden in ['private-token', 'https://', 'C:/', 'private-input-file', 'Traceback']:
                self.assertNotIn(forbidden, output.getvalue())

    def test_transcribe_uses_official_gpu_signature_and_consumes_segment_generator(self):
        calls = []
        consumed = []
        class WhisperModel:
            def __init__(self, model_size_or_path, device='auto', device_index=0,
                    compute_type='default', download_root=None):
                calls.append((model_size_or_path, device, device_index, compute_type, download_root))
            def transcribe(self, audio, beam_size=5, vad_filter=False):
                calls.append((audio, beam_size, vad_filter))
                def segments():
                    consumed.append(True)
                    yield types.SimpleNamespace(text='原典音声', start=2.5, end=4.0)
                return segments(), None
        models = worker.Models({}, {}, 'rules', Path('.'), 10000)
        with patch.dict(sys.modules, {'faster_whisper': types.SimpleNamespace(WhisperModel=WhisperModel)}), patch.object(models, 'unload'):
            result = models.transcribe(Path('test.mp3'))
        self.assertEqual(calls[0][:4], ('small', 'cuda', 0, 'int8_float16'))
        self.assertEqual(calls[1], ('test.mp3', 5, True))
        self.assertEqual(consumed, [True])
        self.assertEqual(result, [{'text': '原典音声', 'start': 2.5, 'end': 4.0}])

    def test_quota_failure_posts_all_results_and_finishes(self):
        records = []
        class FakeAPI:
            expires = __import__('time').time() + 7000
            def __init__(self, config): pass
            def request(self, suffix='', payload=None):
                records.append((suffix, payload))
                if suffix == '':
                    return {'batch_id': 'batch-1', 'jobs': [{'job_id': 'job-1'}, {'job_id': 'job-2'}],
                        'schema': {}, 'config': {}, 'extraction_prompt': 'faithful extraction only'}
                return {}
        with tempfile.TemporaryDirectory() as temporary, patch.object(worker, 'BatchAPI', FakeAPI), patch.object(worker, 'check_gpu', side_effect=AppError('gpu_unavailable', '無料GPUなし')):
            worker.run(claim(), Path(temporary))
        results = [item[1] for item in records if item[0] == '/results']
        self.assertEqual(len(results), 2)
        self.assertTrue(all(item['error']['retryable'] for item in results))
        self.assertEqual(records[-1][0], '/finish')

    @unittest.skipUnless(importlib.util.find_spec('jsonschema') and importlib.util.find_spec('trafilatura') and importlib.util.find_spec('youtube_transcript_api'), 'Extractor test dependencies are unavailable')
    def test_youtube_complete_text_needs_no_audio_or_frames(self):
        from workerlib.extractors import youtube
        extraction = {'evidence': [{'ref_id': 'src_001', 'type': 'youtube_description', 'text': '材料と手順', 'start_seconds': None, 'end_seconds': None}],
            'warnings': [], 'images': [], 'video_url': 'https://www.youtube.com/watch?v=abcdefghijk'}
        models = types.SimpleNamespace(schema={}, structure=lambda *_: ({'title': {'original': '料理'}, 'ingredients': [{}], 'steps': [{}]}, '{}'))
        job = {'job_id': 'job-1', 'input_type': 'youtube', 'input': {'value': extraction['video_url'], 'allow_asr': True, 'frame_seconds': [3]}, 'template': {'source': {}}}
        with tempfile.TemporaryDirectory() as temporary, patch.object(youtube, 'extract_youtube', return_value=extraction), patch.object(youtube, 'download_audio') as audio, patch.object(youtube, 'frames') as frames:
            root = Path(temporary)
            worker.process_job(job, None, models, worker.settings_from({}, root), root)
            audio.assert_not_called()
            frames.assert_not_called()

    @unittest.skipUnless(importlib.util.find_spec('jsonschema') and importlib.util.find_spec('trafilatura') and importlib.util.find_spec('youtube_transcript_api'), 'Extractor test dependencies are unavailable')
    def test_youtube_failed_asr_keeps_incomplete_candidate_for_review(self):
        from workerlib.extractors import youtube
        extraction = {'evidence': [{'ref_id': 'src_001', 'type': 'youtube_description', 'text': '原典', 'start_seconds': None, 'end_seconds': None}],
            'warnings': [], 'images': [], 'video_url': 'https://www.youtube.com/watch?v=abcdefghijk'}
        candidate = {'title': {'original': '原典の料理名'}, 'ingredients': [], 'steps': []}
        models = types.SimpleNamespace(schema={}, structure=lambda *_: (candidate, '{}'))
        job = {'job_id': 'job-1', 'input_type': 'youtube', 'input': {'value': extraction['video_url'], 'allow_asr': True}, 'template': {'source': {}}}
        with tempfile.TemporaryDirectory() as temporary, patch.object(youtube, 'extract_youtube', return_value=extraction), patch.object(youtube, 'download_audio', side_effect=AppError('asr_failed', '取得失敗')), patch.object(youtube, 'frames') as frames:
            root = Path(temporary)
            result = worker.process_job(job, None, models, worker.settings_from({}, root), root)
            self.assertEqual(result['candidate'], candidate)
            frames.assert_not_called()

    @unittest.skipUnless(importlib.util.find_spec('jsonschema') and importlib.util.find_spec('trafilatura') and importlib.util.find_spec('youtube_transcript_api'), 'Extractor test dependencies are unavailable')
    def test_youtube_invalid_json_uses_asr_and_preserves_metadata(self):
        from workerlib.extractors import youtube
        extraction = {'evidence': [{'ref_id': 'src_001', 'type': 'youtube_transcript', 'text': '字幕の一部', 'start_seconds': 0, 'end_seconds': 1}],
            'warnings': [], 'images': [], 'video_url': 'https://www.youtube.com/watch?v=abcdefghijk',
            'title': '取得した動画名', 'creator': '取得したchannel', 'source_id': 'abcdefghijk'}
        seen_templates = []
        def structure(extracted, template):
            seen_templates.append(template)
            if len(seen_templates) == 1:
                raise AppError('llm_not_json', 'JSONではありません', details={'raw_output': 'bad output'})
            return {'title': {'original': '料理'}, 'ingredients': [{}], 'steps': [{}], 'source': template['source']}, '{}'
        models = types.SimpleNamespace(schema={}, structure=structure, transcribe=lambda _: [{'text': '音声の材料と手順', 'start': 2, 'end': 3}])
        job = {'job_id': 'job-1', 'input_type': 'youtube', 'input': {'value': extraction['video_url'], 'allow_asr': True}, 'template': {'source': {'title': None, 'creator': None}}}
        with tempfile.TemporaryDirectory() as temporary, patch.object(youtube, 'extract_youtube', return_value=extraction), patch.object(youtube, 'download_audio', return_value=Path('unused.mp3')) as audio, patch.object(youtube, 'frames') as frames:
            root = Path(temporary)
            result = worker.process_job(job, None, models, worker.settings_from({}, root), root)
            audio.assert_called_once()
            frames.assert_not_called()
            self.assertEqual(result['candidate']['source']['title'], '取得した動画名')
            self.assertEqual(result['candidate']['source']['creator'], '取得したchannel')
            self.assertEqual(result['evidence'][-1]['type'], 'youtube_speech')

    @unittest.skipUnless(importlib.util.find_spec('jsonschema') and importlib.util.find_spec('trafilatura') and importlib.util.find_spec('youtube_transcript_api'), 'Extractor test dependencies are unavailable')
    def test_youtube_generation_cutoff_does_not_request_audio_or_frames(self):
        from workerlib.extractors import youtube
        extraction = {'evidence': [{'ref_id': 'src_001', 'type': 'youtube_transcript', 'text': '原典', 'start_seconds': 0, 'end_seconds': 1}],
            'warnings': [], 'images': [], 'video_url': 'https://www.youtube.com/watch?v=abcdefghijk'}
        failure = AppError('llm_not_json', 'JSON途中切れ', details={'raw_output': '{', 'generation_limit_reached': True})
        models = types.SimpleNamespace(schema={}, structure=Mock(side_effect=failure))
        job = {'job_id': 'job-1', 'input_type': 'youtube', 'input': {'value': extraction['video_url'], 'allow_asr': True, 'frame_seconds': [3]}, 'template': {'source': {}}}
        with tempfile.TemporaryDirectory() as temporary, patch.object(youtube, 'extract_youtube', return_value=extraction), patch.object(youtube, 'download_audio') as audio, patch.object(youtube, 'frames') as frames, redirect_stdout(io.StringIO()):
            root = Path(temporary)
            with self.assertRaises(AppError) as caught:
                worker.process_job(job, None, models, worker.settings_from({}, root), root)
            self.assertIs(caught.exception, failure)
            audio.assert_not_called()
            frames.assert_not_called()

    @unittest.skipUnless(importlib.util.find_spec('jsonschema') and importlib.util.find_spec('trafilatura') and importlib.util.find_spec('youtube_transcript_api'), 'Extractor test dependencies are unavailable')
    def test_six_explicit_frames_only_for_incomplete_candidate(self):
        from workerlib.extractors import youtube
        extraction = {'evidence': [{'ref_id': 'src_001', 'type': 'youtube_transcript', 'text': '一部だけ', 'start_seconds': 0, 'end_seconds': 1}],
            'warnings': [], 'images': [], 'video_url': 'https://www.youtube.com/watch?v=abcdefghijk'}
        calls = []
        def structure(extracted, _):
            calls.append(len(extracted['images']))
            return {'title': {'original': '料理'}, 'ingredients': [{}] if extracted['images'] else [], 'steps': [{}]}, '{}'
        models = types.SimpleNamespace(schema={}, structure=structure)
        seconds = [0, 5, 10, 20, 30, 2700]
        job = {'job_id': 'job-1', 'input_type': 'youtube', 'input': {'value': extraction['video_url'], 'allow_asr': False, 'frame_seconds': seconds}, 'template': {'source': {}}}
        images = [{'path': f'frame-{i}.jpg', 'second': second} for i, second in enumerate(seconds)]
        with tempfile.TemporaryDirectory() as temporary, patch.object(youtube, 'extract_youtube', return_value=extraction), patch.object(youtube, 'download_audio') as audio, patch.object(youtube, 'frames', return_value=images) as frames:
            root = Path(temporary)
            result = worker.process_job(job, None, models, worker.settings_from({}, root), root)
            self.assertEqual(calls, [0, 6])
            self.assertEqual(frames.call_args.args[1], seconds)
            self.assertEqual(result['evidence'][-1]['start_seconds'], 2700)
            audio.assert_not_called()


if __name__ == '__main__':
    unittest.main()
