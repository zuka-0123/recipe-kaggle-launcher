"""CPU-only contract tests. Models, network and Kaggle submission are mocked."""
import datetime as dt
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

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
    def test_public_repository_only(self):
        with patch.object(launcher, 'request_json', return_value={'private': True, 'visibility': 'private'}):
            with self.assertRaisesRegex(launcher.LaunchError, 'public_repository_required'):
                launcher.public_repository('owner/repo', 'token')

    def test_running_and_queued_notebooks_block_launch(self):
        for status in ['running', 'queued', 'mystery']:
            with self.subTest(status=status), patch.object(launcher.subprocess, 'run', return_value=types.SimpleNamespace(returncode=0, stdout=f'owner/worker has status "{status}"')) as call:
                with self.assertRaises(launcher.LaunchError):
                    launcher.launch(claim(), 'owner', 'recipe-worker')
                self.assertEqual(call.call_count, 1)

    def test_private_t4_and_temporary_source_only(self):
        inspected = []
        def cli(command, **kwargs):
            if 'status' in command:
                return types.SimpleNamespace(returncode=0, stdout='owner/worker has status "complete"')
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

    def test_urls_do_not_redirect_credentials(self):
        self.assertIsNone(launcher.NoRedirect().redirect_request(None, None, None, None, None, None))
        for url in ['http://recipe.example', 'https://token@recipe.example', 'https://recipe.example/api', 'https://recipe.example/?x=1']:
            with self.subTest(url=url), self.assertRaises(launcher.LaunchError):
                launcher.base_url(url)


class WorkerTests(unittest.TestCase):
    def test_strict_json_and_markdown_json(self):
        self.assertEqual(worker.parse_json('```json\n{"title":null}\n```'), {'title': None})
        for raw in ['[1]', '{"quantity":NaN}', 'Here is your recipe: {}']:
            with self.subTest(raw=raw), self.assertRaises(AppError):
                worker.parse_json(raw)

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
