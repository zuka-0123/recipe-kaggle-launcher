"""One batch, one private Kaggle GPU session. No external inference API."""
import copy
import datetime as dt
import gc
import json
import logging
import math
import os
from pathlib import Path
import re
import shutil
import signal
import time
from types import SimpleNamespace
import urllib.error
import urllib.request
from urllib.parse import urlsplit

from workerlib.errors import AppError

MAX_IMAGE_BYTES = 15 * 1024 * 1024
MAX_BATCH_JSON_BYTES = 8 * 1024 * 1024
MAX_OUTPUT_CHARS = 160_000


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


class BatchTimeout(Exception):
    pass


def parse_json(raw):
    text = raw.strip()
    if text.startswith('```') and text.endswith('```'):
        text = '\n'.join(text.splitlines()[1:-1])
    try:
        value = json.loads(text, parse_constant=lambda _: (_ for _ in ()).throw(ValueError()))
        if not isinstance(value, dict):
            raise ValueError()
        return value
    except (ValueError, TypeError):
        raise AppError('llm_not_json', 'AI出力をJSONとして読み取れませんでした。', 422,
            {'raw_output': raw[:MAX_OUTPUT_CHARS]}) from None


def valid_id(value):
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,100}', value):
        raise AppError('invalid_job', 'batchまたはjob IDが不正です。')
    return value


class BatchAPI:
    def __init__(self, config):
        origin = config['worker_base_url'].rstrip('/')
        url = urlsplit(origin)
        if url.scheme != 'https' or not url.hostname or url.username or url.password or url.path or url.query or url.fragment:
            raise AppError('invalid_worker_url', 'Worker APIの設定が不正です。')
        self.url = origin + '/api/worker/batches/' + valid_id(config['batch_id'])
        self.token = config['worker_token']
        self.expires = dt.datetime.fromisoformat(config['expires_at'].replace('Z', '+00:00')).timestamp()
        if not 0 < self.expires - time.time() <= 7200:
            raise AppError('worker_token_expired', 'batch tokenの期限が切れています。')

    def request(self, suffix='', payload=None, binary=False):
        if time.time() >= self.expires:
            raise AppError('worker_token_expired', 'batch tokenの期限が切れています。')
        request = urllib.request.Request(self.url + suffix,
            headers={'Authorization': 'Bearer ' + self.token, 'Content-Type': 'application/json'},
            data=None if payload is None else json.dumps(payload, ensure_ascii=False, allow_nan=False).encode())
        maximum = MAX_IMAGE_BYTES if binary else MAX_BATCH_JSON_BYTES
        for attempt in range(3):
            try:
                with urllib.request.build_opener(NoRedirect()).open(request, timeout=40) as response:
                    body = response.read(maximum + 1)
                    if len(body) > maximum:
                        raise AppError('input_too_large', 'API入力が上限を超えています。')
                    return body if binary else json.loads(body)
            except urllib.error.HTTPError as error:
                if error.code not in [429, 500, 502, 503, 504] or attempt == 2:
                    raise AppError('worker_api_failed', 'batch APIとの通信に失敗しました。', 502) from None
            except (urllib.error.URLError, ValueError):
                if attempt == 2:
                    raise AppError('worker_api_failed', 'batch APIとの通信に失敗しました。', 502) from None
            time.sleep(2 ** attempt)
        raise AppError('worker_api_failed', 'batch APIとの通信に失敗しました。', 502)


def check_gpu():
    import torch
    if not torch.cuda.is_available() or 'T4' not in torch.cuda.get_device_name(0):
        raise AppError('gpu_unavailable', '無料T4 GPUを利用できません。後で再試行します。', 503)


def settings_from(config, root):
    return SimpleNamespace(max_input_chars=min(int(config.get('max_input_chars', 120000)), 120000),
        max_video_seconds=min(int(config.get('max_video_seconds', 2700)), 2700),
        data_dir=root, ffmpeg=shutil.which('ffmpeg'))


def frame_seconds(value, maximum):
    if value is None:
        return []
    if not isinstance(value, list) or len(value) > 6 or any(
        not isinstance(second, (int, float)) or isinstance(second, bool) or
        not math.isfinite(second) or not 0 <= second <= maximum for second in value):
        raise AppError('invalid_frame_seconds', 'フレームは動画上限内の最大6か所を数値で指定してください。')
    return value


def model_name(value, default):
    value = value or default
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+', value):
        raise AppError('invalid_model', 'オープンウェイトmodel IDの設定が不正です。')
    return value


class Models:
    """Only one LLM, VLM or ASR is resident at a time; retain across equal jobs."""
    def __init__(self, config, schema, rules, root, deadline):
        self.config, self.schema, self.rules = config, schema, rules
        self.root, self.deadline = root, deadline
        self.kind = self.model = self.processor = None

    def unload(self):
        self.model = self.processor = None
        self.kind = None
        gc.collect()
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def load(self, kind):
        if self.kind == kind:
            return
        self.unload()
        import torch
        from transformers import AutoTokenizer, AutoProcessor, AutoModelForCausalLM, Qwen2_5_VLForConditionalGeneration
        common = {'cache_dir': str(self.root / 'models'), 'trust_remote_code': False,
            'use_safetensors': True, 'torch_dtype': torch.float16,
            'device_map': {'': 'cuda:0'}, 'attn_implementation': 'sdpa'}
        if kind == 'llm':
            name = model_name(self.config.get('llm_model'), 'Qwen/Qwen2.5-3B-Instruct')
            self.processor = AutoTokenizer.from_pretrained(name, trust_remote_code=False, cache_dir=common['cache_dir'])
            self.model = AutoModelForCausalLM.from_pretrained(name, **common)
        elif kind == 'vlm':
            name = model_name(self.config.get('vlm_model'), 'Qwen/Qwen2.5-VL-3B-Instruct')
            self.processor = AutoProcessor.from_pretrained(name, trust_remote_code=False,
                cache_dir=common['cache_dir'], min_pixels=256*28*28, max_pixels=1280*28*28)
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(name, **common)
        else:
            raise AppError('invalid_model', 'model種別が不正です。')
        self.model.eval()
        self.kind = kind

    def structure(self, extraction, template):
        import torch
        images = extraction.get('images', [])
        self.load('vlm' if images else 'llm')
        system = self.rules + '\nJSON Schema:\n' + json.dumps(self.schema, ensure_ascii=False, separators=(',', ':'))
        prompt = json.dumps({'source': template['source'], 'template': template,
            'evidence': extraction['evidence']}, ensure_ascii=False, separators=(',', ':'))
        if images:
            from qwen_vl_utils import process_vision_info
            content = [{'type': 'text', 'text': prompt}]
            for image in images:
                path = (self.root / image['path']).resolve()
                if not path.is_relative_to(self.root.resolve()):
                    raise AppError('invalid_image', '画像パスが不正です。')
                content += [{'type': 'text', 'text': '画像根拠ID: ' + image['ref_id']},
                    {'type': 'image', 'image': str(path)}]
            messages = [{'role': 'system', 'content': system}, {'role': 'user', 'content': content}]
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, _ = process_vision_info(messages)
            inputs = self.processor(text=[text], images=image_inputs, padding=True, return_tensors='pt').to('cuda:0')
        else:
            messages = [{'role': 'system', 'content': system}, {'role': 'user', 'content': prompt}]
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.processor([text], return_tensors='pt').to('cuda:0')
        tokens = inputs['input_ids'].shape[-1]
        if tokens > 24000:
            raise AppError('input_too_large', 'model入力上限を超えています。本文を短くしてください。')
        remaining = self.deadline - time.time()
        if remaining < 30:
            raise BatchTimeout()
        with torch.inference_mode():
            output = self.model.generate(**inputs, max_new_tokens=6144, do_sample=False,
                max_time=min(360, remaining - 20))
        generated = output[0, tokens:]
        decoder = self.processor.tokenizer if images else self.processor
        raw = decoder.decode(generated, skip_special_tokens=True)
        del inputs, output
        candidate = parse_json(raw)
        # IDs and origin metadata belong to this job, never to a generated answer.
        for key in ['recipe_id', 'schema_version', 'created_at', 'updated_at', 'source']:
            candidate[key] = copy.deepcopy(template[key])
        candidate['user_corrections'] = []
        enforce_evidence(candidate, extraction['evidence'])
        return candidate, raw[:MAX_OUTPUT_CHARS]

    def transcribe(self, audio):
        self.unload()
        from faster_whisper import WhisperModel
        name = self.config.get('asr_model', 'small')
        if name not in ['tiny', 'base', 'small', 'medium']:
            raise AppError('invalid_model', 'ASR model設定が不正です。')
        model = WhisperModel(name, device='cuda', device_index=0, compute_type='int8_float16',
            download_root=str(self.root / 'models'))
        try:
            segments, _ = model.transcribe(str(audio), beam_size=5, vad_filter=True)
            result = [{'text': s.text, 'start': s.start, 'end': s.end} for s in segments]
            if not result:
                raise AppError('asr_failed', '音声から文字を読み取れませんでした。')
            return result
        finally:
            del model
            gc.collect()


def enforce_evidence(candidate, evidence):
    """Fix locations only from existing evidence, reject invented references."""
    available = {item['ref_id']: item for item in evidence}
    refs = candidate.get('source_refs', [])
    if not isinstance(refs, list):
        raise AppError('invalid_evidence', 'AIが返した根拠の形式が不正です。')
    for ref in refs:
        if not isinstance(ref, dict) or ref.get('ref_id') not in available:
            raise AppError('invalid_evidence', '原典にない根拠IDをAIが返しました。')
        original = available[ref['ref_id']]
        for key in ['type', 'start_seconds', 'end_seconds']:
            ref[key] = original[key]
        if ref.get('text') and original.get('text') and ref['text'] not in original['text']:
            raise AppError('invalid_evidence', 'AIの引用が原典と一致しません。')
    for section in ['ingredients', 'steps']:
        for item in candidate.get(section, []) if isinstance(candidate.get(section), list) else []:
            if isinstance(item, dict) and item.get('source_ref') is not None and item['source_ref'] not in available:
                raise AppError('invalid_evidence', '原典にない根拠IDをAIが返しました。')


def needs_more(candidate):
    if not isinstance(candidate, dict):
        return True
    title = candidate.get('title')
    return not isinstance(title, dict) or not title.get('original') or not candidate.get('ingredients') or not candidate.get('steps')


def append_transcript(extraction, segments, max_chars):
    from workerlib.extractors.common import evidence
    for row in segments:
        extraction['evidence'].append(evidence(f"src_{len(extraction['evidence'])+1:03d}",
            'youtube_speech', row['text'], row['start'], row['end']))
    if sum(len(item.get('text') or '') for item in extraction['evidence']) > max_chars:
        raise AppError('input_too_large', '音声の文字起こしが入力上限を超えています。')


def extract_image(job, api, directory, root):
    from PIL import Image, ImageOps
    from workerlib.extractors.common import evidence
    original = directory / 'original-image'
    original.write_bytes(api.request('/inputs/' + valid_id(job['job_id']), binary=True))
    try:
        Image.MAX_IMAGE_PIXELS = 24_000_000
        with Image.open(original) as image:
            if image.format not in ['JPEG', 'PNG', 'WEBP'] or image.width * image.height > 24_000_000:
                raise ValueError()
            image.load()
            image = ImageOps.exif_transpose(image).convert('RGB')
            image.thumbnail((2000, 2000))
            path = directory / 'image.jpg'
            image.save(path, 'JPEG', quality=95)
    except Exception:
        raise AppError('image_analysis_failed', '画像を読み取れませんでした。JPEG/PNG/WebPを確認してください。') from None
    return {'evidence': [evidence('src_001', 'image_text')], 'images': [
        {'path': path.relative_to(root).as_posix(), 'mime': 'image/jpeg', 'ref_id': 'src_001'}], 'warnings': []}


def process_job(job, api, models, settings, root):
    from workerlib.extractors.text import extract_text
    from workerlib.extractors.web import extract_web
    from workerlib.extractors.youtube import extract_youtube, download_audio, frames
    from workerlib.extractors.common import evidence
    directory = root / ('job-' + valid_id(job['job_id']))
    directory.mkdir(exist_ok=True)
    value = job.get('input', {}).get('value', '')
    template = copy.deepcopy(job['template'])
    try:
        kind = job['input_type']
        if kind == 'text':
            extraction = extract_text(value, settings.max_input_chars)
        elif kind == 'web':
            extraction = extract_web(value, settings, job['input'].get('recipe_index', 0))
        elif kind == 'youtube':
            extraction = extract_youtube(value, settings)
        elif kind == 'image':
            extraction = extract_image(job, api, directory, root)
        else:
            raise AppError('invalid_input_type', '入力種別が不正です。')
        for key in ['title', 'creator', 'source_id']:
            if extraction.get(key):
                template['source'][key] = extraction[key]
        candidate = raw = None
        first_structure_error = None
        requested = frame_seconds(job.get('input', {}).get('frame_seconds'), settings.max_video_seconds)
        if extraction['evidence'] and (kind != 'youtube' or any(e.get('text') for e in extraction['evidence'])):
            try:
                candidate, raw = models.structure(extraction, template)
            except AppError as error:
                if kind != 'youtube' or error.code != 'llm_not_json' or not (job['input'].get('allow_asr', True) or requested):
                    raise
                # A malformed text candidate can still mean insufficient subtitles.
                first_structure_error = error
                raw = error.details.get('raw_output')
        if kind == 'youtube' and needs_more(candidate) and job['input'].get('allow_asr', True):
            # Use descriptions and original subtitles first; obtain audio only for gaps.
            try:
                audio = download_audio(extraction['video_url'], directory / 'audio', settings)
                append_transcript(extraction, models.transcribe(audio), settings.max_input_chars)
                candidate, raw = models.structure(extraction, template)
            except AppError as error:
                if first_structure_error and raw and 'raw_output' not in error.details:
                    error.details['raw_output'] = raw
                if candidate is None and not requested:
                    raise
                extraction['warnings'].append('音声処理に失敗しました。取得済みの根拠を使います。')
        if kind == 'youtube' and needs_more(candidate) and requested:
            extraction['images'] = frames(extraction['video_url'], requested, directory / 'frames', settings)
            for image in extraction['images']:
                ref_id = f"src_{len(extraction['evidence'])+1:03d}"
                image['ref_id'] = ref_id
                extraction['evidence'].append(evidence(ref_id, 'youtube_overlay', start=image['second'], end=image['second']))
            candidate, raw = models.structure(extraction, template)
        if candidate is None:
            if first_structure_error:
                raise first_structure_error
            raise AppError('extraction_empty', 'レシピ情報を取得できません。本文入力やフレーム指定を利用してください。')
        # Validation errors stay editable in the draft; the cloud validates again on save.
        from jsonschema import Draft202012Validator, FormatChecker
        errors = list(Draft202012Validator(models.schema, format_checker=FormatChecker()).iter_errors(candidate))
        result = {'job_id': job['job_id'], 'candidate': candidate,
            'evidence': extraction['evidence'], 'raw_output': raw}
        # Keep API payload within the common result contract; cloud reports validation errors.
        return result
    finally:
        shutil.rmtree(directory, ignore_errors=True)


RETRYABLE = {'gpu_unavailable', 'worker_api_failed', 'model_load_failed', 'batch_timeout', 'notebook_failed'}


def error_result(job, error):
    if isinstance(error, BatchTimeout):
        error = AppError('batch_timeout', '無料GPU batchの処理時間上限に達しました。', 503)
    if not isinstance(error, AppError):
        error = AppError('model_load_failed', 'GPU処理に失敗しました。quota、メモリ、model取得状況を確認してください。', 503)
    result = {'job_id': job['job_id'], 'error': {'code': error.code, 'message': error.message,
        'retryable': error.code in RETRYABLE}}
    if error.details.get('raw_output'):
        result['raw_output'] = error.details['raw_output']
    return result


def run(config, root):
    # Prevent library diagnostics from exposing sources/URLs in Notebook outputs.
    logging.disable(logging.CRITICAL)
    os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
    os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
    os.environ['HF_HOME'] = str(root / 'models')
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    api = BatchAPI(config)
    batch = api.request()
    if batch.get('batch_id') != config['batch_id']:
        raise AppError('invalid_batch', 'batchが一致しません。')
    jobs = batch['jobs']
    if not isinstance(jobs, list) or len(jobs) > 10:
        raise AppError('invalid_batch', 'batch件数が不正です。')
    api.request('/start', {})
    settings = settings_from(batch['config'], root)
    maximum = min(int(batch['config'].get('max_batch_seconds', 5400)), 5400)
    # Leave time for bounded HTTP retries and the final result/finish callbacks.
    deadline = min(time.time() + maximum, api.expires - 180)
    rules = batch.get('extraction_prompt')
    if not isinstance(rules, str) or not rules.strip():
        for job in jobs:
            api.request('/results', error_result(job, AppError('missing_prompt', '抽出ルールが未設定です。')))
        api.request('/finish', {})
        return
    models = Models(batch['config'], batch['schema'], rules, root, deadline)
    try:
        check_gpu()
    except AppError as error:
        for job in jobs:
            api.request('/results', error_result(job, error))
        api.request('/finish', {})
        return
    def timeout_handler(signum, frame):
        raise BatchTimeout()
    previous = signal.signal(signal.SIGALRM, timeout_handler)
    try:
        for job in jobs:
            if time.time() >= deadline - 30:
                result = error_result(job, BatchTimeout())
            else:
                signal.alarm(max(1, int(deadline - time.time())))
                try:
                    result = process_job(job, api, models, settings, root)
                except Exception as error:
                    result = error_result(job, error)
                finally:
                    signal.alarm(0)
            api.request('/results', result)
        api.request('/finish', {})
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)
        models.unload()


def fail_batch(config, code):
    """Dependency/bootstrap failures can report without importing GPU packages."""
    api = BatchAPI(config)
    batch = api.request()
    api.request('/start', {})
    for job in batch['jobs']:
        api.request('/results', error_result(job, AppError(code, 'Kaggle Notebookを実行できませんでした。', 503)))
    api.request('/finish', {})


# The launcher calls run() inside the private Notebook. Importing is safe for tests.
