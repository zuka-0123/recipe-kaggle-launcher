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
import subprocess
import sys
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


SAFE_EXCEPTION_CLASSES = {'Exception', 'RuntimeError', 'ImportError', 'ModuleNotFoundError',
    'OSError', 'ValueError', 'JSONDecodeError', 'TypeError', 'KeyError', 'TimeoutError',
    'TimeoutExpired', 'HTTPError', 'URLError', 'AppError', 'BatchTimeout', 'KeyboardInterrupt'}
SAFE_DIAGNOSTIC_CODES = {'bootstrap_failed', 'worker_failed', 'report_failed', 'worker_api_failed',
    'worker_token_expired', 'invalid_worker_url', 'invalid_batch', 'missing_prompt',
    'pip_install_failed', 'pip_no_matching_distribution', 'pip_resolution_failed',
    'pip_network_failed', 'pip_build_failed', 'pip_install_timeout', 'pip_oserror'}
SAFE_STAGES = {'dependency_install', 'worker_run', 'failure_report',
    'api_get', 'api_start', 'api_results', 'api_finish', 'api_input', 'api_request'}
SAFE_NETWORK_REASON_CLASSES = {'gaierror', 'SSLCertVerificationError', 'SSLError',
    'ConnectionRefusedError', 'ConnectionResetError', 'ConnectionAbortedError',
    'TimeoutError', 'OSError', 'str'}
SAFE_HTTP_RESPONSE_KINDS = {'edge_challenge', 'cloudflare_html', 'application_json', 'html_other', 'http_other'}
SAFE_APPLICATION_ERROR_CODES = {'unauthorized', 'not_found', 'batch_not_found', 'invalid_json',
    'invalid_batch', 'batch_expired', 'worker_token_expired', 'batch_already_claimed',
    'batch_already_finished', 'validation_failed', 'internal_error'}


def classify_http_error(error):
    """Read a small response locally and return only fixed diagnostic labels."""
    headers = error.headers or {}
    if str(headers.get('cf-mitigated', '')).lower() == 'challenge':
        return {'response_kind': 'edge_challenge'}
    try:
        body = error.read(16 * 1024)
    except Exception:
        body = b''
    content_type = str(headers.get('content-type', '')).lower()
    if 'application/json' in content_type:
        result = {'response_kind': 'application_json'}
        try:
            data = json.loads(body)
            code = data.get('error', {}).get('code') if isinstance(data, dict) and isinstance(data.get('error'), dict) else data.get('code') if isinstance(data, dict) else None
            if isinstance(code, str) and code in SAFE_APPLICATION_ERROR_CODES:
                result['application_error_code'] = code
        except (ValueError, TypeError):
            pass
        return result
    lowered = body.lower()
    if 'text/html' in content_type:
        if b'cloudflare' in lowered and (b'just a moment' in lowered or b'cf-chl-' in lowered):
            return {'response_kind': 'edge_challenge'}
        if b'cloudflare' in lowered and (b'ray id' in lowered or b'cloudflare ray' in lowered or b'attention required' in lowered):
            return {'response_kind': 'cloudflare_html'}
        return {'response_kind': 'html_other'}
    return {'response_kind': 'http_other'}


def network_reason_class(error):
    value = type(error.reason).__name__ if isinstance(error, urllib.error.URLError) else None
    return value if value in SAFE_NETWORK_REASON_CLASSES else None


def safe_diagnostic(stage, error, code=None, http_status=None, response_details=None):
    """Only fixed labels, exception class and numeric HTTP status reach stdout."""
    stage = stage if stage in SAFE_STAGES else 'worker_run'
    details = error.details if isinstance(error, AppError) else {}
    kind = details.get('cause_class', type(error).__name__)
    kind = kind if kind in SAFE_EXCEPTION_CLASSES else 'Exception'
    code = code or (error.code if isinstance(error, AppError) else 'worker_failed')
    code = code if code in SAFE_DIAGNOSTIC_CODES else 'worker_failed'
    status = http_status if http_status is not None else details.get('http_status')
    line = 'recipe_diag stage=' + stage + ' code=' + code + ' exception=' + kind
    if isinstance(status, int) and not isinstance(status, bool) and 100 <= status <= 599:
        line += ' http_status=' + str(status)
    reason = details.get('reason_class') or network_reason_class(error)
    if reason in SAFE_NETWORK_REASON_CLASSES:
        line += ' reason_class=' + reason
    response_details = response_details or details
    response_kind = response_details.get('response_kind')
    if response_kind in SAFE_HTTP_RESPONSE_KINDS:
        line += ' response_kind=' + response_kind
    application_code = response_details.get('application_error_code')
    if application_code in SAFE_APPLICATION_ERROR_CODES:
        line += ' application_error_code=' + application_code
    print(line, flush=True)


def pip_failure_code(stderr):
    # Inspect captured text locally; neither package names nor raw output are logged.
    if 'No matching distribution found' in stderr or 'Could not find a version' in stderr:
        return 'pip_no_matching_distribution'
    if 'ResolutionImpossible' in stderr or 'conflicting dependencies' in stderr:
        return 'pip_resolution_failed'
    if any(label in stderr for label in ['NewConnectionError', 'Temporary failure in name resolution', 'ConnectionError']):
        return 'pip_network_failed'
    if 'Failed building wheel' in stderr or 'subprocess-exited-with-error' in stderr:
        return 'pip_build_failed'
    return 'pip_install_failed'


def bootstrap(config, root):
    """Imports before pip need only stdlib and bundled workerlib.errors."""
    stage = 'dependency_install'
    code = 'pip_install_failed'
    try:
        try:
            install = subprocess.run([sys.executable, '-m', 'pip', 'install',
                '--disable-pip-version-check', '-q', '-r', str(root / 'requirements-worker.txt')],
                capture_output=True, text=True, timeout=600)
        except subprocess.TimeoutExpired:
            code = 'pip_install_timeout'
            raise
        except OSError:
            code = 'pip_oserror'
            raise
        if install.returncode:
            code = pip_failure_code(install.stderr or '')
            raise RuntimeError('dependency_install_failed')
        stage, code = 'worker_run', None
        run(config, root)
    except BaseException as error:
        safe_diagnostic(stage, error, code)
        try:
            fail_batch(config, 'notebook_failed')
        except BaseException as report_error:
            safe_diagnostic('failure_report', report_error, 'report_failed')


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
        stage = {'': 'api_get', '/start': 'api_start', '/results': 'api_results',
            '/finish': 'api_finish'}.get(suffix, 'api_input' if suffix.startswith('/inputs/') else 'api_request')
        if time.time() >= self.expires:
            raise AppError('worker_token_expired', 'batch tokenの期限が切れています。')
        request = urllib.request.Request(self.url + suffix,
            headers={'Authorization': 'Bearer ' + self.token, 'Content-Type': 'application/json',
                'User-Agent': 'PersonalRecipeKB-worker/1.0',
                'Accept': 'application/octet-stream' if binary else 'application/json'},
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
                    response_details = classify_http_error(error)
                    safe_diagnostic(stage, error, 'worker_api_failed', error.code, response_details)
                    raise AppError('worker_api_failed', 'batch APIとの通信に失敗しました。', 502,
                        {'http_status': error.code, 'cause_class': 'HTTPError', **response_details}) from None
            except (urllib.error.URLError, ValueError) as error:
                if attempt == 2:
                    safe_diagnostic(stage, error, 'worker_api_failed')
                    raise AppError('worker_api_failed', 'batch APIとの通信に失敗しました。', 502,
                        {'cause_class': type(error).__name__ if type(error).__name__ in SAFE_EXCEPTION_CLASSES else 'Exception',
                         'reason_class': network_reason_class(error)}) from None
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


COMPLETENESS_RULES = '''原典で読める材料をすべてingredientsへ、調理工程をすべてstepsへ記録してください。先頭の材料だけや代表的な工程だけに省略しないでください。
材料欄と手順欄を最初から最後まで読み、材料は1項目につき1件、工程は原典の順番に並べてください。読めない項目は推測しません。
templateやitem_shapesはJSONの形を示すものです。配列の空・1件という例に合わせず、原典に存在する件数を出力してください。
各stepのdurationとtemperatureは必ず{"value":null,"unit":null,"raw_text":null}形式のobjectを保持します。不明だからといってobject全体をnullにしないでください。
材料amountもobjectを保持し、value・unit・raw_textの3キーを必ず含めます。raw_textは読めた分量表現を数量・単位・語順ごと原文通り保存します。ほかの欄から原文を作り直しません。
unitは提示したCanonical enumの値だけです。日本語の「大さじ」「小さじ」「個」「本」「少々」をunitへそのまま入れないでください。大さじはtbsp、小さじはtsp、個・本の個数はpieceへ表記統一できます。対応が判断できない単位はunit=null、raw_textに原文を残します。
「少々」「適量」「ひとつまみ」「お好みで」等はamount.value=null、amount.unit=null、amount.raw_textに読めた原文を残してください。pinch等のenumにない単位や、推測した数値を返さないでください。
heatは各工程の原文に火加減が明記される場合だけ設定します。未記載の工程はheat=nullです。前の工程の火加減を引き継がず、煮詰める等の動詞から推測しません。
料理名や見出しの「2人分」等も原典の人数情報です。title.originalの原文を保持し、yield.quantity=2、unit="serving"、raw_text="2人分"のようにyieldにも記録してください。記載がなければyieldの各値はnullです。
時間・温度もその工程で明記された値だけを記録します。「片面3分ずつ」を合計6分へ計算するなど、原典にない値を作らないでください。
出力前に原典の材料欄と工程欄をもう一度照合し、材料の抜け、工程の抜け、object全体のnull、他工程から移したheatがないことを確認してください。
出力はCanonical Recipe JSONオブジェクト1個だけです。確認作業の説明や別形式の中間データは出力しません。'''


def resolve_schema(node, schema):
    if '$ref' not in node:
        return node
    value = schema
    reference = node['$ref']
    if not reference.startswith('#/'):
        raise AppError('invalid_schema', 'Canonical Schemaの参照形式が不正です。')
    for part in reference[2:].split('/'):
        value = value[part.replace('~1', '/').replace('~0', '~')]
    return value


def schema_shape(node, schema):
    """Show Canonical item fields, without inventing any ingredient/step content."""
    node = resolve_schema(node, schema)
    if 'properties' in node:
        return {key: schema_shape(child, schema) for key, child in node['properties'].items()}
    types = node.get('type', [])
    types = [types] if isinstance(types, str) else types
    if 'null' in types:
        return None
    if 'array' in types:
        return []
    if 'boolean' in types:
        return False
    if 'integer' in types or 'number' in types:
        return 1
    if 'const' in node:
        return node['const']
    if 'enum' in node:
        return node['enum'][0]
    return '' if 'string' in types else None


def schema_enums(schema):
    values = {}
    def visit(node, path, seen=()):
        reference = node.get('$ref')
        if reference in seen:
            return
        node = resolve_schema(node, schema)
        if 'enum' in node:
            values[path] = node['enum']
        if 'const' in node:
            values[path] = [node['const']]
        for key, child in node.get('properties', {}).items():
            visit(child, path + '.' + key if path else key, seen + ((reference,) if reference else ()))
        if isinstance(node.get('items'), dict):
            visit(node['items'], path + '[]', seen + ((reference,) if reference else ()))
    visit(schema, '')
    return values


def build_structure_prompts(rules, schema, template, extraction, schema_mode='compact'):
    if schema_mode not in ['compact', 'full']:
        raise AppError('invalid_prompt_mode', 'prompt Schema modeはcompactまたはfullを指定してください。')
    system = rules + '\n\n' + COMPLETENESS_RULES
    if schema_mode == 'full':
        system += '\nJSON Schema:\n' + json.dumps(schema, ensure_ascii=False, separators=(',', ':'))
    else:
        system += '\nCanonicalのenum許可値:\n' + json.dumps(schema_enums(schema), ensure_ascii=False, separators=(',', ':'))
    # These are shapes of the existing Canonical arrays, not a new output format.
    shapes = {key: schema_shape(schema['properties'][key]['items'], schema)
        for key in ['ingredients', 'steps', 'source_refs']}
    if extraction['evidence']:
        shapes['source_refs']['ref_id'] = extraction['evidence'][0]['ref_id']
        shapes['source_refs']['type'] = extraction['evidence'][0]['type']
    payload = {'source': template['source'], 'template': template,
        'item_shapes': shapes, 'evidence': extraction['evidence']}
    prompt = json.dumps(payload, ensure_ascii=False, separators=(',', ':'))
    prompt += '\n原典の材料と工程を最後まで照合してください。すべての材料・工程をCanonicalへ含め、各工程のduration/temperatureはobject、未記載heatはnull、人数は見出しも確認します。'
    return system, prompt


def repair_literal_absence(candidate):
    """Restore required containers for literal null; never convert evidence values."""
    if not isinstance(candidate, dict):
        return candidate
    steps = candidate.get('steps')
    for step in steps if isinstance(steps, list) else []:
        if isinstance(step, dict):
            for key in ['duration', 'temperature']:
                if key in step and step[key] is None:
                    step[key] = {'value': None, 'unit': None, 'raw_text': None}
    names = [candidate.get('title')]
    ingredients = candidate.get('ingredients')
    if isinstance(ingredients, list):
        names.extend(item.get('name') for item in ingredients if isinstance(item, dict))
    for name in names:
        if isinstance(name, dict) and name.get('normalized') == '':
            name['normalized'] = None
    return candidate


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
        system, prompt = build_structure_prompts(self.rules, self.schema, template, extraction,
            self.config.get('prompt_schema_mode', 'compact'))
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
        repair_literal_absence(candidate)
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


def job_failure_reason(error):
    """Inspect error text locally and return one fixed label; never return text."""
    try:
        message = str(error).lower()[:8192]
    except Exception:
        message = ''
    if type(error).__name__ == 'OutOfMemoryError' or ('cuda' in message and 'out of memory' in message) or any(
            label in message for label in ['cudnn_status_alloc_failed', 'cublas_status_alloc_failed']):
        return 'gpu_memory'
    if any(label in message for label in ['libcudnn', 'libcublas', 'cudnn library',
            'could not load cudnn', 'could not load cuda', 'cuda driver version is insufficient']):
        return 'cuda_library'
    if any(label in message for label in ['couldn\'t connect to \'https://huggingface.co',
            'cannot find the requested files in the disk cache', 'gated repo',
            'failed to download model', 'model download failed']):
        return 'model_download'
    return 'runtime_other'


def safe_job_diagnostic(stage, error):
    stage = stage if stage in {'extract', 'structure', 'asr', 'frames', 'validate'} else 'extract'
    kind = type(error).__name__
    if not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]{0,79}', kind):
        kind = 'Exception'
    print('recipe_job_diag stage=' + stage + ' exception=' + kind + ' reason=' + job_failure_reason(error), flush=True)


def process_job(job, api, models, settings, root):
    directory = root / ('job-' + valid_id(job['job_id']))
    directory.mkdir(exist_ok=True)
    value = job.get('input', {}).get('value', '')
    template = copy.deepcopy(job['template'])
    stage = 'extract'
    try:
        from workerlib.extractors.text import extract_text
        from workerlib.extractors.web import extract_web
        from workerlib.extractors.youtube import extract_youtube, download_audio, frames
        from workerlib.extractors.common import evidence
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
            stage = 'structure'
            try:
                candidate, raw = models.structure(extraction, template)
            except AppError as error:
                if kind != 'youtube' or error.code != 'llm_not_json' or not (job['input'].get('allow_asr', True) or requested):
                    raise
                safe_job_diagnostic(stage, error)
                # A malformed text candidate can still mean insufficient subtitles.
                first_structure_error = error
                raw = error.details.get('raw_output')
        if kind == 'youtube' and needs_more(candidate) and job['input'].get('allow_asr', True):
            # Use descriptions and original subtitles first; obtain audio only for gaps.
            stage = 'asr'
            try:
                audio = download_audio(extraction['video_url'], directory / 'audio', settings)
                append_transcript(extraction, models.transcribe(audio), settings.max_input_chars)
                stage = 'structure'
                candidate, raw = models.structure(extraction, template)
            except AppError as error:
                if first_structure_error and raw and 'raw_output' not in error.details:
                    error.details['raw_output'] = raw
                if candidate is None and not requested:
                    raise
                safe_job_diagnostic(stage, error)
                extraction['warnings'].append('音声処理に失敗しました。取得済みの根拠を使います。')
        if kind == 'youtube' and needs_more(candidate) and requested:
            stage = 'frames'
            extraction['images'] = frames(extraction['video_url'], requested, directory / 'frames', settings)
            for image in extraction['images']:
                ref_id = f"src_{len(extraction['evidence'])+1:03d}"
                image['ref_id'] = ref_id
                extraction['evidence'].append(evidence(ref_id, 'youtube_overlay', start=image['second'], end=image['second']))
            stage = 'structure'
            candidate, raw = models.structure(extraction, template)
        if candidate is None:
            if first_structure_error:
                raise first_structure_error
            raise AppError('extraction_empty', 'レシピ情報を取得できません。本文入力やフレーム指定を利用してください。')
        # Validation errors stay editable in the draft; the cloud validates again on save.
        stage = 'validate'
        from jsonschema import Draft202012Validator, FormatChecker
        errors = list(Draft202012Validator(models.schema, format_checker=FormatChecker()).iter_errors(candidate))
        result = {'job_id': job['job_id'], 'candidate': candidate,
            'evidence': extraction['evidence'], 'raw_output': raw}
        # Keep API payload within the common result contract; cloud reports validation errors.
        return result
    except Exception as error:
        safe_job_diagnostic(stage, error)
        raise
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
