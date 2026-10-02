import shutil
import subprocess
from pathlib import Path

import httpx
import requests
import yt_dlp
from youtube_transcript_api import YouTubeTranscriptApi

from ..errors import AppError
from .common import evidence, youtube_id

class QuietLogger:
    def debug(self, message): pass
    def warning(self, message): pass
    def error(self, message): pass

class TimedSession(requests.Session):
    def request(self, *args, **kwargs):
        kwargs.setdefault('timeout', 25)
        return super().request(*args, **kwargs)

def options(settings):
    opts = {'quiet': True, 'no_warnings': True, 'logger': QuietLogger(), 'noplaylist': True,
            'socket_timeout': 25, 'retries': 1, 'extractor_retries': 1,
            'skip_download': True}
    if settings.ffmpeg:
        opts['ffmpeg_location'] = settings.ffmpeg
    node = shutil.which('node')
    if node:
        opts['js_runtimes'] = {'node': {'path': node}}
    return opts

def metadata(url, settings):
    with yt_dlp.YoutubeDL(options(settings)) as ydl:
        return ydl.extract_info(url, download=False)

def transcript(video_id):
    with TimedSession() as session:
        api = YouTubeTranscriptApi(http_client=session)
        available = list(api.list(video_id))
        # 原語字幕を使用します。自動翻訳で原典を改変しません。
        available.sort(key=lambda t: (t.language_code not in ['ja', 'en'], t.language_code != 'ja', t.is_generated))
        if not available:
            raise AppError('transcript_unavailable', '字幕がありません。')
        return available[0].fetch()

def extract_youtube(url, settings):
    vid = youtube_id(url)
    canonical_url = 'https://www.youtube.com/watch?v=' + vid
    warnings, items = [], []
    info = {}
    try:
        info = metadata(canonical_url, settings) or {}
    except Exception:
        warnings.append('動画メタデータ・概要欄の取得に失敗しました。字幕の取得を続けます。')
        try:
            with httpx.Client(timeout=20, trust_env=False) as client:
                r = client.get('https://www.youtube.com/oembed', params={'url': canonical_url, 'format': 'json'})
                r.raise_for_status()
                data = r.json()
                info = {'title': data.get('title'), 'channel': data.get('author_name')}
        except (httpx.HTTPError, ValueError):
            warnings.append('動画タイトルとチャンネル名も取得できませんでした。')
    if info.get('description'):
        items.append(evidence('src_001', 'youtube_description', info['description']))
    has_transcript = False
    try:
        rows = transcript(vid)
        for row in rows:
            items.append(evidence(f'src_{len(items)+1:03d}', 'youtube_transcript', row.text,
                                  row.start, row.start + row.duration))
        has_transcript = bool(rows)
    except Exception:
        warnings.append('字幕を取得できませんでした。概要欄を先に使い、不足する場合は音声文字起こしを試します。')
    total_chars = sum(len(x['text'] or '') for x in items)
    if total_chars > settings.max_input_chars:
        raise AppError('input_too_large', '字幕が長すぎます。必要なレシピ部分をテキスト入力してください。')
    return {'evidence': items, 'warnings': warnings, 'images': [], 'title': info.get('title'),
            'creator': info.get('channel') or info.get('uploader'), 'source_id': vid,
            'duration': info.get('duration'), 'has_transcript': has_transcript, 'video_url': canonical_url}

def download_audio(url, directory, settings):
    if not settings.ffmpeg:
        raise AppError('asr_failed', '音声文字起こしにはffmpegが必要です。概要欄や字幕のテキスト入力も利用できます。')
    directory.mkdir(parents=True, exist_ok=True)
    opts = options(settings)
    opts.update({'skip_download': False, 'format': 'bestaudio/best',
                 'outtmpl': str(directory / 'audio.%(ext)s'),
                 'postprocessors': [{'key': 'FFmpegExtractAudio', 'preferredcodec': 'mp3', 'preferredquality': '64'}]})
    def duration_filter(info, *, incomplete=False):
        duration = info.get('duration')
        if duration is None or duration > settings.max_video_seconds:
            return '動画時間が不明、または設定した上限を超えています。'
        return None
    opts['match_filter'] = duration_filter
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.extract_info(url, download=True)
        path = directory / 'audio.mp3'
        if not path.exists() or path.stat().st_size > 24 * 1024 * 1024:
            raise AppError('asr_failed', '音声を取得できないか、文字起こし用の上限24MBを超えています。')
        return path
    except AppError:
        raise
    except Exception:
        raise AppError('asr_failed', '動画の音声を取得できませんでした。本文の貼り付けを利用してください。', 502) from None

def frames(url, seconds, directory, settings):
    if not seconds:
        return []
    if not settings.ffmpeg:
        raise AppError('frame_failed', '動画フレームの取得にはffmpegが必要です。')
    try:
        opts = options(settings)
        opts['format'] = 'bestvideo[height<=720]/best[height<=720]/best'
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=False)
        duration = info.get('duration')
        if duration is None or any(s > duration for s in seconds):
            raise AppError('frame_failed', '動画時間を確認できないか、指定した秒数が動画の範囲外です。')
        stream = info.get('url')
        if not stream:
            raise AppError('frame_failed', '動画フレーム用のURLを取得できませんでした。')
        directory.mkdir(parents=True, exist_ok=True)
        results = []
        for index, second in enumerate(seconds):
            path = directory / f'frame_{index}_{second:g}.jpg'
            # 指定秒数へシークして1枚だけ取得し、動画全体の保存・VLM解析は行いません。
            subprocess.run([settings.ffmpeg, '-hide_banner', '-loglevel', 'error', '-y',
                            '-ss', str(second), '-i', stream, '-frames:v', '1', '-vf', 'scale=1280:-2', str(path)],
                           check=True, timeout=75, capture_output=True)
            if not path.exists() or path.stat().st_size == 0:
                raise AppError('frame_failed', '指定箇所のフレームを取得できませんでした。')
            results.append({'path': path.relative_to(settings.data_dir).as_posix(), 'mime': 'image/jpeg', 'second': second})
        return results
    except AppError:
        raise
    except Exception:
        raise AppError('frame_failed', '指定箇所の動画フレームを取得できませんでした。', 502) from None
