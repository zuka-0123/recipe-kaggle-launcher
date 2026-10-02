import ipaddress
import re
import socket
from urllib.parse import parse_qs, urlencode, urlsplit, urlunsplit

import httpx

from ..errors import AppError

def youtube_id(url):
    p = urlsplit(url.strip())
    host = (p.hostname or '').lower()
    if p.scheme not in ['http', 'https'] or p.username or p.password:
        raise AppError('invalid_url', 'HTTPまたはHTTPSのYouTube URLを指定してください。')
    if host in ['youtu.be', 'www.youtu.be']:
        vid = p.path.strip('/').split('/')[0]
    elif host in ['youtube.com', 'www.youtube.com', 'm.youtube.com', 'music.youtube.com', 'www.youtube-nocookie.com']:
        parts = p.path.strip('/').split('/')
        vid = parts[1] if len(parts) > 1 and parts[0] in ['shorts', 'embed', 'live'] else parse_qs(p.query).get('v', [''])[0]
    else:
        raise AppError('invalid_url', 'YouTubeの動画URLを指定してください。')
    if not re.fullmatch(r'[A-Za-z0-9_-]{11}', vid):
        raise AppError('invalid_url', 'YouTubeのvideo IDを読み取れませんでした。')
    return vid

def normalized_url(url):
    if not url:
        return None
    p = urlsplit(url.strip())
    host = (p.hostname or '').lower()
    if host in ['youtube.com', 'www.youtube.com', 'm.youtube.com', 'music.youtube.com', 'youtu.be', 'www.youtu.be', 'www.youtube-nocookie.com']:
        return 'https://www.youtube.com/watch?v=' + youtube_id(url)
    # 出典URLそのものは変更せず、比較用キーだけ追跡パラメータを除去します。
    query = [(k, v) for k, vals in parse_qs(p.query, keep_blank_values=True).items()
             if not k.lower().startswith('utm_') and k.lower() not in ['fbclid', 'gclid'] for v in vals]
    port = ':' + str(p.port) if p.port and not (p.scheme == 'http' and p.port == 80 or p.scheme == 'https' and p.port == 443) else ''
    return urlunsplit((p.scheme.lower(), host + port, p.path or '/', urlencode(sorted(query)), ''))

def source_key(source):
    try:
        return normalized_url(source.get('url'))
    except ValueError:
        raise AppError('invalid_url', 'URLを読み取れませんでした。形式を確認してください。') from None

def validate_url(url):
    try:
        p = urlsplit(url)
        if p.scheme not in ['https', 'http'] or not p.hostname or p.username or p.password or p.port not in [None, 80, 443]:
            raise ValueError()
        addresses = socket.getaddrinfo(p.hostname, p.port or (443 if p.scheme == 'https' else 80), type=socket.SOCK_STREAM)
        if not addresses or any(not ipaddress.ip_address(x[4][0]).is_global for x in addresses):
            raise ValueError()
    except (ValueError, OSError):
        raise AppError('invalid_url', '取得可能な公開HTTP/HTTPS URLを指定してください。') from None
    return url

def fetch_html(url):
    try:
        with httpx.Client(timeout=30, follow_redirects=False, trust_env=False,
                          headers={'User-Agent': 'PersonalRecipeKB/1.0'}) as client:
            current = url
            for _ in range(6):
                validate_url(current)
                with client.stream('GET', current) as response:
                    if response.is_redirect:
                        current = str(response.url.join(response.headers['location']))
                        continue
                    response.raise_for_status()
                    content_type = response.headers.get('content-type', '').lower()
                    if content_type and not any(t in content_type for t in ['text/html', 'application/xhtml+xml', 'text/plain']):
                        raise AppError('fetch_failed', 'HTMLまたはテキストのページを指定してください。')
                    body = bytearray()
                    for chunk in response.iter_bytes():
                        body.extend(chunk)
                        if len(body) > 5 * 1024 * 1024:
                            raise AppError('fetch_failed', 'ページが大きすぎます。レシピ本文を貼り付けてください。')
                    # 日本語ページのcharset宣言も尊重します。
                    from bs4 import UnicodeDammit
                    html = UnicodeDammit(bytes(body), is_html=True).unicode_markup
                    if html is None:
                        raise AppError('fetch_failed', 'ページの文字コードを読み取れませんでした。')
                    return html
            raise AppError('fetch_failed', 'リダイレクトが多すぎます。')
    except httpx.HTTPError:
        raise AppError('fetch_failed', 'ページを取得できませんでした。URLを確認するか本文を貼り付けてください。', 502) from None

def evidence(ref_id, kind, text=None, start=None, end=None):
    return {'ref_id': ref_id, 'type': kind, 'text': text, 'start_seconds': start, 'end_seconds': end}
