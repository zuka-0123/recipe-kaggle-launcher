import json
import re
from urllib.parse import urljoin

import trafilatura
from bs4 import BeautifulSoup

from ..errors import AppError
from .common import evidence, fetch_html

def recipes_in(value):
    if isinstance(value, list):
        for item in value:
            yield from recipes_in(item)
    elif isinstance(value, dict):
        types = value.get('@type', [])
        if isinstance(types, str):
            types = [types]
        if any(str(t).split('/')[-1] == 'Recipe' for t in types):
            yield value
            return
        for child in value.values():
            if isinstance(child, (dict, list)):
                yield from recipes_in(child)

def parse_web(html, max_chars, recipe_index=0, base_url=None):
    soup = BeautifulSoup(html, 'html.parser')
    recipes, warnings = [], []
    for script in soup.find_all('script', attrs={'type': 'application/ld+json'}):
        try:
            for recipe in recipes_in(json.loads(script.get_text())):
                if recipe not in recipes:
                    recipes.append(recipe)
        except (ValueError, TypeError):
            warnings.append('読み取れないJSON-LDがありました。取得できた構造化データまたは本文を使います。')
    selected = None
    if recipes:
        if not 0 <= recipe_index < len(recipes):
            raise AppError('invalid_recipe_index', 'ページ内のレシピ番号が範囲外です。')
        selected = recipes[recipe_index]
        if len(recipes) > 1:
            warnings.append(f'ページ内に{len(recipes)}件のRecipeがあります。{recipe_index + 1}件目を選びました。必要なら入力欄の番号を変更してください。')
    image_urls = []
    def add_image(value):
        if isinstance(value, list):
            for item in value:
                add_image(item)
        elif isinstance(value, dict):
            add_image(value.get('contentUrl') or value.get('url'))
        elif isinstance(value, str) and base_url:
            url = urljoin(base_url, value)
            if url.startswith(('https://', 'http://')) and url not in image_urls:
                image_urls.append(url)
    def step_images(value):
        if isinstance(value, list):
            for step in value:
                step_images(step)
        elif isinstance(value, dict):
            add_image(value.get('image'))
            step_images(value.get('itemListElement'))
    if selected:
        add_image(selected.get('image'))
        step_images(selected.get('recipeInstructions'))
    else:
        for element in soup.select('nav, aside, footer, header, form, [role="navigation"], [role="complementary"], .advertisement, .ads, .related'):
            element.decompose()
        main = soup.find('article') or soup.find('main')
        if main:
            for image in main.find_all('img'):
                try:
                    if int(image.get('width', '400')) < 200 or int(image.get('height', '300')) < 120:
                        continue
                except ValueError:
                    pass
                add_image(image.get('data-src') or image.get('src'))
    items = []
    if selected:
        items.append(evidence('src_001', 'web_structured_data', json.dumps(selected, ensure_ascii=False)))
    complete = selected and selected.get('recipeIngredient') and selected.get('recipeInstructions')
    if not complete or (len(recipes) == 1 and not selected.get('recipeYield')):
        # 材料・手順が揃っていても、人数は本文だけに記載される場合があります。
        # 複数Recipeのページでは、選んでいない料理の本文を混ぜません。
        for element in soup.select('script, style, nav, aside, footer, header, form, [role="navigation"], [role="complementary"], .advertisement, .ads, .related'):
            element.decompose()
        body = trafilatura.extract(str(soup), include_comments=False, include_tables=True, favor_precision=True)
        if not body:
            main = soup.find('main') or soup.find('article')
            body = main.get_text('\n', strip=True) if main else None
        if selected and not selected.get('recipeYield'):
            # 本文抽出器が短い人数表示だけを落とす場合も、原文のまま資料へ残します。
            for element in soup.find_all(['p', 'span']):
                text = element.get_text(' ', strip=True)
                if re.fullmatch(r'[（(]?\s*\d+(?:\.\d+)?(?:\s*[〜～~\-]\s*\d+(?:\.\d+)?)?\s*人分\s*[）)]?', text) and text not in (body or ''):
                    body = (body or '') + '\n' + text
        if body:
            items.append(evidence(f'src_{len(items)+1:03d}', 'web_text', body))
    if not items:
        raise AppError('web_extraction_failed', 'レシピ本文を取得できませんでした。本文を貼り付けてください。', 422)
    if sum(len(x['text'] or '') for x in items) > max_chars:
        raise AppError('input_too_large', 'ページ本文が長すぎます。必要なレシピ部分を貼り付けてください。')
    title = selected.get('name') if selected else None
    if not title:
        title = soup.title.get_text(strip=True) if soup.title else None
    author = selected.get('author') if selected else None
    if isinstance(author, list):
        author = author[0] if author else None
    if isinstance(author, dict):
        author = author.get('name')
    return {'attachment_urls': image_urls[:4], 'evidence': items, 'warnings': warnings, 'images': [],
            'title': title if isinstance(title, str) else None,
            'creator': author if isinstance(author, str) else None}

def extract_web(url, settings, recipe_index=0):
    return parse_web(fetch_html(url), settings.max_input_chars, recipe_index, url)
