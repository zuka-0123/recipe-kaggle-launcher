from ..errors import AppError
from .common import evidence

def extract_text(text, max_chars):
    if not text or not text.strip():
        raise AppError('empty_input', 'レシピ本文を入力してください。')
    if len(text) > max_chars:
        raise AppError('input_too_large', f'本文は{max_chars:,}文字以内にしてください。自動で切り捨ては行いません。')
    return {'evidence': [evidence('src_001', 'manual_input', text)], 'warnings': [], 'images': []}
