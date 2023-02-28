import os
from pathlib import Path

from PIL import Image

from download_openimages import build_bucket

_CACHE_DIR = Path.home() / 'ImageD_OpenImages'


def _download_image(image_id, split):
    bucket = build_bucket()
    bucket.download_file(f'{split}/{image_id}.jpg',
                         os.path.join(_CACHE_DIR, f'{image_id}.jpg'))


def _ensure_cache_dir():
    if not _CACHE_DIR.exists():
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)


def load_image(image_id, split="train") -> Image.Image:
    _ensure_cache_dir()
    file_path = _CACHE_DIR / f'{image_id}.jpg'
    if not file_path.exists():
        _download_image(image_id, split)
    img = Image.open(file_path)
    img = img.convert('RGB')
    return img

