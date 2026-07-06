# -*- coding: utf-8 -*-
"""
Handwriting_Wikipedia_Dataset_Generator.py
==========================================
تولیدکننده دیتاست تصاویر دست‌نویس مصنوعی از متن ویکی‌پدیا (تک‌فایل، CPU-Only)

نصب وابستگی‌ها:
    pip install requests pillow numpy arabic-reshaper python-bidi fonttools

    (arabic-reshaper و python-bidi فقط برای زبان‌های راست‌به‌چپ مانند فارسی لازم‌اند)
    (fonttools اختیاری اما به‌شدت توصیه‌شده: بررسی دقیق پشتیبانی فونت از حروف،
     تا از تولید تصاویر «مربع‌مربع/Tofu» جلوگیری شود)

ساختار پوشه‌ها (کنار همین فایل):
    Base_Handwrite_Font/      ← فونت‌های دست‌خط (ttf/otf) - خودکار شناسایی می‌شوند
    Base_Handwrite_BGpaper/   ← تصاویر کاغذ (png/jpg) - پس‌زمینه تصادفی
    handwrite_dataset/        ← خروجی (خودکار ساخته می‌شود)

نمونه اجرا:
    python Handwriting_Wikipedia_Dataset_Generator.py \
        --lang fa --keywords "سلول خورشیدی" "پروسکایت" فیزیک \
        --max-unique-words 50000 --window 1 --stride 1 \
        --samples-per-class 20 --workers 8 --metadata

پارامترهای اصلی:
    --lang               کد زبان ویکی‌پدیا (fa, en, de, ...)          [پیش‌فرض: fa]
    --keywords           یک یا چند کلیدواژه اولیه برای شروع جستجو      [اجباری]
    --max-unique-words   حداقل تعداد واژه/عبارت یکتای موردنیاز          [پیش‌فرض: 50000]
    --window             تعداد کلمات هر تصویر (Sliding Window)         [پیش‌فرض: 1]
    --stride             گام پنجره لغزان                               [پیش‌فرض: 1]
    --samples-per-class  تعداد نمونه تصویر برای هر کلاس                [پیش‌فرض: 10]
    --min-word-len       حداقل طول واژه معتبر                          [پیش‌فرض: 2]
    --workers            تعداد پردازش‌های موازی (0 = تعداد هسته‌ها)     [پیش‌فرض: 0]
    --phash-threshold    آستانه فاصله همینگ برای حذف تصاویر مشابه       [پیش‌فرض: 4]
    --metadata           ذخیره فایل JSON اطلاعات کنار هر تصویر
    --numbers            افزودن ارقام 0 تا 9 به کلاس‌های دیتاست
                         (برای زبان‌های راست‌به‌چپ، ارقام فارسی ۰ تا ۹ نیز اضافه می‌شود)
    --shape              افزودن بیش از ۲۰ شکل پرکاربرد ریاضی (دایره، مثلث، مربع، ...)
                         به‌صورت دست‌کشیده؛ نام پوشه این کلاس‌ها پسوند __SHAPE دارد
    --fonts-dir / --bg-dir / --output-dir   مسیرهای سفارشی

نکته: هر فونت قبل از استفاده بررسی می‌شود که تمام حروفِ متن را پشتیبانی کند؛
در غیر این صورت فونت دیگری انتخاب می‌شود تا خروجی خراب (Tofu/□□□) ساخته نشود.

قابلیت ادامه (Resume): اجرای مجدد فقط نمونه‌های باقی‌مانده را تولید می‌کند و
واژه‌های جمع‌آوری‌شده در فایل state ذخیره می‌شوند تا دوباره از ویکی‌پدیا دریافت نشوند.
"""

import argparse
import hashlib
import json
import math
import os
import random
import re
import sys
import time
import unicodedata
from collections import deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path

import numpy as np
import requests
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont

try:
    import arabic_reshaper
    from bidi.algorithm import get_display
    HAS_RTL_LIBS = True
except ImportError:
    HAS_RTL_LIBS = False

try:
    from fontTools.ttLib import TTFont
    HAS_FONTTOOLS = True
except ImportError:
    HAS_FONTTOOLS = False

Image.MAX_IMAGE_PIXELS = None
RTL_LANGS = {"fa", "ar", "ur", "ps", "ckb", "he", "yi", "ug", "pnb", "sd"}
STATE_FILE = "_state.json"

# ============================================================================
# ۱) نرمال‌سازی متن
# ============================================================================

_PERSIAN_MAP = str.maketrans({
    "ي": "ی", "ك": "ک", "ة": "ه", "ۀ": "ه", "أ": "ا", "إ": "ا", "ؤ": "و",
    "ئ": "ی", "٤": "۴", "٥": "۵", "٦": "۶",
    "0": "", "1": "", "2": "", "3": "", "4": "", "5": "", "6": "", "7": "",
    "8": "", "9": "",
})
_ARABIC_DIACRITICS = re.compile(r"[\u064B-\u065F\u0670\u0640\u06D6-\u06ED]")
_ZWNJ = "\u200c"


def normalize_text(text: str, lang: str) -> str:
    """نرمال‌سازی کامل متن متناسب با زبان."""
    text = unicodedata.normalize("NFKC", text)
    if lang in RTL_LANGS:
        text = _ARABIC_DIACRITICS.sub("", text)          # حذف اعراب و کشیده
        text = text.translate(_PERSIAN_MAP)               # یکسان‌سازی حروف عربی/فارسی
        text = re.sub(r"[\u200b\u200d\u200e\u200f\ufeff]", "", text)
        text = re.sub(rf"{_ZWNJ}+", _ZWNJ, text)          # یکسان‌سازی نیم‌فاصله
        text = re.sub(rf"\s*{_ZWNJ}\s*", _ZWNJ, text)
    else:
        text = text.lower()
        text = re.sub(r"[\u200b-\u200f\ufeff]", "", text)
    text = re.sub(r"\s+", " ", text)                      # حذف فاصله‌های اضافی
    return text.strip()


def tokenize(text: str, lang: str, min_len: int):
    """استخراج واژه‌های معتبر از متن نرمال‌شده."""
    if lang in RTL_LANGS:
        pattern = re.compile(rf"[\u0600-\u06FF{_ZWNJ}]+")
    else:
        pattern = re.compile(r"[a-zA-ZÀ-ÖØ-öø-ÿĀ-žΑ-Ωа-яА-ЯЁё']+")
    tokens = []
    for tok in pattern.findall(text):
        tok = tok.strip(_ZWNJ + "'")
        if len(tok.replace(_ZWNJ, "")) >= min_len:
            tokens.append(tok)
    return tokens


# ============================================================================
# ۲) جمع‌آوری متن از ویکی‌پدیا
# ============================================================================

class WikipediaCollector:
    """از کلیدواژه‌ها شروع می‌کند و تا رسیدن به تعداد واژه یکتای هدف،
    مقاله دریافت، پردازش و لینک‌های مرتبط را دنبال می‌کند."""

    def __init__(self, lang: str, min_word_len: int):
        self.lang = lang
        self.min_word_len = min_word_len
        self.api = f"https://{lang}.wikipedia.org/w/api.php"
        self.session = requests.Session()
        self.session.headers["User-Agent"] = (
            "HandwritingDatasetGenerator/1.0 (research; contact: local)"
        )
        self.visited = set()
        self.queue = deque()
        self.articles_processed = 0

    def _get(self, params: dict):
        params = {"format": "json", "action": "query", **params}
        for attempt in range(4):
            try:
                r = self.session.get(self.api, params=params, timeout=30)
                r.raise_for_status()
                return r.json()
            except Exception as e:
                if attempt == 3:
                    print(f"  [!] خطای شبکه: {e}")
                    return None
                time.sleep(2 * (attempt + 1))

    def search(self, keyword: str, limit: int = 25):
        data = self._get({"list": "search", "srsearch": keyword,
                          "srlimit": limit, "srnamespace": 0})
        if not data:
            return []
        return [it["title"] for it in data.get("query", {}).get("search", [])]

    def fetch_extract(self, title: str) -> str:
        data = self._get({"prop": "extracts", "explaintext": 1,
                          "titles": title, "redirects": 1,
                          "exlimit": 1})
        if not data:
            return ""
        pages = data.get("query", {}).get("pages", {})
        for p in pages.values():
            return p.get("extract", "") or ""
        return ""

    def fetch_links(self, title: str, limit: int = 100):
        data = self._get({"prop": "links", "titles": title,
                          "pllimit": limit, "plnamespace": 0})
        if not data:
            return []
        pages = data.get("query", {}).get("pages", {})
        links = []
        for p in pages.values():
            for l in p.get("links", []):
                links.append(l["title"])
        return links

    def collect_unique_units(self, keywords, target: int,
                             window: int, stride: int,
                             resume: dict = None, save_cb=None):
        """جمع‌آوری واژه/عبارت یکتا تا رسیدن به مقدار هدف.
        resume: ادامه جمع‌آوری ناتمام از state قبلی.
        save_cb: ذخیره دوره‌ای پیشرفت برای Resume."""
        unique_units = {}   # unit -> None (dict حفظ ترتیب)
        if resume:
            unique_units = dict.fromkeys(resume.get("units", []))
            self.visited = set(resume.get("visited", []))
            self.queue = deque(resume.get("queue", []))
            self.articles_processed = resume.get("articles", 0)
            print(f"[Resume] ادامه جمع‌آوری از {len(unique_units):,} واژه "
                  f"و {self.articles_processed} مقاله قبلی.")

        if not self.queue:
            for kw in keywords:
                for t in self.search(kw):
                    if t not in self.visited:
                        self.queue.append(t)
        while len(unique_units) < target:
            if not self.queue:
                print("  [!] صف مقالات خالی شد؛ جستجوی مجدد کلیدواژه‌ها...")
                for kw in keywords:
                    for t in self.search(kw, limit=50):
                        if t not in self.visited:
                            self.queue.append(t)
                if not self.queue:
                    print("  [!] مقاله جدیدی یافت نشد. توقف با تعداد فعلی.")
                    break

            title = self.queue.popleft()
            if title in self.visited:
                continue
            self.visited.add(title)

            raw = self.fetch_extract(title)
            if not raw or len(raw) < 200:
                continue

            text = normalize_text(raw, self.lang)
            tokens = tokenize(text, self.lang, self.min_word_len)
            if not tokens:
                continue

            self.articles_processed += 1
            before = len(unique_units)

            if window <= 1:
                for tok in tokens:
                    if tok not in unique_units:
                        unique_units[tok] = None
                        if len(unique_units) >= target:
                            break
            else:
                for i in range(0, max(1, len(tokens) - window + 1), stride):
                    phrase = " ".join(tokens[i:i + window])
                    if len(tokens[i:i + window]) == window and phrase not in unique_units:
                        unique_units[phrase] = None
                        if len(unique_units) >= target:
                            break

            gained = len(unique_units) - before
            print(f"  [{self.articles_processed:>4}] «{title[:45]}» "
                  f"(+{gained})  مجموع: {len(unique_units):,}/{target:,}")

            # تغذیه صف با لینک‌های مرتبط وقتی صف کوتاه است
            if len(self.queue) < 30:
                for l in self.fetch_links(title):
                    if l not in self.visited:
                        self.queue.append(l)

            # ذخیره دوره‌ای پیشرفت برای امکان ادامه پس از قطع شدن
            if save_cb and self.articles_processed % 20 == 0:
                save_cb(list(unique_units), self)

        return list(unique_units.keys())


# ============================================================================
# ۳) ابزارهای تصویر: هش ادراکی، وارپ، پرسپکتیو
# ============================================================================

def dhash(img: Image.Image, hash_size: int = 8) -> int:
    g = img.convert("L").resize((hash_size + 1, hash_size), Image.LANCZOS)
    px = np.asarray(g, dtype=np.int16)
    bits = (px[:, 1:] > px[:, :-1]).flatten()
    return int(np.packbits(bits).tobytes().hex() or "0", 16)


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def soft_vertical_warp(alpha: np.ndarray, rng: random.Random) -> np.ndarray:
    """اعوجاج سینوسی بسیار ملایم (شبیه حرکت طبیعی دست)."""
    h, w = alpha.shape
    amp = rng.uniform(0.5, max(1.0, h * 0.02))
    freq = rng.uniform(0.5, 1.6)
    phase = rng.uniform(0, 2 * math.pi)
    xs = np.arange(w)
    offs = np.round(amp * np.sin(2 * math.pi * freq * xs / max(w, 1) + phase)).astype(int)
    rows = (np.arange(h)[:, None] - offs[None, :]).clip(0, h - 1)
    return alpha[rows, np.arange(w)[None, :]]


def _perspective_coeffs(src, dst):
    A, B = [], []
    for (x, y), (u, v) in zip(dst, src):
        A.append([x, y, 1, 0, 0, 0, -u * x, -u * y])
        A.append([0, 0, 0, x, y, 1, -v * x, -v * y])
        B.extend([u, v])
    res = np.linalg.lstsq(np.array(A, dtype=np.float64),
                          np.array(B, dtype=np.float64), rcond=None)[0]
    return res.tolist()


def slight_perspective(img: Image.Image, rng: random.Random) -> Image.Image:
    w, h = img.size
    d = min(w, h) * rng.uniform(0.0, 0.02)
    src = [(0, 0), (w, 0), (w, h), (0, h)]
    dst = [(rng.uniform(-d, d), rng.uniform(-d, d)) for _ in range(4)]
    dst = [(sx + ox, sy + oy) for (sx, sy), (ox, oy) in zip(src, dst)]
    coeffs = _perspective_coeffs(src, dst)
    fill = img.getpixel((min(3, w - 1), min(3, h - 1)))  # جلوگیری از گوشه سیاه
    return img.transform((w, h), Image.PERSPECTIVE, coeffs,
                         resample=Image.BICUBIC, fillcolor=fill)


# ============================================================================
# ۴) موتور رندر دست‌خط (درون هر پردازش کارگر)
# ============================================================================

_G = {}  # کش سراسری هر پردازش: فونت‌ها، پس‌زمینه‌ها


def _init_worker(font_paths, bg_paths, cfg):
    _G["font_paths"] = font_paths
    _G["font_cache"] = {}
    _G["charsets"] = {}       # font_path -> set(codepoints) | None
    _G["mask_cache"] = {}     # (font_path, char) -> mask bytes
    _G["cfg"] = cfg
    bgs = []
    for p in bg_paths:
        try:
            im = Image.open(p).convert("RGB")
            # محدود کردن اندازه برای کنترل حافظه
            if max(im.size) > 1600:
                im.thumbnail((1600, 1600), Image.LANCZOS)
            bgs.append(im)
        except Exception:
            pass
    _G["bgs"] = bgs


def _get_font(path: str, size: int) -> ImageFont.FreeTypeFont:
    size = max(14, int(round(size / 2) * 2))  # گرد کردن برای کش موثر
    key = (path, size)
    if key not in _G["font_cache"]:
        if len(_G["font_cache"]) > 256:
            _G["font_cache"].clear()
        _G["font_cache"][key] = ImageFont.truetype(path, size)
    return _G["font_cache"][key]


_NOTDEF_PROBE = "\U000F0000"  # کاراکتری که تقریباً هیچ فونتی ندارد → گلیف .notdef


def _font_supports(path: str, text: str) -> bool:
    """بررسی می‌کند فونت تمام حروف متن را دارد تا خروجی مربع‌مربع (Tofu) نشود."""
    chars = {c for c in text if c not in (" ", _ZWNJ, "\n")}
    if not chars:
        return False

    # روش دقیق: خواندن cmap فونت با fontTools (یک‌بار، سپس کش)
    if path not in _G["charsets"]:
        cs = None
        if HAS_FONTTOOLS:
            try:
                tf = TTFont(path, lazy=True, fontNumber=0)
                cs = set(tf.getBestCmap().keys())
                tf.close()
            except Exception:
                cs = None
        _G["charsets"][path] = cs
    cs = _G["charsets"][path]
    if cs is not None:
        return all(ord(c) in cs for c in chars)

    # روش جایگزین (بدون fontTools): مقایسه رندر حرف با گلیف .notdef
    try:
        font = _get_font(path, 36)
        nd_key = (path, _NOTDEF_PROBE)
        if nd_key not in _G["mask_cache"]:
            _G["mask_cache"][nd_key] = font.getmask(_NOTDEF_PROBE).tobytes()
        notdef = _G["mask_cache"][nd_key]
        for c in chars:
            k = (path, c)
            if k not in _G["mask_cache"]:
                if len(_G["mask_cache"]) > 4096:
                    _G["mask_cache"] = {nd_key: notdef}
                _G["mask_cache"][k] = font.getmask(c).tobytes()
            m = _G["mask_cache"][k]
            if len(m) == 0 or m == notdef:
                return False
        return True
    except Exception:
        return False


def _choose_font(display_text: str, rng: random.Random):
    """انتخاب تصادفی فونتی که تمام حروف متن را پشتیبانی کند."""
    paths = _G["font_paths"]
    sample = rng.sample(paths, min(len(paths), 10))
    for p in sample:
        if _font_supports(p, display_text):
            return p
    for p in paths:  # جستجوی کامل در صورت شکست نمونه تصادفی
        if _font_supports(p, display_text):
            return p
    return None


_INK_COLORS = [
    ((15, 15, 20), 0.40),     # مشکی
    ((25, 45, 130), 0.30),    # آبی خودکار
    ((15, 25, 75), 0.18),     # سرمه‌ای
    ((40, 40, 45), 0.07),     # مداد/ذغالی
    ((20, 70, 40), 0.05),     # سبز تیره
]


def _pick_ink(rng: random.Random):
    r = rng.random()
    acc = 0.0
    for color, p in _INK_COLORS:
        acc += p
        if r <= acc:
            base = color
            break
    else:
        base = _INK_COLORS[0][0]
    jitter = lambda c: int(np.clip(c + rng.gauss(0, 8), 0, 120))
    return tuple(jitter(c) for c in base)


def _shape_for_display(text: str, lang: str) -> str:
    if lang in RTL_LANGS and HAS_RTL_LIBS:
        return get_display(arabic_reshaper.reshape(text))
    return text


def _render_word_tile(word: str, font, ink, rng: random.Random):
    """رندر یک کلمه روی لایه RGBA با تغییرات هندسی جزئی خودِ کلمه."""
    stroke = rng.choice([0, 0, 0, 1])                      # ضخامت جزئی
    bbox = font.getbbox(word, stroke_width=stroke)
    if bbox[2] - bbox[0] <= 0 or bbox[3] - bbox[1] <= 0:
        return None
    pad = 8 + stroke
    w = bbox[2] - bbox[0] + 2 * pad
    h = bbox[3] - bbox[1] + 2 * pad
    tile = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(tile)
    d.text((pad - bbox[0], pad - bbox[1]), word, font=font,
           fill=ink + (255,), stroke_width=stroke, stroke_fill=ink + (255,))

    # تغییر عرض/ارتفاع حروف (مقیاس ناهمسان جزئی)
    sx = rng.uniform(0.94, 1.06)
    sy = rng.uniform(0.94, 1.06)
    tile = tile.resize((max(1, int(w * sx)), max(1, int(h * sy))), Image.BICUBIC)

    # چرخش بسیار جزئی هر کلمه
    ang = rng.gauss(0, 0.8)
    tile = tile.rotate(ang, resample=Image.BICUBIC, expand=True)
    return tile


def _apply_pressure(canvas: Image.Image, rng: random.Random) -> Image.Image:
    """فشار متفاوت قلم: مدولاسیون کم‌فرکانس شفافیت جوهر."""
    a = np.array(canvas.split()[3], dtype=np.float32)
    noise = (np.random.default_rng(rng.getrandbits(32)).random((6, 12)) * 255)
    low = np.array(Image.fromarray(noise.astype(np.uint8))
                   .resize(canvas.size, Image.BICUBIC), dtype=np.float32) / 255.0
    a = a * (0.78 + 0.22 * low)
    canvas.putalpha(Image.fromarray(a.clip(0, 255).astype(np.uint8)))
    return canvas


def _compose_line(words, lang, rng: random.Random, ink):
    """چیدمان کلمات با فاصله‌گذاری طبیعی + شیب و اعوجاج کلی."""
    cfg = _G["cfg"]

    display_words = [_shape_for_display(w, lang) for w in words]
    if lang in RTL_LANGS:
        display_words = display_words[::-1]  # چیدمان بصری راست‌به‌چپ

    # انتخاب فونتی که همه حروف را دارد (جلوگیری از خروجی مربع‌مربع)
    all_chars = "".join(display_words)
    font_path = _choose_font(all_chars, rng)
    if font_path is None:
        return None, None, 0
    size = rng.randint(cfg["font_min"], cfg["font_max"])
    font = _get_font(font_path, size)

    tiles = []
    for w in display_words:
        # فاصله بین حروف: از طریق کشیدگی افقی جزئی شبیه‌سازی می‌شود
        t = _render_word_tile(w, font, ink, rng)
        if t is not None:
            tiles.append(t)
    if not tiles:
        return None, font_path, size

    space_base = int(size * rng.uniform(0.28, 0.5))
    gaps = [max(2, int(space_base * rng.uniform(0.8, 1.25))) for _ in tiles]
    total_w = sum(t.width for t in tiles) + sum(gaps[:-1]) if tiles else 0
    max_h = max(t.height for t in tiles)
    canvas = Image.new("RGBA", (total_w + 20, max_h + 30), (0, 0, 0, 0))

    x = 10
    for i, t in enumerate(tiles):
        y = (canvas.height - t.height) // 2 + int(rng.gauss(0, size * 0.03))
        canvas.alpha_composite(t, (x, max(0, y)))
        x += t.width + (gaps[i] if i < len(gaps) - 1 else 0)

    # شیب دست‌خط (Shear) بسیار کم
    shear = rng.uniform(-0.08, 0.08)
    nw = canvas.width + int(abs(shear) * canvas.height) + 2
    canvas = canvas.transform(
        (nw, canvas.height), Image.AFFINE,
        (1, shear, -shear * canvas.height if shear > 0 else 0, 0, 1, 0),
        resample=Image.BICUBIC)

    # چرخش کلی جزئی
    canvas = canvas.rotate(rng.gauss(0, 1.0), resample=Image.BICUBIC, expand=True)

    # اعوجاج ملایم موضعی روی آلفا
    arr = np.array(canvas)
    arr[:, :, 3] = soft_vertical_warp(arr[:, :, 3], rng)
    canvas = Image.fromarray(arr)

    canvas = _apply_pressure(canvas, rng)

    # اعتبارسنجی: خروجی خالی/خراب ذخیره نشود
    if np.asarray(canvas.split()[3], dtype=np.uint32).sum() < 2000:
        return None, None, 0
    return canvas, os.path.basename(font_path), size


# ----------------------------------------------------------------------------
# اشکال ریاضی دست‌کشیده (--shape)
# ----------------------------------------------------------------------------

SHAPES = [
    "circle", "ellipse", "square", "rectangle", "triangle", "right_triangle",
    "isosceles_triangle", "rhombus", "parallelogram", "trapezoid", "pentagon",
    "hexagon", "heptagon", "octagon", "star5", "star6", "semicircle", "ring",
    "cross", "arrow", "line_segment", "arc", "sector", "kite",
]


def _regular_poly(n, cx, cy, r, rot=-math.pi / 2):
    return [(cx + r * math.cos(rot + 2 * math.pi * i / n),
             cy + r * math.sin(rot + 2 * math.pi * i / n)) for i in range(n)]


def _star(n, cx, cy, r_out, r_in, rot=-math.pi / 2):
    pts = []
    for i in range(2 * n):
        r = r_out if i % 2 == 0 else r_in
        a = rot + math.pi * i / n
        pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    return pts


def _ellipse_pts(cx, cy, rx, ry, a0=0.0, a1=2 * math.pi, n=72):
    return [(cx + rx * math.cos(a0 + (a1 - a0) * i / n),
             cy + ry * math.sin(a0 + (a1 - a0) * i / n)) for i in range(n + 1)]


def _shape_paths(name: str, cx, cy, r, rng: random.Random):
    """برمی‌گرداند: لیستی از (points, closed) برای هر مسیر شکل."""
    j = lambda a, b: rng.uniform(a, b)
    if name == "circle":
        return [(_ellipse_pts(cx, cy, r, r * j(0.96, 1.04)), True)]
    if name == "ellipse":
        return [(_ellipse_pts(cx, cy, r, r * j(0.5, 0.7)), True)]
    if name == "square":
        return [([(cx - r, cy - r), (cx + r, cy - r),
                  (cx + r, cy + r), (cx - r, cy + r)], True)]
    if name == "rectangle":
        ry = r * j(0.5, 0.7)
        return [([(cx - r, cy - ry), (cx + r, cy - ry),
                  (cx + r, cy + ry), (cx - r, cy + ry)], True)]
    if name == "triangle":
        return [(_regular_poly(3, cx, cy, r), True)]
    if name == "right_triangle":
        return [([(cx - r, cy + r * 0.8), (cx + r, cy + r * 0.8),
                  (cx - r, cy - r * 0.8)], True)]
    if name == "isosceles_triangle":
        return [([(cx, cy - r), (cx + r * 0.65, cy + r * 0.85),
                  (cx - r * 0.65, cy + r * 0.85)], True)]
    if name == "rhombus":
        return [([(cx, cy - r), (cx + r * 0.62, cy),
                  (cx, cy + r), (cx - r * 0.62, cy)], True)]
    if name == "parallelogram":
        s = r * 0.45
        return [([(cx - r + s, cy - r * 0.55), (cx + r + s, cy - r * 0.55),
                  (cx + r - s, cy + r * 0.55), (cx - r - s, cy + r * 0.55)], True)]
    if name == "trapezoid":
        return [([(cx - r * 0.55, cy - r * 0.6), (cx + r * 0.55, cy - r * 0.6),
                  (cx + r, cy + r * 0.6), (cx - r, cy + r * 0.6)], True)]
    if name in ("pentagon", "hexagon", "heptagon", "octagon"):
        n = {"pentagon": 5, "hexagon": 6, "heptagon": 7, "octagon": 8}[name]
        return [(_regular_poly(n, cx, cy, r), True)]
    if name == "star5":
        return [(_star(5, cx, cy, r, r * 0.42), True)]
    if name == "star6":
        return [(_star(6, cx, cy, r, r * 0.55), True)]
    if name == "semicircle":
        pts = _ellipse_pts(cx, cy, r, r, math.pi, 2 * math.pi)
        pts.append(pts[0])
        return [(pts, False)]
    if name == "ring":
        return [(_ellipse_pts(cx, cy, r, r), True),
                (_ellipse_pts(cx, cy, r * 0.55, r * 0.55), True)]
    if name == "cross":
        a = r * 0.35
        return [([(cx - a, cy - r), (cx + a, cy - r), (cx + a, cy - a),
                  (cx + r, cy - a), (cx + r, cy + a), (cx + a, cy + a),
                  (cx + a, cy + r), (cx - a, cy + r), (cx - a, cy + a),
                  (cx - r, cy + a), (cx - r, cy - a), (cx - a, cy - a)], True)]
    if name == "arrow":
        h = r * 0.35
        return [([(cx - r, cy - h * 0.5), (cx + r * 0.3, cy - h * 0.5),
                  (cx + r * 0.3, cy - h), (cx + r, cy),
                  (cx + r * 0.3, cy + h), (cx + r * 0.3, cy + h * 0.5),
                  (cx - r, cy + h * 0.5)], True)]
    if name == "line_segment":
        a = rng.uniform(-0.35, 0.35)
        dx, dy = r * math.cos(a), r * math.sin(a)
        return [([(cx - dx, cy - dy), (cx + dx, cy + dy)], False)]
    if name == "arc":
        a0 = rng.uniform(0, math.pi)
        return [(_ellipse_pts(cx, cy, r, r, a0, a0 + rng.uniform(1.6, 3.6)), False)]
    if name == "sector":
        a0 = rng.uniform(0, 2 * math.pi)
        a1 = a0 + rng.uniform(0.9, 2.2)
        pts = [(cx, cy)] + _ellipse_pts(cx, cy, r, r, a0, a1, 40) + [(cx, cy)]
        return [(pts, False)]
    if name == "kite":
        return [([(cx, cy - r), (cx + r * 0.6, cy - r * 0.2),
                  (cx, cy + r), (cx - r * 0.6, cy - r * 0.2)], True)]
    return [(_ellipse_pts(cx, cy, r, r), True)]  # پیش‌فرض ایمن


def _draw_hand_path(draw, pts, closed, ink, rng: random.Random, width):
    """ترسیم مسیر با لرزش طبیعی دست."""
    if closed:
        pts = list(pts) + [pts[0]]
    dense = []
    for (x1, y1), (x2, y2) in zip(pts, pts[1:]):
        n = max(2, int(math.hypot(x2 - x1, y2 - y1) / 6))
        for t in range(n):
            u = t / n
            dense.append((x1 + (x2 - x1) * u + rng.gauss(0, 0.9),
                          y1 + (y2 - y1) * u + rng.gauss(0, 0.9)))
    dense.append(pts[-1])
    draw.line(dense, fill=ink + (255,), width=width, joint="curve")


def _render_shape(name: str, rng: random.Random, ink):
    """رندر یک شکل ریاضی به‌صورت دست‌کشیده روی لایه RGBA."""
    S = rng.randint(170, 320)
    pad = 30
    tile = Image.new("RGBA", (S + 2 * pad, S + 2 * pad), (0, 0, 0, 0))
    d = ImageDraw.Draw(tile)
    width = rng.randint(2, 4)
    for pts, closed in _shape_paths(name, tile.width / 2, tile.height / 2,
                                    S / 2, rng):
        _draw_hand_path(d, pts, closed, ink, rng, width)
        if rng.random() < 0.25:  # دوباره‌کشی جزئی برخی خطوط (طبیعی)
            _draw_hand_path(d, pts, closed, ink, rng, max(1, width - 1))

    tile = tile.rotate(rng.gauss(0, 3.0), resample=Image.BICUBIC, expand=True)
    arr = np.array(tile)
    arr[:, :, 3] = soft_vertical_warp(arr[:, :, 3], rng)
    tile = _apply_pressure(Image.fromarray(arr), rng)
    if np.asarray(tile.split()[3], dtype=np.uint32).sum() < 2000:
        return None
    return tile


def _make_background(text_w: int, text_h: int, rng: random.Random):
    """برش تصادفی از یک کاغذ واقعی (یا کاغذ ساده در صورت نبود تصویر)."""
    m_l = int(text_w * rng.uniform(0.06, 0.35))
    m_r = int(text_w * rng.uniform(0.06, 0.35))
    m_t = int(text_h * rng.uniform(0.15, 0.9))
    m_b = int(text_h * rng.uniform(0.15, 0.9))
    W, H = text_w + m_l + m_r, text_h + m_t + m_b

    bgs = _G["bgs"]
    if bgs:
        bg = rng.choice(bgs)
        if bg.width < W or bg.height < H:
            scale = max(W / bg.width, H / bg.height) * 1.05
            bg = bg.resize((int(bg.width * scale) + 1, int(bg.height * scale) + 1),
                           Image.BICUBIC)
        x = rng.randint(0, bg.width - W)
        y = rng.randint(0, bg.height - H)
        crop = bg.crop((x, y, x + W, y + H)).copy()
        if rng.random() < 0.5:
            crop = crop.transpose(rng.choice([
                Image.FLIP_LEFT_RIGHT, Image.FLIP_TOP_BOTTOM, Image.ROTATE_180]))
    else:
        tint = rng.randint(238, 252)
        crop = Image.new("RGB", (W, H),
                         (tint, tint - rng.randint(0, 6), tint - rng.randint(0, 12)))
    return crop, (m_l, m_t)


def _photometric(img: Image.Image, rng: random.Random) -> Image.Image:
    """شبیه‌سازی دوربین/اسکنر: نور، کنتراست، گاما، بلور، سایه، فشرده‌سازی."""
    # سایه ملایم خطی
    if rng.random() < 0.6:
        w, h = img.size
        grad = np.linspace(rng.uniform(0.90, 0.99), rng.uniform(0.99, 1.03),
                           w if rng.random() < 0.5 else h, dtype=np.float32)
        arr = np.asarray(img, dtype=np.float32)
        if len(grad) == w:
            arr *= grad[None, :, None]
        else:
            arr *= grad[:, None, None]
        img = Image.fromarray(arr.clip(0, 255).astype(np.uint8))

    img = ImageEnhance.Brightness(img).enhance(rng.uniform(0.9, 1.08))
    img = ImageEnhance.Contrast(img).enhance(rng.uniform(0.9, 1.1))
    img = ImageEnhance.Color(img).enhance(rng.uniform(0.9, 1.1))

    # گاما
    gamma = rng.uniform(0.9, 1.12)
    if abs(gamma - 1.0) > 0.02:
        lut = [int(255 * (i / 255) ** gamma) for i in range(256)] * 3
        img = img.point(lut)

    # بلور بسیار کم (فوکوس/حرکت)
    if rng.random() < 0.7:
        img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.0, 0.7)))

    # پرسپکتیو / زاویه کاغذ بسیار کم
    if rng.random() < 0.4:
        img = slight_perspective(img, rng)

    # کیفیت دوربین/اسکنر: رفت‌وبرگشت JPEG با کیفیت بالا
    if rng.random() < 0.85:
        buf = BytesIO()
        img.save(buf, "JPEG", quality=rng.randint(68, 95))
        buf.seek(0)
        img = Image.open(buf).convert("RGB")

    # تغییر جزئی نسبت ابعاد
    if rng.random() < 0.5:
        w, h = img.size
        img = img.resize((max(8, int(w * rng.uniform(0.97, 1.03))),
                          max(8, int(h * rng.uniform(0.97, 1.03)))), Image.BICUBIC)

    # برش طبیعی جزئی اطراف
    if rng.random() < 0.5:
        w, h = img.size
        cl = int(w * rng.uniform(0, 0.02)); cr = int(w * rng.uniform(0, 0.02))
        ct = int(h * rng.uniform(0, 0.02)); cb = int(h * rng.uniform(0, 0.02))
        if w - cl - cr > 10 and h - ct - cb > 10:
            img = img.crop((cl, ct, w - cr, h - cb))
    return img


def render_sample(text: str, lang: str, rng: random.Random, kind: str = "text"):
    """تولید یک نمونه کامل: متن/شکل دست‌نویس روی کاغذ + افکت‌های عکاسی."""
    ink = _pick_ink(rng)
    if kind == "shape":
        line = _render_shape(text, rng, ink)
        font_name, size = "(hand-drawn shape)", 0
    else:
        line, font_name, size = _compose_line(text.split(" "), lang, rng, ink)
    if line is None:
        return None, {}

    bg, (mx, my) = _make_background(line.width, line.height, rng)
    # جابه‌جایی تصادفی محل متن روی کاغذ
    px = mx + int(rng.gauss(0, mx * 0.15)) if mx > 2 else mx
    py = my + int(rng.gauss(0, my * 0.15)) if my > 2 else my
    px = max(0, min(px, bg.width - line.width))
    py = max(0, min(py, bg.height - line.height))
    bg.paste(Image.new("RGB", line.size, ink), (px, py), line)

    img = _photometric(bg, rng)
    meta = {"kind": kind, "font": font_name, "font_size": size,
            "ink_rgb": list(ink), "text": text}
    return img, meta


# ============================================================================
# ۵) وظیفه کارگر: تولید نمونه‌های یک کلاس با حذف تکراری‌ها
# ============================================================================

def generate_class(task):
    """(class_dir, text, lang, kind, n_target, phash_thr, save_meta, seed)"""
    class_dir, text, lang, kind, n_target, phash_thr, save_meta, seed = task
    rng = random.Random(seed)
    cdir = Path(class_dir)
    cdir.mkdir(parents=True, exist_ok=True)

    # Resume: هش تصاویر موجود
    hashes, existing = [], []
    for p in sorted(cdir.glob("*.png")):
        try:
            hashes.append(dhash(Image.open(p)))
            existing.append(p.name)
        except Exception:
            pass

    generated = skipped = 0
    idx = len(existing)
    attempts = 0
    max_attempts = (n_target - len(existing)) * 6 + 10

    none_streak = 0
    while len(existing) + generated < n_target and attempts < max_attempts:
        attempts += 1
        try:
            img, meta = render_sample(text, lang, rng, kind)
        except Exception:
            continue
        if img is None:
            none_streak += 1
            if none_streak >= 5:   # هیچ فونتی این متن را پشتیبانی نمی‌کند
                break
            continue
        none_streak = 0
        h = dhash(img)
        if any(hamming(h, h2) <= phash_thr for h2 in hashes):
            skipped += 1
            continue
        hashes.append(h)
        idx += 1
        out = cdir / f"{idx:05d}.png"
        img.save(out, "PNG", optimize=False)
        if save_meta:
            with open(out.with_suffix(".json"), "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False)
        generated += 1

    no_font = (generated == 0 and len(existing) == 0 and none_streak >= 5)
    return generated, skipped, no_font


# ============================================================================
# ۶) برنامه اصلی
# ============================================================================

def sanitize_class_name(text: str) -> str:
    name = re.sub(r'[\\/:*?"<>|\s]+', "_", text).strip("._")
    if len(name) > 60 or not name:
        name = (name[:48] + "_" + hashlib.md5(text.encode()).hexdigest()[:8]).strip("_")
    return name


def parse_args():
    ap = argparse.ArgumentParser(
        description="تولید دیتاست تصاویر دست‌نویس مصنوعی از ویکی‌پدیا (CPU-Only)")
    ap.add_argument("--lang", default="fa", help="کد زبان ویکی‌پدیا (fa, en, ...)")
    ap.add_argument("--keywords", nargs="+", required=True, help="کلیدواژه‌های اولیه")
    ap.add_argument("--max-unique-words", type=int, default=50000,
                    help="حداقل تعداد واژه/عبارت یکتای هدف")
    ap.add_argument("--window", type=int, default=1, help="تعداد کلمات هر تصویر")
    ap.add_argument("--stride", type=int, default=1, help="گام پنجره لغزان")
    ap.add_argument("--samples-per-class", type=int, default=10)
    ap.add_argument("--min-word-len", type=int, default=2)
    ap.add_argument("--workers", type=int, default=0, help="0 = تعداد هسته‌ها")
    ap.add_argument("--phash-threshold", type=int, default=4)
    ap.add_argument("--metadata", action="store_true")
    ap.add_argument("--numbers", action="store_true",
                    help="افزودن ارقام 0-9 (و ۰-۹ برای زبان‌های راست‌به‌چپ) به دیتاست")
    ap.add_argument("--shape", action="store_true",
                    help="افزودن بیش از ۲۰ شکل ریاضی دست‌کشیده (پوشه‌ها با پسوند __SHAPE)")
    ap.add_argument("--font-min", type=int, default=34)
    ap.add_argument("--font-max", type=int, default=64)
    ap.add_argument("--fonts-dir", default="Base_Handwrite_Font")
    ap.add_argument("--bg-dir", default="Base_Handwrite_BGpaper")
    ap.add_argument("--output-dir", default="handwrite_dataset")
    return ap.parse_args()


def main():
    args = parse_args()
    t0 = time.time()

    if args.lang in RTL_LANGS and not HAS_RTL_LIBS:
        print("[!] برای زبان‌های راست‌به‌چپ نصب کنید: pip install arabic-reshaper python-bidi")
        sys.exit(1)

    # فونت‌ها
    fonts_dir = Path(args.fonts_dir)
    font_paths = sorted(str(p) for ext in ("*.ttf", "*.otf", "*.TTF", "*.OTF")
                        for p in fonts_dir.glob(ext))
    if not font_paths:
        print(f"[!] هیچ فونتی در «{fonts_dir}» یافت نشد.")
        sys.exit(1)

    # پس‌زمینه‌ها
    bg_dir = Path(args.bg_dir)
    bg_paths = sorted(str(p) for ext in ("*.png", "*.jpg", "*.jpeg", "*.webp",
                                          "*.PNG", "*.JPG", "*.JPEG")
                      for p in bg_dir.glob(ext)) if bg_dir.exists() else []
    if not bg_paths:
        print(f"[!] هشدار: تصویری در «{bg_dir}» نیست؛ از کاغذ ساده استفاده می‌شود.")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    state_path = out_dir / STATE_FILE

    # ---------- مرحله ۱: جمع‌آوری واژه‌ها (با قابلیت Resume) ----------
    sig = f"{args.lang}|{args.max_unique_words}|{args.window}|{args.stride}|{args.min_word_len}"
    units, articles, resume_partial = None, 0, None
    if state_path.exists():
        try:
            st = json.loads(state_path.read_text(encoding="utf-8"))
            if st.get("sig") == sig:
                if st.get("done"):
                    units = st["units"]
                    articles = st.get("articles", 0)
                    print(f"[Resume] {len(units):,} واژه/عبارت از state بارگذاری شد.")
                else:
                    resume_partial = st  # جمع‌آوری ناتمام → ادامه از همان‌جا
        except Exception:
            pass

    if units is None:
        print(f"— جمع‌آوری متن از ویکی‌پدیای «{args.lang}» —")
        col = WikipediaCollector(args.lang, args.min_word_len)

        def save_partial(u, c):
            state_path.write_text(json.dumps(
                {"sig": sig, "done": False, "units": u,
                 "articles": c.articles_processed,
                 "visited": list(c.visited)[-5000:],
                 "queue": list(c.queue)[:800]},
                ensure_ascii=False), encoding="utf-8")

        units = col.collect_unique_units(args.keywords, args.max_unique_words,
                                         args.window, args.stride,
                                         resume=resume_partial,
                                         save_cb=save_partial)
        articles = col.articles_processed
        state_path.write_text(json.dumps(
            {"sig": sig, "done": True, "units": units, "articles": articles},
            ensure_ascii=False), encoding="utf-8")
        print(f"— جمع‌آوری کامل شد: {len(units):,} واحد یکتا از {articles} مقاله —")

    # ---------- مرحله ۲: ساخت کلاس‌ها (متن + ارقام + اشکال) ----------
    entries = [(u, "text") for u in units]

    if args.numbers:  # افزودن ارقام 0-9 به‌عنوان کلاس‌های مستقل
        digits = list("0123456789")
        if args.lang in RTL_LANGS:
            digits += list("۰۱۲۳۴۵۶۷۸۹")
        entries += [(d, "text") for d in digits]
        print(f"— {len(digits)} کلاس رقم اضافه شد (--numbers) —")

    if args.shape:  # افزودن اشکال ریاضی دست‌کشیده
        entries += [(s, "shape") for s in SHAPES]
        print(f"— {len(SHAPES)} کلاس شکل ریاضی اضافه شد (--shape) —")

    class_map, used = {}, set()
    for text, kind in entries:
        name = sanitize_class_name(text)
        if kind == "shape":
            name = f"{name}__SHAPE"   # جداسازی پوشه اشکال از واژه‌ها
        if name in used:
            name = f"{name}_{hashlib.md5(text.encode()).hexdigest()[:6]}"
        used.add(name)
        class_map[name] = {"text": text, "kind": kind}
    (out_dir / "_classes.json").write_text(
        json.dumps(class_map, ensure_ascii=False, indent=1), encoding="utf-8")

    tasks = []
    base_seed = 1234567
    for i, (name, info) in enumerate(class_map.items()):
        cdir = out_dir / name
        # Resume: اگر کلاس کامل است، وظیفه ساخته نشود؛ در غیر این صورت فقط ادامه
        if cdir.exists() and len(list(cdir.glob("*.png"))) >= args.samples_per_class:
            continue
        tasks.append((str(cdir), info["text"], args.lang, info["kind"],
                      args.samples_per_class, args.phash_threshold,
                      args.metadata, base_seed + i))

    print(f"— تولید تصاویر: {len(tasks):,} کلاس ناقص از {len(class_map):,} کلاس —")

    cfg = {"font_min": args.font_min, "font_max": args.font_max}
    workers = args.workers or os.cpu_count() or 4
    total_gen = total_skip = total_nofont = 0

    if tasks:
        with ProcessPoolExecutor(max_workers=workers,
                                 initializer=_init_worker,
                                 initargs=(font_paths, bg_paths, cfg)) as ex:
            futures = [ex.submit(generate_class, t) for t in tasks]
            done = 0
            for fut in as_completed(futures):
                g, s, nf = fut.result()
                total_gen += g
                total_skip += s
                total_nofont += int(nf)
                done += 1
                if done % 50 == 0 or done == len(tasks):
                    el = time.time() - t0
                    print(f"  کلاس {done:,}/{len(tasks):,} | تصاویر: {total_gen:,} "
                          f"| سرعت: {total_gen / max(el, 1):.1f} img/s")

    # ---------- گزارش نهایی ----------
    elapsed = time.time() - t0
    total_images = sum(1 for _ in out_dir.rglob("*.png"))
    print("\n" + "=" * 58)
    print("گزارش نهایی")
    print("=" * 58)
    print(f"  مقالات پردازش‌شده          : {articles:,}")
    print(f"  واژه/عبارت یکتا            : {len(units):,}")
    print(f"  کلاس‌های ساخته‌شده          : {len(class_map):,}")
    print(f"  تصاویر تولیدشده (این اجرا) : {total_gen:,}")
    print(f"  کل تصاویر دیتاست           : {total_images:,}")
    print(f"  فونت‌های استفاده‌شده        : {len(font_paths)}")
    print(f"  حذف‌شده به دلیل شباهت       : {total_skip:,}")
    print(f"  کلاس رد‌شده (بدون فونت سازگار): {total_nofont:,}")
    print(f"  مدت اجرا                   : {elapsed / 60:.1f} دقیقه")
    print(f"  سرعت متوسط                 : {total_gen / max(elapsed, 1):.2f} تصویر/ثانیه")
    print("=" * 58)


if __name__ == "__main__":
    main()
