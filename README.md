# Handwriting-Wikipedia-Dataset-Generator
A wikipedia based Handwriting dataset generator, coded with python. A high-quality, curated dataset featuring words and sentences that are highly natural and closely resemble human handwriting.

==========================================


تولیدکننده دیتاست تصاویر دست‌نویس مصنوعی از متن ویکی‌پدیا (تک‌فایل، CPU-Only)
 
## نصب وابستگی‌ها:
```
pip install requests pillow numpy arabic-reshaper python-bidi
 ```
(arabic-reshaper و python-bidi فقط برای زبان‌های راست‌به‌چپ مانند فارسی لازم‌اند)
 
## ساختار پوشه‌ها (کنار همین فایل):
```
    Base_Handwrite_Font/      ← فونت‌های دست‌خط (ttf/otf) - خودکار شناسایی می‌شوند
    Base_Handwrite_BGpaper/   ← تصاویر کاغذ (png/jpg) - پس‌زمینه تصادفی
    handwrite_dataset/        ← خروجی (خودکار ساخته می‌شود)
 ```
## نمونه اجرا:
```    
python Handwriting_Wikipedia_Dataset_Generator.py \
        --lang fa --keywords "سلول خورشیدی" "پروسکایت" فیزیک \
        --max-unique-words 50000 --window 1 --stride 1 \
        --samples-per-class 20 --workers 8 --metadata
 ```
## پارامترهای اصلی:
```    
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
    --fonts-dir / --bg-dir / --output-dir   مسیرهای سفارشی
 ```
### قابلیت ادامه (Resume): اجرای مجدد فقط نمونه‌های باقی‌مانده را تولید می‌کند و
واژه‌های جمع‌آوری‌شده در فایل state ذخیره می‌شوند تا دوباره از ویکی‌پدیا دریافت نشوند.
