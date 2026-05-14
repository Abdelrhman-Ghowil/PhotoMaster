import re
from io import BytesIO
from zipfile import ZipFile

import requests
import streamlit as st
from bs4 import BeautifulSoup
from PIL import Image, ImageOps, UnidentifiedImageError
from pdf2image import convert_from_bytes
from transformers import pipeline
from transformers.modeling_utils import PreTrainedModel


@st.cache_data
def convert_drive_link(link):
    match_d = re.search(r"/d/([^/]+)", link)
    if match_d:
        file_id = match_d.group(1)
        return f"https://drive.google.com/uc?export=download&id={file_id}"

    match_id = re.search(r"id=([^&]+)|([-\\w]{25,})", link)
    if match_id:
        file_id = match_id.group(1) if match_id.group(1) else match_id.group(2)
        return f"https://drive.google.com/uc?export=download&id={file_id}"

    match_postimg = re.search(r"postimg.cc/([^/]+)", link)
    if match_postimg:
        image_id = match_postimg.group(1)
        return f"https://i.postimg.cc/{image_id}/your-image-name.jpg"

    if "imgg.io" in link or "ibb.co" in link:
        try:
            response = requests.get(link)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, "html.parser")
                img_tag = soup.find("a", {"class": "btn btn-download default"})
                if img_tag and "href" in img_tag.attrs:
                    return img_tag["href"]
            else:
                st.error(f"Failed to access the page. Status code: {response.status_code}")
        except Exception as e:
            st.error(f"An error occurred: {e}")

    return link


@st.cache_data
def download_image(url):
    response = requests.get(url)
    if response.status_code == 200:
        return response.content
    st.error(f"Failed to download image. Status code: {response.status_code}")
    return None


@st.cache_data
def resize_image(image_content, size=(1024, 1024), aspect_ratio_threshold=2):
    try:
        image = Image.open(BytesIO(image_content))
        if image.mode not in ["RGB", "RGBA"]:
            image = image.convert("RGB")
        if image.mode == "RGBA":
            image = image.convert("RGB")

        original_width, original_height = image.size
        aspect_ratio = original_width / original_height
        inverse_threshold = 1 / aspect_ratio_threshold

        if aspect_ratio < inverse_threshold:
            new_height = int(original_width / inverse_threshold)
            crop_top = (original_height - new_height) // 2
            image = image.crop((0, crop_top, original_width, crop_top + new_height))
        elif aspect_ratio > aspect_ratio_threshold:
            new_width = int(original_height * aspect_ratio_threshold)
            crop_left = (original_width - new_width) // 2
            image = image.crop((crop_left, 0, crop_left + new_width, original_height))

        image = image.resize(size)
        img_byte_arr = BytesIO()
        image.save(img_byte_arr, format="JPEG")
        return img_byte_arr.getvalue()
    except UnidentifiedImageError:
        return None


if not hasattr(PreTrainedModel, "all_tied_weights_keys"):
    PreTrainedModel.all_tied_weights_keys = {}


@st.cache_resource
def get_rmbg_pipeline():
    return pipeline(
        "image-segmentation",
        model="briaai/RMBG-1.4",
        trust_remote_code=True,
        model_kwargs={"low_cpu_mem_usage": False},
    )


@st.cache_data
def remove_background(image_content):
    try:
        image = Image.open(BytesIO(image_content))
        output_img = get_rmbg_pipeline()(image)
        img_byte_arr = BytesIO()
        output_img.save(img_byte_arr, format="PNG")
        return img_byte_arr.getvalue()
    except UnidentifiedImageError:
        return None


@st.cache_data
def combine_with_background(foreground_content, background_content, resize_foreground=False):
    try:
        foreground = Image.open(BytesIO(foreground_content)).convert("RGBA")
        background = Image.open(BytesIO(background_content)).convert("RGBA")
        background = background.resize((1024, 1024))

        if resize_foreground:
            fg_area = foreground.width * foreground.height
            bg_area = background.width * background.height
            scale_factor = (0.8 * bg_area / fg_area) ** 0.5
            new_width = int(foreground.width * scale_factor)
            new_height = int(foreground.height * scale_factor)
            foreground = foreground.resize((new_width, new_height))
            dimensions = (new_width, new_height)
        else:
            dimensions = (foreground.width, foreground.height)

        fg_width, fg_height = foreground.size
        bg_width, bg_height = background.size
        position = ((bg_width - fg_width) // 2, (bg_height - fg_height) // 2)
        combined = background.copy()
        combined.paste(foreground, position, foreground)

        img_byte_arr = BytesIO()
        combined.save(img_byte_arr, format="PNG")
        return img_byte_arr.getvalue(), dimensions
    except UnidentifiedImageError:
        return None, None


def flip_image(image, flip_horizontal=False, flip_vertical=False):
    if flip_horizontal:
        image = ImageOps.mirror(image)
    if flip_vertical:
        image = ImageOps.flip(image)
    return image


def zoom_image(image, zoom_factor=1.0):
    if zoom_factor == 1.0:
        return image

    base = image.convert("RGBA")
    width, height = base.size

    if zoom_factor > 1.0:
        crop_w = max(1, int(width / zoom_factor))
        crop_h = max(1, int(height / zoom_factor))
        left = (width - crop_w) // 2
        top = (height - crop_h) // 2
        cropped = base.crop((left, top, left + crop_w, top + crop_h))
        return cropped.resize((width, height), Image.LANCZOS)

    # Zoom out: shrink image and center it on a transparent canvas.
    new_w = max(1, int(width * zoom_factor))
    new_h = max(1, int(height * zoom_factor))
    resized = base.resize((new_w, new_h), Image.LANCZOS)
    canvas = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    paste_x = (width - new_w) // 2
    paste_y = (height - new_h) // 2
    canvas.paste(resized, (paste_x, paste_y), resized)
    return canvas


def move_image(image, offset_x=0, offset_y=0):
    if offset_x == 0 and offset_y == 0:
        return image

    base = image.convert("RGBA")
    width, height = base.size
    canvas = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    canvas.paste(base, (int(offset_x), int(offset_y)), base)
    return canvas


def fit_image_to_canvas(image, canvas_size=(1024, 1024)):
    base = image.convert("RGBA")
    canvas_w, canvas_h = canvas_size
    img_w, img_h = base.size
    if img_w == 0 or img_h == 0:
        return Image.new("RGBA", canvas_size, (0, 0, 0, 0))

    scale = min(canvas_w / img_w, canvas_h / img_h)
    new_w = max(1, int(img_w * scale))
    new_h = max(1, int(img_h * scale))
    resized = base.resize((new_w, new_h), Image.LANCZOS)

    canvas = Image.new("RGBA", canvas_size, (0, 0, 0, 0))
    x = (canvas_w - new_w) // 2
    y = (canvas_h - new_h) // 2
    canvas.paste(resized, (x, y), resized)
    return canvas


def convert_and_compress_image(image_content, output_format="png", quality=90):
    image = Image.open(BytesIO(image_content))
    target_format = str(output_format).lower()
    quality = max(1, min(100, int(quality)))

    if target_format in ("jpg", "jpeg"):
        ext = "jpg"
        mime = "image/jpeg"
        fmt = "JPEG"
        if image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info):
            alpha = image.convert("RGBA")
            background = Image.new("RGB", alpha.size, (255, 255, 255))
            background.paste(alpha, mask=alpha.split()[-1])
            image_to_save = background
        else:
            image_to_save = image.convert("RGB")
        save_kwargs = {"quality": quality, "optimize": True}
    elif target_format == "webp":
        ext = "webp"
        mime = "image/webp"
        fmt = "WEBP"
        image_to_save = image.convert("RGBA")
        save_kwargs = {"quality": quality, "method": 6}
    else:
        ext = "png"
        mime = "image/png"
        fmt = "PNG"
        image_to_save = image.convert("RGBA")
        compress_level = int(round((100 - quality) * 9 / 100))
        save_kwargs = {"optimize": True, "compress_level": compress_level}

    out = BytesIO()
    image_to_save.save(out, format=fmt, **save_kwargs)
    return out.getvalue(), ext, mime


@st.cache_data
def convert_pdf_to_images(pdf_content):
    try:
        return convert_from_bytes(pdf_content)
    except Exception as e:
        st.error(f"Error converting PDF to images: {e}")
        return []


@st.cache_data
def download_all_images_as_zip(
    images_info,
    remove_bg=False,
    add_bg=False,
    bg_image=None,
    resize_foreground=False,
    threshold=2,
    flip_horizontal=False,
    flip_vertical=False,
    enable_compress_convert=False,
    output_format="png",
    quality=90,
):
    zip_buffer = BytesIO()
    with ZipFile(zip_buffer, "w") as zf:
        for name, url_or_file in images_info:
            if isinstance(url_or_file, str):
                url = convert_drive_link(url_or_file)
                image_content = download_image(url)
            elif isinstance(url_or_file, bytes):
                image_content = url_or_file
            else:
                image_content = url_or_file.read()

            if image_content:
                if remove_bg:
                    processed_image = remove_background(image_content)
                    ext = "png"
                else:
                    size = (1290, 789) if "banner" in name.lower() else (1024, 1024)
                    processed_image = resize_image(
                        image_content, size=size, aspect_ratio_threshold=threshold
                    )
                    ext = "png"

                if add_bg and bg_image:
                    processed_image, _ = combine_with_background(
                        processed_image, bg_image, resize_foreground=resize_foreground
                    )
                    ext = "png"

                if flip_horizontal or flip_vertical:
                    image = Image.open(BytesIO(image_content))
                    processed_image = flip_image(image, flip_horizontal, flip_vertical)
                    img_byte_arr = BytesIO()
                    processed_image.save(img_byte_arr, format="PNG")
                    processed_image = img_byte_arr.getvalue()

                if processed_image:
                    if enable_compress_convert:
                        processed_image, ext, _ = convert_and_compress_image(
                            processed_image,
                            output_format=output_format,
                            quality=quality,
                        )
                    zf.writestr(f"{name.rsplit('.', 1)[0]}.{ext}", processed_image)

    zip_buffer.seek(0)
    return zip_buffer
