import re
import os
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import requests
import streamlit as st
from bs4 import BeautifulSoup
from PIL import Image, ImageOps, UnidentifiedImageError
from pdf2image import convert_from_bytes
import torch
from torch.hub import download_url_to_file
from transformers import pipeline
from transformers.modeling_utils import PreTrainedModel


OBJECT_ERASER_MODEL_URL = os.environ.get(
    "OBJECT_ERASER_MODEL_URL",
    "https://huggingface.co/spaces/aryadytm/remove-photo-object/resolve/main/assets/big-lama.pt",
)

try:
    torch.set_num_threads(max(1, int(os.environ.get("TORCH_NUM_THREADS", "2"))))
except (RuntimeError, ValueError):
    pass


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


def _ceil_modulo(value, modulo):
    if value % modulo == 0:
        return value
    return (value // modulo + 1) * modulo


def _pad_chw_to_modulo(array, modulo):
    channels, height, width = array.shape
    out_height = _ceil_modulo(height, modulo)
    out_width = _ceil_modulo(width, modulo)
    return np.pad(
        array,
        ((0, 0), (0, out_height - height), (0, out_width - width)),
        mode="symmetric",
    )


def _resize_long_side(image, size_limit, resample=Image.BICUBIC):
    width, height = image.size
    if max(width, height) <= size_limit:
        return image

    scale = size_limit / max(width, height)
    new_size = (max(1, int(width * scale + 0.5)), max(1, int(height * scale + 0.5)))
    return image.resize(new_size, resample)


def _normalize_image(image):
    array = np.asarray(image)
    if array.ndim == 2:
        array = array[:, :, np.newaxis]
    array = np.transpose(array, (2, 0, 1))
    return array.astype("float32") / 255.0


def _resolve_object_eraser_model_path():
    model_path = Path(
        os.environ.get("OBJECT_ERASER_MODEL_PATH", "assets/big-lama.pt")
    ).expanduser()
    if not model_path.is_absolute():
        model_path = Path.cwd() / model_path

    if model_path.exists():
        return model_path

    try:
        model_path.parent.mkdir(parents=True, exist_ok=True)
        download_url_to_file(OBJECT_ERASER_MODEL_URL, str(model_path), progress=True)
    except Exception as exc:
        st.error(
            "Object Eraser model is missing and could not be downloaded. "
            f"Put big-lama.pt at {model_path} or set OBJECT_ERASER_MODEL_PATH. Error: {exc}"
        )
        return None

    return model_path if model_path.exists() else None


@st.cache_resource
def get_object_eraser_model():
    model_path = _resolve_object_eraser_model_path()
    if model_path is None:
        return None

    try:
        model = torch.jit.load(str(model_path), map_location="cpu")
        model.eval()
        return model
    except Exception as exc:
        st.error(f"Object Eraser model could not be loaded: {exc}")
        return None


def _mask_to_png_bytes(mask):
    mask_image = mask.convert("L") if isinstance(mask, Image.Image) else Image.open(BytesIO(mask)).convert("L")
    out = BytesIO()
    mask_image.save(out, format="PNG")
    return out.getvalue()


def erase_object(image_content, mask):
    try:
        return _erase_object_cached(
            image_content,
            _mask_to_png_bytes(mask),
            int(os.environ.get("OBJECT_ERASER_SIZE_LIMIT", "1280")),
        )
    except UnidentifiedImageError:
        return None


@st.cache_data(show_spinner=False, max_entries=32)
def _erase_object_cached(image_content, mask_content, size_limit):
    try:
        original_image = Image.open(BytesIO(image_content)).convert("RGB")
        mask_image = Image.open(BytesIO(mask_content)).convert("L")
    except UnidentifiedImageError:
        return None

    if mask_image.size != original_image.size:
        mask_image = mask_image.resize(original_image.size, Image.NEAREST)

    model = get_object_eraser_model()
    if model is None:
        return None

    work_image = _resize_long_side(original_image, size_limit, Image.BICUBIC)
    work_mask = mask_image.resize(work_image.size, Image.NEAREST)

    image_array = _pad_chw_to_modulo(_normalize_image(work_image), 8)
    mask_array = _pad_chw_to_modulo(_normalize_image(work_mask), 8)
    mask_array = (mask_array > 0).astype("float32")

    image_tensor = torch.from_numpy(image_array).unsqueeze(0)
    mask_tensor = torch.from_numpy(mask_array).unsqueeze(0)

    try:
        with torch.no_grad():
            output = model(image_tensor, mask_tensor)
    except Exception as exc:
        st.error(f"Object Eraser failed: {exc}")
        return None

    width, height = work_image.size
    result = output[0].permute(1, 2, 0).detach().cpu().numpy()
    result = result[:height, :width, :]
    result = np.clip(result * 255, 0, 255).astype("uint8")
    result_image = Image.fromarray(result).convert("RGB")
    if result_image.size != original_image.size:
        result_image = result_image.resize(original_image.size, Image.BICUBIC)

    out = BytesIO()
    result_image.save(out, format="PNG")
    return out.getvalue()


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


def crop_image(image, left_pct=0, right_pct=0, top_pct=0, bottom_pct=0):
    base = image.convert("RGBA")
    width, height = base.size

    left_px = int(width * max(0, left_pct) / 100)
    right_px = int(width * max(0, right_pct) / 100)
    top_px = int(height * max(0, top_pct) / 100)
    bottom_px = int(height * max(0, bottom_pct) / 100)

    crop_left = min(left_px, width - 1)
    crop_top = min(top_px, height - 1)
    crop_right = max(crop_left + 1, width - right_px)
    crop_bottom = max(crop_top + 1, height - bottom_px)

    if crop_right <= crop_left or crop_bottom <= crop_top:
        return base
    return base.crop((crop_left, crop_top, crop_right, crop_bottom))


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
