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
import torch.nn as nn
import torch.nn.functional as F
from torch.hub import download_url_to_file
from transformers import pipeline
from transformers.modeling_utils import PreTrainedModel


OBJECT_ERASER_MODEL_URL = os.environ.get(
    "OBJECT_ERASER_MODEL_URL",
    "https://huggingface.co/spaces/aryadytm/remove-photo-object/resolve/main/assets/big-lama.pt",
)
REALESRGAN_X4PLUS_MODEL_URL = (
    "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
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


def _resolve_realesrgan_model_path():
    model_path = Path(
        os.environ.get("REALESRGAN_MODEL_PATH", "assets/models/RealESRGAN_x4plus.pth")
    ).expanduser()
    if not model_path.is_absolute():
        model_path = Path.cwd() / model_path

    if model_path.exists():
        return model_path

    try:
        model_path.parent.mkdir(parents=True, exist_ok=True)
        download_url_to_file(REALESRGAN_X4PLUS_MODEL_URL, str(model_path), progress=True)
    except Exception as exc:
        st.error(
            "Real-ESRGAN model is missing and could not be downloaded. "
            f"Put RealESRGAN_x4plus.pth at {model_path} or set REALESRGAN_MODEL_PATH. Error: {exc}"
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


class _ResidualDenseBlock(nn.Module):
    def __init__(self, num_feat=64, num_grow_ch=32):
        super().__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class _RRDB(nn.Module):
    def __init__(self, num_feat=64, num_grow_ch=32):
        super().__init__()
        self.rdb1 = _ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = _ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = _ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * 0.2 + x


class _RRDBNet(nn.Module):
    def __init__(self, num_block=23, num_feat=64, num_grow_ch=32):
        super().__init__()
        self.conv_first = nn.Conv2d(3, num_feat, 3, 1, 1)
        self.body = nn.Sequential(*[_RRDB(num_feat, num_grow_ch) for _ in range(num_block)])
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, 3, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        feat = self.conv_first(x)
        body_feat = self.conv_body(self.body(feat))
        feat = feat + body_feat
        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode="nearest")))
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode="nearest")))
        out = self.conv_last(self.lrelu(self.conv_hr(feat)))
        return out


class _LocalRealESRGANUpsampler:
    def __init__(self, model_path, tile=256, tile_pad=10):
        self.scale = 4
        self.tile = tile
        self.tile_pad = tile_pad
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = _RRDBNet().to(self.device)
        try:
            state = torch.load(str(model_path), map_location=self.device, weights_only=True)
        except TypeError:
            state = torch.load(str(model_path), map_location=self.device)
        state_dict = state.get("params_ema") or state.get("params") or state
        self.model.load_state_dict(state_dict, strict=True)
        self.model.eval()

    def _process_tensor(self, img):
        _, _, height, width = img.shape
        if not self.tile:
            return self.model(img)

        output = img.new_zeros((1, 3, height * self.scale, width * self.scale))
        for y in range(0, height, self.tile):
            for x in range(0, width, self.tile):
                y1 = min(y + self.tile, height)
                x1 = min(x + self.tile, width)
                y0_pad = max(y - self.tile_pad, 0)
                x0_pad = max(x - self.tile_pad, 0)
                y1_pad = min(y1 + self.tile_pad, height)
                x1_pad = min(x1 + self.tile_pad, width)

                input_tile = img[:, :, y0_pad:y1_pad, x0_pad:x1_pad]
                output_tile = self.model(input_tile)

                out_y0 = y * self.scale
                out_x0 = x * self.scale
                out_y1 = y1 * self.scale
                out_x1 = x1 * self.scale
                tile_y0 = (y - y0_pad) * self.scale
                tile_x0 = (x - x0_pad) * self.scale
                tile_y1 = tile_y0 + (y1 - y) * self.scale
                tile_x1 = tile_x0 + (x1 - x) * self.scale
                output[:, :, out_y0:out_y1, out_x0:out_x1] = output_tile[
                    :, :, tile_y0:tile_y1, tile_x0:tile_x1
                ]
        return output

    def enhance(self, image, outscale=2):
        outscale = int(outscale)
        has_alpha = image.ndim == 3 and image.shape[2] == 4
        if image.ndim == 2:
            image = np.repeat(image[:, :, np.newaxis], 3, axis=2)
        rgb_image = image[:, :, :3] if has_alpha else image
        alpha = image[:, :, 3] if has_alpha else None

        img = rgb_image.astype(np.float32) / 255.0
        img = torch.from_numpy(np.transpose(img, (2, 0, 1))).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self._process_tensor(img)

        output = output.squeeze(0).float().cpu().clamp_(0, 1).numpy()
        output = np.transpose(output, (1, 2, 0))
        output = (output * 255.0).round().astype(np.uint8)

        if outscale != self.scale:
            output_size = (
                max(1, int(output.shape[1] * outscale / self.scale)),
                max(1, int(output.shape[0] * outscale / self.scale)),
            )
            output = np.asarray(Image.fromarray(output).resize(output_size, Image.BICUBIC))

        if alpha is not None:
            alpha = np.asarray(
                Image.fromarray(alpha).resize((output.shape[1], output.shape[0]), Image.BICUBIC)
            )
            output = np.dstack((output, alpha))

        return output, None


@st.cache_resource
def get_realesrgan_upsampler():
    model_path = _resolve_realesrgan_model_path()
    if model_path is None:
        return None

    try:
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from realesrgan import RealESRGANer
    except Exception:
        try:
            return _LocalRealESRGANUpsampler(model_path, tile=256, tile_pad=10)
        except Exception as exc:
            st.error(f"Real-ESRGAN upscaler could not be loaded: {exc}")
            return None

    try:
        model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=4,
        )
        use_cuda = torch.cuda.is_available()
        return RealESRGANer(
            scale=4,
            model_path=str(model_path),
            model=model,
            tile=256,
            tile_pad=10,
            pre_pad=0,
            half=use_cuda,
            gpu_id=0 if use_cuda else None,
        )
    except Exception as exc:
        st.error(f"Real-ESRGAN upscaler could not be loaded: {exc}")
        return None


@st.cache_data(show_spinner=False, max_entries=16)
def upscale_image(image_content, scale=2):
    upsampler = get_realesrgan_upsampler()
    if upsampler is None:
        return None

    try:
        scale = int(scale)
        if scale not in (2, 4):
            scale = 2
        input_image = Image.open(BytesIO(image_content))
        if input_image.mode == "RGBA":
            image = np.asarray(input_image)
        else:
            image = np.asarray(input_image.convert("RGB"))

        with torch.no_grad():
            output, _ = upsampler.enhance(image, outscale=scale)

        out = BytesIO()
        Image.fromarray(output).save(out, format="PNG")
        return out.getvalue()
    except UnidentifiedImageError:
        st.error("Real-ESRGAN could not read this image.")
        return None
    except RuntimeError as exc:
        st.error(f"Real-ESRGAN upscale failed. Try a smaller image or 2x scale. Error: {exc}")
        return None
    except Exception as exc:
        st.error(f"Real-ESRGAN upscale failed: {exc}")
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
