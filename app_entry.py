import os
import re
import shutil
from collections import defaultdict
from io import BytesIO

# Silence TensorFlow/oneDNN startup warnings when using transformers with PyTorch.
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import pandas as pd
import pypdfium2 as pdfium
import streamlit as st
import streamlit.elements.image as st_image
from PIL import Image


def _patch_drawable_canvas_image_to_url():
    if hasattr(st_image, "image_to_url"):
        return

    try:
        from streamlit.elements.lib.image_utils import image_to_url
        from streamlit.elements.lib.layout_utils import create_layout_config
    except ModuleNotFoundError:
        return

    def _canvas_image_to_url(image, width, clamp, channels, output_format, image_id):
        layout_config = create_layout_config(width=width, allow_content_width=True)
        return image_to_url(image, layout_config, clamp, channels, output_format, image_id)

    st_image.image_to_url = _canvas_image_to_url


_patch_drawable_canvas_image_to_url()

from streamlit_drawable_canvas import st_canvas

from services.excel_service import extract_links, rename_images_based_on_sheet
from services.google_drive_service import (
    authenticate_gdrive,
    convert_drive_file,
    download_drive_file,
    get_files_from_folder,
    scan_drive_image_library,
)
from services.image_service import (
    combine_with_background,
    convert_and_compress_image,
    convert_drive_link,
    convert_pdf_to_images,
    crop_image,
    download_all_images_as_zip,
    download_image,
    erase_object,
    fit_image_to_canvas,
    flip_image,
    move_image,
    remove_background,
    resize_image,
    zoom_image,
)


ADVANCED_OPTION_PREFIXES = (
    "per_add_bg_",
    "remove_bg_",
    "disable_auto_resize_",
    "zoom_enabled_",
    "zoom_value_",
    "move_enabled_",
    "move_x_",
    "move_y_",
    "adv_resize_enabled_",
    "adv_resize_threshold_",
    "adv_resize_custom_size_",
    "adv_resize_width_",
    "adv_resize_height_",
    "crop_enabled_",
    "crop_left_",
    "crop_right_",
    "crop_top_",
    "crop_bottom_",
    "object_erase_",
    "object_erase_brush_",
)

DEFAULT_IMAGE_LIBRARY_FOLDER_LINK = "https://drive.google.com/drive/folders/1rOMd51BaqDQ38l3SzygRwWtGH7rGeHtY"


def _reset_advanced_options_state():
    keys_to_clear = [
        key
        for key in list(st.session_state.keys())
        if any(key.startswith(prefix) for prefix in ADVANCED_OPTION_PREFIXES)
    ]
    for key in keys_to_clear:
        del st.session_state[key]


def _reset_advanced_options_for_index(index):
    for prefix in ADVANCED_OPTION_PREFIXES:
        key = f"{prefix}{index}"
        if key in st.session_state:
            del st.session_state[key]
    erased_key = f"object_erased_image_{index}"
    if erased_key in st.session_state:
        del st.session_state[erased_key]


def _extract_drive_folder_id(folder_link):
    match = re.search(r"/folders/([a-zA-Z0-9_-]+)", folder_link or "")
    if match:
        return match.group(1)

    match = re.search(r"[?&]id=([a-zA-Z0-9_-]+)", folder_link or "")
    if match:
        return match.group(1)

    stripped = str(folder_link or "").strip()
    if re.fullmatch(r"[a-zA-Z0-9_-]{20,}", stripped):
        return stripped

    return None


def _dedupe_image_info_names(images_info):
    counts = defaultdict(int)
    deduped = []
    duplicates = 0

    for name, content in images_info:
        safe_name = str(name or "image").strip() or "image"
        if "." in safe_name:
            base, ext = safe_name.rsplit(".", 1)
            ext = f".{ext}"
        else:
            base, ext = safe_name, ""

        key = safe_name.lower()
        count = counts[key]
        counts[key] += 1
        if count:
            safe_name = f"{base}_{count}{ext}"
            duplicates += 1
        deduped.append((safe_name, content))

    return deduped, duplicates


def _render_drive_image_library():
    with st.expander("Google Drive Image Library", expanded=False):
        st.caption("Search nested Drive folders and add only the selected image to the current batch.")
        library_link = st.text_input(
            "Library folder link",
            value=DEFAULT_IMAGE_LIBRARY_FOLDER_LINK,
            key="drive_library_link",
        )
        folder_id = _extract_drive_folder_id(library_link)

        if st.button("Scan Library", key="drive_library_scan"):
            if not folder_id:
                st.error("Enter a valid Google Drive folder link.")
            else:
                with st.spinner("Scanning Drive folders..."):
                    st.session_state["drive_library_scan_result"] = scan_drive_image_library(folder_id)

        scan_result = st.session_state.get("drive_library_scan_result")
        if not scan_result:
            return []

        stats = scan_result["stats"]
        stat_cols = st.columns(4)
        stat_cols[0].metric("Images", stats.get("images_found", 0))
        stat_cols[1].metric("Folders", stats.get("folders_scanned", 0))
        stat_cols[2].metric("Renamed", stats.get("duplicates_renamed", 0))
        stat_cols[3].metric("Skipped", stats.get("inaccessible_folders", 0))

        for warning in scan_result.get("warnings", [])[:5]:
            st.warning(warning)
        if len(scan_result.get("warnings", [])) > 5:
            st.warning(f"{len(scan_result['warnings']) - 5} more folders were skipped.")

        image_files = [
            file
            for file in scan_result["files"]
            if file.get("mimeType", "").startswith("image/")
        ]
        if not image_files:
            st.info("No image files were found in this library.")
            return st.session_state.get("drive_library_selected_images", [])

        search_query = st.text_input("Search images", key="drive_library_search").strip().lower()
        if search_query:
            image_files = [
                file
                for file in image_files
                if search_query in file["name"].lower() or search_query in file["path"].lower()
            ]
        if not image_files:
            st.info("No images match this search.")
            return st.session_state.get("drive_library_selected_images", [])

        st.caption(f"Showing {min(len(image_files), 12)} of {len(image_files)} matching images.")
        gallery_files = image_files[:12]

        gallery_cols = st.columns(4)
        checked_file_ids = []
        for index, file in enumerate(gallery_files):
            with gallery_cols[index % 4]:
                is_checked = st.checkbox(
                    file["name"],
                    key=f"drive_library_checked_{file['id']}",
                )
                st.caption(file["path"])
                if is_checked:
                    checked_file_ids.append(file["id"])

        checked_files = [file for file in gallery_files if file["id"] in checked_file_ids]
        if checked_files:
            st.caption(f"{len(checked_files)} image(s) checked.")
            if st.button("Preview First Checked Image", key="drive_library_preview_checked"):
                preview_file = checked_files[0]
                with st.spinner(f"Loading preview for {preview_file['name']}..."):
                    try:
                        st.session_state["drive_library_preview"] = (
                            preview_file["id"],
                            download_drive_file(preview_file["id"]),
                        )
                    except Exception as exc:
                        st.error(f"Could not preview {preview_file['name']}: {exc}")

            preview = st.session_state.get("drive_library_preview")
            preview_file = next((file for file in checked_files if preview and file["id"] == preview[0]), None)
            if preview_file:
                st.image(preview[1], caption=preview_file["name"], use_container_width=True)

            if st.button("Add Checked Images", key="drive_library_add_checked"):
                selected_images = st.session_state.setdefault("drive_library_selected_images", [])
                selected_file_ids = st.session_state.setdefault("drive_library_selected_file_ids", [])
                existing_file_ids = set(selected_file_ids)
                added_count = 0
                skipped_count = 0
                for file in checked_files:
                    if file["id"] in existing_file_ids:
                        skipped_count += 1
                        continue

                    preview = st.session_state.get("drive_library_preview")
                    if preview and preview[0] == file["id"]:
                        image_content = preview[1]
                    else:
                        with st.spinner(f"Downloading {file['name']}..."):
                            try:
                                image_content = download_drive_file(file["id"])
                            except Exception as exc:
                                st.error(f"Could not download {file['name']}: {exc}")
                                image_content = None
                    if image_content:
                        selected_images.append((file["name"], image_content))
                        selected_file_ids.append(file["id"])
                        existing_file_ids.add(file["id"])
                        added_count += 1
                selected_images, duplicate_count = _dedupe_image_info_names(selected_images)
                st.session_state["drive_library_selected_images"] = selected_images
                st.session_state["drive_library_selected_file_ids"] = selected_file_ids
                if duplicate_count:
                    st.info(f"Renamed {duplicate_count} duplicate selected image name.")
                if added_count:
                    st.success(f"Added {added_count} image(s) to the current workflow.")
                elif skipped_count:
                    st.info("No new images were added because the selected images already exist.")
                if added_count and skipped_count:
                    st.info(f"Skipped {skipped_count} image(s) that already exist.")

        selected_images = st.session_state.get("drive_library_selected_images", [])
        if selected_images:
            st.write(f"{len(selected_images)} selected image(s) ready in this workflow.")
            if st.button("Clear Selected Library Images", key="drive_library_clear_selected"):
                st.session_state["drive_library_selected_images"] = []
                st.session_state["drive_library_selected_file_ids"] = []
                st.rerun()

        return st.session_state.get("drive_library_selected_images", [])


def _build_object_eraser_mask(canvas_image_data, background_image):
    if canvas_image_data is None:
        return None

    canvas_image = Image.fromarray(canvas_image_data.astype("uint8"), "RGBA").convert("RGB")
    background = background_image.convert("RGB")
    if canvas_image.size != background.size:
        background = background.resize(canvas_image.size, Image.LANCZOS)

    canvas_pixels = canvas_image.load()
    background_pixels = background.load()
    mask = Image.new("L", canvas_image.size, 0)
    mask_pixels = mask.load()

    has_strokes = False
    for y in range(canvas_image.height):
        for x in range(canvas_image.width):
            cr, cg, cb = canvas_pixels[x, y]
            br, bg, bb = background_pixels[x, y]
            diff = abs(cr - br) + abs(cg - bg) + abs(cb - bb)
            if diff > 60 and cr > 180 and cb > 180 and cg < 140:
                mask_pixels[x, y] = 255
                has_strokes = True

    return mask if has_strokes else None


def _render_object_eraser(image_content, index, name):
    original = Image.open(BytesIO(image_content)).convert("RGB")
    max_canvas_width = 360
    max_canvas_height = 420
    scale = min(
        1.0,
        max_canvas_width / original.width,
        max_canvas_height / original.height,
    )
    canvas_width = max(1, int(original.width * scale))
    canvas_height = max(1, int(original.height * scale))
    canvas_background = original.resize((canvas_width, canvas_height), Image.LANCZOS)

    brush_size = st.slider(
        "Brush size",
        min_value=5,
        max_value=120,
        value=35,
        step=5,
        key=f"object_erase_brush_{index}",
    )
    canvas_result = st_canvas(
        fill_color="rgba(255, 0, 255, 0.35)",
        stroke_width=brush_size,
        stroke_color="#FF00FF",
        background_image=canvas_background,
        update_streamlit=True,
        height=canvas_height,
        width=canvas_width,
        drawing_mode="freedraw",
        display_toolbar=True,
        key=f"object_erase_canvas_{index}",
    )

    if not st.button("Apply Object Eraser", key=f"object_erase_apply_{index}"):
        return None

    mask = _build_object_eraser_mask(canvas_result.image_data, canvas_background)
    if mask is None:
        st.warning(f"Brush over the object in {name}, then apply Object Eraser.")
        return None

    if mask.size != original.size:
        mask = mask.resize(original.size, Image.NEAREST)

    with st.spinner(f"Erasing object in {name}..."):
        erased_image = erase_object(image_content, mask)
    if erased_image:
        st.session_state[f"object_erased_image_{index}"] = erased_image
    return erased_image


def run_app():
    st.set_page_config(page_title="PhotoMaster", page_icon="🖼️")
    st.markdown(
        """
        <style>
        [data-testid="stCheckbox"] p {
            font-size: 0.82rem !important;
            line-height: 1.15 !important;
        }
        .pm-title {
            display: flex;
            align-items: center;
            gap: 0.55rem;
            margin: 0.1rem 0 0.9rem 0;
            font-size: 2.15rem;
            font-weight: 700;
            line-height: 1.05;
        }
        .pm-title img {
            height: 1em;
            width: auto;
            display: inline-block;
        }
        .pm-title svg {
            height: 1em;
            width: auto;
            display: inline-block;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="pm-title">
            <svg xmlns:xlink="http://www.w3.org/1999/xlink" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 112 36" width="134" height="44"><g fill="#39C1CB" clip-path="url(#Logo_svg__a)"><path d="M26.87 17.582c.354-1.543.537-2.925.558-4.178.14-5.293-2.48-7.865-6.54-7.875-3.91 0-6.444 1.907-8.484 5.871l1.117-4.896H3.415L0 21.61l2.524-.097-1.44 6.332H11.33l1.525-6.739-2.524.097.956-4.243c.634-2.122 1.665-3.236 3.35-3.215 1.429 0 2.213.815 2.16 2.508-.012.439-.087.985-.194 1.542L15.862 21l2.524-.096-1.579 6.964h10.245l1.665-7.393-2.524.107.677-2.978zM39.274 0c-3.49 0-6.004 1.66-6.057 4.04-.054 2.1 2.03 3.567 5.004 3.567 3.426 0 5.907-1.714 5.971-4.168C44.235 1.457 42.205.011 39.274 0M42.506 9.075H32.261l-2.567 11.379 2.524-.108-1.697 7.479h10.245l1.805-7.907-2.524.096zM70.213 27.825l2.052-9.075-2.524.096.3-1.296c.377-1.543.538-2.925.591-4.179.14-5.292-2.502-7.842-6.55-7.842-3.91 0-6.466 1.885-8.485 5.85l1.117-4.897H46.587l-3.04 13.425 2.525-.107-1.805 8.014h10.246l1.911-8.443-2.502.108.57-2.55c.58-2.111 1.642-3.215 3.307-3.225 1.428 0 2.212.814 2.18 2.507-.022.46-.086 1.007-.193 1.543l-.355 1.5 2.524-.097-1.965 8.657h10.223zM81.597 7.596c3.394.011 5.885-1.703 5.95-4.146C87.6 1.436 85.592 0 82.67 0c-3.49 0-6.046 1.65-6.11 4.029-.054 2.1 2.03 3.567 5.025 3.567zM75.572 9.032l-2.169 9.718 2.502-.107-2.05 9.043c-.355 1.564-1.14 2.207-2.364 2.218-.698 0-1.127-.204-1.514-.472l-1.31 5.7c1.138.46 3.018.857 5.037.857 6.078 0 9.128-2.475 10.256-7.607l2.255-9.986 2.341-.096c-.215.343-.44.707-.59 1.029-2.846 6.096 1.578 9.342 5.412 9.342 3.265.011 5.434-1.403 7.109-3.492l-.569 2.657h9.848L112 17.346l-2.502.108c.086-.611.161-1.222.15-1.8.172-6.333-3.877-9.975-10.954-9.975-4.468 0-7.571 1.103-10.17 2.56l-1.611 7.007c1.955-1.146 4.736-1.907 7.346-1.928 3.264.01 5.337 1.371 5.262 4.371v.097c-1.138-1.854-3.093-2.797-5.96-2.818-3.147.01-5.96 1.114-7.475 3.257l-2.352.096 2.084-9.289zm22.682 14.904c-1.525-.011-2.427-.836-2.395-1.94.021-1.446 1.407-2.325 3.136-2.325 1.471 0 2.405.804 2.395 1.918-.054 1.425-1.429 2.347-3.136 2.357z" fill="#39C1CB"></path></g><defs><clipPath id="Logo_svg__a"><path fill="#fff" d="M0 0h112v36H0z"></path></clipPath></defs></svg>
            <span>PhotoMaster</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    threshold = 2.0
    global_zoom_enabled = False
    global_zoom_value = 1.0
    disable_auto_resize = False
    compress_convert_enabled = False
    object_erase_enabled = False
    output_format = "png"
    output_quality = 90

    option_row1 = st.columns(4)
    with option_row1[0]:
        remove_bg = st.checkbox("📷 Remove BG")
    with option_row1[1]:
        add_bg = st.checkbox("📜 Add BG")
    with option_row1[2]:
        resize_fg = st.checkbox("🔛 Resize")
    with option_row1[3]:
        disable_auto_resize = st.checkbox("🚫 Disable Auto Resize")

    option_row2 = st.columns(4)
    with option_row2[0]:
        global_zoom_enabled = st.checkbox("🔍 Global Zoom")
    with option_row2[1]:
        compress_convert_enabled = st.checkbox("👊 Compress/Convert")

    if global_zoom_enabled:
        global_zoom_value = st.slider("Global zoom level", 0.5, 2.0, 1.0, 0.05)
    if resize_fg:
        advanced_cols = st.columns([1, 2])
        with advanced_cols[0]:
            udvanced = st.checkbox("💎Advanced Resize")
        with advanced_cols[1]:
            if udvanced:
                threshold = st.slider("Aspect Ratio Threshold", 1.0, 2.5, 1.5)
    if compress_convert_enabled:
        compress_cols = st.columns([1, 2])
        with compress_cols[0]:
            output_format = st.selectbox(
                "Output format",
                options=["png", "jpeg", "webp"],
                index=0,
            )
        with compress_cols[1]:
            output_quality = st.slider(
                "Output quality",
                min_value=40,
                max_value=100,
                value=90,
                step=1,
            )

    uploaded_files = st.file_uploader(
        "",
        type=["xlsx", "csv", "jpg", "jpeg", "png", "jfif", "avif", "webp", "heic", "NEF", "ARW", "tiff", "pdf"],
        accept_multiple_files=True,
    )
    folder_link = st.text_input("Enter Google Drive Link for (**Larger Files**)")
    selected_library_images = _render_drive_image_library()

    images_info = []
    if selected_library_images:
        images_info.extend(selected_library_images)

    if uploaded_files:
        if len(uploaded_files) == 1 and uploaded_files[0].name.endswith((".xlsx", ".csv")):
            st.write("Select the type of images in the Excel file:")
            images_type = st.radio("Images are:", ["Links of images", "Embedded in Excel file"])
            file_type = "excel"
        elif all(file.type.startswith("image/") for file in uploaded_files):
            file_type = "images"
        elif any(file.type == "application/pdf" for file in uploaded_files):
            file_type = "pdf"
        else:
            file_type = "mixed"

        if file_type == "mixed":
            st.error("You should work with one type of file: either an Excel file, images, or a PDF.")
        else:
            if file_type == "excel" and images_type == "Links of images":
                uploaded_file = uploaded_files[0]
                if uploaded_file.name.endswith(".xlsx"):
                    xl = pd.ExcelFile(uploaded_file)

                    for sheet_name in xl.sheet_names:
                        st.write(f"Processing sheet: {sheet_name}")
                        df = xl.parse(sheet_name)
                        if "links" in df.columns and "name" in df.columns:
                            df.dropna(subset=["links"], inplace=True)
                            name_count = defaultdict(int)
                            empty_count = 0
                            unique_images_info = []
                            links = extract_links(uploaded_file, links_column="links")

                            for name, link in zip(df["name"], links):
                                if pd.isna(name) or str(name).strip() == "":
                                    empty_name = f"empty_{empty_count}" if empty_count > 0 else "empty"
                                    name = empty_name
                                    empty_count += 1
                                if name_count[name] > 0:
                                    unique_name = f"{name}_{name_count[name]}"
                                else:
                                    unique_name = name
                                unique_images_info.append((unique_name, link))
                                name_count[name] += 1

                            images_info.extend(unique_images_info)
                            if empty_count > 0:
                                st.warning(f"Number of empty cells in 'name' column: {empty_count}")
                        else:
                            st.error(f"The sheet '{sheet_name}' must contain 'links' and 'name' columns.")
                else:
                    df = pd.read_csv(uploaded_file)
                    name_col = "name" if "name" in df.columns else "names" if "names" in df.columns else None
                    if "links" in df.columns and name_col:
                        df.dropna(subset=["links"], inplace=True)
                        name_count = defaultdict(int)
                        empty_count = 0
                        unique_images_info = []
                        for name, link in zip(df[name_col], df["links"]):
                            if pd.isna(name) or str(name).strip() == "":
                                empty_name = f"empty_{empty_count}" if empty_count > 0 else "empty"
                                name = empty_name
                                empty_count += 1
                            if name_count[name] > 0:
                                unique_name = f"{name}_{name_count[name]}"
                            else:
                                unique_name = name
                            unique_images_info.append((unique_name, link))
                            name_count[name] += 1
                        images_info.extend(unique_images_info)
                        if empty_count > 0:
                            st.warning(f"Number of empty cells in 'name' column: {empty_count}")
                    else:
                        st.error("The uploaded file must contain 'links' and 'name' columns.")

            elif file_type == "excel" and images_type == "Embedded in Excel file":
                temp_dir = "temp"
                if not os.path.exists(temp_dir):
                    os.makedirs(temp_dir)

                file_path = os.path.join(temp_dir, uploaded_files[0].name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_files[0].getbuffer())

                output_dir = os.path.join(temp_dir, "extracted_images")
                if os.path.exists(output_dir):
                    shutil.rmtree(output_dir)
                os.makedirs(output_dir, exist_ok=True)

                rename_images_based_on_sheet(file_path, output_dir)
                images_info = [
                    (image, open(os.path.join(output_dir, image), "rb").read())
                    for image in os.listdir(output_dir)
                ]

            elif file_type == "images":
                images_info = [(file.name, file) for file in uploaded_files]

            elif file_type == "pdf":
                images_info = []
                for uploaded_file in uploaded_files:
                    pdf = pdfium.PdfDocument(uploaded_file)
                    fn = uploaded_file.name
                    for i in range(len(pdf)):
                        page = pdf[i]
                        image = page.render(scale=1.45).to_pil()
                        img_byte_arr = BytesIO()
                        image.save(img_byte_arr, format="JPEG")
                        if i == 0:
                            images_info.append((f"{fn.rsplit('.', 1)[0]}.jpg", img_byte_arr.getvalue()))
                        else:
                            images_info.append((f"{fn.rsplit('.', 1)[0]}_page_{i + 1}.jpg", img_byte_arr.getvalue()))

    if folder_link:
        folder_id_match = re.search(r"/folders/([a-zA-Z0-9_-]+)", folder_link)
        if folder_id_match:
            folder_id = folder_id_match.group(1)
            service = authenticate_gdrive()
            files = get_files_from_folder(folder_id, service)

            if files:
                st.write(f"Found {len(files)} files in the folder.")
                for file in files:
                    file_id = file["id"]
                    file_name = file["name"]
                    mime_type = file["mimeType"]

                    if mime_type.startswith("image/"):
                        image_url = convert_drive_file(file_id)
                        image_content = download_image(image_url)
                        if image_content:
                            images_info.append((file_name, image_content))

                    elif mime_type == "application/pdf":
                        pdf_url = convert_drive_file(file_id)
                        pdf_content = download_image(pdf_url)
                        if pdf_content:
                            pdf_images = convert_pdf_to_images(pdf_content)
                            for img in pdf_images:
                                img_byte_arr = BytesIO()
                                img.save(img_byte_arr, format="PNG")
                                images_info.append((f"{file_name.rsplit('.', 1)[0]}.png", img_byte_arr.getvalue()))

    if images_info:
        images_info, duplicate_count = _dedupe_image_info_names(images_info)
        if duplicate_count:
            st.info(f"Renamed {duplicate_count} duplicate filename(s) in the current workflow.")

    if images_info:
        bg_image = None
        any_per_image_add_bg = any(
            bool(value) for key, value in st.session_state.items() if key.startswith("per_add_bg_")
        )
        if add_bg or any_per_image_add_bg:
            bg_file = st.file_uploader("Upload background image", type=["jpg", "jpeg", "png"])
            if bg_file:
                bg_image = resize_image(bg_file.read())
            else:
                with open("./Bg.png", "rb") as file:
                    bg_image = resize_image(file.read())

        st.markdown("## Preview")
        flip_horizontal_global = False
        flip_vertical_global = False
        cols = st.columns(2)

        for i, (name, url_or_file) in enumerate(images_info):
            col = cols[i % 2]
            with col:
                original_name = name
                if isinstance(url_or_file, str):
                    url = convert_drive_link(url_or_file)
                    image_content = download_image(url)
                elif isinstance(url_or_file, bytes):
                    image_content = url_or_file
                else:
                    image_content = url_or_file.read()

                if image_content:
                    flip_col1, flip_col2, rename_col3 = st.columns(3)
                    with flip_col1:
                        flip_horizontal = st.checkbox("Flip H🔁", key=f"flip_horizontal_{i}")
                    with flip_col2:
                        flip_vertical = st.checkbox("Flip V🔃", key=f"flip_vertical_{i}")
                    with rename_col3:
                        rename_image = st.checkbox("Rename", key=f"rename_checkbox_{i}")
                    renamed_base = None

                    adv_resize_enabled = False
                    per_image_disable_auto_resize = False
                    per_image_remove_bg = False
                    per_image_add_bg = False
                    custom_zoom_enabled = False
                    custom_zoom_value = 1.0
                    crop_enabled = False
                    crop_left = 0
                    crop_right = 0
                    crop_top = 0
                    crop_bottom = 0
                    per_image_object_erase = False
                    move_enabled = False
                    move_x = 0
                    move_y = 0
                    per_image_threshold = threshold
                    use_custom_size = False
                    custom_width = 1024
                    custom_height = 1024
                    if st.button("Reset", key=f"reset_adv_{i}"):
                        _reset_advanced_options_for_index(i)
                        st.rerun()
                    with st.expander("Avanced Options (Per Image)", expanded=False):
                        per_image_add_bg = st.checkbox(
                            "Add BG",
                            key=f"per_add_bg_{i}",
                        )
                        per_image_remove_bg = st.checkbox(
                            "Remove BG",
                            key=f"remove_bg_{i}",
                        )
                        per_image_disable_auto_resize = st.checkbox(
                            "Disable auto resize (1024x1024)",
                            key=f"disable_auto_resize_{i}",
                        )
                        custom_zoom_enabled = st.checkbox("Zoom", key=f"zoom_enabled_{i}")
                        if custom_zoom_enabled:
                            custom_zoom_value = st.slider(
                                "Zoom",
                                0.5,
                                2.0,
                                1.0,
                                0.05,
                                key=f"zoom_value_{i}",
                                label_visibility="collapsed",
                            )
                        crop_enabled = st.checkbox("Crop Tool", key=f"crop_enabled_{i}")
                        if crop_enabled:
                            crop_left = st.slider("Crop Left %", 0, 45, 0, 1, key=f"crop_left_{i}")
                            crop_right = st.slider("Crop Right %", 0, 45, 0, 1, key=f"crop_right_{i}")
                            crop_top = st.slider("Crop Top %", 0, 45, 0, 1, key=f"crop_top_{i}")
                            crop_bottom = st.slider("Crop Bottom %", 0, 45, 0, 1, key=f"crop_bottom_{i}")
                        per_image_object_erase = st.checkbox(
                            "Object Eraser",
                            key=f"object_erase_{i}",
                        )
                        move_enabled = st.checkbox("Move", key=f"move_enabled_{i}")
                        if move_enabled:
                            move_x = st.slider(
                                "L/R",
                                -300,
                                300,
                                0,
                                1,
                                key=f"move_x_{i}",
                            )
                            move_y = st.slider(
                                "T/B",
                                -300,
                                300,
                                0,
                                1,
                                key=f"move_y_{i}",
                            )
                        adv_resize_enabled = st.checkbox(
                            "Enable advanced resize",
                            key=f"adv_resize_enabled_{i}",
                        )
                        if adv_resize_enabled:
                            per_image_threshold = st.slider(
                                "Aspect ratio threshold",
                                1.0,
                                2.5,
                                float(threshold),
                                0.1,
                                key=f"adv_resize_threshold_{i}",
                            )
                            use_custom_size = st.checkbox(
                                "Set custom output size",
                                key=f"adv_resize_custom_size_{i}",
                            )
                            if use_custom_size:
                                custom_width = st.number_input(
                                    "Width",
                                    min_value=128,
                                    max_value=4096,
                                    value=1024,
                                    step=1,
                                    key=f"adv_resize_width_{i}",
                                )
                                custom_height = st.number_input(
                                    "Height",
                                    min_value=128,
                                    max_value=4096,
                                    value=1024,
                                    step=1,
                                    key=f"adv_resize_height_{i}",
                                )

                    if rename_image:
                        new_name = st.text_input(
                            f"Enter new name for {name} (without extension):",
                            key=f"rename_input_{i}",
                        )
                        if new_name and new_name.strip():
                            renamed_base = new_name.strip()

                    effective_object_erase = object_erase_enabled or per_image_object_erase
                    erased_image_key = f"object_erased_image_{i}"
                    if effective_object_erase:
                        st.caption("Brush the object area, then apply the eraser.")
                        erased_image = _render_object_eraser(image_content, i, name)
                        if erased_image:
                            image_content = erased_image
                        elif erased_image_key in st.session_state:
                            image_content = st.session_state[erased_image_key]

                    effective_remove_bg = remove_bg or per_image_remove_bg
                    if effective_remove_bg:
                        processed_image = remove_background(image_content)
                        ext = "png"
                    elif disable_auto_resize or per_image_disable_auto_resize:
                        image = Image.open(BytesIO(image_content))
                        image = fit_image_to_canvas(image, (1024, 1024))
                        img_byte_arr = BytesIO()
                        image.save(img_byte_arr, format="PNG")
                        processed_image = img_byte_arr.getvalue()
                        ext = "png"
                    else:
                        size = (1290, 789) if "banner" in original_name.lower() else (1024, 1024)
                        resize_threshold = threshold
                        if adv_resize_enabled:
                            resize_threshold = per_image_threshold
                            if use_custom_size:
                                size = (int(custom_width), int(custom_height))
                        processed_image = resize_image(
                            image_content,
                            size=size,
                            aspect_ratio_threshold=resize_threshold,
                        )
                        ext = "png"

                    if (
                        effective_remove_bg
                        and (disable_auto_resize or per_image_disable_auto_resize)
                        and processed_image
                    ):
                        image = Image.open(BytesIO(processed_image))
                        image = fit_image_to_canvas(image, (1024, 1024))
                        img_byte_arr = BytesIO()
                        image.save(img_byte_arr, format="PNG")
                        processed_image = img_byte_arr.getvalue()
                        ext = "png"

                    effective_add_bg = add_bg or per_image_add_bg
                    if effective_add_bg and bg_image and processed_image:
                        processed_image, _ = combine_with_background(
                            processed_image, bg_image, resize_foreground=resize_fg
                        )
                        ext = "png"

                    if crop_enabled and (crop_left or crop_right or crop_top or crop_bottom):
                        image = Image.open(BytesIO(processed_image))
                        processed_image = crop_image(
                            image,
                            left_pct=crop_left,
                            right_pct=crop_right,
                            top_pct=crop_top,
                            bottom_pct=crop_bottom,
                        )
                        img_byte_arr = BytesIO()
                        processed_image.save(img_byte_arr, format="PNG")
                        processed_image = img_byte_arr.getvalue()
                        ext = "png"

                    if flip_horizontal or flip_vertical:
                        image = Image.open(BytesIO(processed_image))
                        processed_image = flip_image(image, flip_horizontal, flip_vertical)
                        img_byte_arr = BytesIO()
                        processed_image.save(img_byte_arr, format="PNG")
                        processed_image = img_byte_arr.getvalue()

                    effective_zoom = global_zoom_value if global_zoom_enabled else 1.0
                    if custom_zoom_enabled:
                        effective_zoom = custom_zoom_value
                    if effective_zoom != 1.0:
                        image = Image.open(BytesIO(processed_image))
                        processed_image = zoom_image(image, effective_zoom)
                        img_byte_arr = BytesIO()
                        processed_image.save(img_byte_arr, format="PNG")
                        processed_image = img_byte_arr.getvalue()

                    if move_enabled and (move_x != 0 or move_y != 0):
                        image = Image.open(BytesIO(processed_image))
                        processed_image = move_image(image, move_x, move_y)
                        img_byte_arr = BytesIO()
                        processed_image.save(img_byte_arr, format="PNG")
                        processed_image = img_byte_arr.getvalue()

                    if processed_image:
                        mime = f"image/{ext}"
                        if compress_convert_enabled:
                            processed_image, ext, mime = convert_and_compress_image(
                                processed_image,
                                output_format=output_format,
                                quality=output_quality,
                            )
                        if renamed_base:
                            name = f"{renamed_base}.{ext}"
                        images_info[i] = (name, processed_image)
                        st.image(processed_image, caption=name)
                        st.download_button(
                            label=f"Download {name.rsplit('.', 1)[0]}",
                            data=processed_image,
                            file_name=f"{name.rsplit('.', 1)[0]}.{ext}",
                            mime=mime,
                            key=f"download_{i}",
                        )

                    flip_horizontal_global = flip_horizontal_global or flip_horizontal
                    flip_vertical_global = flip_vertical_global or flip_vertical

        if st.button("Download All Images", key="download_all"):
            zip_buffer = download_all_images_as_zip(
                images_info,
                remove_bg=remove_bg,
                add_bg=add_bg,
                bg_image=bg_image,
                resize_foreground=resize_fg,
                threshold=threshold,
                flip_horizontal=flip_horizontal_global,
                flip_vertical=flip_vertical_global,
                enable_compress_convert=compress_convert_enabled,
                output_format=output_format,
                quality=output_quality,
            )
            st.download_button(
                label="Download All Images as ZIP",
                data=zip_buffer,
                file_name="all_images.zip",
                mime="application/zip",
            )
