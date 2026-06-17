from collections import defaultdict
from io import BytesIO
import os

import streamlit as st
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseDownload
from google.oauth2 import service_account
from googleapiclient.discovery import build


FOLDER_MIME_TYPE = "application/vnd.google-apps.folder"


@st.cache_resource
def authenticate_gdrive():
    credentials_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "credentials.json")
    credentials = service_account.Credentials.from_service_account_file(credentials_path)
    return build("drive", "v3", credentials=credentials)


def get_files_from_folder(folder_id, service):
    results = service.files().list(
        q=f"'{folder_id}' in parents and (mimeType contains 'image/' or mimeType contains 'application/pdf')",
        pageSize=1000,
        fields="files(id, name, mimeType)",
    ).execute()
    return results.get("files", [])


def convert_drive_file(file_id):
    return f"https://drive.google.com/uc?export=download&id={file_id}"


def download_drive_file(file_id):
    service = authenticate_gdrive()
    request = service.files().get_media(fileId=file_id, supportsAllDrives=True)
    file_buffer = BytesIO()
    downloader = MediaIoBaseDownload(file_buffer, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    file_buffer.seek(0)
    return file_buffer.getvalue()


def _safe_drive_name(name):
    return str(name or "untitled").strip() or "untitled"


def _dedupe_file_name(file_name, seen_names):
    safe_name = _safe_drive_name(file_name)
    if "." in safe_name:
        base, ext = safe_name.rsplit(".", 1)
        ext = f".{ext}"
    else:
        base, ext = safe_name, ""

    count = seen_names[safe_name.lower()]
    seen_names[safe_name.lower()] += 1
    if count == 0:
        return safe_name, False
    return f"{base}_{count}{ext}", True


def _list_children(folder_id, service):
    children = []
    page_token = None

    while True:
        response = service.files().list(
            q=(
                f"'{folder_id}' in parents and trashed = false and "
                f"(mimeType contains 'image/' or mimeType = 'application/pdf' or mimeType = '{FOLDER_MIME_TYPE}')"
            ),
            pageSize=1000,
            pageToken=page_token,
            fields="nextPageToken, files(id, name, mimeType)",
            supportsAllDrives=True,
            includeItemsFromAllDrives=True,
        ).execute()
        children.extend(response.get("files", []))
        page_token = response.get("nextPageToken")
        if not page_token:
            break

    return children


@st.cache_data(show_spinner=False, ttl=900)
def scan_drive_image_library(folder_id):
    service = authenticate_gdrive()
    stack = [(folder_id, "")]
    visited_folder_ids = set()
    seen_names = defaultdict(int)
    files = []
    stats = {
        "folders_scanned": 0,
        "inaccessible_folders": 0,
        "duplicates_renamed": 0,
        "images_found": 0,
        "pdfs_found": 0,
    }
    warnings = []

    while stack:
        current_folder_id, current_path = stack.pop()
        if current_folder_id in visited_folder_ids:
            continue
        visited_folder_ids.add(current_folder_id)

        try:
            children = _list_children(current_folder_id, service)
        except HttpError as exc:
            stats["inaccessible_folders"] += 1
            warnings.append(f"Skipped inaccessible folder: {current_path or current_folder_id} ({exc.resp.status})")
            continue
        except Exception as exc:
            stats["inaccessible_folders"] += 1
            warnings.append(f"Skipped inaccessible folder: {current_path or current_folder_id} ({exc})")
            continue

        stats["folders_scanned"] += 1

        for child in children:
            child_id = child["id"]
            child_name = _safe_drive_name(child.get("name"))
            mime_type = child.get("mimeType", "")
            child_path = f"{current_path}/{child_name}" if current_path else child_name

            if mime_type == FOLDER_MIME_TYPE:
                stack.append((child_id, child_path))
                continue

            if not mime_type.startswith("image/") and mime_type != "application/pdf":
                continue

            safe_name, was_duplicate = _dedupe_file_name(child_name, seen_names)
            if was_duplicate:
                stats["duplicates_renamed"] += 1

            if mime_type.startswith("image/"):
                stats["images_found"] += 1
            elif mime_type == "application/pdf":
                stats["pdfs_found"] += 1

            files.append(
                {
                    "id": child_id,
                    "name": safe_name,
                    "original_name": child_name,
                    "mimeType": mime_type,
                    "path": child_path,
                    "download_url": convert_drive_file(child_id),
                }
            )

    files.sort(key=lambda file: (file["path"].lower(), file["name"].lower()))
    stats["files_found"] = len(files)
    return {"files": files, "stats": stats, "warnings": warnings}
