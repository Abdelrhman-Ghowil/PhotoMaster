import streamlit as st
from google.oauth2 import service_account
from googleapiclient.discovery import build


@st.cache_data
def authenticate_gdrive():
    credentials = service_account.Credentials.from_service_account_file("credentials.json")
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
