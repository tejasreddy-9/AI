import os
import shutil
from tempfile import NamedTemporaryFile

UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER", "/app/uploaded_files")

def save_file(uploaded_file):
    """Save uploaded file to the upload folder and return the path."""
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)

    file_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    return file_path