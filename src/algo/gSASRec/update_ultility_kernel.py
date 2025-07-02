import json
import os
import subprocess
import time

from kaggle.api.kaggle_api_extended import KaggleApi
from slugify import slugify  # pip install python-slugify

# Khởi tạo API
api = KaggleApi()
api.authenticate()  # Load từ ~/.kaggle/kaggle.json

utility_files = [
    "config.py",
    "config_electronics.py",
    "model.py",
    "utils_dataset.py",
    "utils_validation.py",
    "pre_process_data.py",
    "transformer_decoder.py",
    "utils.py",
]

with open("kernel-metadata.json", "r") as f:
    metadata = json.load(f)

for file in utility_files:
    if not os.path.exists(file):
        print(f"⚠️ File {file} không tồn tại, bỏ qua!")
        continue
    
    base_name = os.path.splitext(file)[0]
    clean_slug = slugify(base_name)
    kernel_id = f"hodinhtrieu/{clean_slug}"  # Đúng format yêu cầu

    metadata.update({
        "id": kernel_id,
        "title": base_name.replace("_", " ").title(),
        "language": "python",
        "kernel_type": "script",
        "code_file": file,
        "is_private": True,
        "enable_gpu": False,
        "enable_internet": True,
        "keywords": ["util-script"],
    })

    if file == "utils.py":
        metadata["kernel_sources"] = ["hodinhtrieu/config"] 

    # Lưu metadata
    with open("kernel-metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    try:
        print(f"🚀 Uploading {file} as {kernel_id}...")
        # Sử dụng kernels_push với tham số rõ ràng
        api.kernels_push(
            folder=".",
        )
        print(f"✅ Uploaded {kernel_id} successfully!")
        time.sleep(5)  # Đảm bảo không vượt rate limit
        
    except Exception as e:
        print(f"❌ Failed to upload {kernel_id}: {str(e)}")
        if "rate limit" in str(e).lower():
            print("🕒 Hit rate limit, waiting 60 seconds...")
            time.sleep(60)