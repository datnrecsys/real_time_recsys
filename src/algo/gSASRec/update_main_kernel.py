import json
import os

from kaggle.api.kaggle_api_extended import KaggleApi
from slugify import slugify

# 1. CẤU HÌNH THÔNG TIN UPLOAD
USERNAME = "hodinhtrieu"  # Thay bằng username Kaggle của bạn
SCRIPT_FILE = "train_gsasrec.py"  # File script bạn muốn upload
KERNEL_TITLE = "Train gSASRec"  # Tiêu đề hiển thị trên Kaggle
IS_PRIVATE = True  # True nếu muốn private, False để public

# 2. KHỞI TẠO API
api = KaggleApi()
api.authenticate()  # Yêu cầu file kaggle.json trong ~/.kaggle/

# 3. TẠO METADATA
if not os.path.exists(SCRIPT_FILE):
    raise FileNotFoundError(f"File {SCRIPT_FILE} không tồn tại!")

slug = slugify(KERNEL_TITLE.lower())
kernel_id = f"{USERNAME}/{slug}"

metadata = {
    "id": kernel_id,
    "title": KERNEL_TITLE,
    "code_file": SCRIPT_FILE,
    "language": "python",
    "kernel_type": "script",
    "is_private": IS_PRIVATE,
    "enable_gpu": True,
    "enable_internet": True,
    "sources": [SCRIPT_FILE],
    "keywords": ["gsasrec", "sasrec", "recommender-system"],
    "kernel_sources": ["hodinhtrieu/config","hodinhtrieu/config-electronics","hodinhtrieu/utils-validation",
                       "hodinhtrieu/model","hodinhtrieu/utils-dataset","hodinhtrieu/utils","hodinhtrieu/pre-process-data"],  # Thay đổi nếu cần thiết
    "dataset_sources": ["hodinhtrieu/amazon-electronics-0-1"],  # Danh sách các dataset nếu có
}

# 4. LƯU METADATA
with open("kernel-metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

# 5. UPLOAD LÊN KAGGLE
try:
    print(f"🚀 Đang upload {SCRIPT_FILE}...")
    api.kernels_push(
        folder=".")
    print(f"✅ Upload thành công! Truy cập tại: https://kaggle.com/{kernel_id}")
except Exception as e:
    print(f"❌ Lỗi khi upload: {str(e)}")