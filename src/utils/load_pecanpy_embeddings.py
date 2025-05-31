import numpy as np
import torch

from src.utils.embedding_id_mapper import IDMapper

def load_pecanpy_embeddings(
    emb_path: str,
    id_mapper: IDMapper,
    item_num: int,
    hidden_units: int,
    padding_idx: int
) -> torch.FloatTensor:
    """
    Đọc file .emb do PecanPy sinh ra, gán vào một ma trận numpy (sau đổi sang torch).
    - emb_path: đường dẫn tới file .emb
    - id_mapper: instance của IDMapper đã được load mapping ASIN→index
    - item_num: số lượng item (không kể padding), các chỉ số item hợp lệ là 0..item_num-1
    - hidden_units: chiều của vector embedding (ví dụ 128)
    - padding_idx: giá trị index dành cho padding (thường bằng item_num)
    Trả về Tensor shape = (item_num+1, hidden_units)
    """

    # 1. Khởi tạo ma trận (item_num+1)×hidden_units, mặc định zero
    #    phần tử ở index = padding_idx sẽ giữ zero.
    emb_matrix = np.zeros((item_num + 1, hidden_units), dtype=np.float32)

    with open(emb_path, "r") as f:
        first = f.readline().strip().split()
        try:
            num_nodes = int(first[0])
            dim       = int(first[1])
            if dim != hidden_units:
                raise ValueError(f"File .emb dimension ({dim}) != hidden_units ({hidden_units})")
        except (ValueError, IndexError):
            f.seek(0)

        for line in f:
            parts = line.strip().split()
            if len(parts) != hidden_units + 1:
                continue

            asin = parts[0]               # ví dụ "B01K8B8YA8"
            vec_vals = parts[1:]          # 128 giá trị dưới dạng chuỗi

            # Map ASIN sang chỉ số item_index
            idx = id_mapper.get_item_index(asin)
            if idx < 0 or idx >= item_num:
                continue

            vec = np.array([float(x) for x in vec_vals], dtype=np.float32)
            if vec.shape[0] != hidden_units:
                continue

            emb_matrix[idx] = vec

    # Đảm bảo padding_idx (thường = item_num) là zero-vector
    emb_matrix[padding_idx] = np.zeros(hidden_units, dtype=np.float32)

    return torch.from_numpy(emb_matrix)  # shape = (item_num+1, hidden_units)
