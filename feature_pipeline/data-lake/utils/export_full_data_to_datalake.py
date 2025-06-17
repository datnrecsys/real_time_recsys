import io
from datasets import load_dataset

def main():
    
    # Load rating dataset from Hugging Face
    amz_rating_raw = load_dataset(
        "0core_rating_only_Electronics",
        "raw_meta_Electronics",
        split = "full",
        trust_remote_code= True
    )
    # 1) Serialize DataFrame to Parquet in a BytesIO buffer
    buf = io.BytesIO()
    amz_rating_raw.to_parquet(buf, engine="pyarrow", index=False)
    buf.seek(0)  # rewind to the start
    
    


if __name__ == "__main__":
    main()
    