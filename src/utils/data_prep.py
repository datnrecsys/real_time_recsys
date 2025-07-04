import json
from typing import Callable, List

import numpy as np
import pandas as pd
from loguru import logger
from scipy.sparse import issparse
from sqlalchemy import Engine
from sqlalchemy.types import JSON
from tqdm.auto import tqdm


def parse_dt(df: pd.DataFrame, cols: List[str] = ["timestamp"]) -> pd.DataFrame:
    """Convert specified columns in the DataFrame to datetime format."""
    return df.assign(**{col: lambda df: pd.to_datetime(df[col].astype(int), unit = "ms") for col in cols})


def handle_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Convert the 'rating' column in the DataFrame to float type."""
    return df.assign(**{"rating": lambda df: df["rating"].astype(float), 
                        "parent_asin": lambda df: df["parent_asin"].astype(str)})


def chunk_ingest_decorator(chunk_size: int = 1000) -> Callable:
    """Decorator to ingest a DataFrame in chunks into an OLTP database."""

    def decorator(func: Callable) -> Callable:
        def wrapper(df: pd.DataFrame, engine: Engine, schema: str, table_name: str, **kwargs) -> None:
            """Wrapper function to handle chunking."""
            progress_bar = tqdm(range(0, len(df), chunk_size), desc="Ingesting chunks")

            for start in progress_bar:
                end = min(start + chunk_size, len(df))
                chunk_df = df.iloc[start:end]
                func(
                    chunk_df, engine, schema, table_name, **kwargs
                )  # Call the original function with the chunk

        return wrapper

    return decorator


@chunk_ingest_decorator(chunk_size=1000)
def insert_chunk_to_oltp(
    chunk_df: pd.DataFrame, engine: Engine, schema: str, table_name: str, **kwargs
) -> None:
    """Insert a chunk of the DataFrame into the OLTP database."""
    try:
        if "dtype" in kwargs:
            dtype = kwargs.get("dtype")
            if JSON in dtype.values():
                json_cols = [col for col, col_type in dtype.items() if col_type == JSON]
            
            # Handle JSON columns
            chunk_df = chunk_df.assign(
                **{col: lambda df: df[col].apply(json.loads) for col in json_cols}
            )

        chunk_df.to_sql(table_name, engine, schema=schema, if_exists="append", index=False, **kwargs)
    except Exception as e:
        logger.error(f"Error inserting chunk into OLTP: {e}")


def chunk_transform(df, pipeline, chunk_size=1000):
    transformed_chunks = []

    progress_bar = tqdm(range(0, df.shape[0], chunk_size), desc="Transforming chunks")

    # Iterate through the DataFrame in chunks
    for start in progress_bar:
        end = min(start + chunk_size, df.shape[0])
        chunk_df = df.iloc[start:end]

        # Apply the pipeline transformation to the chunk
        transformed_chunk = pipeline.transform(chunk_df)

        # Check if the transformed output is sparse, and convert to dense
        if issparse(transformed_chunk):
            transformed_chunk = transformed_chunk.toarray()

        # Collect the transformed chunk
        transformed_chunks.append(transformed_chunk)

    # Concatenate the transformed chunks into a single NumPy array
    transformed_full = np.vstack(transformed_chunks)

    return transformed_full
