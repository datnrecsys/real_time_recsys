import pandas as pd

from src.utils.embedding_id_mapper import IDMapper


def create_rec_df(df, 
                  idm: IDMapper,
                  user_col: str = "user_id",
                  item_col: str = "parent_asin",
                  ):
    """
    Create a DataFrame for recommendations.
    
    Args:
        df (pd.DataFrame): The DataFrame containing the recommendations.
        idm (IDMapper): The IDMapper object to map IDs.
        user_col (str): The name of the user column in the DataFrame.
        item_col (str): The name of the item column in the DataFrame.
        rating_col (str): The name of the rating column in the DataFrame.
        timestamp_col (str): The name of the timestamp column in the DataFrame.

    Returns:
        pd.DataFrame: A DataFrame with mapped user and item IDs and recommendation.
    """
    return df.assign(
        rec_ranking = lambda df:(
            df.groupby("user_indice", as_index=False)["score"].rank(
                ascending=False, method="first"
            )
        ),
        **{
            user_col: lambda df: df["user_indice"].apply(
                lambda user_indice: idm.get_user_id(user_indice)
            ),
            item_col: lambda df: df["recommendation"].apply(
                lambda parent_asin: idm.get_item_id(parent_asin)
            ),

        }
    )

def create_label_df(
        df,
        user_col: str = "user_id",
        item_col: str = "parent_asin",
        rating_col: str = "rating",
        timestamp_col: str = "timestamp",
):
    """
    Create a DataFrame for labels.

    Args:
        df (pd.DataFrame): The DataFrame containing the labels.
        user_col (str): The name of the user column in the DataFrame.
        item_col (str): The name of the item column in the DataFrame.
        rating_col (str): The name of the rating column in the DataFrame.
        timestamp_col (str): The name of the timestamp column in the DataFrame.
    Returns:
        pd.DataFrame: A DataFrame with sorted user and item IDs and labels.
    """
    label_df = (
        df.sort_values([timestamp_col], ascending=False)
        .assign(
            rating_rank = lambda df: df.groupby(user_col)[rating_col].rank(
                ascending=False, method="first"
            )
        )
        .sort_values(["rating_rank"], ascending=[True])[[user_col, item_col, rating_col, "rating_rank"]]
    )
    return label_df

def merge_recs_with_target(
    recs_df,
    label_df,
    k=10,
    user_col="user_id",
    item_col="parent_asin",
    rating_col="rating",
):
    
    return (
        recs_df.pipe(
            lambda df: pd.merge(
                df,
                label_df[[user_col, item_col, rating_col, "rating_rank"]],
                on=[user_col, item_col],
                how="outer",
            )
        )
        .assign(
            rating=lambda df: df[rating_col].fillna(0).astype(int),
            # Fill the recall with ranking = top_K + 1 so that the recall calculation is correct
            rec_ranking=lambda df: df["rec_ranking"].fillna(k + 1).astype(int),
        )
        .sort_values([user_col, "rec_ranking"])
    )