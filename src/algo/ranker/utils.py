import numpy as np
import time

def pad_timestamp_sequence(inp: str, sequence_length=10, padding_value=-1):
    if inp is None:
        return [padding_value] * sequence_length
    inp_list = [int(x) for x in inp.split(",")]
    padding_needed = sequence_length - len(inp_list)
    output = np.pad(
        inp_list,
        (padding_needed, 0),  # Add padding at the beginning
        "constant",
        constant_values=padding_value,
    )
    return output

def bucketize_seconds_diff(seconds: int):
    if seconds < 60 * 10:
        return 0
    if seconds < 60 * 60:
        return 1
    if seconds < 60 * 60 * 24:
        return 2
    if seconds < 60 * 60 * 24 * 7:
        return 3
    if seconds < 60 * 60 * 24 * 30:
        return 4
    if seconds < 60 * 60 * 24 * 365:
        return 5
    if seconds < 60 * 60 * 24 * 365 * 3:
        return 6
    if seconds < 60 * 60 * 24 * 365 * 5:
        return 7
    if seconds < 60 * 60 * 24 * 365 * 10:
        return 8
    return 9


def from_ts_to_bucket(ts, current_ts: int = None):
    if current_ts is None:
        current_ts = int(time.time())
    return bucketize_seconds_diff(current_ts - ts)


def calc_sequence_timestamp_bucket(row):
    ts = row["timestamp_unix"]
    output = []
    for x in row["item_sequence_ts"]:
        x_i = int(x)
        if x_i == -1:
            # Keep padding (blank) element
            output.append(x_i)
        else:
            bucket = from_ts_to_bucket(x_i, ts)
            output.append(bucket)
    return output


def generate_item_sequences(
    df,
    user_col,
    item_col,
    timestamp_col,
    sequence_length,
    padding=True,
    padding_value=-1,
):
    """
    Generates a column 'item_sequence' containing lists of previous item indices for each user.

    Parameters:
    - df: DataFrame containing the data
    - user_col: The name of the user column
    - item_col: The name of the item column
    - timestamp_col: The name of the timestamp column
    - sequence_length: The maximum length of the item sequence to keep
    - padding: whether to pad the item sequence with `padding_value` if it's shorter than sequence_length
    - padding_value: value used for padding

    Returns:
    - DataFrame with an additional column 'item_sequence'
    """

    def get_item_sequence(sub_df):
        sequences = []
        for i in range(len(sub_df)):
            prev_df = sub_df.loc[
                lambda df: df[timestamp_col].lt(sub_df[timestamp_col].iloc[i])
            ]
            # Get item indices up to the current row (excluding the current row)
            sequence = prev_df[item_col].tolist()[-sequence_length:]
            if padding:
                padding_needed = sequence_length - len(sequence)
                sequence = np.pad(
                    sequence,
                    (padding_needed, 0),  # Add padding at the beginning
                    "constant",
                    constant_values=padding_value,
                )
            sequences.append(sequence)
        return sequences

    agg_df = df.sort_values([user_col, timestamp_col])
    item_sequences = agg_df.groupby(user_col, group_keys=True).apply(get_item_sequence)
    item_sequences_flatten = (
        item_sequences.to_frame("item_sequence").reset_index().explode("item_sequence")
    )
    agg_df["item_sequence"] = (
        # Need to use .values to avoid auto index mapping between agg_df and item_sequences_flatten which causes miss join
        item_sequences_flatten["item_sequence"]
        .fillna("")
        .apply(list)
        .values
    )

    return agg_df
