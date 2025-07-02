from feast import FeatureService

from user.rating_stats import user_rating_stats_fresh_fv

sequence_stats_fs_v1 = FeatureService(
    name="sequence_stats_v1",
    features=[
        user_rating_stats_fresh_fv[["user_rating_list_10_recent_asin", "user_rating_list_10_recent_asin_timestamp"]],
    ]
)