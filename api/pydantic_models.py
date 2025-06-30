from datetime import datetime
from typing import Dict, List, Optional, Any

from pydantic import BaseModel, Field


# class FeatureRequest(BaseModel):
#     entities: Dict[str, List[str]]
#     features: List[str]


# class FeatureRequestFeature(BaseModel):
#     feature_view: str
#     feature_name: str

#     def get_full_name(self, fresh: bool, is_request: bool):
#         feature_view = self.feature_view
#         delimiter = ":"
#         if fresh:
#             feature_view += "_fresh"
#         if not is_request:
#             delimiter = "__"
#         return f"{feature_view}{delimiter}{self.feature_name}"


# class FeatureRequestResult(BaseModel):
#     class Metadata(BaseModel):
#         feature_names: List[str]

#     class Result(BaseModel):
#         values: List[Optional[str]]
#         statuses: List[str]
#         event_timestamps: List[datetime]

#     metadata: Metadata
#     results: List[Result]

#     def get_feature_value(self, feature: FeatureRequestFeature):
#         fresh_idx = self.metadata.feature_names.index(
#             feature.get_full_name(fresh=True, is_request=False)
#         )
#         common_idx = self.metadata.feature_names.index(
#             feature.get_full_name(fresh=False, is_request=False)
#         )

#         feature_results = self.results
#         get_feature_values = lambda idx: feature_results[idx].values[0]
#         fresh_feature_sequence_value_str = get_feature_values(fresh_idx)
#         if fresh_feature_sequence_value_str is not None:
#             feature_sequence_value = fresh_feature_sequence_value_str.split(",")
#         else:
#             common_feature_sequence_value_str = get_feature_values(common_idx)
#             feature_sequence_value = common_feature_sequence_value_str.split(",")
#         return feature_sequence_value

#     def get_feature_value_no_fresh(self, feature: FeatureRequestFeature):
#         """Get normal feature not including fresh source"""
#         common_idx = self.metadata.feature_names.index(
#             feature.get_full_name(fresh=False, is_request=False)
#         )

#         feature_results = self.results
#         get_feature_values = lambda idx: feature_results[idx].values[0]
#         feature_value = get_feature_values(common_idx)
#         return feature_value


class TitleSearchRequest(BaseModel):
    text: str = Field(description="Text to search in item titles")
    limit: int = Field(default=50, ge=1, le=100, description="Maximum number of items to return")
    debug: bool = False


class SearchItem(BaseModel):
    parent_asin: str
    score: float = 1.0


class TitleSearchResponse(BaseModel):
    items: List[SearchItem]


# Feature store request model for fetching features
class FeatureRequest(BaseModel):
    feature_service: str = Field(
        default="sequence_stats_v1",
        description="Name of the feature service to fetch features from",
    )
    entities: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="Entities to fetch features for, e.g., {'user_id': ['user1', 'user2']}",
    )


class FeatureRequestResult(BaseModel):
    class Metadata(BaseModel):
        feature_names: List[str]

    class Result(BaseModel):
        values: List[Optional[str]]
        statuses: List[str]
        event_timestamps: List[datetime]

    metadata: Metadata
    results: List[Result]

    def get_feature(self, feature_name: str) -> Any:
        try:
            feature_idx = self.metadata.feature_names.index(feature_name)
        except ValueError:
            raise ValueError(f"Feature '{feature_name}' not found.")

        feature_results = self.results

        def get_feature_values(idx):
            return feature_results[idx].values[0]

        feature_value = get_feature_values(feature_idx)

        if "list" in feature_name and isinstance(feature_value, str):
            # If the feature is a list, we need to convert it to a Python list
            if feature_value is not None:
                feature_value = feature_value.split(",")

        return feature_value
