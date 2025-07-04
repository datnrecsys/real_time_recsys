import asyncio
import json
import os
import random
import sys
from typing import Any, Dict, List, Optional
import numpy as np
import httpx
import redis
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchText

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from .load_examples import custom_openapi
from .logging_utils import RequestIDMiddleware
from .pydantic_models import (
    FeatureRequest,
    FeatureRequestResult,
    TitleSearchRequest,
    TitleSearchResponse,
    SearchItem,
    ItemSequenceInput,
    RetrieveContext
)

from .utils import debug_logging_decorator
import time
from datetime import datetime


app = FastAPI()
app.add_middleware(RequestIDMiddleware)

logger.remove()
logger.add(
    sys.stderr,
    format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<8} | {name}:{function}:{line} | request_id: {extra[rec_id]} - {message}",
)

# Global flag to control user_tag_pref usage
# USE_USER_TAG_PREF = os.getenv("USE_USER_TAG_PREF", "false").lower() == "true"

# SEQRP_MODEL_SERVER_URL = os.getenv("SEQRP_MODEL_SERVER_URL", "http://localhost:3000")
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = os.getenv("REDIS_PORT", 6379)
FEAST_ONLINE_SERVER_HOST = os.getenv("FEAST_ONLINE_SERVER_HOST", "138.2.61.6")
FEAST_ONLINE_SERVER_PORT = os.getenv("FEAST_ONLINE_SERVER_PORT", 6566)

# seqrp_url = f"{SEQRP_MODEL_SERVER_URL}/predict"
redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0, decode_responses=True)
redis_output_i2i_key_prefix = "output:i2i:"
# redis_feature_recent_items_key_prefix = "feature:user:recent_items:"
redis_output_popular_key = "output:popularitems"
# redis_item_tag_key_prefix = "dim:tag_item_map:"

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = os.getenv("QDRANT_PORT", 6333)
qdrant_client = AsyncQdrantClient(
    url=f"http://{QDRANT_HOST}:{QDRANT_PORT}",
    prefer_grpc=False,  # Use HTTP instead of gRPC
)

QDRANT_ITEM_COLLECTION = "item2vec_item"

seq_retriever_model_server_url = "http://138.2.61.6:30000"

# Set the custom OpenAPI schema with examples
app.openapi = lambda: custom_openapi(
    app,
    redis_client,
    redis_output_i2i_key_prefix,
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # React dev server
        "http://localhost:5173",  # Vite dev server
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:8000",
        "http://127.0.0.1:8002",
        "http://127.0.0.1:8001",  # Self
        "http://138.2.61.6:8000",
        "http://138.2.61.6:20000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_recommendations_from_redis(redis_key: str, count: Optional[int]) -> Dict[str, Any]:
    rec_data = redis_client.get(redis_key)
    if not rec_data:
        error_message = f"[DEBUG] No recommendations found for key: {redis_key}"
        logger.error(error_message)
        raise HTTPException(status_code=404, detail=error_message)
    rec_data_json = json.loads(rec_data)
    rec_item_ids = rec_data_json.get("rec_item_ids", [])
    rec_scores = rec_data_json.get("rec_scores", [])
    if count is not None:
        rec_item_ids = rec_item_ids[:count]
        rec_scores = rec_scores[:count]
    logger.info("hehe")
    return {"rec_item_ids": rec_item_ids, "rec_scores": rec_scores}


# def get_items_from_tag_redis(
#     redis_key: str, count: Optional[int] = 100
# ) -> Dict[str, Any]:
#     items = redis_client.smembers(redis_key)
#     if not items:
#         error_message = f"[DEBUG] No items found for key: {redis_key}"
#         logger.error(error_message)
#         raise HTTPException(status_code=404, detail=error_message)
#     random.shuffle(items)
#     return {"items": items[:count], "redis_key": redis_key}


@app.get("/recs/i2i")
@debug_logging_decorator
async def get_recommendations_i2i(
    item_id: str = Query(..., description="ID of the item to get recommendations for"),
    count: Optional[int] = Query(10, description="Number of recommendations to return"),
    debug: bool = Query(False, description="Enable debug logging"),
):
    redis_key = f"{redis_output_i2i_key_prefix}{item_id}"
    recommendations = get_recommendations_from_redis(redis_key, count)
    return {
        "item_id": item_id,
        "recommendations": recommendations,
    }


@app.get("/recs/popular")
@debug_logging_decorator
async def get_recommendations_popular(
    count: Optional[int] = Query(10, description="Number of popular items to return"),
    debug: bool = Query(False, description="Enable debug logging"),
):
    recommendations = get_recommendations_from_redis(redis_output_popular_key, count)
    return {"recommendations": recommendations}


@app.post("/search/title", summary="Search for items by title", response_model=TitleSearchResponse)
async def search_title(
    input: TitleSearchRequest = Query(..., description="Search query and parameters"),
):
    """
    Search for items by title using Qdrant's text search capabilities.
    """
    try:
        records, _ = await qdrant_client.scroll(
            collection_name=QDRANT_ITEM_COLLECTION,
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key="title", match=MatchText(text=input.text)  # full-text match
                    )
                ]
            ),
            with_payload=True,
            limit=input.limit,
        )
        item_ids = [r.payload["parent_asin"] for r in records]
        search_items = [SearchItem(parent_asin=item_id, score=1.0) for item_id in item_ids]

        response = TitleSearchResponse(items=search_items)

        return response
    except Exception as e:
        error_message = f"[DEBUG] Error during search: {str(e)}"
        logger.error(error_message)
        raise HTTPException(status_code=500, detail=error_message)


@app.post("/feature_store/fetch")
async def fetch_features(request: FeatureRequest):
    # Define the URL for the feature store's endpoint
    feature_store_url = (
        f"http://{FEAST_ONLINE_SERVER_HOST}:{FEAST_ONLINE_SERVER_PORT}/get-online-features"
    )
    logger.info(f"Sending request to {feature_store_url}...")

    # Make the POST request to the feature store
    async with httpx.AsyncClient() as client:
        response = await client.post(feature_store_url, json=request.model_dump())

    # Check if the request was successful
    if response.status_code == 200:
        return response.json()
    else:
        raise HTTPException(
            status_code=response.status_code,
            detail=f"Error fetching features: {response.text}",
        )


@app.get("/feature_store/fetch/item_sequence")
@debug_logging_decorator
async def feature_store_fetch_item_sequence(
    user_id: str = Query(
        "AE22236AFRRSMQIKGG7TPTB75QEA", description="ID of the user to fetch item sequences for"
    )
):
    """
    Quick work around to get feature sequences from both streaming sources and common online sources
    """
    feature_service = "sequence_stats_v1"

    item_sequence_feature = "user_rating_list_10_recent_asin"
    item_sequence_ts_feature = "user_rating_list_10_recent_asin_timestamp"

    fr = FeatureRequest(
        entities={"user_id": [user_id]},
        feature_service=feature_service,
    )
    response = await fetch_features(fr)

    result = FeatureRequestResult.model_validate(response)

    item_sequence = result.get_feature(item_sequence_feature)
    item_sequence_ts = result.get_feature(item_sequence_ts_feature)

    return {
        "user_id": user_id,
        "item_sequence": item_sequence,
        "item_sequence_ts": item_sequence_ts,
    }


@app.get(
    "/recs/u2i/last_item_i2i",
    summary="Get recommendations for users based on their most recent items",
)
@debug_logging_decorator
async def get_recommendations_u2i_last_item_i2i(
    user_id: str = Query(..., description="ID of the user"),
    count: Optional[int] = Query(10, description="Number of recommendations to return"),
    debug: bool = Query(False, description="Enable debug logging"),
):
    logger.debug(f"Getting recent items for user_id: {user_id}")

    # Get the recent items for the user
    item_sequences = await feature_store_fetch_item_sequence(user_id)
    
    try:
        last_item_id = item_sequences["item_sequence"][-1]

        logger.debug(f"Most recently interacted item: {last_item_id}")

        # Call the i2i endpoint internally to get recommendations for that item
        recommendations = await get_recommendations_i2i(last_item_id, count, debug)

        # Step 3: Format and return the output
        result = {
            "user_id": user_id,
            "last_item_id": last_item_id,
            "recommendations": recommendations["recommendations"],
        }
        
    except Exception as e:
        error_message = f"[DEBUG] Error retrieving last item for user {user_id}: {str(e)}. Falling back to popular recommendations."
        logger.debug(error_message)
        
        # Fallback to popular recommendations if no recent items are found
        popular_recs = await get_recommendations_popular(count, debug)
        result = {
            "user_id": user_id,
            "last_item_id": None,
            "recommendations": popular_recs["recommendations"],
        }
    
    return result

@app.post(
    "/feature_store/push/item_sequence",
    summary="Push new item sequence for a user to the feature store",
)
async def push_new_item_sequence(
    input: ItemSequenceInput,
):
    response = await feature_store_fetch_item_sequence(input.user_id)

    item_sequences = response.get("item_sequence")
    
    if not item_sequences:
        item_sequences = []
    
    new_item_sequences = item_sequences + input.new_items
    new_item_sequences = new_item_sequences[-input.sequence_length:]
    new_item_sequences_str = ",".join(new_item_sequences)

    item_sequence_tss = response.get("item_sequence_ts")
    if not item_sequence_tss:
        item_sequence_tss = []
    new_item_sequence_tss = item_sequence_tss + [int(time.time())]
    new_item_sequence_tss = new_item_sequence_tss[-input.sequence_length:]
    new_item_sequence_tss_str = ",".join([str(ts) for ts in new_item_sequence_tss])

    event_dict = {
        "user_id": [input.user_id],
        "timestamp": [str(datetime.now())],
        "dedup_rn": [
            1
        ],  # Mock to conform with current offline schema TODO: Remove this column in the future
        "user_rating_cnt_90d": [1],  # Mock
        "user_rating_avg_prev_rating_90d": [4.5],  # Mock
        "user_rating_list_10_recent_asin": [new_item_sequences_str],
        "user_rating_list_10_recent_asin_timestamp": [new_item_sequence_tss_str],
    }
    push_data = {
        "push_source_name": "user_rating_stats_push_source",
        "df": event_dict,
        "to": "online",
    }
    logger.info(f"Event data to be pushed to feature store PUSH API {event_dict}")
    
    async with httpx.AsyncClient() as client:
        try:
            r = await client.post(
                f"http://{FEAST_ONLINE_SERVER_HOST}:{FEAST_ONLINE_SERVER_PORT}/push",
                json=push_data,
            )
            if r.status_code == 200:
                logger.info(f"Successfully pushed data to feature store: {r.json()}")
                return {"status": "success", "message": "Data pushed successfully"}
            else:
                logger.error(f"Error: {r.status_code} {r.text}")
                raise HTTPException(status_code=r.status_code, detail=f"Feature store error: {r.text}")
        except httpx.RequestError as e:
            # Network/connection errors
            error_message = f"[DEBUG] Network error pushing data to feature store: {str(e)}"
            logger.error(error_message)
            raise HTTPException(status_code=503, detail="Feature store service unavailable")
        except httpx.HTTPStatusError as e:
            # HTTP status errors (4xx, 5xx)
            error_message = f"[DEBUG] HTTP error pushing data to feature store: {e.response.status_code} {e.response.text}"
            logger.error(error_message)
            raise HTTPException(status_code=e.response.status_code, detail=f"Feature store error: {e.response.text}")
        except Exception as e:
            # Any other unexpected errors
            error_message = f"[DEBUG] Unexpected error pushing data to feature store: {str(e)}"
            logger.error(error_message)
            raise HTTPException(status_code=500, detail="Internal server error")
@app.post(
    "/model_server/call",
    summary="Call external sequence retriever model server",
)
async def call_seq_retriever(
        ctx: RetrieveContext, endpoint: str
    ):
        user_ids = ctx.user_ids
        item_seq = ctx.item_sequences

        logger.debug(
            f"Calling seq_rating_predicting with user_ids: {user_ids}, item_seq: {item_seq}"
        )

        # Prepare the payload for the external service
        payload = {"input_data": ctx.model_dump()}

        # Using json.dumps to format payload as json string so that later can extract from logs and rebuild the data easily
        logger.debug(
            f"[COLLECT] Payload prepared: <features>{json.dumps(payload)}</features>"
        )

        # Make the POST request to the external service
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{seq_retriever_model_server_url}/{endpoint}",
                    json=payload,
                    headers={
                        "accept": "application/json",
                        "Content-Type": "application/json",
                    },
                )

            # Handle response
            if response.status_code == 200:
                logger.debug(
                    f"[COLLECT] Response from external service: <result>{json.dumps(response.json())}</result>"
                )
                result = response.json()
                return result
            else:
                error_message = (
                    f"[DEBUG] External service returned an error: {response.text}"
                )
                logger.error(error_message)
                raise HTTPException(
                    status_code=response.status_code,
                    detail=error_message,
                )

        except httpx.HTTPError as e:
            error_message = f"[DEBUG] Error connecting to external service: {str(e)}"
            logger.error(error_message)
            raise HTTPException(status_code=500, detail=error_message) from e

@app.get(
    "/recs/u2i/two_tower_retrieve",
    summary="Get recommendations for users based on their item sequences",
)
@debug_logging_decorator
async def retrieve_recommendations(
    user_id: str = Query(..., description="ID of the user"),
    count: Optional[int] = Query(10, description="Number of recommendations to return"),
    debug: bool = Query(False, description="Enable debug logging"),
) -> Dict[str, Any]:
    """
    Retrieve recommendations for a user based on their item sequences using a two-tower model.
    """
    logger.debug(f"Retrieving recommendations for user_id: {user_id} using two-tower model")

    # Step 1: Fetch item sequences for the user
    item_sequences = await feature_store_fetch_item_sequence(user_id)
    item_sequences = item_sequences["item_sequence"]
    items_to_exclude = set()
    if not item_sequences or len(item_sequences) == 0:
        logger.debug(f"No item sequences found for user_id: {user_id}. Falling back to popular recommendations.")
        # Fallback to popular recommendations if no item sequences are found
        popular_recs = await get_recommendations_popular(count, debug)
        
        result = {
            "user_id": user_id,
            "item_sequence": [],
            "recommendations": popular_recs["recommendations"],
        }
    
    else:
        items_to_exclude.update(item_sequences)
        
        # Step 2: Get query embedding for the user's item sequences
        ctx = RetrieveContext(
            item_sequences=[item_sequences]
        )
        
        query_embedding_result = await call_seq_retriever(
            ctx, "get_query_embeddings"
        )
        
        query_embedding =  np.array(query_embedding_result.get("query_embedding")[0])
        logger.info(f"[DEBUG] {query_embedding.shape=}")
        
        buffer_count = count + len(items_to_exclude)
        
        hits = await qdrant_client.search(
            collection_name="two_tower_sequence_item_embedding",
            query_vector=query_embedding.tolist(),
            limit=buffer_count,
            with_payload=True,
            with_vectors=False,  # We don't need vectors in the response
        )
        
        rec_item_ids = []
        rec_scores = []
        for hit in hits:
            # TODO: This knowledge of using parent_asin as item id should be clear to developers...
            item_id = hit.payload.get("parent_asin", "")
            if item_id not in items_to_exclude:
                rec_item_ids.append(item_id)
                
                # Re-asign score
                # ctx = RetrieveContext(
                #     item_sequences=[item_sequences],
                #     item_ids=[item_id]
                # )
                
                # score = await call_seq_retriever(
                #     ctx, "predict"
                # )
                
                # score = score["scores"][0]
                
                # rec_scores.append(score)
                rec_scores.append(hit.model_dump()["score"])
                
                if len(rec_item_ids) >= count:
                    break
        result = {
            "user_id": user_id,
            "item_sequence": item_sequences,
            "recommendations": {
                "rec_item_ids": rec_item_ids,
                "rec_scores": rec_scores,
            },
        }
    return result
                
# @app.get("/recs/u2i/rerank", summary="Get recommendations for users")
# @debug_logging_decorator
# async def get_recommendations_u2i_rerank(
#     user_id: str = Query(
#         ..., description="ID of the user to provide recommendations for"
#     ),
#     top_k_retrieval: Optional[int] = Query(
#         100, description="Number of retrieval results to use"
#     ),
#     count: Optional[int] = Query(10, description="Number of recommendations to return"),
#     debug: bool = Query(False, description="Enable debug logging"),
# ):
#     # Step 1: Get popular and i2i recommendations concurrently
#     popular_recs, last_item_i2i_recs = await asyncio.gather(
#         get_recommendations_popular(count=top_k_retrieval, debug=False),
#         get_recommendations_u2i_last_item_i2i(
#             user_id=user_id, count=top_k_retrieval, debug=False
#         ),
#     )

#     # Step 2: Merge popular and i2i recommendations
#     all_items = set(popular_recs["recommendations"]["rec_item_ids"]).union(
#         set(last_item_i2i_recs["recommendations"]["rec_item_ids"])
#     )
#     all_items = list(all_items)

#     logger.debug("Retrieved items: {}", all_items)

#     # Step 3: Get item_sequence features
#     item_sequences = await feature_store_fetch_item_sequence(user_id)
#     item_sequences = item_sequences["item_sequence"]

#     # Step 4: Remove rated items
#     set_item_sequences = set(item_sequences)
#     set_all_items = set(all_items)
#     already_rated_items = list(set_item_sequences.intersection(set_all_items))
#     logger.debug(
#         f"Removing {len(already_rated_items)} items already rated by this user: {already_rated_items}..."
#     )
#     all_items = list(set_all_items - set_item_sequences)

#     # Step 5: Rerank
#     reranked_recs = await score_seq_rating_prediction(
#         user_ids=[user_id] * len(all_items),
#         item_sequences=[item_sequences] * len(all_items),
#         item_ids=all_items,
#     )

#     # Step 6: Extract scores from the result
#     scores = reranked_recs.get("scores", [])
#     returned_items = reranked_recs.get("item_ids", [])
#     reranked_metadata = reranked_recs.get("metadata", {})
#     if not scores or len(scores) != len(all_items):
#         error_message = "[DEBUG] Mismatch sizes between returned scores and all items"
#         logger.debug(error_message)
#         raise HTTPException(status_code=500, detail=error_message)

#     # Create a list of tuples (item_id, score)
#     item_scores = list(zip(returned_items, scores))

#     # Sort the items based on the scores in descending order
#     item_scores.sort(key=lambda x: x[1], reverse=True)

#     # Unzip the sorted items and scores
#     sorted_item_ids, sorted_scores = zip(*item_scores)

#     # Step 7: Return the reranked recommendations
#     result = {
#         "user_id": user_id,
#         "features": {"item_sequence": item_sequences},
#         "recommendations": {
#             "rec_item_ids": list(sorted_item_ids)[:count],
#             "rec_scores": list(sorted_scores)[:count],
#         },
#         "metadata": {"rerank": reranked_metadata},
#     }

#     return result


# @app.get(
#     "/recs/u2i/rerank_v2",
#     summary="End-to-end retrieve-rerank flow from user to item recommendations",
# )
# @debug_logging_decorator
# async def get_recommendations_u2i_rerank_v2(
#     user_id: str = Query(
#         ..., description="ID of the user to provide recommendations for"
#     ),
#     top_k_retrieval: Optional[int] = Query(
#         100, description="Number of retrieval results to use"
#     ),
#     count: Optional[int] = Query(10, description="Number of recommendations to return"),
#     debug: bool = Query(False, description="Enable debug logging"),
# ):
#     rec_title = "Recommended For You"
#     retrievers = []

#     # Conditionally include user_tag_pref retrieval based on the flag
#     if USE_USER_TAG_PREF:
#         popular_recs, last_item_i2i_recs, user_tag_pref = await asyncio.gather(
#             get_recommendations_popular(count=top_k_retrieval, debug=False),
#             get_recommendations_u2i_last_item_i2i(
#                 user_id=user_id, count=top_k_retrieval, debug=False
#             ),
#             retrieve_user_tag_pref(user_id=user_id, count=10, debug=False),
#         )
#     else:
#         popular_recs, last_item_i2i_recs = await asyncio.gather(
#             get_recommendations_popular(count=top_k_retrieval, debug=False),
#             get_recommendations_u2i_last_item_i2i(
#                 user_id=user_id, count=top_k_retrieval, debug=False
#             ),
#         )
#         user_tag_pref = {"data": []}

#     # Prioritize user_tag_pref retrieve if available
#     if user_tags := user_tag_pref.get("data"):
#         logger.debug(f"Creating retrieve based on user tag preferences {user_tags}...")
#         # Get top 5 tags. The list user tags is sorted by score already.
#         user_tags = user_tags[:5]
#         # Select random one tag as retrieve key
#         chosen = random.choice(user_tags)
#         tag = chosen["tag"]
#         redis_key = redis_item_tag_key_prefix + tag
#         logger.debug(f"Calling redis with key {redis_key}...")
#         all_items = get_items_from_tag_redis(redis_key, count=top_k_retrieval).get(
#             "items", []
#         )
#         rec_title = f"Based on Your Interest in {tag} Titles"
#         retrievers.append("user_tag_pref")
#     else:
#         logger.debug("Merging popular and last_item_i2i recommendations...")
#         # Merge popular and i2i recommendations
#         all_items = set(popular_recs["recommendations"]["rec_item_ids"]).union(
#             set(last_item_i2i_recs["recommendations"]["rec_item_ids"])
#         )
#         all_items = list(all_items)
#         retrievers.extend(["popular", "last_item_i2i"])

#     logger.debug("Retrieved items: {}", all_items)

#     # Get item_sequence features
#     item_sequences = await feature_store_fetch_item_sequence(user_id)
#     item_sequences = item_sequences["item_sequence"]

#     # Remove rated items
#     set_item_sequences = set(item_sequences)
#     set_all_items = set(all_items)
#     already_rated_items = list(set_item_sequences.intersection(set_all_items))
#     logger.debug(
#         f"Removing {len(already_rated_items)} items already rated by this user: {already_rated_items}..."
#     )
#     all_items = list(set_all_items - set_item_sequences)

#     # Rerank
#     reranked_recs = await score_seq_rating_prediction(
#         user_ids=[user_id] * len(all_items),
#         item_sequences=[item_sequences] * len(all_items),
#         item_ids=all_items,
#     )

#     # Extract scores from the result
#     scores = reranked_recs.get("scores", [])
#     returned_items = reranked_recs.get("item_ids", [])
#     reranked_metadata = reranked_recs.get("metadata", {})
#     if not scores or len(scores) != len(all_items):
#         error_message = "[DEBUG] Mismatch sizes between returned scores and all items"
#         logger.debug(error_message)
#         raise HTTPException(status_code=500, detail=error_message)

#     # Create a list of tuples (item_id, score)
#     item_scores = list(zip(returned_items, scores))

#     # Sort the items based on the scores in descending order
#     item_scores.sort(key=lambda x: x[1], reverse=True)

#     # Unzip the sorted items and scores
#     sorted_item_ids, sorted_scores = zip(*item_scores)

#     # Return the reranked recommendations
#     result = {
#         "user_id": user_id,
#         "features": {"item_sequence": item_sequences},
#         "recommendations": {
#             "rec_item_ids": list(sorted_item_ids)[:count],
#             "rec_scores": list(sorted_scores)[:count],
#         },
#         "rec_title": rec_title,
#         "metadata": {"retrieve": retrievers, "rerank": reranked_metadata},
#     }

#     return result


# @app.get("/recs/popular")
# @debug_logging_decorator
# async def get_recommendations_popular(
#     count: Optional[int] = Query(10, description="Number of popular items to return"),
#     debug: bool = Query(False, description="Enable debug logging"),
# ):
#     recommendations = get_recommendations_from_redis(redis_output_popular_key, count)
#     return {"recommendations": recommendations}


# @app.get("/recs/retrieve/user_tag_pref")
# @debug_logging_decorator
# async def retrieve_user_tag_pref(
#     user_id: str = Query(
#         ..., description="ID of the user to provide recommendations for"
#     ),
#     count: Optional[int] = Query(10, description="Number of items to return"),
#     debug: bool = Query(False, description="Enable debug logging"),
# ):
#     # If the feature flag is off, simply return an empty result.
#     if not USE_USER_TAG_PREF:
#         logger.info("User tag preference feature is disabled.")
#         return {"data": []}

#     feature_view = "user_tag_pref"
#     user_tag_pref_feature = FeatureRequestFeature(
#         feature_view=feature_view, feature_name="user_tag_pref_score_full_list"
#     )

#     fr = FeatureRequest(
#         entities={"user_id": [user_id]},
#         features=[user_tag_pref_feature.get_full_name(fresh=False, is_request=True)],
#     )
#     response = await fetch_features(fr)

#     result = FeatureRequestResult(
#         metadata=response["metadata"], results=response["results"]
#     )
#     feature_value = result.get_feature_value_no_fresh(user_tag_pref_feature)

#     if not feature_value:
#         return {"data": []}

#     # Example feature_value: Classic__4.0,Multiplayer__4.0
#     output = []
#     for tag_score in feature_value.split(","):
#         tag, score = tag_score.split("__")
#         output.append({"tag": tag, "score": score})

#     output = sorted(output, key=lambda x: x["score"], reverse=True)[:count]

#     return {"data": output}


# # New endpoint to connect to external service
# @app.post("/score/seq_rating_prediction")
# @debug_logging_decorator
# async def score_seq_rating_prediction(
#     user_ids: List[str],
#     item_sequences: List[List[str]],
#     item_ids: List[str],
#     debug: bool = Query(False, description="Enable debug logging"),
# ):
#     logger.debug(
#         f"Calling seq_rating_predicting with user_ids: {user_ids}, item_sequences: {item_sequences} and item_ids: {item_ids}"
#     )

#     # Step 1: Prepare the payload for the external service
#     payload = {
#         "input_data": {
#             "user_ids": user_ids,
#             "item_sequences": item_sequences,
#             "item_ids": item_ids,
#         }
#     }

#     # Using json.dumps to format payload as json string so that later can extract from logs and rebuild the data easily
#     logger.debug(
#         f"[COLLECT] Payload prepared: <features>{json.dumps(payload)}</features>"
#     )

#     # Step 2: Make the POST request to the external service
#     try:
#         async with httpx.AsyncClient() as client:
#             response = await client.post(
#                 seqrp_url,
#                 json=payload,
#                 headers={
#                     "accept": "application/json",
#                     "Content-Type": "application/json",
#                 },
#             )

#         # Step 3: Handle response
#         if response.status_code == 200:
#             logger.debug(
#                 f"[COLLECT] Response from external service: <result>{json.dumps(response.json())}</result>"
#             )
#             result = response.json()
#             return result
#         else:
#             error_message = (
#                 f"[DEBUG] External service returned an error: {response.text}"
#             )
#             logger.error(error_message)
#             raise HTTPException(
#                 status_code=response.status_code,
#                 detail=error_message,
#             )

#     except httpx.HTTPError as e:
#         error_message = f"[DEBUG] Error connecting to external service: {str(e)}"
#         logger.error(error_message)
#         raise HTTPException(status_code=500, detail=error_message)


# @app.post("/feature_store/fetch")
# async def fetch_features(request: FeatureRequest):
#     # Define the URL for the feature store's endpoint
#     feature_store_url = f"http://{FEAST_ONLINE_SERVER_HOST}:{FEAST_ONLINE_SERVER_PORT}/get-online-features"
#     logger.info(f"Sending request to {feature_store_url}...")

#     # Create the payload to send to the feature store
#     payload_fresh = {
#         "entities": request.entities,
#         "features": request.features,
#         "full_feature_names": True,
#     }

#     # Make the POST request to the feature store
#     async with httpx.AsyncClient() as client:
#         response = await client.post(feature_store_url, json=payload_fresh)

#     # Check if the request was successful
#     if response.status_code == 200:
#         return response.json()
#     else:
#         raise HTTPException(
#             status_code=response.status_code,
#             detail=f"Error fetching features: {response.text}",
#         )


# @app.get("/feature_store/fetch/item_sequence")
# @debug_logging_decorator
# async def feature_store_fetch_item_sequence(user_id: str):
#     """
#     Quick work around to get feature sequences from both streaming sources and common online sources
#     """
#     feature_view = "user_rating_stats"
#     item_sequence_feature = FeatureRequestFeature(
#         feature_view=feature_view, feature_name="user_rating_list_10_recent_asin"
#     )
#     item_sequence_ts_feature = FeatureRequestFeature(
#         feature_view=feature_view,
#         feature_name="user_rating_list_10_recent_asin_timestamp",
#     )

#     fr = FeatureRequest(
#         entities={"user_id": [user_id]},
#         features=[
#             item_sequence_feature.get_full_name(fresh=True, is_request=True),
#             item_sequence_feature.get_full_name(fresh=False, is_request=True),
#             item_sequence_ts_feature.get_full_name(fresh=True, is_request=True),
#             item_sequence_ts_feature.get_full_name(fresh=False, is_request=True),
#         ],
#     )
#     response = await fetch_features(fr)

#     result = FeatureRequestResult(
#         metadata=response["metadata"], results=response["results"]
#     )
#     item_sequence = result.get_feature_value(item_sequence_feature)
#     item_sequence_ts = result.get_feature_value(item_sequence_ts_feature)

#     return {
#         "user_id": user_id,
#         "item_sequence": item_sequence,
#         "item_sequence_ts": item_sequence_ts,
#     }
