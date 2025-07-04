from fastapi.openapi.utils import get_openapi


def get_sample_id_from_redis(redis_client, key_prefix):
    # Use SCAN to get keys with the item_id prefix
    keys = redis_client.scan_iter(match=key_prefix + "*", count=1)
    try:
        first_key = next(keys)
        # Extract item_id from the key
        sample_item_id = first_key[len(key_prefix) :]
        return sample_item_id
    except StopIteration:
        return "sample_id_not_found"


def custom_openapi(
    app,
    redis_client,
    redis_output_i2i_key_prefix,
):
    if app.openapi_schema:
        return app.openapi_schema

    # Fetch sample item_id and user_id from Redis
    sample_item_id = get_sample_id_from_redis(redis_client, redis_output_i2i_key_prefix)
    # sample_user_id = get_sample_id_from_redis(
    #     redis_client, redis_feature_recent_items_key_prefix
    # )
    sample_user_id = "AE22236AFRRSMQIKGG7TPTB75QEA"  # Placeholder for user_id

    openapi_schema = get_openapi(
        title="Recommendation API",
        version="1.0.0",
        description="API for item-to-item and user-to-item recommendations",
        routes=app.routes,
    )

    # Modify the parameters to include examples
    for path in openapi_schema["paths"]:
        for method in openapi_schema["paths"][path]:
            operation = openapi_schema["paths"][path][method]
            if "parameters" in operation:
                for param in operation["parameters"]:
                    if param["name"] == "item_id":
                        param["example"] = sample_item_id
                    elif param["name"] == "user_id":
                        param["example"] = sample_user_id

    app.openapi_schema = openapi_schema
    return app.openapi_schema
