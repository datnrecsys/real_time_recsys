from fastapi import FastAPI
import psycopg2
import json
import random

items = ["B09X1NC7P1", "B0771FXGH7", "B09XMDJKR6", "B0BZCGF172", "B09Y22H121", "B01E7IQ6V2", "B07YX4WG1Y", "B000RNS2UA", "B00R4XMEFU", "B072JNFG8P"]
confidence_scores = [0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45]

app = FastAPI()

@app.get("/api/user/{user_id}")
def post_uid(user_id):
    # Mock API: returns items and scores in random order

    combined = list(zip(items, confidence_scores))
    random.shuffle(combined)
    shuffled_items, shuffled_scores = zip(*combined)

    return {
        "userid": user_id,
        "recommendations": list(shuffled_items),
        "score": list(shuffled_scores)
    }
