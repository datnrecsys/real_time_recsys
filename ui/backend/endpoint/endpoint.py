from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import asyncpg
import json
import httpx
import asyncio
from typing import Optional, Dict, Any

# Database connection pool
db_pool = None

async def init_db_pool():
    """Initialize the database connection pool"""
    global db_pool
    db_pool = await asyncpg.create_pool(
        database="amazon_rating",
        user="resys-user",
        password="hehehe",
        host="localhost",
        port=5432,
        min_size=1,
        max_size=10
    )

app = FastAPI()

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
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/endpoint/item_details")
async def get_item_details(item_id: str) -> Optional[Dict[str, Any]]:
    """Fetch item details from database using async streaming query"""
    if not db_pool:
        await init_db_pool()
    
    try:
        async with db_pool.acquire() as conn:
            # Use async query with streaming
            query = "SELECT * FROM oltp.raw_metadata WHERE parent_asin = $1 LIMIT 1"
            row = await conn.fetchrow(query, item_id)
            
            if row:
                # Parse JSON fields safely
                try:
                    num_reviews = json.loads(row[3]) if row[3] else 0
                except (json.JSONDecodeError, TypeError):
                    num_reviews = 0
                
                try:
                    image_data = json.loads(row[7].replace("None", "null")) if row[7] else {}
                    image_large = image_data.get("large", []) if isinstance(image_data, dict) else []
                    image_hires = image_data.get("hi_res", []) if isinstance(image_data, dict) else []
                    image_urls = image_large + image_hires
                    image_urls = [url for url in (image_large + image_hires) if isinstance(url, str)]
                except (json.JSONDecodeError, TypeError):
                    image_urls = []
                
                return {
                    "item_id": item_id,
                    "main_category": row[0],
                    "name": row[1],
                    "rating": row[2],
                    "num_reviews": num_reviews,
                    "price": row[6],
                    "image_urls": image_urls,
                }
            return None
            
    except Exception as error:
        print(f"Database error for item {item_id}: {error}")
        return None


@app.on_event("startup")
async def startup_event():
    """Initialize database pool on startup"""
    await init_db_pool()

@app.on_event("shutdown")
async def shutdown_event():
    """Close database pool on shutdown"""
    global db_pool
    if db_pool:
        await db_pool.close()

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "Recommendation Endpoint Service is running", "status": "healthy"}

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "service": "recommendation_endpoint",
        "database_connected": db_pool is not None
    }

@app.get("/api/endpoint")
async def fetch_rec(user_id: str, page: int = 0, limit: int = 10):
    """Fetch recommendations using async HTTP and database calls with pagination"""
    try:
        # Use async HTTP client for non-blocking requests
        async with httpx.AsyncClient() as client:
            response = await client.get(f"http://localhost:8000/recs/popular?count=50")
            response.raise_for_status()
            model_data = response.json()
        recommendations = model_data.get("recommendations", [])
        # print(recommendations)
        recommendations_ids = recommendations["rec_item_ids"] if "rec_item_ids" in recommendations else []
        scores = recommendations["rec_scores"] if "rec_scores" in recommendations else []

        # Calculate pagination
        start_index = page * limit
        end_index = start_index + limit
        
        # Debug logging
        print(f"Pagination request: user_id={user_id}, page={page}, limit={limit}")
        print(f"Total items available: {len(recommendations_ids)}")
        print(f"Requesting slice: [{start_index}:{end_index}]")
        
        # Get the paginated slice
        paginated_ids = recommendations_ids[start_index:end_index]
        paginated_scores = scores[start_index:end_index]
        
        print(f"Returning {len(paginated_ids)} items for page {page}")
        
        # Fetch item details concurrently using asyncio.gather
        tasks = [get_item_details(item) for item in paginated_ids]
        items = await asyncio.gather(*tasks)
        
        return {
            "user_id": user_id,
            "recommendations": items,
            "score": paginated_scores,
            "pagination": {
                "page": page,
                "limit": limit,
                "total_items": len(recommendations_ids),
                "has_more": end_index < len(recommendations_ids)
            }
        }
        
    except httpx.HTTPError as e:
        print(f"HTTP error fetching recommendations for user {user_id}: {e}")
        return {
            "user_id": user_id,
            "recommendations": [],
            "score": [],
            "pagination": {
                "page": page,
                "limit": limit,
                "total_items": 0,
                "has_more": False
            }
        }
    except Exception as e:
        print(f"Error fetching recommendations for user {user_id}: {e}")
        return {
            "user_id": user_id,
            "recommendations": [],
            "score": [],
            "pagination": {
                "page": page,
                "limit": limit,
                "total_items": 0,
                "has_more": False
            }
        }


@app.get("/api/endpoint/i2i")
async def fetch_i2i_rec(item_id: str, count: int = 10):
    """Fetch item-to-item recommendations using async HTTP and database calls"""
    try:
        # Use async HTTP client for non-blocking requests to external i2i API
        async with httpx.AsyncClient() as client:
            response = await client.get(f"http://138.2.61.6:8000/recs/i2i?item_id={item_id}&count={count}&debug=false")
            response.raise_for_status()
            i2i_data = response.json()
        
        print(f"I2I API response for item {item_id}:", i2i_data)
        
        # Extract item IDs and scores from the response
        if i2i_data and "recommendations" in i2i_data and isinstance(i2i_data["recommendations"], dict):
            rec_item_ids = i2i_data["recommendations"].get("rec_item_ids", [])
            rec_scores = i2i_data["recommendations"].get("rec_scores", [])
        else:
            print(f"Unexpected i2i response format: {i2i_data}")
            return {
                "item_id": item_id,
                "recommendations": [],
                "score": [],
                "count": 0
            }
        
        print(f"Found {len(rec_item_ids)} i2i recommendations for item {item_id}")
        
        # Fetch item details concurrently using asyncio.gather
        tasks = [get_item_details(item_id) for item_id in rec_item_ids]
        items = await asyncio.gather(*tasks)
        
        # Filter out None items (items not found in database)
        valid_items = []
        valid_scores = []
        for i, item in enumerate(items):
            if item is not None:
                valid_items.append(item)
                if i < len(rec_scores):
                    valid_scores.append(rec_scores[i])
                else:
                    valid_scores.append(0.5)  # Default score if missing
        
        print(f"Returning {len(valid_items)} valid i2i items with database details")
        
        return {
            "item_id": item_id,
            "recommendations": valid_items,
            "score": valid_scores,
            "count": len(valid_items),
            "metadata": i2i_data.get("metadata", {})
        }
        
    except httpx.HTTPError as e:
        print(f"HTTP error fetching i2i recommendations for item {item_id}: {e}")
        return {
            "item_id": item_id,
            "recommendations": [],
            "score": [],
            "count": 0,
            "error": f"HTTP error: {str(e)}"
        }
    except Exception as e:
        print(f"Error fetching i2i recommendations for item {item_id}: {e}")
        return {
            "item_id": item_id,
            "recommendations": [],
            "score": [],
            "count": 0,
            "error": f"Server error: {str(e)}"
        }

@app.get("/api/endpoint/unified")
async def fetch_unified_rec(user_id: str, page: int = 0, limit: int = 10, last_item_id: Optional[str] = None):
    """Fetch unified recommendations with optimized performance"""
    try:
        # If we have a last_item_id, prioritize i2i recommendations
        if last_item_id:
            print(f"Fetching i2i recommendations for item: {last_item_id}")
            try:
                async with httpx.AsyncClient(timeout=5.0) as client:
                    i2i_response = await client.get(f"http://138.2.61.6:8000/recs/i2i?item_id={last_item_id}&count={limit}&debug=false")
                    if i2i_response.status_code == 200:
                        i2i_data = i2i_response.json()
                        
                        if i2i_data and "recommendations" in i2i_data and isinstance(i2i_data["recommendations"], dict):
                            rec_item_ids = i2i_data["recommendations"].get("rec_item_ids", [])
                            rec_scores = i2i_data["recommendations"].get("rec_scores", [])
                            
                            if rec_item_ids:
                                print(f"Found {len(rec_item_ids)} i2i recommendations, fetching details...")
                                
                                # Fetch item details concurrently
                                tasks = [get_item_details(item_id) for item_id in rec_item_ids]
                                items = await asyncio.gather(*tasks)
                                
                                # Build response with valid items
                                recommendations_batch = []
                                scores_batch = []
                                
                                for i, item in enumerate(items):
                                    if item is not None:
                                        recommendations_batch.append(item)
                                        score = rec_scores[i] if i < len(rec_scores) else 0.5
                                        scores_batch.append(score)
                                
                                print(f"Returning {len(recommendations_batch)} i2i items")
                                
                                return {
                                    "user_id": user_id,
                                    "recommendations": recommendations_batch,
                                    "score": scores_batch,
                                    "pagination": {
                                        "page": page,
                                        "limit": limit,
                                        "total_items": len(recommendations_batch),
                                        "has_more": False,  # i2i typically returns a fixed set
                                        "source_type": "i2i",
                                        "last_item_id": last_item_id
                                    }
                                }
            except Exception as e:
                print(f"I2I fetch failed: {e}, falling back to main API")
        
        # Fallback to main API (original logic but optimized)
        print(f"Fetching main recommendations for user {user_id}, page {page}")
        
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"http://localhost:8000/recs/popular?count=50")
            response.raise_for_status()
            model_data = response.json()
        
        recommendations = model_data.get("recommendations", [])
        recommendations_ids = recommendations["rec_item_ids"] if "rec_item_ids" in recommendations else []
        scores = recommendations["rec_scores"] if "rec_scores" in recommendations else []

        # Calculate pagination
        start_index = page * limit
        end_index = start_index + limit
        
        # Get the paginated slice
        paginated_ids = recommendations_ids[start_index:end_index]
        paginated_scores = scores[start_index:end_index]
        
        print(f"Fetching details for {len(paginated_ids)} main recommendations...")
        
        # Fetch item details concurrently
        tasks = [get_item_details(item) for item in paginated_ids]
        items = await asyncio.gather(*tasks)
        
        return {
            "user_id": user_id,
            "recommendations": items,
            "score": paginated_scores,
            "pagination": {
                "page": page,
                "limit": limit,
                "total_items": len(recommendations_ids),
                "has_more": end_index < len(recommendations_ids),
                "source_type": "main"
            }
        }
        
    except httpx.HTTPError as e:
        print(f"HTTP error in unified recommendations for user {user_id}: {e}")
        return {
            "user_id": user_id,
            "recommendations": [],
            "score": [],
            "pagination": {
                "page": page,
                "limit": limit,
                "total_items": 0,
                "has_more": False,
                "source_type": "error"
            }
        }
    except Exception as e:
        print(f"Error in unified recommendations for user {user_id}: {e}")
        return {
            "user_id": user_id,
            "recommendations": [],
            "score": [],
            "pagination": {
                "page": page,
                "limit": limit,
                "total_items": 0,
                "has_more": False,
                "source_type": "error"
            }
        }