from src.infrastructure.qdrant.client import get_qdrant_client

if __name__ == "__main__":
    client = get_qdrant_client()
    
    print(client.get_collection("two_tower_item"))