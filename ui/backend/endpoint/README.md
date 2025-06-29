# Recommendation Endpoint Service

An async FastAPI service that provides product recommendations by fetching data from a machine learning model and enriching it with product details from a PostgreSQL database.

## Features

- **Async Database Operations**: Uses asyncpg for non-blocking database queries
- **Connection Pooling**: Efficient database connection management
- **Concurrent Processing**: Fetches multiple product details simultaneously
- **Async HTTP Calls**: Non-blocking requests to the ML model API
- **Error Handling**: Graceful error handling for database and HTTP failures
- **Performance Optimized**: Streaming queries and concurrent execution

## Architecture

```
Frontend → Endpoint Service → ML Model API
    ↓                ↓
Database (PostgreSQL)  ↓
    ↑___________________|
```

## Setup

### Prerequisites

- Python 3.8+
- PostgreSQL database with product metadata
- ML model API running on port 8000
- Virtual environment (recommended)

### Installation

1. Create and activate virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Database Configuration

The service connects to PostgreSQL with these settings:
- **Database**: `amazon_rating`
- **User**: `resys-user`
- **Password**: `hehehe`
- **Host**: `localhost`
- **Port**: `5432`

Expected table structure:
```sql
-- oltp.raw_metadata table should contain:
-- Column 0: main_category
-- Column 1: name
-- Column 2: rating
-- Column 3: num_reviews (JSON)
-- Column 6: price
-- Column 7: image_urls (JSON with 'large' key)
```

### Running the Service

#### Option 1: Using the start script
```bash
./start.sh
```

#### Option 2: Direct execution
```bash
uvicorn endpoint:app --host 127.0.0.1 --port 8001 --reload
```

The service will be available at `http://127.0.0.1:8001`

## API Endpoints

### Get Recommendations
```
GET /api/rec/{user_id}
```

**Parameters:**
- `user_id` (string): User identifier

**Response:**
```json
{
  "userid": "123",
  "recommendations": [
    {
      "main_category": "Electronics",
      "name": "Product Name",
      "rating": 4.5,
      "num_reviews": 100,
      "price": "29.99",
      "image_urls": ["https://...", "https://..."]
    }
  ],
  "score": [0.95, 0.87, 0.82]
}
```

## Performance Features

### 1. Async Database Queries
- Uses `asyncpg` for non-blocking PostgreSQL queries
- Connection pooling (1-10 connections)
- Streaming query execution

### 2. Concurrent Processing
- Fetches multiple product details simultaneously using `asyncio.gather()`
- Non-blocking HTTP requests with `httpx.AsyncClient`

### 3. Error Handling
- Database connection failures
- HTTP request timeouts
- JSON parsing errors
- Graceful degradation

## Monitoring

### Health Check
The service includes FastAPI's automatic health monitoring and can be extended with:
- Database connection health
- Response time metrics
- Error rate tracking

### Logging
Errors are logged to stdout with detailed information:
- Database errors with item IDs
- HTTP errors with user IDs
- General exception handling

## Production Considerations

1. **Environment Variables**: Move database credentials to environment variables
2. **SSL/TLS**: Configure SSL for database connections
3. **Rate Limiting**: Add rate limiting for API endpoints
4. **Caching**: Implement Redis caching for frequently requested items
5. **Monitoring**: Add APM tools like Datadog or New Relic
6. **Load Balancing**: Use multiple instances behind a load balancer

## Development

### Local Development
```bash
# Install development dependencies
pip install -r requirements.txt

# Run with auto-reload
uvicorn endpoint:app --reload --host 127.0.0.1 --port 8001
```

### Testing
```bash
# Test the endpoint
curl http://127.0.0.1:8001/api/rec/123

# Check API documentation
open http://127.0.0.1:8001/docs
```

## Troubleshooting

### Common Issues

1. **Database Connection Failed**:
   - Verify PostgreSQL is running
   - Check database credentials
   - Ensure database and table exist

2. **ML Model API Unavailable**:
   - Check if model API is running on port 8000
   - Verify network connectivity
   - Check API endpoint URL

3. **Performance Issues**:
   - Monitor database connection pool usage
   - Check query execution times
   - Consider adding database indexes

4. **Memory Issues**:
   - Adjust connection pool size
   - Monitor asyncio task creation
   - Consider pagination for large result sets

## Dependencies

- **FastAPI**: Web framework
- **asyncpg**: Async PostgreSQL driver
- **httpx**: Async HTTP client
- **uvicorn**: ASGI server
