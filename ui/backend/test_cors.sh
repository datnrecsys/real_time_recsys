#!/bin/bash

echo "Testing CORS configuration..."

# Test the endpoint service
echo "Testing endpoint service at 127.0.0.1:8001..."
curl -X GET \
  -H "Origin: http://localhost:5173" \
  -H "Access-Control-Request-Method: GET" \
  -H "Access-Control-Request-Headers: Content-Type" \
  -v \
  http://127.0.0.1:8001/

echo -e "\n\nTesting analytics service at localhost:8001..."
curl -X GET \
  -H "Origin: http://localhost:5173" \
  -H "Access-Control-Request-Method: GET" \
  -H "Access-Control-Request-Headers: Content-Type" \
  -v \
  http://localhost:8001/

echo -e "\n\nCORS test completed!"
