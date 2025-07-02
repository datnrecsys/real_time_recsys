# Analytics Integration Setup Guide

This guide explains how to set up and test the analytics integration between the React frontend and Python backend service with Kafka.

## 🏗️ Architecture Overview

```
React Frontend → FastAPI Analytics Service → Kafka → Consumer Processing
```

- **Frontend**: Tracks user interactions (impressions, view details, add to cart, checkout)
- **Analytics Service**: REST API that receives events and publishes to Kafka
- **Kafka**: Message broker for real-time event streaming
- **Consumer**: Processes events for analytics and ML model updates

## 🚀 Quick Start

### 1. Start the Analytics Infrastructure

```bash
cd /home/ubuntu/real_time_recsys/analytics-service
./start.sh
```

This will start:
- Kafka & Zookeeper
- Analytics FastAPI service
- Kafka UI for monitoring

### 2. Start the Frontend

```bash
cd /home/ubuntu/real_time_recsys/ui
npm run dev
```

### 3. Test the Integration

```bash
cd /home/ubuntu/real_time_recsys/analytics-service
python test_analytics.py
```

### 4. Monitor Events

In a new terminal, start the consumer:
```bash
cd /home/ubuntu/real_time_recsys/analytics-service
python consumer.py
```

## 📊 Analytics Events Tracked

### 1. **Impression Events**
- Triggered when products are displayed
- Tracks: product ID, name, score, price, position
- Used for: recommendation performance analysis

### 2. **View Details Events**
- Triggered when users click "View Details"
- Tracks: product information, user interaction
- Used for: interest analysis, CTR calculation

### 3. **Add to Cart Events**
- Triggered when users add items to cart
- Tracks: product, quantity, options, source
- Used for: conversion funnel analysis

### 4. **Checkout Events**
- Triggered when users complete purchase
- Tracks: cart total, items, quantities
- Used for: revenue analysis, purchase patterns

## 🔧 Service URLs

- **Analytics API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Kafka UI**: http://localhost:8080
- **Health Check**: http://localhost:8000/health

## 📋 Testing Checklist

1. ✅ Start analytics infrastructure
2. ✅ Verify health check endpoint
3. ✅ Start frontend application
4. ✅ Start event consumer
5. ✅ Interact with shopping page:
   - Load products (impressions)
   - View product details
   - Add items to cart
   - Complete checkout
6. ✅ Verify events in consumer logs
7. ✅ Check Kafka UI for message flow

## 🐛 Troubleshooting

### Service Not Starting
```bash
# Check Docker status
docker ps

# View service logs
docker-compose logs analytics-service
docker-compose logs kafka
```

### Frontend Not Sending Events
- Check browser console for errors
- Verify Vite proxy configuration
- Check network tab for API calls

### Events Not Reaching Kafka
- Check analytics service logs
- Verify Kafka connection
- Use Kafka UI to inspect topics

### Consumer Not Processing Events
- Check Python dependencies
- Verify Kafka topic exists
- Check consumer group configuration

## 🔄 Development Workflow

1. **Make frontend changes** → Events automatically tracked
2. **Update analytics schema** → Modify Pydantic models
3. **Change event processing** → Update consumer logic
4. **Test changes** → Use test script and manual testing

## 📈 Production Considerations

- Add authentication to analytics API
- Implement rate limiting
- Set up monitoring and alerting
- Configure Kafka for high availability
- Add data retention policies
- Implement event deduplication
