import { useState, useEffect, useRef, useCallback } from 'react'
import ProductCard from './ProductCard'
import LoadingSpinner from './LoadingSpinner'
import ErrorMessage from './ErrorMessage'
import Notification from './Notification'
import ProductDetails from './ProductDetails'
import ShoppingCart from './ShoppingCart'
import './ShoppingPage.css'

const ShoppingPage = () => {
  const [recommendations, setRecommendations] = useState([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [userId, setUserId] = useState(() => {
    // Get or create a persistent user ID for this session
    try {
      const storedUserId = localStorage.getItem('shopping_user_id')
      if (storedUserId) {
        console.log('Retrieved stored user ID:', storedUserId)
        return parseInt(storedUserId)
      } else {
        const newUserId = Math.floor(Math.random() * 10000) + 1
        localStorage.setItem('shopping_user_id', newUserId.toString())
        console.log('Created new user ID:', newUserId)
        return newUserId
      }
    } catch (error) {
      console.warn('Failed to access localStorage for user ID:', error)
      return Math.floor(Math.random() * 10000) + 1
    }
  })
  const [notifications, setNotifications] = useState([])
  const [selectedProduct, setSelectedProduct] = useState(null)
  const [isProductDetailsOpen, setIsProductDetailsOpen] = useState(false)
  const [cartItems, setCartItems] = useState([])
  const [isCartOpen, setIsCartOpen] = useState(false)
  
  // User interaction tracking for item2item recommendations
  const [lastInteractedItem, setLastInteractedItem] = useState(null)
  const [isUsingItem2Item, setIsUsingItem2Item] = useState(false)
  const [hasRestoredI2I, setHasRestoredI2I] = useState(false)
  
  // Lazy loading state
  const [currentPage, setCurrentPage] = useState(0)
  const [totalItems, setTotalItems] = useState(0)
  const [loadingMore, setLoadingMore] = useState(false)
  const [hasMoreItems, setHasMoreItems] = useState(true)
  const observerRef = useRef()
  const lastItemRef = useRef()

  // Persistence functions for i2i state
  const saveI2IStateToStorage = (item, isUsing) => {
    try {
      const i2iState = {
        lastInteractedItem: item,
        isUsingItem2Item: isUsing,
        userId: userId,
        timestamp: Date.now()
      }
      localStorage.setItem('shopping_i2i_state', JSON.stringify(i2iState))
      console.log('Saved i2i state to localStorage:', i2iState)
    } catch (error) {
      console.warn('Failed to save i2i state to localStorage:', error)
    }
  }

  const loadI2IStateFromStorage = () => {
    try {
      const storedState = localStorage.getItem('shopping_i2i_state')
      if (storedState) {
        const i2iState = JSON.parse(storedState)
        
        // Check if the stored state is recent (within 24 hours) and for the same user
        const isRecent = (Date.now() - i2iState.timestamp) < (24 * 60 * 60 * 1000)
        const isSameUser = i2iState.userId === userId
        
        if (isRecent && isSameUser && i2iState.lastInteractedItem) {
          console.log('Restored i2i state from localStorage:', i2iState)
          setLastInteractedItem(i2iState.lastInteractedItem)
          setIsUsingItem2Item(i2iState.isUsingItem2Item)
          return true
        } else {
          console.log('Stored i2i state is stale or for different user, clearing')
          localStorage.removeItem('shopping_i2i_state')
        }
      }
    } catch (error) {
      console.warn('Failed to load i2i state from localStorage:', error)
      localStorage.removeItem('shopping_i2i_state')
    }
    return false
  }

  const clearI2IStateFromStorage = () => {
    try {
      localStorage.removeItem('shopping_i2i_state')
      console.log('Cleared i2i state from localStorage')
    } catch (error) {
      console.warn('Failed to clear i2i state from localStorage:', error)
    }
  }

  // Cart persistence functions
  const saveCartToStorage = (cart) => {
    try {
      const cartState = {
        cartItems: cart,
        userId: userId,
        timestamp: Date.now()
      }
      localStorage.setItem('shopping_cart', JSON.stringify(cartState))
      console.log('Saved cart to localStorage:', cartState)
    } catch (error) {
      console.warn('Failed to save cart to localStorage:', error)
    }
  }

  const loadCartFromStorage = () => {
    try {
      const storedCart = localStorage.getItem('shopping_cart')
      if (storedCart) {
        const cartState = JSON.parse(storedCart)
        
        // Check if the stored cart is recent (within 7 days) and for the same user
        const isRecent = (Date.now() - cartState.timestamp) < (7 * 24 * 60 * 60 * 1000)
        const isSameUser = cartState.userId === userId
        
        if (isRecent && isSameUser && cartState.cartItems) {
          console.log('Restored cart from localStorage:', cartState)
          setCartItems(cartState.cartItems)
          if (cartState.cartItems.length > 0) {
            addNotification(`Restored ${cartState.cartItems.length} items from your previous session`, 'info')
          }
          return true
        } else {
          console.log('Stored cart is stale or for different user, clearing')
          localStorage.removeItem('shopping_cart')
        }
      }
    } catch (error) {
      console.warn('Failed to load cart from localStorage:', error)
      localStorage.removeItem('shopping_cart')
    }
    return false
  }

  const clearCartFromStorage = () => {
    try {
      localStorage.removeItem('shopping_cart')
      console.log('Cleared cart from localStorage')
    } catch (error) {
      console.warn('Failed to clear cart from localStorage:', error)
    }
  }

  // API call function to get recommendations with pagination and i2i priority
  const fetchRecommendations = async (userId, page = 0, append = false, lastItemId = null) => {
    try {
      if (!append) {
        setLoading(true)
      } else {
        setLoadingMore(true)
      }
      setError(null)
      
      // Build URL with optional last_item_id parameter for i2i prioritization
      let apiUrl = `http://127.0.0.1:8001/api/endpoint/unified?user_id=${userId}&page=${page}&limit=10`
      if (lastItemId) {
        apiUrl += `&last_item_id=${lastItemId}`
        console.log(`Requesting with i2i priority for item: ${lastItemId}`)
      }
      
      const response = await fetch(apiUrl, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
      })
      
      if (!response.ok) {
        if (response.status === 0 || !response.status) {
          throw new Error('Unable to connect to recommendation service. Please check if the service is running and CORS is properly configured.')
        }
        throw new Error(`API request failed with status ${response.status}: ${response.statusText}`)
      }
      
      const data = await response.json()
      
      // Validate API response structure
      if (!data.recommendations || !Array.isArray(data.recommendations)) {
        throw new Error('Invalid API response format')
      }
      
      // Debug: Log pagination info
      console.log('Unified API response pagination:', data.pagination)
      console.log(`Page ${page}: received ${data.recommendations.length} items, source: ${data.pagination?.source_type}`)
      console.log(`Has more items: ${data.pagination?.has_more}`)
      
      // Update state based on source type
      if (data.pagination?.source_type === 'i2i' || data.pagination?.source_type === 'mixed') {
        setIsUsingItem2Item(true)
      }
      
      // Transform items to ensure consistent format
      const itemsWithScores = data.recommendations
        .map((item, index) => {
          if (!item) return null
          
          return {
            id: item.name ? `${item.name.slice(0, 20)}_p${page}_i${index}_${Date.now()}` : `product_${page}_${index}_${Date.now()}`,
            item_id: item.item_id || item.parent_asin || item.asin || item.id || `item_${page * 10 + index}`,
            name: item.name || `Product ${page * 10 + index + 1}`,
            score: data.score && data.score[index] !== undefined ? data.score[index] : 0,
            price: item.price && item.price !== "None" ? parseFloat(item.price) : Math.floor(Math.random() * 200) + 20,
            image: item.image_urls && Array.isArray(item.image_urls) && item.image_urls.length > 0 
              ? item.image_urls[0]
              : `https://picsum.photos/300/300?random=${page * 10 + index + 1}`,
            description: item.name ? `${item.main_category || 'Product'} - ${item.name}` : `High-quality product with excellent features and customer reviews.`,
            rating: item.rating || 0,
            num_reviews: item.num_reviews || 0,
            main_category: item.main_category || 'General',
            image_urls: item.image_urls || [],
            isItem2Item: data.pagination?.source_type === 'i2i' || data.pagination?.source_type === 'mixed'
          }
        })
        .filter(item => item !== null)
      
      const sortedItems = itemsWithScores.sort((a, b) => b.score - a.score)
      
      if (append) {
        // Append new items to existing recommendations
        setRecommendations(prev => {
          const existingIds = new Set(prev.map(item => item.item_id || item.id))
          const uniqueNewItems = sortedItems.filter(item => 
            !existingIds.has(item.item_id) && !existingIds.has(item.id)
          )
          console.log(`Appending ${uniqueNewItems.length} unique items from ${data.pagination?.source_type} source`)
          return [...prev, ...uniqueNewItems]
        })
      } else {
        // Replace recommendations with new items
        setRecommendations(sortedItems)
      }
      
      // Update pagination state
      setCurrentPage(page)
      setTotalItems(data.pagination?.total_items || sortedItems.length)
      setHasMoreItems(data.pagination?.has_more || false)
      
    } catch (err) {
      if (err.name === 'TypeError' && err.message.includes('Failed to fetch')) {
        setError('Cannot connect to recommendation service. Please ensure the backend service is running on http://127.0.0.1:8001')
      } else {
        setError(err.message || 'Failed to fetch recommendations. Please try again.')
      }
      console.error('Error fetching recommendations:', err)
    } finally {
      setLoading(false)
      setLoadingMore(false)
    }
  }



  // Load more items from the paginated backend API
  // Load more items with i2i prioritization
  const loadMoreItems = async () => {
    if (loadingMore) return

    console.log(`loadMoreItems called: currentPage=${currentPage}, hasMoreItems=${hasMoreItems}, isUsingItem2Item=${isUsingItem2Item}`)
    
    try {
      setLoadingMore(true)
      
      // Determine the best item for i2i recommendations
      let priorityItemId = null
      
      // Priority 1: Last interacted item (user showed interest)
      if (lastInteractedItem) {
        priorityItemId = lastInteractedItem.item_id || lastInteractedItem.id
        console.log(`Using last interacted item for i2i priority: ${priorityItemId}`)
      }
      // Priority 2: Random item from top 3 current recommendations (highest scored items)
      else if (recommendations.length > 0) {
        const topItems = recommendations.slice(0, 3) // Use top 3 highest-scored items
        const randomIndex = Math.floor(Math.random() * topItems.length)
        priorityItemId = topItems[randomIndex].item_id || topItems[randomIndex].id
        console.log(`Using random top item for i2i priority: ${priorityItemId}`)
      }
      
      // Always try to fetch with i2i priority if we have an item
      const nextPage = currentPage + 1
      console.log(`Attempting to load page ${nextPage} with i2i priority`)
      
      // Call unified API with i2i prioritization
      await fetchRecommendations(userId, nextPage, true, priorityItemId)
      
      setCurrentPage(nextPage)
      console.log(`Successfully loaded page ${nextPage} with i2i prioritization`)

    } catch (err) {
      console.error('Error loading more items:', err)
      setHasMoreItems(false)
    } finally {
      setLoadingMore(false)
    }
  }

  useEffect(() => {
    // Reset lazy loading state when user changes
    setCurrentPage(0)
    setTotalItems(0)
    setHasMoreItems(true)
    setHasRestoredI2I(false)
    
    // Load item2item state from localStorage on mount
    const isRestored = loadI2IStateFromStorage()
    
    // Load cart from localStorage on mount
    loadCartFromStorage()
    
    // Always fetch initial recommendations first
    fetchRecommendations(userId, 0, false)
  }, [userId])

  // Separate effect to handle restored i2i state after initial recommendations are loaded
  useEffect(() => {
    if (lastInteractedItem && userId && recommendations.length > 0 && !hasRestoredI2I) {
      console.log('Appending restored i2i recommendations for:', lastInteractedItem.name)
      setHasRestoredI2I(true)
      fetchAndAppendI2IRecommendations(lastInteractedItem)
    }
  }, [lastInteractedItem, userId, recommendations.length, hasRestoredI2I])

  // Intersection Observer for infinite scroll with debouncing
  const lastItemElementRef = useCallback(node => {
    if (loadingMore) return
    if (observerRef.current) observerRef.current.disconnect()
    
    observerRef.current = new IntersectionObserver(entries => {
      if (entries[0].isIntersecting && hasMoreItems && !loadingMore) {
        console.log('Intersection observer triggered: loading more items')
        // Debounce to prevent multiple rapid calls
        setTimeout(() => {
          if (hasMoreItems && !loadingMore) {
            loadMoreItems()
          }
        }, 100)
      }
    }, {
      threshold: 0.1,
      rootMargin: '100px'
    })
    
    if (node) observerRef.current.observe(node)
  }, [loadingMore, hasMoreItems])

  const handleRefresh = () => {
    // Reset lazy loading state
    setCurrentPage(0)
    setTotalItems(0)
    setHasMoreItems(true)
    setLoadingMore(false)
    setIsUsingItem2Item(false)
    setLastInteractedItem(null)
    setHasRestoredI2I(false)
    
    // Clear persisted i2i state when explicitly refreshing
    clearI2IStateFromStorage()
    
    fetchRecommendations(userId, 0, false)
  }

  const generateNewUserSession = () => {
    const newUserId = Math.floor(Math.random() * 10000) + 1
    setUserId(newUserId)
    
    try {
      localStorage.setItem('shopping_user_id', newUserId.toString())
      console.log('Generated new user session with ID:', newUserId)
      addNotification(`New user session created: User #${newUserId}`, 'info')
    } catch (error) {
      console.warn('Failed to save new user ID to localStorage:', error)
    }
    
    // Clear all state for new user session
    clearI2IStateFromStorage()
    clearCartFromStorage()
    setIsUsingItem2Item(false)
    setLastInteractedItem(null)
    setHasRestoredI2I(false)
    setCartItems([])
    setIsCartOpen(false)
  }

  const handleUserIdChange = (e) => {
    const newUserId = parseInt(e.target.value) || 1
    setUserId(newUserId)
    
    // Persist the new user ID to localStorage
    try {
      localStorage.setItem('shopping_user_id', newUserId.toString())
      console.log('Updated stored user ID:', newUserId)
    } catch (error) {
      console.warn('Failed to save user ID to localStorage:', error)
    }
  }

  // Notification management
  const addNotification = (message, type = 'success', duration = 3000) => {
    const id = Date.now()
    const notification = { id, message, type, duration }
    // Replace all existing notifications with the new one
    setNotifications([notification])
  }

  const removeNotification = (id) => {
    setNotifications(prev => prev.filter(notif => notif.id !== id))
  }

  // Function to fetch and append i2i recommendations
  const fetchAndAppendI2IRecommendations = async (product) => {
    try {
      setLoadingMore(true)
      const itemId = product.item_id || product.id
      console.log(`Fetching i2i recommendations for item: ${itemId}`)
      
      // Find the next available page number for appending
      const nextPage = Math.floor(recommendations.length / 10)
      
      // Fetch i2i recommendations and append them
      await fetchRecommendations(userId, nextPage, true, itemId)
      
      // Removed notification for i2i recommendations
    } catch (error) {
      console.error('Error fetching i2i recommendations:', error)
      addNotification('Failed to load related items', 'error')
    } finally {
      setLoadingMore(false)
    }
  }

  // Product details management
  const handleViewProductDetails = (product) => {
    setSelectedProduct(product)
    setIsProductDetailsOpen(true)
    // Auto-collapse cart to prevent overlap with product details modal
    setIsCartOpen(false)
    
    // Track user interaction for item2item recommendations
    setLastInteractedItem(product)
    setIsUsingItem2Item(true)
    saveI2IStateToStorage(product, true)
    console.log(`User viewed product details: ${product.name} (${product.item_id})`)
    
    // Fetch and append related items
    fetchAndAppendI2IRecommendations(product)
  }

  const handleCloseProductDetails = () => {
    setIsProductDetailsOpen(false)
    setSelectedProduct(null)
  }

  // Cart management
  const handleAddToCart = (product) => {
    // Track user interaction for item2item recommendations
    setLastInteractedItem(product)
    setIsUsingItem2Item(true)
    saveI2IStateToStorage(product, true)
    console.log(`User added to cart: ${product.name} (${product.item_id})`)
    
    // Fetch and append related items
    fetchAndAppendI2IRecommendations(product)
    
    // For quick add from card
    const cartItem = {
      ...product,
      quantity: 1,
      cartId: `${product.id}-${Date.now()}` // Unique cart ID
    }

    setCartItems(prev => {
      const existingItem = prev.find(item => 
        item.id === product.id
      )

      let newCart
      if (existingItem) {
        addNotification(`Updated quantity for "${product.name}" in cart`, 'info')
        newCart = prev.map(item =>
          item.cartId === existingItem.cartId
            ? { ...item, quantity: item.quantity + 1 }
            : item
        )
      } else {
        addNotification(`Added "${product.name}" to cart!`, 'success')
        newCart = [...prev, cartItem]
      }
      
      // Save to localStorage
      saveCartToStorage(newCart)
      return newCart
    })
  }

  const handleAddToCartFromDetails = (productWithOptions) => {
    // Track user interaction for item2item recommendations
    setLastInteractedItem(productWithOptions)
    setIsUsingItem2Item(true)
    saveI2IStateToStorage(productWithOptions, true)
    console.log(`User added to cart from details: ${productWithOptions.name} (${productWithOptions.item_id})`)
    
    // Fetch and append related items
    fetchAndAppendI2IRecommendations(productWithOptions)
    
    const cartItem = {
      ...productWithOptions,
      cartId: `${productWithOptions.id}-${Date.now()}`
    }

    setCartItems(prev => {
      const existingItem = prev.find(item => 
        item.id === productWithOptions.id
      )

      let newCart
      if (existingItem) {
        addNotification(`Updated quantity for "${productWithOptions.name}" in cart`, 'info')
        newCart = prev.map(item =>
          item.cartId === existingItem.cartId
            ? { ...item, quantity: item.quantity + productWithOptions.quantity }
            : item
        )
      } else {
        addNotification(`Added "${productWithOptions.name}" to cart!`, 'success')
        newCart = [...prev, cartItem]
      }
      
      // Save to localStorage
      saveCartToStorage(newCart)
      return newCart
    })

    handleCloseProductDetails()
  }

  const handleUpdateCartQuantity = (cartId, newQuantity) => {
    setCartItems(prev => {
      const newCart = prev.map(item =>
        item.cartId === cartId
          ? { ...item, quantity: newQuantity }
          : item
      )
      // Save to localStorage
      saveCartToStorage(newCart)
      return newCart
    })
  }

  const handleRemoveFromCart = (cartId) => {
    setCartItems(prev => {
      const item = prev.find(item => item.cartId === cartId)
      if (item) {
        addNotification(`Removed "${item.name}" from cart`, 'warning')
      }
      const newCart = prev.filter(item => item.cartId !== cartId)
      // Save to localStorage
      saveCartToStorage(newCart)
      return newCart
    })
  }

  const handleClearCart = () => {
    setCartItems([])
    clearCartFromStorage()
    addNotification('Cart cleared', 'info')
  }

  const toggleCart = () => {
    setIsCartOpen(!isCartOpen)
  }

  const handleCheckout = () => {
    if (cartItems.length === 0) {
      addNotification('Cart is empty', 'warning')
      return
    }

    const cartTotal = cartItems.reduce((sum, item) => sum + (item.price * item.quantity), 0)

    // Simulate checkout process
    addNotification(`Checkout successful! Total: $${cartTotal.toFixed(2)}`, 'success')
    setCartItems([])
    clearCartFromStorage()
    setIsCartOpen(false)
  }

  // Debug function to manually trigger item2item recommendations
  const handleTestItem2Item = async () => {
    console.log('Manual item2item test triggered')
    
    let targetItem = null
    if (lastInteractedItem) {
      targetItem = lastInteractedItem
    } else if (recommendations.length > 0) {
      // Use the highest-rated item from current recommendations
      const topRatedItem = recommendations
        .filter(item => item.score || item.rating)
        .sort((a, b) => (b.score || b.rating) - (a.score || a.rating))[0]
      targetItem = topRatedItem || recommendations[0]
    }
    
    if (targetItem) {
      console.log(`Testing item2item with item: ${targetItem.item_id || targetItem.id}`)
      await fetchAndAppendI2IRecommendations(targetItem)
    } else {
      addNotification('No items available for item2item recommendations', 'warning')
    }
  }

  // Debug function to manually clear i2i context
  const handleClearI2IContext = () => {
    setIsUsingItem2Item(false)
    setLastInteractedItem(null)
    setHasRestoredI2I(false)
    clearI2IStateFromStorage()
    addNotification('Item2Item context cleared - showing main recommendations', 'info')
    console.log('Manual i2i context clear triggered')
  }

  return (
    <>
      {/* Notifications - positioned relative to viewport */}
      <div className="notifications-container">
        {notifications.map(notification => (
          <Notification
            key={notification.id}
            message={notification.message}
            type={notification.type}
            duration={notification.duration}
            onClose={() => removeNotification(notification.id)}
          />
        ))}
      </div>

      <div className="shopping-page">

      <header className="shopping-header">
        <h1>🛍️ Personalized Shopping Recommendations</h1>
        <p className="subtitle">
          Discover products tailored for User #{userId} • Session persists across refreshes
        </p>
        
        <div className="controls">
          <div className="user-control">
            <label htmlFor="userId">User ID:</label>
            <input
              id="userId"
              type="number"
              value={userId}
              onChange={handleUserIdChange}
              min="1"
              className="user-input"
            />
            <button 
              onClick={generateNewUserSession} 
              className="refresh-btn"
              style={{marginLeft: '8px', fontSize: '0.8rem', padding: '4px 8px'}}
              title="Generate new user session"
            >
              🎲 New User
            </button>
          </div>
          <button onClick={handleRefresh} className="refresh-btn" disabled={loading}>
            🔄 Refresh Recommendations
          </button>
          <button onClick={handleTestItem2Item} className="refresh-btn" disabled={loading || loadingMore}>
            🔗 Test Item2Item
          </button>
          {(lastInteractedItem || isUsingItem2Item) && (
            <button onClick={handleClearI2IContext} className="refresh-btn" style={{backgroundColor: '#ff6b6b'}}>
              🗑️ Clear i2i Context
            </button>
          )}
          <button onClick={toggleCart} className="cart-toggle-btn">
            🛒 Cart ({cartItems.reduce((sum, item) => sum + item.quantity, 0)})
          </button>
          {cartItems.length > 0 && (
            <button onClick={handleCheckout} className="checkout-btn">
              💳 Checkout (${cartItems.reduce((sum, item) => sum + (item.price * item.quantity), 0).toFixed(2)})
            </button>
          )}
        </div>
      </header>

      <main className="shopping-content">
        {loading && <LoadingSpinner />}
        
        {error && <ErrorMessage message={error} onRetry={handleRefresh} />}
        
        {!loading && !error && recommendations.length > 0 && (
          <>
            <div className="recommendations-info">
              <h2>Recommended for User #{userId} (Persistent Session)</h2>
              <p className="items-count">
                {hasMoreItems && "Scroll down for more"}
                {isUsingItem2Item && (
                  <span style={{ color: '#667eea', fontWeight: 'bold' }}>
                    {lastInteractedItem ? 
                      ` • Including related items based on "${lastInteractedItem.name}" (click items to add more)` : 
                      ' • Including related items based on your browsing (click items to add more)'}
                  </span>
                )}
              </p>
            </div>
            
            <div className="products-grid">
              {recommendations.map((item, index) => {
                // Create unique key combining item.id with index to prevent duplicates
                const uniqueKey = `${item.id}-${index}`
                
                // Add ref to the last item for infinite scroll
                if (index === recommendations.length - 1) {
                  return (
                    <div key={uniqueKey} ref={lastItemElementRef}>
                      <ProductCard 
                        product={item} 
                        onViewDetails={handleViewProductDetails}
                        onAddToCart={handleAddToCart}
                      />
                    </div>
                  )
                }
                return (
                  <ProductCard 
                    key={uniqueKey} 
                    product={item} 
                    onViewDetails={handleViewProductDetails}
                    onAddToCart={handleAddToCart}
                  />
                )
              })}
            </div>
            
            {/* Loading indicator for infinite scroll */}
            {loadingMore && (
              <div style={{ 
                display: 'flex', 
                justifyContent: 'center', 
                padding: '20px',
                color: '#666'
              }}>
                <div style={{ 
                  display: 'flex', 
                  alignItems: 'center', 
                  gap: '10px',
                  fontSize: '1rem'
                }}>
                  <div style={{
                    width: '20px',
                    height: '20px',
                    border: '2px solid #e0e0e0',
                    borderTop: '2px solid #667eea',
                    borderRadius: '50%',
                    animation: 'spin 1s linear infinite'
                  }}></div>
                  {isUsingItem2Item && lastInteractedItem ? 
                    `Loading items related to "${lastInteractedItem.name}"...` : 
                    isUsingItem2Item ? 
                    'Loading related items...' : 
                    'Loading more items...'}
                </div>
              </div>
            )}
            
            {/* End of results indicator */}
            {!hasMoreItems && recommendations.length > 10 && (
              <div style={{ 
                display: 'flex', 
                justifyContent: 'center', 
                padding: '20px',
                color: '#999',
                fontSize: '0.9rem'
              }}>
                🎉 You've reached the end! No more items to load.
              </div>
            )}
          </>
        )}
        
        {!loading && !error && recommendations.length === 0 && (
          <div className="empty-state">
            <h3>No recommendations found</h3>
            <p>Try refreshing or changing the user ID</p>
          </div>
        )}
      </main>
    </div>

      {/* Product Details Modal - rendered outside main container for proper viewport positioning */}
      <ProductDetails
        product={selectedProduct}
        isOpen={isProductDetailsOpen}
        onClose={handleCloseProductDetails}
        onAddToCart={handleAddToCartFromDetails}
      />

      {/* Shopping Cart Panel */}
      <ShoppingCart
        isOpen={isCartOpen}
        onClose={() => setIsCartOpen(false)}
        cartItems={cartItems}
        onUpdateQuantity={handleUpdateCartQuantity}
        onRemoveItem={handleRemoveFromCart}
        onClearCart={handleClearCart}
      />
    </>
  )
}

export default ShoppingPage
