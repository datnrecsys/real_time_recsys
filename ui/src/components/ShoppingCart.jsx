import { useState } from 'react'
import './ShoppingCart.css'

const ShoppingCart = ({ isOpen, onClose, cartItems, onUpdateQuantity, onRemoveItem, onClearCart }) => {
  const [isAnimating, setIsAnimating] = useState(false)

  const totalPrice = cartItems.reduce((sum, item) => sum + (item.price * item.quantity), 0)
  const totalItems = cartItems.reduce((sum, item) => sum + item.quantity, 0)

  const handleQuantityChange = (cartId, newQuantity) => {
    if (newQuantity < 1) {
      onRemoveItem(cartId)
    } else {
      onUpdateQuantity(cartId, newQuantity)
    }
  }

  const handleCheckout = () => {
    setIsAnimating(true)
    setTimeout(() => {
      alert('Checkout functionality would be implemented here!')
      setIsAnimating(false)
    }, 1000)
  }

  return (
    <div className={`shopping-cart-overlay ${isOpen ? 'open' : ''}`}>
      <div className="shopping-cart-panel">
        <div className="cart-header">
          <div className="cart-title">
            <h2>🛒 Shopping Cart</h2>
            <span className="cart-count">({totalItems} items)</span>
          </div>
          <button className="cart-close-btn" onClick={onClose}>×</button>
        </div>

        <div className="cart-content">
          {cartItems.length === 0 ? (
            <div className="empty-cart">
              <div className="empty-cart-icon">🛒</div>
              <h3>Your cart is empty</h3>
              <p>Add some items to get started!</p>
            </div>
          ) : (
            <>
              <div className="cart-items">
                {cartItems.map((item) => (
                  <div key={item.cartId} className="cart-item">
                    <img src={item.image} alt={item.name} className="cart-item-image" />
                    
                    <div className="cart-item-details">
                      <h4 className="cart-item-name">{item.name}</h4>
                      <div className="cart-item-options">
                        <span>Size: {item.selectedSize}</span>
                        <span>Color: {item.selectedColor}</span>
                      </div>
                      <div className="cart-item-score">
                        Score: {(item.score * 100).toFixed(0)}%
                      </div>
                    </div>

                    <div className="cart-item-controls">
                      <div className="cart-quantity-controls">
                        <button 
                          className="cart-qty-btn"
                          onClick={() => handleQuantityChange(item.cartId, item.quantity - 1)}
                        >
                          -
                        </button>
                        <span className="cart-quantity">{item.quantity}</span>
                        <button 
                          className="cart-qty-btn"
                          onClick={() => handleQuantityChange(item.cartId, item.quantity + 1)}
                        >
                          +
                        </button>
                      </div>
                      
                      <div className="cart-item-price">${item.price}</div>
                      <div className="cart-item-total">${(item.price * item.quantity).toFixed(2)}</div>
                      
                      <button 
                        className="remove-item-btn"
                        onClick={() => onRemoveItem(item.cartId)}
                        title="Remove item"
                      >
                        🗑️
                      </button>
                    </div>
                  </div>
                ))}
              </div>

              <div className="cart-summary">
                <div className="cart-total">
                  <div className="total-row">
                    <span>Subtotal ({totalItems} items):</span>
                    <span>${totalPrice.toFixed(2)}</span>
                  </div>
                  <div className="total-row">
                    <span>Shipping:</span>
                    <span>FREE</span>
                  </div>
                  <div className="total-row total-final">
                    <span>Total:</span>
                    <span>${totalPrice.toFixed(2)}</span>
                  </div>
                </div>

                <div className="cart-actions">
                  <button 
                    className="clear-cart-btn"
                    onClick={onClearCart}
                    disabled={cartItems.length === 0}
                  >
                    Clear Cart
                  </button>
                  <button 
                    className={`checkout-btn ${isAnimating ? 'animating' : ''}`}
                    onClick={handleCheckout}
                    disabled={cartItems.length === 0 || isAnimating}
                  >
                    {isAnimating ? 'Processing...' : 'Checkout'}
                  </button>
                </div>
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  )
}

export default ShoppingCart
