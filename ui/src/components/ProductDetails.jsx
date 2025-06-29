import { useState, useEffect } from 'react'
import './ProductDetails.css'

const ProductDetails = ({ product, isOpen, onClose, onAddToCart }) => {
  const [quantity, setQuantity] = useState(1)
  const [currentImageIndex, setCurrentImageIndex] = useState(0)

  // Handle body scroll locking when modal is open
  useEffect(() => {
    if (isOpen) {
      document.body.classList.add('modal-open')
    } else {
      document.body.classList.remove('modal-open')
    }

    // Cleanup on unmount
    return () => {
      document.body.classList.remove('modal-open')
    }
  }, [isOpen])

  // Reset thumbnail state when product changes
  useEffect(() => {
    setCurrentImageIndex(0)
  }, [product])

  if (!product) return null

  const { name, score, price, image, description, rating, num_reviews, main_category, image_urls } = product

  // Use image_urls array if available, otherwise fallback to single image
  const productImages = image_urls && image_urls.length > 0 ? image_urls : [image]

  const getScoreColor = (score) => {
    if (score >= 0.9) return '#4CAF50'
    if (score >= 0.8) return '#FF9800'
    if (score >= 0.7) return '#2196F3'
    return '#9E9E9E'
  }

  const getScoreLabel = (score) => {
    if (score >= 0.9) return 'Highly Recommended'
    if (score >= 0.8) return 'Recommended'
    if (score >= 0.7) return 'Good Match'
    return 'Fair Match'
  }

  const handleAddToCart = () => {
    onAddToCart({
      ...product,
      quantity,
      totalPrice: price * quantity
    })
  }

  const handlePreviousImage = () => {
    setCurrentImageIndex(prev => 
      prev === 0 ? productImages.length - 1 : prev - 1
    )
  }

  const handleNextImage = () => {
    setCurrentImageIndex(prev => 
      prev === productImages.length - 1 ? 0 : prev + 1
    )
  }

  return (
    <div className={`product-details-overlay ${isOpen ? 'open' : ''}`} onClick={onClose}>
      <div className="product-details-panel" onClick={(e) => e.stopPropagation()}>
        <div className="product-details-header">
          <h2>Product Details</h2>
          <button className="close-btn" onClick={onClose}>×</button>
        </div>

        <div className="product-details-content">
          <div className="product-details-image">
            <div className="main-image-container">
              <img src={productImages[currentImageIndex]} alt={name} />
              <div 
                className="score-badge-large" 
                style={{ backgroundColor: getScoreColor(score) }}
              >
                {(score * 100).toFixed(0)}%
              </div>
            </div>
            
            {productImages.length > 1 && (
              <div className="image-thumbnails">
                {productImages.map((imgUrl, index) => (
                  <img 
                    key={index}
                    src={imgUrl} 
                    alt={`${name} ${index + 1}`}
                    className={`thumbnail ${index === currentImageIndex ? 'active' : ''}`}
                    onClick={() => setCurrentImageIndex(index)}
                  />
                ))}
              </div>
            )}
          </div>

          <div className="product-details-info">
            <div className="category-badge-large">{main_category}</div>
            <h1 className="product-title">{name}</h1>
            
            {rating > 0 && (
              <div className="product-rating-large">
                <div className="stars-large">
                  {[...Array(5)].map((_, i) => (
                    <span 
                      key={i} 
                      className={`star-large ${i < Math.floor(rating) ? 'filled' : ''}`}
                    >
                      ★
                    </span>
                  ))}
                </div>
                <span className="rating-text-large">
                  {rating.toFixed(1)} out of 5 ({num_reviews} review{num_reviews !== 1 ? 's' : ''})
                </span>
              </div>
            )}
            
            <div className="product-price-large">
              {price && price > 0 ? `$${price}` : 'Price not available'}
            </div>
            
            <div className="score-section">
              <span className="score-label-large">{getScoreLabel(score)}</span>
              <div className="score-bar-large">
                <div 
                  className="score-fill-large" 
                  style={{ 
                    width: `${score * 100}%`,
                    backgroundColor: getScoreColor(score)
                  }}
                ></div>
              </div>
            </div>

            <p className="product-description-large">{description}</p>

            <div className="product-options">
              <div className="option-group">
                <label>Quantity:</label>
                <div className="quantity-controls">
                  <button 
                    className="qty-btn" 
                    onClick={() => setQuantity(Math.max(1, quantity - 1))}
                  >
                    -
                  </button>
                  <span className="quantity">{quantity}</span>
                  <button 
                    className="qty-btn" 
                    onClick={() => setQuantity(quantity + 1)}
                  >
                    +
                  </button>
                </div>
              </div>
            </div>

            <div className="product-actions">
              <div className="total-price">
                Total: ${(price * quantity).toFixed(2)}
              </div>
              <button className="add-to-cart-large" onClick={handleAddToCart}>
                Add to Cart
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default ProductDetails
