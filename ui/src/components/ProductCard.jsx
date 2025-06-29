import './ProductCard.css'

const ProductCard = ({ product, onViewDetails, onAddToCart }) => {
  const { name, score, price, image, description, rating, num_reviews, main_category } = product

  const handleAddToCart = (e) => {
    e.stopPropagation()
    onAddToCart(product)
  }

  const handleViewDetails = () => {
    onViewDetails(product)
  }

  const getScoreColor = (score) => {
    if (score >= 0.9) return '#4CAF50' // Green
    if (score >= 0.8) return '#FF9800' // Orange
    if (score >= 0.7) return '#2196F3' // Blue
    return '#9E9E9E' // Gray
  }

  const getScoreLabel = (score) => {
    if (score >= 0.9) return 'Highly Recommended'
    if (score >= 0.8) return 'Recommended'
    if (score >= 0.7) return 'Good Match'
    return 'Fair Match'
  }

  return (
    <div className="product-card" onClick={handleViewDetails}>
      <div className="product-image-container">
        <img src={image} alt={name} className="product-image" />
        <div 
          className="score-badge" 
          style={{ backgroundColor: getScoreColor(score) }}
          title={`Recommendation Score: ${(score * 100).toFixed(1)}%`}
        >
          {(score * 100).toFixed(0)}%
        </div>
      </div>
      
      <div className="product-info">
        <div className="category-badge">{main_category}</div>
        <h3 className="product-name" title={name}>{name}</h3>
        
        {rating > 0 && (
          <div className="product-rating">
            <div className="stars">
              {[...Array(5)].map((_, i) => (
                <span 
                  key={i} 
                  className={`star ${i < Math.floor(rating) ? 'filled' : ''}`}
                >
                  ★
                </span>
              ))}
            </div>
            <span className="rating-text">
              {rating.toFixed(1)} ({num_reviews} review{num_reviews !== 1 ? 's' : ''})
            </span>
          </div>
        )}
        
        <div className="product-meta">
          <div className="score-info">
            <span className="score-label">{getScoreLabel(score)}</span>
            <div className="score-bar">
              <div 
                className="score-fill" 
                style={{ 
                  width: `${score * 100}%`,
                  backgroundColor: getScoreColor(score)
                }}
              ></div>
            </div>
          </div>
        </div>
        
        <div className="product-footer">
          <span className="product-price">
            {price && price > 0 ? `$${price}` : 'Price N/A'}
          </span>
          <button 
            className="add-to-cart-btn"
            onClick={handleAddToCart}
          >
            Add to Cart
          </button>
        </div>
      </div>
    </div>
  )
}

export default ProductCard
