# Shopping Recommendation UI

A modern React + Vite application for displaying personalized shopping recommendations with a beautiful, responsive interface.

## Features

- 🛍️ **Personalized Recommendations**: Fetches and displays recommended items for specific users
- 📊 **Smart Sorting**: Items are automatically sorted by recommendation scores (highest first)
- 🎨 **Modern Design**: Beautiful, responsive UI with gradient backgrounds and smooth animations
- 🔄 **Real-time Updates**: Refresh recommendations and change user IDs dynamically
- 📱 **Mobile Responsive**: Optimized for all screen sizes
- ⚡ **Fast Loading**: Built with Vite for optimal performance

## API Integration

The application expects API responses in the following JSON format:

```json
{
  "userid": 123,
  "recommendations": ["Product A", "Product B", "Product C"],
  "score": [0.95, 0.87, 0.73]
}
```

## Getting Started

### Prerequisites
- Node.js (version 14 or higher)
- npm or yarn

### Installation

1. Install dependencies:
```bash
npm install
```

2. Start the development server:
```bash
npm run dev
```

3. Open your browser and navigate to `http://localhost:5173`

### Building for Production

```bash
npm run build
```

### Preview Production Build

```bash
npm run preview
```

## Project Structure

```
src/
├── components/
│   ├── ShoppingPage.jsx      # Main shopping page component
│   ├── ProductCard.jsx       # Individual product card
│   ├── LoadingSpinner.jsx    # Loading indicator
│   ├── ErrorMessage.jsx      # Error handling component
│   └── *.css                 # Component-specific styles
├── App.jsx                   # Root component
├── App.css                   # Global styles
└── main.jsx                  # Application entry point
```

## Customization

### API Integration
Replace the mock API call in `src/components/ShoppingPage.jsx` with your actual API endpoint:

```javascript
// Replace this mock call
const response = await fetch(`/api/recommendations?userId=${userId}`)
const data = await response.json()
```

### Styling
The application uses modern CSS with gradients, shadows, and animations. You can customize:
- Color schemes in the CSS files
- Layout and spacing
- Animation effects
- Responsive breakpoints

## Technologies Used

- **React 18**: Modern React with hooks
- **Vite**: Next-generation frontend tooling
- **CSS3**: Modern styling with flexbox and grid
- **ESLint**: Code quality and consistency

## License

This project is open source and available under the [MIT License](LICENSE).
