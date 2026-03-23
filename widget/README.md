# RAG Chatbot Widget

An embeddable web chat widget for your AI chatbot. Simply paste a JavaScript snippet on any website to add a chat interface.

## Features

- 🚀 Easy integration - Single script tag
- 🎨 Customizable colors and styling
- 📱 Mobile responsive
- 💬 Real-time streaming responses
- 🔧 Configurable via data attributes
- 🌐 CORS-enabled API support

## Quick Start

### 1. Deploy Widget Files

Upload the widget files to your web server:
- `chatbot-widget.js` - Main widget JavaScript
- `loader.js` - Auto-loader script
        ``      
### 2. Update API Service

Make sure your API service (`src/api_service.py`) is running and accessible. The widget will connect to:
- `/query/stream` - Streaming endpoint for chat
- `/query` - Non-streaming endpoint (fallback)

### 3. Embed on Your Website

Add this single line to your HTML:

```html
<script src="https://your-domain.com/widget/loader.js" 
        data-api-url="https://api.example.com"
        data-position="bottom-right"
        data-primary-color="#1f77b4"></script>
```

## Configuration

All configuration is done via data attributes on the script tag:

| Attribute | Description | Default |
|-----------|-------------|---------|
| `data-api-url` | API endpoint URL | `http://localhost:8000` |
| `data-subject` | Filter by subject | `null` (all subjects) |
| `data-model-key` | AI model to use | `null` (default) |
| `data-position` | Widget position | `bottom-right` |
| `data-primary-color` | Primary color (hex) | `#1f77b4` |
| `data-text-color` | Text color (hex) | `#333333` |
| `data-background-color` | Background color (hex) | `#ffffff` |
| `data-button-text` | Chat button text | `💬 Chat` |
| `data-welcome-message` | Welcome message | `Xin chào! Tôi có thể giúp gì cho bạn?` |
| `data-placeholder` | Input placeholder | `Nhập câu hỏi của bạn...` |

## Examples

### Basic Usage

```html
<script src="https://your-domain.com/widget/loader.js" 
        data-api-url="https://api.example.com"></script>
```

### Custom Styling

```html
<script src="https://your-domain.com/widget/loader.js" 
        data-api-url="https://api.example.com"
        data-primary-color="#FF5733"
        data-position="bottom-left"
        data-button-text="💬 Hỏi đáp"></script>
```

### Subject-Specific Widget

```html
<script src="https://your-domain.com/widget/loader.js" 
        data-api-url="https://api.example.com"
        data-subject="Mathematics"
        data-welcome-message="Xin chào! Tôi có thể giúp gì cho bạn về môn Toán?"></script>
```

## API Requirements

Your API service must:

1. **Enable CORS** - The widget makes cross-origin requests
2. **Provide `/query/stream` endpoint** - For streaming responses
3. **Accept JSON requests** with:
   ```json
   {
     "question": "User question",
     "subject": "optional subject",
     "model_key": "optional model",
     "use_history": true
   }
   ```
4. **Return Server-Sent Events (SSE)** with format:
   ```
   data: {"chunk": "text chunk"}
   data: {"done": true}
   ```

## Manual Initialization

For advanced use cases, you can manually initialize the widget:

```html
<script src="https://your-domain.com/widget/chatbot-widget.js"></script>
<script>
    const widget = new ChatbotWidget({
        apiUrl: 'https://api.example.com',
        subject: 'Mathematics',
        position: 'bottom-right',
        primaryColor: '#1f77b4'
    });
    
    // Access widget instance
    window.myChatbot = widget;
</script>
```

## Browser Support

- Chrome/Edge (latest)
- Firefox (latest)
- Safari (latest)
- Mobile browsers (iOS Safari, Chrome Mobile)

## Troubleshooting

### Widget doesn't appear
- Check browser console for errors
- Verify `loader.js` URL is accessible
- Ensure API URL is correct

### API connection fails
- Check CORS settings on API server
- Verify API endpoint is accessible
- Check network tab for request/response details

### Streaming not working
- Ensure `/query/stream` endpoint exists
- Check API returns proper SSE format
- Verify API server supports streaming

## Development

To test locally:

1. Start your API server:
   ```bash
   python -m uvicorn src.api_service:app --reload
   ```

2. Serve widget files (use a simple HTTP server):
   ```bash
   cd widget
   python -m http.server 8080
   ```

3. Open `example.html` in your browser

## License

Same as the main project.