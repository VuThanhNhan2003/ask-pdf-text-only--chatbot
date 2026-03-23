# Quick Start Guide - Embeddable Chat Widget

## 🚀 Get Started in 3 Steps

### Step 1: Deploy Widget Files

Upload these files to your web server:
- `chatbot-widget.js` → Main widget code
- `loader.js` → Auto-loader script

**Example locations:**
- `https://your-domain.com/widget/chatbot-widget.js`
- `https://your-domain.com/widget/loader.js`

### Step 2: Start Your API Server

Make sure your FastAPI server is running and accessible:

```bash
python -m uvicorn src.api_service:app --host 0.0.0.0 --port 8000
```

The API should be accessible at: `http://your-api-domain.com:8000`

### Step 3: Paste This Code on Your Website

Add this single line before the closing `</body>` tag:

```html
<script src="https://your-domain.com/widget/loader.js" 
        data-api-url="https://your-api-domain.com:8000"></script>
```

**That's it!** The chat widget will appear in the bottom-right corner.

---

## 🎨 Customization Examples

### Change Colors
```html
<script src="https://your-domain.com/widget/loader.js" 
        data-api-url="https://api.example.com"
        data-primary-color="#FF5733"
        data-background-color="#F5F5F5"></script>
```

### Change Position
```html
<script src="https://your-domain.com/widget/loader.js" 
        data-api-url="https://api.example.com"
        data-position="bottom-left"></script>
```

### Subject-Specific Widget
```html
<script src="https://your-domain.com/widget/loader.js" 
        data-api-url="https://api.example.com"
        data-subject="Mathematics"
        data-welcome-message="Ask me anything about Math!"></script>
```

### Custom Button Text
```html
<script src="https://your-domain.com/widget/loader.js" 
        data-api-url="https://api.example.com"
        data-button-text="💬 Hỏi đáp"
        data-placeholder="Nhập câu hỏi..."></script>
```

---

## 🔧 Generate Your Unique Snippet

1. Open `snippet-generator.html` in your browser
2. Fill in your configuration
3. Click "Copy Snippet"
4. Paste it on your website

---

## ✅ Testing Checklist

- [ ] Widget files are accessible via HTTPS/HTTP
- [ ] API server is running and accessible
- [ ] CORS is enabled on API server
- [ ] `/query/stream` endpoint works
- [ ] Widget appears on test page
- [ ] Chat messages send successfully
- [ ] Streaming responses work
- [ ] Mobile view looks good

---

## 🐛 Troubleshooting

**Widget doesn't appear:**
- Check browser console (F12) for errors
- Verify script URL is correct
- Check if API URL is accessible

**API connection fails:**
- Verify CORS is enabled: `app.add_middleware(CORSMiddleware, ...)`
- Check API endpoint: `/query/stream`
- Test API directly: `curl -X POST http://api/query/stream ...`

**Streaming not working:**
- Ensure endpoint returns Server-Sent Events (SSE)
- Check response format: `data: {"chunk": "text"}`

---

## 📱 Mobile Support

The widget automatically adapts to mobile screens:
- Full-screen on devices < 480px width
- Touch-friendly buttons
- Responsive layout

---

## 🔒 Security Notes

For production:
1. Restrict CORS origins (don't use `allow_origins=["*"]`)
2. Use HTTPS for all URLs
3. Consider adding API authentication
4. Rate limit API endpoints

---

## 📚 Full Documentation

See `README.md` for complete documentation.