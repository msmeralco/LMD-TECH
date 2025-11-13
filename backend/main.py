import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",  # ← Points to the 'app' object in app/main.py
        host="0.0.0.0",
        port=8000,
        reload=True,  # Auto-reload when you edit code
    )