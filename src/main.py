from fastapi import FastAPI

from src.api.routes import router

app = FastAPI(title="Insurance QnA Bot", version="0.1.0")
app.include_router(router)

# Serve React frontend in production:
# from fastapi.staticfiles import StaticFiles
# app.mount("/", StaticFiles(directory="frontend/build", html=True), name="frontend")
