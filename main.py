# Standard and third-party imports
from fastapi import FastAPI
import uvicorn

# Local imports
from api import router as api_router

def create_app() -> FastAPI:
    """
    Create and configure an instance of the FastAPI application.
    """
    app = FastAPI()
    app.include_router(api_router, prefix="/api")
    return app

app = create_app()

if __name__ == "__main__": 
    uvicorn.run(app, host="0.0.0.0", port=8000)
