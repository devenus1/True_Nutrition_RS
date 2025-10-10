import asyncio
import time
import httpx
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic_settings import BaseSettings
import os
from dotenv import load_dotenv
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import settings

load_dotenv()

app = FastAPI()

# CORS middleware 
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Runpod constants
RUNPOD_ENDPOINT_ID = os.getenv("RUNPOD_ENDPOINT_ID")
RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY")
RUNPOD_RECOMMENDATIONS_URL = f"https://api.runpod.ai/v2/{RUNPOD_ENDPOINT_ID}/run"
HEADERS = {
    "Authorization": f"Bearer {RUNPOD_API_KEY}",
    "Content-Type": "application/json",
}

AUTHORIZATION_TOKEN = os.getenv("AUTHORIZATION_TOKEN")

# Security
security = HTTPBearer()

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.credentials != AUTHORIZATION_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing token",
        )

async def send_request_to_runpod_recommendations(
    response_id: str,
    client: httpx.AsyncClient,
    url=RUNPOD_RECOMMENDATIONS_URL,
    headers=HEADERS,
):
    payload = {"input": {"response_id": response_id}}
    response = await client.post(url, json=payload, headers=headers)
    response.raise_for_status()  # Raise an exception for bad status codes
    data = response.json()
    return data.get("id")

async def poll_status(
    task_id: str,
    client: httpx.AsyncClient,
    endpoint_id: str = RUNPOD_ENDPOINT_ID,
    headers=HEADERS,
    timeout: int = 500,
    interval: int = 3,
):
    """Polls the status of the task asynchronously."""
    url_status = f"https://api.runpod.ai/v2/{endpoint_id}/status/{task_id}"
    start_time = time.time()

    while time.time() - start_time < timeout:
        response = await client.get(url_status, headers=headers)
        if response.status_code == 200:
            status_data = response.json()
            status = status_data.get("status")
            if status == "COMPLETED":
                output = status_data.get("output", {})
                if "error" in output:
                    raise HTTPException(status_code=500, detail=f"Runpod handler error: {output['error']}")
                return output
            elif status in {"FAILED", "CANCELLED", "ERROR"}:
                error_msg = status_data.get("error", "Unknown error")
                raise HTTPException(status_code=500, detail=f"Runpod task failed: {error_msg}")
        
        await asyncio.sleep(interval)

    raise HTTPException(status_code=504, detail=f"Task {task_id} timed out after {timeout} seconds.")

@app.get("/get-recommendations/{response_id}")
async def get_recommendations(
    response_id: str,
    credentials: HTTPAuthorizationCredentials = Depends(verify_token),
):
    async with httpx.AsyncClient() as client:
        try:
            task_id = await send_request_to_runpod_recommendations(response_id, client)
            if not task_id:
                raise HTTPException(status_code=500, detail="Failed to submit job to Runpod")
            
            result = await poll_status(task_id, client)
            return result
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")