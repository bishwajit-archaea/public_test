import time
import asyncio
from typing import List, Optional
from uuid import uuid4, UUID

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, status, Query
from pydantic import BaseModel, Field, EmailStr
from datetime import datetime

# --- 1. THE APP INSTANCE ---
app = FastAPI(
    title="Gemini's FastAPI Showcase",
    description="A high-performance API featuring Async, Pydantic, and DI.",
    version="2.0.0"
)

# --- 2. DATA MODELS (Pydantic) ---
class Task(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    title: str = Field(..., min_length=3, max_length=50, example="Launch Rocket")
    description: Optional[str] = None
    priority: int = Field(ge=1, le=5, description="Priority from 1-5")
    created_at: datetime = Field(default_factory=datetime.now)

class User(BaseModel):
    username: str
    email: EmailStr

# --- 3. IN-MEMORY DATABASE ---
db = {
    "tasks": [],
    "logs": []
}

# --- 4. DEPENDENCY INJECTION ---
# Simulates checking an API key or User Session
async def get_active_user(api_key: str = Query(...)):
    if api_key != "secret-token":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, 
            detail="Invalid API Key"
        )
    return {"user": "Admin", "role": "Superuser"}

# --- 5. BACKGROUND TASKS ---
def write_log(message: str):
    # Simulates a slow IO operation (like writing to a file or sending email)
    time.sleep(2) 
    db["logs"].append(f"LOG [{datetime.now()}]: {message}")

# --- 6. ROUTES ---

@app.get("/", tags=["General"])
async def root():
    """Welcome endpoint with basic health check."""
    return {"status": "online", "engine": "FastAPI", "speed": "Ludicrous"}

@app.post("/tasks/", response_model=Task, status_code=status.HTTP_201_CREATED, tags=["Tasks"])
async def create_task(
    task: Task, 
    background_tasks: BackgroundTasks,
    user: dict = Depends(get_active_user)
):
    """
    Creates a task, validates the input via Pydantic, 
    and triggers a background logging process.
    """
    db["tasks"].append(task)
    background_tasks.add_task(write_log, f"Task '{task.title}' created by {user['user']}")
    return task

@app.get("/tasks/", response_model=List[Task], tags=["Tasks"])
async def get_tasks(priority: Optional[int] = None):
    """Fetch all tasks with optional filtering."""
    if priority:
        return [t for t in db["tasks"] if t.priority == priority]
    return db["tasks"]

@app.get("/slow-operation", tags=["Performance"])
async def slow_async():
    """Demonstrates non-blocking async wait."""
    await asyncio.sleep(3)
    return {"message": "This didn't block other users while waiting!"}

@app.get("/logs", tags=["Admin"])
def get_logs():
    return {"logs": db["logs"]}
    #test
