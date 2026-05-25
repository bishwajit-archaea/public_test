pip install fastapi uvicorn
    ```
2.  **Start the server:**
    ```bash
    uvicorn main:app --reload
    ```
    *   `main`: refers to the file name (`main.py`).
    *   `app`: refers to the object created inside the file (`app = FastAPI()`).
    *   `--reload`: automatically restarts the server when you save changes to your code.

---

### 3. Key Features to Check Out
Once the server is running, FastAPI provides two things out-of-the-box that make life much easier:

*   **Interactive API Docs (Swagger UI):** Go to `[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)`. You can test your endpoints directly from the browser.
*   **Alternative Docs (ReDoc):** Go to `[http://127.0.0.1:8000](http://127.0.0.1:8000)Setting up a FastAPI application is remarkably straightforward. It’s built on modern Python type hints, which makes it fast to write and even faster to run.

Below is a complete, "ready-to-run" example of a basic FastAPI file.

### 1. Create the File: `main.py`
```python
from fastapi import FastAPI
from typing import Optional
from pydantic import BaseModel
#commend add
# Initialize the app
app = FastAPI()
# Optional: Define a data model for POST requests
class Item(BaseModel):
    name: str
    price: float
    is_offer: Optional[bool] = None
#ok
@app.get("/")
def read_root():
    return {"message": "Hello, FastAPI!"}

@app.get("/items/{item_id}")
def read_item(item_id: int, q: Optional[str] = None):
    return {"item_id": item_id, "query_param": q}

@app.post("/items/")
def create_item(item: Item):
    return {"message": f"Item {item.name} created", "data": item}
