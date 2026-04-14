from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional

# 1. Initialize the app
app = FastAPI()

# 2. Define a data model using Pydantic
class Item(BaseModel):
    name: str
    price: float
    is_offer: Optional[bool] = None

# 3. Create a GET route (Basic)
@app.get("/")
def read_root():
    return {"Hello": "World"}

# 4. Create a GET route with path and query parameters
@app.get("/items/{item_id}")
def read_item(item_id: int, q: Optional[str] = None):
    return {"item_id": item_id, "q": q}

# 5. Create a POST route to receive JSON data
@app.post("/items/")
def create_item(item: Item):
    return {"item_name": item.name, "item_price": item.price}
