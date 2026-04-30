from fastapi import FastAPI, HTTPException

app = FastAPI()

items = {"1": "Machine Learning Primer"}

@app.get("/items/{item_id}")
def read_item(item_id: str):
    if item_id not in items:
        # This manually triggers a 404 error
        raise HTTPException(status_code=404, detail="Item not found")
    return {"item": items[item_id]}
