"""Background helpers shared by the API."""
import time
from datetime import datetime

db = {"tasks": [], "logs": []}
LOG_DELAY_SECONDS = 2  # how long write_log waits


def write_log(message: str):
    # Simulates a slow IO operation (like writing to a file or sending email)
    time.sleep(2) 
    db["logs"].append(f"LOG [{datetime.now()}]: {message}")
