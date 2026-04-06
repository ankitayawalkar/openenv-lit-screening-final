from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict
import json

app = FastAPI()

# ---------- LOAD DATA ----------
with open("data.json") as f:
    ORIGINAL_DATA = json.load(f)

data = []
index = 0

# ---------- RESET (POST REQUIRED) ----------
@app.post("/reset")
def reset():
    global data, index
    data = ORIGINAL_DATA.copy()
    index = 0
    return {"status": "ok"}

# ---------- STATE ----------
@app.get("/state")
def state():
    global index, data
    if index >= len(data):
        return None
    return data[index]

# ---------- STEP REQUEST ----------
class StepRequest(BaseModel):
    action: str

# ---------- STEP ----------
@app.post("/step")
def step(request: StepRequest):
    global index, data

    if index >= len(data):
        return {
            "observation": {},
            "reward": 0.0,
            "done": True
        }

    action = request.action
    true_label = data[index].get("label", "include")

    reward = 1.0 if action == true_label else 0.0

    index += 1
    done = index >= len(data)

    return {
        "observation": {},
        "reward": reward,
        "done": done
    }

# ---------- GRADER ----------
class PredictionRequest(BaseModel):
    predictions: List[Dict]

@app.post("/grader")
def grader(request: PredictionRequest):
    preds = request.predictions
    score = 0

    for pred, item in zip(preds, ORIGINAL_DATA):
        if pred.get("label") == item.get("label"):
            score += 1

    final_score = score / len(ORIGINAL_DATA)
    return {"score": final_score}
