from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict
import json

app = FastAPI()

with open("data.json") as f:
    ORIGINAL_DATA = json.load(f)

data = []
index = 0

@app.post("/reset")
def reset():
    global data, index
    data = ORIGINAL_DATA.copy()
    index = 0
    return {"status": "ok"}

@app.get("/state")
def state():
    global index, data
    if index >= len(data):
        return None
    return data[index]

class StepRequest(BaseModel):
    action: str

@app.post("/step")
def step(request: StepRequest):
    global index, data

    if index >= len(data):
        return {"observation": {}, "reward": 0.0, "done": True}

    action = request.action
    true_label = data[index].get("label", "include")

    reward = 1.0 if action == true_label else 0.0

    index += 1
    done = index >= len(data)

    return {"observation": {}, "reward": reward, "done": done}

class PredictionRequest(BaseModel):
    predictions: List[Dict]

@app.post("/grader")
def grader(request: PredictionRequest):
    preds = request.predictions
    score = 0

    for pred, item in zip(preds, ORIGINAL_DATA):
        if pred.get("label") == item.get("label"):
            score += 1

    return {"score": score / len(ORIGINAL_DATA)}
