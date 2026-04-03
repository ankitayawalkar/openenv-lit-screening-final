from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List, Dict
import json

app = FastAPI()

# ---------- ROOT ----------
@app.get("/", response_class=HTMLResponse)
def root():
    return "<h2>API Running</h2>"

# ---------- LOAD DATA ----------
with open("data.json") as f:
    ORIGINAL_DATA = json.load(f)

data = []
index = 0

# ---------- RESET (FIXED: POST) ----------
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

# ---------- REQUEST MODEL FOR STEP ----------
class StepRequest(BaseModel):
    action: str

# ---------- STEP (FIXED FORMAT) ----------
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

    # simple reward logic (can refine later)
    reward = 1.0 if action == data[index].get("label") else 0.0

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
    try:
        score = grade(request.predictions, ORIGINAL_DATA)
        return {"score": score}
    except Exception as e:
        return {"error": str(e)}

# ---------- GRADING FUNCTION ----------
def grade(predictions, data, task="medium"):
    score = 0

    for pred, paper in zip(predictions, data):
        if pred.get("label") == paper.get("label"):
            score += 1

    return score / len(data)