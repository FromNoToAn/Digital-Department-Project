from fastapi import FastAPI, Query
from pydantic import BaseModel
from stopDetector import StopDetector
import json
app = FastAPI()

@app.get("/test")
async def crash():

    response = {
        "message": "server work",
    }
    return response

@app.get("/stop")
async def crash(url: str):
    response = []
    if url is not None:
        stopDetector = StopDetector('./yolov8s.onnx')
        try:
            result = stopDetector.run(url)
        except ValueError:
            response ={
                "message": ValueError
            }
        response ={
            "message": result
        }
    else:
        response = {
            "message": 'video not font'
        }
    return response


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)