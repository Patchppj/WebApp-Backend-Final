from typing import Union

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routers import diabetes, hypertention, submit_data #, save_to_excel  

app = FastAPI(
    title="Cardiometabolic Risk Prediction API",
    description="API for predicting cardiometabolic risk using XGBoost model",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#รวมทุก router
app.include_router(diabetes.router)
app.include_router(hypertention.router)
app.include_router(submit_data.router) 
#app.include_router(save_to_excel.router)  

@app.get("/")
def read_root():
    return {
        "message": "Welcome to Cardiometabolic Risk Prediction API",
        "docs_url": "/docs"
    }

@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}
