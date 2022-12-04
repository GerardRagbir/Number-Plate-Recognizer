import time
from cv2 import cv2
from PIL import Image
import uvicorn as uvicorn
from easyocr import easyocr
from fastapi import FastAPI, File, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
import logging
from typing import Tuple
import pandas as pd
from glob import glob

# Config
CUDA = False
DEBUG = False

# API
app = FastAPI(debug=DEBUG, title="Number Plate Detection Server", version="1.0.0")

# CORS Headers
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True
)


reader = easyocr.Reader(['en'], gpu=CUDA)


# Routes
@app.get("/")
async def root():
    return JSONResponse(status_code=status.HTTP_200_OK, content=jsonable_encoder({'detail': 'System Running'}))


@app.post("/upload")
async def upload(image: UploadFile = File(...)):
    global blob_name
    try:
        blob_name = f'../detector/blobs/{time.time()}.png'
        with open(blob_name, 'wb') as f:
            while contents := \
                    image.file.read():
                f.write(contents)

    except StarletteHTTPException as e:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=jsonable_encoder({'detail': e.detail}))
    finally:
        _, text = predictor(blob_name)
        print(text)
        image.file.close()
        return JSONResponse(
            status_code=status.HTTP_202_ACCEPTED,
            content=jsonable_encoder({'detail': 'Succeeded', 'plate': str(text).strip('\nName: Text, dtype: object')}))


dataset = glob('../detector/images/*')


def predictor(img):
    try:
        inference = reader.readtext(img)
        df = pd.DataFrame(inference, columns=['Bounds', 'Text', 'Confidence'])
        return df, df.loc[:, 'Text']
    except TypeError:
        logging.warning("Predictor cannot be given NONE")


# Main
if __name__ == "__main__":
    uvicorn.run(
        app=app,
        host="0.0.0.0",
        port=8000,
        reload=True,
        use_colors=True,
        access_log=True,
    )
