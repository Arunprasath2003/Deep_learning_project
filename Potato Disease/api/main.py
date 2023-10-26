from fastapi import FastAPI,File,UploadFile
import uvicorn #Uvicorn is an ASGI(Asynchronous Server Gateway Interface) web server implementation for Python.
import numpy as np
from io import BytesIO
from PIL import Image #PIL is a module used to read images in Python
import tensorflow as tf

app=FastAPI()
MODEL = tf.keras.models.load_model("D:\7TH_SEM\CSE4088 DL\PROJECT\Potato Disease\models\1")
CLASS_NAMES = ["Early Blight","Late Blight","Healthy"]

@app.get("/ping")
async def ping():
    return "Hello, I am alive"

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

@app.post("/predict")
async def ping(
    file: UploadFile=File(...)
):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image,0)
    predictions = MODEL.predict(image)

    predicted_class=CLASS_NAMES[np.argmax(predictions[0])]
    confidence=np.max(predictions[0])
    return{
        'class':predicted_class
    }

if __name__=="__main__":
    uvicorn.run(app,host='localhost',port=8000)