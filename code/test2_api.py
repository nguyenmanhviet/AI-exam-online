import uvicorn
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from prediction import *
import time

app = FastAPI()

origins = [
    "*",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class MyItem(BaseModel):
    file: str


@app.post('/')
async def predict(item:MyItem):
    # print(item)
    start = time.time()
    print(type(item.file))
    image = read_image(item.file)

    image = pil_cv(image)

    detection = detect(image)
    
#    save_image(image)
    end = time.time()
    print('\n --------------------- \n time:',end - start)
    print(detection)
    return detection
    #return True
@app.get('/sum')
def hamcong():
    # print(item)
    return "ham cong"

    # return "ok"

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)
