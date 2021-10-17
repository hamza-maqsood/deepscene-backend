from fastapi import Depends, FastAPI

import routers.common_sense as common_sense
import routers.entity as entity
import routers.nlp as nlp


from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

app = FastAPI()

origins = ["*"]
app.add_middleware(GZipMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(common_sense.router)
app.include_router(entity.router)
app.include_router(nlp.router)


@app.get("/")
async def root():
    return {"message": "I AM FYP"}
