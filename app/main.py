from fastapi import FastAPI

import app.routers.common_sense as common_sense
import app.routers.entity as entity
import app.routers.nlp as nlp
import json

from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

"""

entities:

Test sentences
 - A cat running across the field
 - A frog flying across the river
 - A panda throwing a ball
 - An apple falling from a tree
 - A panda running on the road
 - A giraffe carrying a bucket over her head
 - A panda carrying a sword
 - A bird flying over a river
 - A rabbit is on the roof of a running bus on the road

 prepositions: {"across", "from", "on", "over", "roof(not a preposition, but related here)"}
 actions: {"running", "flying", "throwing", "falling", "running", "carrying"}
 backgrounds: {"default", "field", "river", "road", "", ""}
 nouns: {"cat", "frog", "panda", "ball", "tree", "giraffe", "bucket", "torch", "rabbit", "bus", "bird"}

"""

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

from gensim.models import KeyedVectors
# load word2vec
model = KeyedVectors.load_word2vec_format('./app/resources/word2vec-model.bin', binary=True)

# load distance and direction GCNs
# save_base_path = './app/resources/'
# distance_model = DistanceModel()
# distance_model.load_state_dict(torch.load(save_base_path + 'distance_model.pt'))

# direction_model = DirectionModel()
# direction_model.load_state_dict(torch.load(save_base_path + 'direction_model.pt'))


# with open("./app/resources/dataset.json", "r") as dataset:
with open("./app/resources/mannan.json", "r") as dataset:
    global json_store
    json_store = json.load(dataset)
    print("dataset size: " + str(len(json_store['array'])))
    dataset.close()






@app.get("/", tags=["root"])
async def root() -> dict:
    return {"message": "I AM FYP"}
