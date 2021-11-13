from fastapi import APIRouter

from quickdraw import QuickDrawData
from fastapi.encoders import jsonable_encoder

router = APIRouter()


@router.get("/entity/{name}", tags=["entity"])
async def get_entity_by_id(name: str):

    """GET: entity of required name returned from quickdraw

    Returns:
        QuickDrawData: a data object of a quick draw image
    """

    # make empty object to return
    qd = QuickDrawData()
    # get drawing from parameter
    
    entity = qd.get_drawing(name)
#  nouns: {"cat", "frog", "boy", "ball", "tree", "woman", "bucket", "torch", "rabbit", "bus"}

    # hard coded images
    if name == "cat":
        qd.get_drawing(name, index=1)
    elif name == "forg":
        qd.get_drawing(name, index=7)
    elif name == "panda":
        qd.get_drawing(name, index=124)
    elif name == "ball":
        qd.get_drawing("basketball", index=190)
    elif name == "tree":
        qd.get_drawing(name, index=21)
    elif name == "giraffe":
        qd.get_drawing(name, index=10)
    elif name == "bucket":
        qd.get_drawing(name, index=0)
    elif name == "sword":
        qd.get_drawing(name, index=37)
    elif name == "rabbit":
        qd.get_drawing(name, index=133)
    elif name == "bus":
        qd.get_drawing(name, index=86)
    elif name == "bird":
        qd.get_drawing(name, index=25)
    else:
        qd.get_drawing(name)
    # convert entity to json to return
    json_compatible_item_data = jsonable_encoder(entity)

    return json_compatible_item_data
