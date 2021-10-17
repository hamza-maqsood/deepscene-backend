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
    # convert entity to json to return
    json_compatible_item_data = jsonable_encoder(entity)

    return json_compatible_item_data
