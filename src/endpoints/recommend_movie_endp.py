from fastapi import APIRouter, Form, File, UploadFile
from src.operations.recommend_movie import recommend_movie


router = APIRouter(
    prefix="/api/recommend_movie",
    tags=["recommend_movie"],
    responses={404: {"description": "Not found"}},
)



@router.get("/")
async def get_recommend_movie(movie_name: str):
    return recommend_movie(movie_name)

   



    
