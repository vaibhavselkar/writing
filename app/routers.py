from fastapi import APIRouter, Request
from app.services import process_text
from pydantic import BaseModel

router = APIRouter()

class TextRequest(BaseModel):
    text: str

@router.post("/process_text/")
async def process_text_endpoint(request: TextRequest):
    uncategorized_words = process_text(request.text)
    return {"uncategorized_words": uncategorized_words}