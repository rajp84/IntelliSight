from fastapi import APIRouter

router = APIRouter()

@router.get("/health")
def health():
    return {"status": "example 2 ok"}
