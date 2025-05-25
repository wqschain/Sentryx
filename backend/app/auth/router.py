from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.security import get_current_user
from . import service, schemas

router = APIRouter()

@router.post("/register", response_model=schemas.User)
async def register(
    user_in: schemas.UserCreate,
    db: AsyncSession = Depends(get_db)
):
    """Register a new user"""
    # Check if user exists
    db_user = await service.get_user_by_email(user_in.email, db)
    if db_user:
        raise HTTPException(
            status_code=400,
            detail="Email already registered"
        )
    
    return await service.create_user(user_in, db)

@router.post("/login", response_model=schemas.Token)
async def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: AsyncSession = Depends(get_db)
):
    """Login user and return access token"""
    user = await service.authenticate_user(
        form_data.username,  # OAuth2 form uses username field for email
        form_data.password,
        db
    )
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token = service.create_access_token(subject=user.email)
    return {"access_token": access_token, "token_type": "bearer"}

@router.get("/me", response_model=schemas.User)
async def read_users_me(
    current_user: schemas.User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get current user information"""
    return current_user 