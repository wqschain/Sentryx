from datetime import datetime, timedelta
from typing import Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from jose import jwt

from app.core.security import verify_password, get_password_hash
from app.core.config import settings
from .models import User
from . import schemas

async def authenticate_user(
    email: str,
    password: str,
    db: AsyncSession
) -> Optional[User]:
    """Authenticate a user by email and password"""
    result = await db.execute(select(User).where(User.email == email))
    user = result.scalar_one_or_none()
    
    if not user or not verify_password(password, user.hashed_password):
        return None
    return user

async def create_user(
    user_create: schemas.UserCreate,
    db: AsyncSession
) -> User:
    """Create a new user"""
    db_user = User(
        email=user_create.email,
        hashed_password=get_password_hash(user_create.password)
    )
    db.add(db_user)
    await db.commit()
    await db.refresh(db_user)
    return db_user

async def get_user_by_email(
    email: str,
    db: AsyncSession
) -> Optional[User]:
    """Get a user by email"""
    result = await db.execute(select(User).where(User.email == email))
    return result.scalar_one_or_none()

def create_access_token(
    subject: str,
    expires_delta: Optional[timedelta] = None
) -> str:
    """Create JWT access token"""
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(
            minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES
        )
    
    to_encode = {"exp": expire, "sub": str(subject)}
    encoded_jwt = jwt.encode(
        to_encode,
        settings.SECRET_KEY,
        algorithm="HS256"
    )
    return encoded_jwt 