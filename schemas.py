from datetime import datetime

from pydantic import BaseModel, ConfigDict, EmailStr
from typing import Optional, List
from fastapi import UploadFile


class UserCreate(BaseModel):
    username: str
    email: str
    password: str


class UserBase(BaseModel):
    id: int
    username: str
    email: str


class UserUpdate(BaseModel):  #НОВОЕ
    username: Optional[str] = None
    email: Optional[EmailStr] = None
    current_password: Optional[str] = None
    new_password: Optional[str] = None


class UserResponse(BaseModel):
    id: int
    username: str
    email: str
    full_name: Optional[str] = None
    profile_picture: Optional[str] = None

    model_config = ConfigDict(from_attributes=True)


class AuthResponse(BaseModel):
    user: UserBase
    access_token: str
    token_type: str = "bearer"

    model_config = ConfigDict(from_attributes=True)


class LoginRequest(BaseModel):
    username: str
    password: str


class ProgressCreate(BaseModel):
    algorithm: str
    completed: bool


class ProgressResponse(ProgressCreate):
    id: int


class ProfileUpdate(BaseModel):
    profile_picture: Optional[str] = None


class ProfileResponse(BaseModel):
    message: str
    profile_picture: Optional[str] = None


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    response: str


class ChatMessageCreate(BaseModel):
    role: str  # "user" или "assistant"
    content: str


class ChatMessageResponse(ChatMessageCreate):
    id: int
    timestamp: datetime
