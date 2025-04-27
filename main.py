import logging
import os
import shutil
from datetime import datetime, timedelta
from typing import Annotated

import httpx
from dotenv import load_dotenv
from fastapi import (
    FastAPI,
    Request,
    Depends,
    HTTPException,
    status,
    File,
    UploadFile,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sqlalchemy import create_engine, select, or_, false, update
from sqlalchemy.ext.asyncio import AsyncSession
import auth
import models
import schemas
from database import database, get_async_session
from models import ChatMessage
from typing import List

# Загрузка переменных окружения
load_dotenv()

# Настройка логгера
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Конфигурация API нейросети
NEURAL_API_URL = os.getenv("NEURAL_API_URL")
API_KEY = os.getenv("AI_API_KEY")
AI_MODEL = os.getenv("AI_MODEL", "deepseek-ai/DeepSeek-R1")


# Валидация конфигурации
def validate_config():
    errors = []
    if not NEURAL_API_URL:
        errors.append("NEURAL_API_URL not set in .env")
    if not API_KEY:
        errors.append("AI_API_KEY not set in .env")

    if errors:
        logger.error("Configuration errors detected:")
        for error in errors:
            logger.error(f" - {error}")
        raise RuntimeError("Invalid server configuration")


validate_config()

# Инициализация FastAPI
app = FastAPI(title="AI Learning Assistant API", redirect_slashes=False)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info(f"Incoming request: {request.method} {request.url}")
    response = await call_next(request)
    logger.info(f"Response status: {response.status_code}")
    return response


# CORS настройки
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Синхронный движок для миграций
sync_engine = create_engine(os.getenv("DATABASE_URL", "sqlite:///./database.db"))

# Создание таблиц
models.Base.metadata.create_all(bind=sync_engine)

# Директория для загрузок
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")


class ChatRequest(BaseModel):
    message: str
    chat_history: list[dict] = []


class ChatResponse(BaseModel):
    response: str
    chat_history: list[dict] = []


@app.on_event("startup")
async def startup():
    await database.connect()
    logger.info("Database connected")
    logger.info(f"AI API Configuration:\n"
                f" - URL: {NEURAL_API_URL}\n"
                f" - Model: {AI_MODEL}\n"
                f" - API Key: {API_KEY[:5]}...{API_KEY[-5:]}")


@app.on_event("shutdown")
async def shutdown():
    await database.disconnect()
    logger.info("Database disconnected")


@app.put("/profile", response_model=schemas.UserResponse)
async def update_profile(
        user_update: schemas.UserUpdate,
        current_user: models.User = Depends(auth.get_current_user),
        db: AsyncSession = Depends(get_async_session)
):
    try:
        result = await db.execute(
            select(models.User).where(models.User.id == current_user.id)
        )
        user = result.scalars().first()

        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )

        if user_update.username is not None:
            existing_user = await db.execute(
                select(models.User)
                .where(models.User.username == user_update.username)
                .where(models.User.id != current_user.id)
            )
            if existing_user.scalars().first():
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Username already taken"
                )
            user.username = user_update.username

        if user_update.email is not None:
            existing_email = await db.execute(
                select(models.User)
                .where(models.User.email == user_update.email)
                .where(models.User.id != current_user.id)
            )
            if existing_email.scalars().first():
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Email already in use"
                )
            user.email = user_update.email

        logger.info(f"current_password: {user_update.current_password}, new_password: {user_update.new_password}")
        if user_update.current_password and user_update.new_password:
            if not auth.verify_password(user_update.current_password, user.hashed_password):
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Current password is incorrect"
                )

            if len(user_update.new_password) < 6:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="New password must be at least 6 characters"
                )

            user.hashed_password = auth.hash_password(user_update.new_password)
            logger.info(f"Password updated for user {user.id}, new hash: {user.hashed_password}")

        # Сохраняем изменения
        await db.commit()

        return user

    except HTTPException:
        await db.rollback()
        raise

    except Exception as e:
        await db.rollback()
        logger.error(f"Error updating profile: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Could not update profile: {str(e)}"
        )


@app.delete("/profile/delete-photo", response_model=schemas.ProfileResponse)  #НОВОЕ удаление фото
async def delete_profile_photo(
        current_user: models.User = Depends(auth.get_current_user),
        db: AsyncSession = Depends(get_async_session)
):
    try:
        if not current_user.profile_picture:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No profile photo to delete"
            )

        # Удаляем файл
        filename = current_user.profile_picture.split("/uploads/")[-1]
        file_path = os.path.join(UPLOAD_DIR, filename)
        if os.path.exists(file_path):
            os.remove(file_path)

        # Обновляем базу данных
        await db.execute(
            models.User.__table__.update()
            .where(models.User.id == current_user.id)
            .values(profile_picture=None)
        )
        await db.commit()

        return schemas.ProfileResponse(
            message="Profile photo deleted successfully",
            profile_picture=None
        )

    except Exception as e:
        logger.error(f"Error deleting profile photo: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not delete profile photo"
        )


@app.post("/chat", response_model=ChatResponse)
async def chat_with_ai(
        chat_request: ChatRequest,
        current_user: models.User = Depends(auth.get_current_user),
):
    try:
        if not API_KEY:
            logger.error("AI_API_KEY not configured")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Server configuration error"
            )

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            *chat_request.chat_history,
            {"role": "user", "content": chat_request.message},
        ]

        logger.debug(f"Sending request to AI API:\n"
                     f"URL: {NEURAL_API_URL}\n"
                     f"Headers: { {'Authorization': f'Bearer {API_KEY[:5]}...'} }\n"
                     f"Body: { {'model': AI_MODEL, 'messages': messages} }")

        timeout = httpx.Timeout(60.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                NEURAL_API_URL,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {API_KEY}",
                },
                json={
                    "model": AI_MODEL,
                    "messages": messages,
                },
            )

            logger.debug(f"AI API response:\n"
                         f"Status: {response.status_code}\n"
                         f"Headers: {response.headers}\n"
                         f"Body: {response.text}")

            response.raise_for_status()
            response_data = response.json()

            if not response_data.get("choices"):
                logger.error("Invalid response format from AI API")
                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY,
                    detail="Invalid response from AI service"
                )

            ai_message = response_data['choices'][0]['message']['content']
            if "</think>" in ai_message:
                ai_message = ai_message.split("</think>\n\n")[1]

            # Сохраняем сообщения в БД
            async for db in get_async_session():
                db.add_all([
                    ChatMessage(user_id=current_user.id, role="user", content=chat_request.message),
                    ChatMessage(user_id=current_user.id, role="assistant", content=ai_message)
                ])
                await db.commit()

            updated_history = [
                *chat_request.chat_history,
                {"role": "user", "content": chat_request.message},
                {"role": "assistant", "content": ai_message},
            ]

            return ChatResponse(
                response=ai_message,
                chat_history=updated_history,
            )

    except httpx.HTTPStatusError as e:
        logger.error(f"AI API Error: {e.response.text}")
        error_detail = "AI service error"

        if e.response.status_code == 401:
            error_detail = "Invalid AI API credentials"
        elif e.response.status_code == 404:
            error_detail = "AI API endpoint not found"
        elif 500 <= e.response.status_code < 600:
            error_detail = "AI service internal error"

        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=error_detail
        )

    except httpx.RequestError as e:
        logger.error(f"Network error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Could not connect to AI service"
        )

    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@app.get("/chat/history", response_model=List[schemas.ChatMessageResponse])
async def get_chat_history(
        current_user: models.User = Depends(auth.get_current_user),
        db: AsyncSession = Depends(get_async_session)
):
    result = await db.execute(
        select(ChatMessage)
        .where(ChatMessage.user_id == current_user.id)
        .order_by(ChatMessage.timestamp)
    )
    messages = result.scalars().all()
    return messages


# Остальные эндпоинты остаются без изменений...

@app.post("/register", response_model=schemas.AuthResponse)
async def register(
        user: schemas.UserCreate,
        db: AsyncSession = Depends(get_async_session)
):
    try:
        # Начало транзакции
        await db.begin()

        # Проверка пароля
        if len(user.password) < 6:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Password must be at least 6 characters",
            )

        # Проверка существующего пользователя
        existing_user = await db.execute(
            select(models.User).where(
                or_(
                    models.User.username == user.username,
                    models.User.email == user.email
                )
            )
        )
        if existing_user.scalars().first():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Username or email already exists",
            )

        # Хеширование пароля
        hashed_password = auth.hash_password(user.password)
        if not hashed_password:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Password hashing failed"
            )

        # Создание пользователя
        new_user = models.User(
            username=user.username,
            email=user.email,
            hashed_password=hashed_password,
        )

        db.add(new_user)
        await db.commit()

        # Обновляем объект, чтобы получить ID
        await db.refresh(new_user)

        # Генерация токена
        access_token = auth.create_access_token(
            data={"sub": new_user.username},
            expires_delta=timedelta(minutes=auth.ACCESS_TOKEN_EXPIRE_MINUTES),
        )

        return schemas.AuthResponse(
            user=schemas.UserBase(
                id=new_user.id,
                username=new_user.username,
                email=new_user.email,
            ),
            access_token=access_token,
            token_type="bearer",
        )

    except HTTPException:
        await db.rollback()
        raise

    except Exception as e:
        await db.rollback()
        logger.error(f"Registration error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Registration failed"
        )


@app.post("/login", response_model=schemas.AuthResponse)
async def login(
        user: schemas.LoginRequest, db: AsyncSession = Depends(get_async_session)
):
    db_user = await auth.authenticate_user(db, user.username, user.password)
    if not db_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
        )

    access_token = auth.create_access_token(
        data={"sub": db_user.username},
        expires_delta=timedelta(minutes=auth.ACCESS_TOKEN_EXPIRE_MINUTES),
    )

    return schemas.AuthResponse(
        user=schemas.UserBase(
            id=db_user.id,
            username=db_user.username,
            email=db_user.email,
        ),
        access_token=access_token,
        token_type="bearer",
    )


@app.get("/profile", response_model=schemas.UserResponse)
async def get_profile(current_user: models.User = Depends(auth.get_current_user)):
    return current_user


@app.post("/profile/upload-photo", response_model=schemas.ProfileResponse)
async def upload_profile_photo(
    file: UploadFile = File(...),
    current_user: models.User = Depends(auth.get_current_user),
    db: AsyncSession = Depends(get_async_session)
):
    try:
        if not file.content_type.startswith('image/'):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="File must be an image"
            )

        # Генерация имени файла
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_extension = os.path.splitext(file.filename)[1]
        filename = f"profile_{current_user.id}_{timestamp}{file_extension}"
        file_path = os.path.join(UPLOAD_DIR, filename)


        # Сохраняем файл
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        print(f"file: {file.file} \n filepath:{file_path} \n buffer: {buffer} \n")
        

        file_url = f"/uploads/{filename}"

        # Теперь правильно обновляем пользователя через ORM
        stmt = (
            update(models.User)
            .where(models.User.id == current_user.id)
            .values(profile_picture=file_url)
        )
        print(f"Preparing to commit the update with profile_picture: {file_url} for user ID: {current_user.id}\n")

        await db.execute(stmt)

        
        print("after execute\n")
        

        await db.commit()

        print("after commit\n")

        return schemas.ProfileResponse(
            message="Profile photo updated successfully",
            profile_picture=file_url
        )

    except Exception as e:
        logger.error(f"Error uploading profile photo: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not upload profile photo"
        )



if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
