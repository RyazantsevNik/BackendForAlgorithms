from sqlalchemy import create_engine, MetaData
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from databases import Database
from typing import AsyncGenerator
import os
import logging

logger = logging.getLogger(__name__)

# Получаем URL базы данных из переменных окружения или используем значение по умолчанию
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite+aiosqlite:///./database.db")

# Создаем базовый класс для моделей
Base = declarative_base()

# Инициализация базы данных
database = Database(DATABASE_URL)
engine = create_async_engine(DATABASE_URL, echo=True)
session_maker = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

async def init_db():
    """Инициализация базы данных"""
    try:
        # Подключаемся к базе данных
        await database.connect()
        
        # Создаем синхронный движок для создания таблиц
        sync_engine = create_engine(DATABASE_URL.replace("+aiosqlite", ""))
        
        # Создаем все таблицы
        Base.metadata.create_all(bind=sync_engine)
        
        logger.info("База данных успешно инициализирована")
    except Exception as e:
        logger.error(f"Ошибка при инициализации базы данных: {str(e)}")
        raise

async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    """Получение асинхронной сессии базы данных"""
    async with session_maker() as session:
        try:
            yield session
        except Exception as e:
            await session.rollback()
            logger.error(f"Ошибка в сессии базы данных: {str(e)}")
            raise
        finally:
            await session.close()
