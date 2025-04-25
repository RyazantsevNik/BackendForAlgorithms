import pytest
from fastapi import status


# Тест требует валидный токен
def test_chat_endpoint(client, auth_token):
    test_messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"}
    ]

    response = client.post(
        "/chat",
        json={"messages": test_messages},
        headers={"Authorization": f"Bearer {auth_token}"}  # Используем реальный токен
    )

    assert response.status_code == status.HTTP_200_OK
    assert "response" in response.json()


# Тест без токена должен возвращать 401
def test_chat_without_token(client):
    response = client.post("/chat", json={"messages": []})
    assert response.status_code == status.HTTP_401_UNAUTHORIZED


# Тест с неверным форматом (после авторизации)
def test_invalid_message_format(client, auth_token):
    response = client.post(
        "/chat",
        json={"messages": [{"wrong_field": "value"}]},
        headers={"Authorization": f"Bearer {auth_token}"}
    )
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


# Тест успешного логина
def test_login_success(client, test_user):
    response = client.post(
        "/login",
        json={"username": test_user["username"], "password": test_user["password"]}
    )
    assert response.status_code == status.HTTP_200_OK
    assert "access_token" in response.json()


# Тест неверных учетных данных
def test_login_invalid_credentials(client):
    response = client.post(
        "/login",
        json={"username": "invalid", "password": "invalid"}
    )
    assert response.status_code == status.HTTP_401_UNAUTHORIZED