from pathlib import Path
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi import APIRouter
import os
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import Depends, HTTPException, status, Request
# Using this as a base for token handling
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from authlib.integrations.starlette_client import OAuth
from starlette.responses import RedirectResponse
from pydantic import BaseModel

# Assuming models_fastapi.py is in backend/
from models_fastapi import User, TokenData

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24

# This scheme can be used in Depends to get the token from the Authorization header
# For cookie-based auth, we'll manually extract from cookies.
# Dummy tokenUrl, we are not using password flow
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Configure Google OAuth
oauth = OAuth()
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
GOOGLE_REDIRECT_URI = os.getenv(
    "GOOGLE_REDIRECT_URI", "http://localhost:8000/auth")
print("Google OAuth redirect URI:", GOOGLE_REDIRECT_URI)
print("Google OAuth client ID:", GOOGLE_CLIENT_ID)
print("Google OAuth client secret:", GOOGLE_CLIENT_SECRET)
if GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET:
    oauth.register(
        name="google",
        server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
        client_id=GOOGLE_CLIENT_ID,
        client_secret=GOOGLE_CLIENT_SECRET,
        client_kwargs={
            "scope": "email openid profile",
            "redirect_url": GOOGLE_REDIRECT_URI,
            "expires_in": 100,
        },
    )
else:
    print("Warning: GOOGLE_CLIENT_ID or GOOGLE_CLIENT_SECRET not set. Google OAuth will not work.")
ROOT_DIR_PROJECT = Path(__file__).parent
print(ROOT_DIR_PROJECT)
templates = Jinja2Templates(directory=ROOT_DIR_PROJECT)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + \
            timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, JWT_SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


async def get_current_user(request: Request) -> User:
    print(request.session.get("user"))
    token = request.headers.get("Authorization")
    print(token)
    if "Bearer " in token:
        token = token.split("Bearer ")[1]
    # token = request.cookies.get("session_token")
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated (no session token)",
            headers={"WWW-Authenticate": "Bearer"},
        )
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        full_name: str = payload.get("name")
        picture: str = payload.get("picture")
        if email is None:
            raise credentials_exception
        token_data = TokenData(email=email)
    except JWTError:
        raise credentials_exception

    # Here you could fetch the user from a database if you had one
    # For now, we'll create a User model from the token payload
    user = User(email=token_data.email, full_name=full_name,
                picture=picture, is_active=True)
    if user is None:  # Should not happen if email is present
        raise credentials_exception
    return user


async def get_current_active_user(current_user: User = Depends(get_current_user)) -> User:
    if not current_user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user


# FastAPI router for auth endpoints
router = APIRouter()


@router.get('/login')
async def login_via_google(request: Request, redirect_path: Optional[str] = "/"):
    if not GOOGLE_CLIENT_ID or not GOOGLE_CLIENT_SECRET:
        raise HTTPException(
            status_code=500, detail="Google OAuth not configured on server.")
    redirect_uri_callback = str(request.url_for('auth_google_callback'))
    print("Redirect URI for Google OAuth:", redirect_uri_callback)
    request.session['final_redirect_path'] = redirect_path

    return await oauth.google.authorize_redirect(request, redirect_uri_callback)


@router.get('/auth', name='auth_google_callback')
async def auth_google_callback(request: Request):
    if not GOOGLE_CLIENT_ID or not GOOGLE_CLIENT_SECRET:
        raise HTTPException(
            status_code=500, detail="Google OAuth not configured on server.")
    try:
        token_data = await oauth.google.authorize_access_token(request)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail=f"Could not authorize Google token: {e}")

    user_info_google = token_data.get('userinfo')
    if not user_info_google:
        # Fallback for some providers
        user_info_google = await oauth.google.parse_id_token(request, token_data)

    if not user_info_google or not user_info_google.get("email"):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                            detail="Could not fetch user info from Google")
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={
            "sub": user_info_google.get("email"),
            "name": user_info_google.get("name"),
            "picture": user_info_google.get("picture")
        },
        expires_delta=access_token_expires
    )
    streamlit_base_url = os.getenv(
        "STREAMLIT_SERVER_ADDRESS", "http://localhost:8080")
    request.session["user"] = access_token
    request.session["session_token"] = access_token
    response = RedirectResponse(url=streamlit_base_url, status_code=302)
    response.set_cookie(
        key="session_token",
        value=access_token,
        httponly=False,
        max_age=int(access_token_expires.total_seconds()),
        samesite="Lax",  # Or "Strict" if FastAPI and Streamlit are same-site
        secure=False,  # Set to True if served over HTTPS
    )
    return templates.TemplateResponse(
        "welcome.html",
        {"request": request, "token": access_token,
         "user": user_info_google, "redirect_url": streamlit_base_url},

    )
    # return response


@router.post("/logout")
# Redirect handled by Streamlit
async def logout(response: RedirectResponse = RedirectResponse(url="/")):
    # In Streamlit, the page will reload or redirect after clearing the token.
    # FastAPI just clears the cookie.
    response = RedirectResponse(url=os.getenv(
        "STREAMLIT_SERVER_ADDRESS", "http://localhost:8501"))
    response.delete_cookie("session_token", httponly=True,
                           samesite="Lax")  # Add secure=True for HTTPS
    return {"message": "Successfully logged out"}


@router.get("/users/me", response_model=User)
async def read_users_me(current_user: User = Depends(get_current_active_user)):
    return current_user
