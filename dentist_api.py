from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session

# --- Database Setup ---
DATABASE_URL = "sqlite:///./appointments.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

class Appointment(Base):
    __tablename__ = 'appointments'
    id = Column(Integer, primary_key=True, index=True)
    date = Column(String, index=True)
    time = Column(String)

Base.metadata.create_all(bind=engine)

# --- FastAPI App ---
app = FastAPI()

# --- CORS Setup (allow frontend port 5500) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5500"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Static & Template Setup ---
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# --- In-memory session storage ---
sessions = {}

# --- Pydantic Models ---
class LoginRequest(BaseModel):
    username: str
    password: str

class AppointmentResponse(BaseModel):
    date: str
    time: str

# --- Dependency Injection ---
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# --- Routes ---
@app.get("/", response_class=HTMLResponse)
def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@app.post("/login")
def login(data: LoginRequest):
    if data.username == "ismail" and data.password == "ismail":
        token = "secure-token"
        sessions[token] = True
        return {"success": True, "token": token}
    raise HTTPException(status_code=401, detail="Invalid credentials")

@app.get("/dashboard", response_class=HTMLResponse)
def dashboard(request: Request):
    return templates.TemplateResponse("dashboard.html", {"request": request})

@app.get("/api/appointments", response_model=List[AppointmentResponse])
def get_appointments(token: str, db: Session = Depends(get_db)):
    if token not in sessions:
        raise HTTPException(status_code=403, detail="Unauthorized")
    appointments = db.query(Appointment).all()
    return [AppointmentResponse(date=a.date, time=a.time) for a in appointments]
