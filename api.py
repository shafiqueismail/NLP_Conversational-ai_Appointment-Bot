from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.orm import sessionmaker, declarative_base

# --------------------
# Database setup
# --------------------
DATABASE_URL = "sqlite:///./appointments.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

class Appointment(Base):
    __tablename__ = "appointments"
    id = Column(Integer, primary_key=True, index=True)
    date = Column(String, index=True)  # Format: YYYY-MM-DD
    time = Column(String)              # Format: e.g., "10:00 AM"
    name = Column(String)

# Create the table(s)
Base.metadata.create_all(bind=engine)

# --------------------
# FastAPI setup
# --------------------
app = FastAPI()

# Allow your frontend (running on localhost:5500)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5500"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------
# API Models
# --------------------
class AppointmentOut(BaseModel):
    date: str
    time: str
    name: str

# --------------------
# GET /api/appointments
# --------------------
@app.get("/api/appointments", response_model=List[AppointmentOut])
def get_appointments(dates: str = Query(..., description="Comma-separated list of dates")):
    date_list = [d.strip() for d in dates.split(",")]
    db = SessionLocal()
    appointments = db.query(Appointment).filter(Appointment.date.in_(date_list)).all()
    return [{"date": a.date, "time": a.time, "name": a.name} for a in appointments]
