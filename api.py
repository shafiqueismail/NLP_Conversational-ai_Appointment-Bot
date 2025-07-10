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
    date = Column(String, index=True)      # Format: YYYY-MM-DD
    time = Column(String)                  # Format: "10:00 AM"
    name = Column(String)
    treatment = Column(String)
    duration = Column(Integer)

Base.metadata.create_all(bind=engine)

# --------------------
# FastAPI setup
# --------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5500"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------
# Pydantic models
# --------------------
class AppointmentOut(BaseModel):
    date: str
    time: str
    name: str
    treatment: str
    duration: int

class NewAppointment(BaseModel):
    name: str
    date: str
    time: str
    treatment: str
    duration: int

# --------------------
# GET: Retrieve appointments by date
# --------------------
@app.get("/api/appointments", response_model=List[AppointmentOut])
def get_appointments(dates: str = Query("", description="Comma-separated list of dates")):
    if not dates:
        return []
    date_list = [d.strip() for d in dates.split(",")]
    db = SessionLocal()
    appointments = db.query(Appointment).filter(Appointment.date.in_(date_list)).all()
    db.close()
    return [{
        "date": a.date,
        "time": a.time,
        "name": a.name,
        "treatment": a.treatment,
        "duration": a.duration
    } for a in appointments]

# --------------------
# POST: Add new appointment (with double-booking check)
# --------------------
@app.post("/api/add_appointment")
def add_appointment(appointment: NewAppointment):
    db = SessionLocal()

    # Step 1: Check if date + time is already booked
    conflict = db.query(Appointment).filter(
        Appointment.date == appointment.date,
        Appointment.time == appointment.time
    ).first()

    if conflict:
        db.close()
        return {"status": "error", "message": "This time slot is already booked."}

    # Step 2: No conflict, proceed to insert
    new_entry = Appointment(
        name=appointment.name,
        date=appointment.date,
        time=appointment.time,
        treatment=appointment.treatment,
        duration=appointment.duration
    )
    db.add(new_entry)
    db.commit()
    db.close()

    return {"status": "success"}
