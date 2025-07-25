from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.orm import sessionmaker, declarative_base
from datetime import datetime, timedelta

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
    allow_origins=["http://127.0.0.1:5500"],  # Adjust if your frontend runs elsewhere
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
# POST: Add new appointment with overlap check
# --------------------
@app.post("/api/add_appointment")
def add_appointment(appointment: NewAppointment):
    db = SessionLocal()

    # Convert new appointment time to datetime
    try:
        new_start = datetime.strptime(appointment.time, "%I:%M %p")
    except ValueError:
        db.close()
        return {"status": "error", "message": "Invalid time format. Use 'HH:MM AM/PM'."}

    new_end = new_start + timedelta(minutes=appointment.duration)

    # Fetch same-day appointments
    same_day_appointments = db.query(Appointment).filter(
        Appointment.date == appointment.date
    ).all()

    # Check for overlap
    for existing in same_day_appointments:
        existing_start = datetime.strptime(existing.time, "%I:%M %p")
        existing_end = existing_start + timedelta(minutes=existing.duration)

        # Overlap rule
        if new_start < existing_end and new_end > existing_start:
            db.close()
            return {
                "status": "error",
                "message": f"Conflict with {existing.name}'s appointment at {existing.time}."
            }

    # No conflict → Save appointment
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
