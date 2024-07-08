from sqlalchemy import Boolean, Column, ForeignKey, Integer, String, Float, DateTime
from database import Base
class PredictionResult(Base):
    __tablename__ = "prediction_results"

    id = Column(Integer, primary_key=True, index=True)
    time = Column(DateTime, index=True)
    prediction = Column(String, index=True)
    
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    password = Column(String)
