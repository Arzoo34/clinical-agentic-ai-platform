from sqlalchemy import Column, Integer, String, Boolean, Float, DateTime, ForeignKey, JSON, Text
from sqlalchemy.orm import relationship
from datetime import datetime
from database import Base

class Seller(Base):
    __tablename__ = "sellers"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)
    preferred_language = Column(String, default="hi")  # hi, gu, ta
    city = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    listings = relationship("Listing", back_populates="seller")
    health_briefs = relationship("HealthBrief", back_populates="seller")

class Listing(Base):
    __tablename__ = "listings"
    
    id = Column(Integer, primary_key=True, index=True)
    seller_id = Column(Integer, ForeignKey("sellers.id"))
    title = Column(String, index=True)
    description = Column(Text)
    category = Column(String, index=True)
    price = Column(Float)
    size_chart = Column(Boolean, default=False)
    photo_count = Column(Integer, default=1)
    fabric_mentioned = Column(Boolean, default=False)
    wash_care = Column(Boolean, default=False)
    cod_enabled = Column(Boolean, default=False)
    pin_code = Column(String, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    seller = relationship("Seller", back_populates="listings")
    risk_scores = relationship("RiskScore", back_populates="listing")
    questions = relationship("BuyerQuestion", back_populates="listing")

class RiskScore(Base):
    __tablename__ = "risk_scores"
    
    id = Column(Integer, primary_key=True, index=True)
    listing_id = Column(Integer, ForeignKey("listings.id"))
    score = Column(Float)  # 0-100
    gap_details = Column(JSON)  # [{label, severity, contribution_pct, explanation}, ...]
    created_at = Column(DateTime, default=datetime.utcnow)
    
    listing = relationship("Listing", back_populates="risk_scores")

class BuyerQuestion(Base):
    __tablename__ = "buyer_questions"
    
    id = Column(Integer, primary_key=True, index=True)
    listing_id = Column(Integer, ForeignKey("listings.id"))
    question_text = Column(Text)
    language = Column(String, default="en")
    cluster_id = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    listing = relationship("Listing", back_populates="questions")
    replies = relationship("QAReply", back_populates="question")

class QAReply(Base):
    __tablename__ = "qa_replies"
    
    id = Column(Integer, primary_key=True, index=True)
    question_id = Column(Integer, ForeignKey("buyer_questions.id"))
    draft_reply = Column(Text)
    status = Column(String, default="pending")  # pending, approved
    created_at = Column(DateTime, default=datetime.utcnow)
    
    question = relationship("BuyerQuestion", back_populates="replies")

class HealthBrief(Base):
    __tablename__ = "health_briefs"
    
    id = Column(Integer, primary_key=True, index=True)
    seller_id = Column(Integer, ForeignKey("sellers.id"))
    week_of = Column(DateTime)
    summary_text = Column(Text)
    recommendations = Column(JSON)  # [{title, description}, ...]
    language = Column(String, default="hi")
    created_at = Column(DateTime, default=datetime.utcnow)
    
    seller = relationship("Seller", back_populates="health_briefs")

class PinCodeRisk(Base):
    __tablename__ = "pin_code_risk"
    
    id = Column(Integer, primary_key=True, index=True)
    pin_code = Column(String, unique=True, index=True)
    rto_rate = Column(Float)  # percentage
    fraud_flag = Column(Boolean, default=False)
    notes = Column(String, nullable=True)

class CategoryReturnBenchmark(Base):
    __tablename__ = "category_return_benchmarks"
    
    id = Column(Integer, primary_key=True, index=True)
    category = Column(String, index=True)
    gap_type = Column(String)  # e.g. "missing_size_chart", "single_photo", "no_fabric", "no_wash_care"
    avg_contribution_pct = Column(Float)  # how much this gap contributes to risk score

class SyntheticReturn(Base):
    __tablename__ = "synthetic_returns"
    
    id = Column(Integer, primary_key=True, index=True)
    listing_id = Column(Integer, ForeignKey("listings.id"))
    reason = Column(String)  # "wrong_size", "not_as_described", "damaged"
    is_cod = Column(Boolean)  # prepaid vs COD
    created_at = Column(DateTime, default=datetime.utcnow)
