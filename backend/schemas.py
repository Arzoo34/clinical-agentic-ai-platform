from pydantic import BaseModel
from datetime import datetime
from typing import Optional, List

class SellerBase(BaseModel):
    name: str
    preferred_language: str
    city: str

class SellerCreate(SellerBase):
    pass

class Seller(SellerBase):
    id: int
    created_at: datetime
    
    class Config:
        from_attributes = True

class ListingBase(BaseModel):
    title: str
    description: str
    category: str
    price: float
    size_chart: bool
    photo_count: int
    fabric_mentioned: bool
    wash_care: bool
    cod_enabled: bool
    pin_code: str

class ListingCreate(BaseModel):
    seller_id: int
    raw_input: str  # transcribed voice or form input
    category: str
    photo_count: Optional[int] = 1
    cod_enabled: Optional[bool] = False
    pin_code: Optional[str] = None

class Listing(ListingBase):
    id: int
    seller_id: int
    created_at: datetime
    
    class Config:
        from_attributes = True

class RiskGap(BaseModel):
    label: str
    severity: str  # HIGH, MEDIUM, LOW
    contribution_pct: float
    explanation: str

class RiskScoreResponse(BaseModel):
    listing_id: int
    score: float
    gaps: List[RiskGap]
    predicted_score_after_fixes: float
    timestamp: datetime

class FraudCheckResponse(BaseModel):
    pin_code: str
    rto_rate: float
    fraud_flag: bool
    risk_level: str  # HIGH, MEDIUM, LOW, NONE
    message: str

class BuyerQuestionBase(BaseModel):
    listing_id: int
    question_text: str
    language: str

class BuyerQuestion(BuyerQuestionBase):
    id: int
    cluster_id: Optional[int]
    created_at: datetime
    
    class Config:
        from_attributes = True

class QAReplyBase(BaseModel):
    question_id: int
    draft_reply: str
    status: str

class QAReply(QAReplyBase):
    id: int
    created_at: datetime
    
    class Config:
        from_attributes = True

class QuestionCluster(BaseModel):
    cluster_id: int
    questions: List[BuyerQuestion]
    draft_reply: str
    listing_fix_suggestion: str

class HealthBriefBase(BaseModel):
    summary_text: str
    recommendations: dict
    language: str

class HealthBrief(HealthBriefBase):
    id: int
    seller_id: int
    week_of: datetime
    created_at: datetime
    
    class Config:
        from_attributes = True
