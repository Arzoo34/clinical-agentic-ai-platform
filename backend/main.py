from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv

load_dotenv()

from database import init_db, get_db
import models
import schemas
from agents import listing_agent, qa_agent, health_agent

app = FastAPI(title="Shuruaat AI - Meesho Seller Co-pilot")

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize database on startup
@app.on_event("startup")
def startup():
    init_db()
    print("Database initialized")

# ============ SELLER ENDPOINTS ============

@app.get("/seller", response_model=schemas.Seller)
def get_current_seller(db: Session = Depends(get_db)):
    """Get the demo seller (Priya)"""
    seller = db.query(models.Seller).filter(models.Seller.name == "Priya").first()
    if not seller:
        raise HTTPException(status_code=404, detail="Seller not found")
    return seller

@app.post("/seller/language")
def update_seller_language(language: str, db: Session = Depends(get_db)):
    """Update seller's preferred language"""
    seller = db.query(models.Seller).filter(models.Seller.name == "Priya").first()
    if not seller:
        raise HTTPException(status_code=404, detail="Seller not found")
    seller.preferred_language = language
    db.commit()
    return {"message": f"Language updated to {language}"}

# ============ LISTING ENDPOINTS ============

@app.post("/listings/generate", response_model=schemas.Listing)
def generate_listing(
    seller_id: int,
    raw_input: str,
    category: str,
    photo_count: int = 1,
    cod_enabled: bool = False,
    pin_code: str = "000000",
    db: Session = Depends(get_db)
):
    """Generate a listing from raw seller input using AI"""
    
    seller = db.query(models.Seller).filter(models.Seller.id == seller_id).first()
    if not seller:
        raise HTTPException(status_code=404, detail="Seller not found")
    
    # Get language preference
    language = seller.preferred_language
    
    # Call listing agent to generate content
    try:
        generated = listing_agent.generate_listing(raw_input, category, language)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"AI generation failed: {str(e)}")
    
    # Create listing record
    listing = models.Listing(
        seller_id=seller_id,
        title=generated.get("title", "Product"),
        description=raw_input,  # Use raw input as base; can be enriched
        category=category,
        price=float(generated.get("suggested_price_range", "500").split("-")[0].replace("₹", "").strip()) if "-" in generated.get("suggested_price_range", "500") else 500,
        size_chart=False,
        photo_count=photo_count,
        fabric_mentioned=False,
        wash_care=False,
        cod_enabled=cod_enabled,
        pin_code=pin_code,
        created_at=datetime.utcnow()
    )
    
    db.add(listing)
    db.commit()
    db.refresh(listing)
    
    return listing

@app.get("/listings", response_model=list[schemas.Listing])
def list_listings(seller_id: int, db: Session = Depends(get_db)):
    """List all listings for a seller"""
    listings = db.query(models.Listing).filter(models.Listing.seller_id == seller_id).all()
    return listings

@app.get("/listings/{listing_id}", response_model=schemas.Listing)
def get_listing(listing_id: int, db: Session = Depends(get_db)):
    """Get a specific listing"""
    listing = db.query(models.Listing).filter(models.Listing.id == listing_id).first()
    if not listing:
        raise HTTPException(status_code=404, detail="Listing not found")
    return listing

@app.put("/listings/{listing_id}")
def update_listing(
    listing_id: int,
    size_chart: bool = None,
    photo_count: int = None,
    fabric_mentioned: bool = None,
    wash_care: bool = None,
    cod_enabled: bool = None,
    db: Session = Depends(get_db)
):
    """Update listing fields (for applying fixes)"""
    listing = db.query(models.Listing).filter(models.Listing.id == listing_id).first()
    if not listing:
        raise HTTPException(status_code=404, detail="Listing not found")
    
    if size_chart is not None:
        listing.size_chart = size_chart
    if photo_count is not None:
        listing.photo_count = photo_count
    if fabric_mentioned is not None:
        listing.fabric_mentioned = fabric_mentioned
    if wash_care is not None:
        listing.wash_care = wash_care
    if cod_enabled is not None:
        listing.cod_enabled = cod_enabled
    
    db.commit()
    db.refresh(listing)
    return {"message": "Listing updated", "listing": listing}

# ============ RISK SCORING ENDPOINTS ============

@app.post("/listings/{listing_id}/risk-score", response_model=schemas.RiskScoreResponse)
def calculate_risk_score(listing_id: int, db: Session = Depends(get_db)):
    """Calculate risk score for a listing"""
    listing = db.query(models.Listing).filter(models.Listing.id == listing_id).first()
    if not listing:
        raise HTTPException(status_code=404, detail="Listing not found")
    
    # Get category benchmarks
    benchmarks = db.query(models.CategoryReturnBenchmark).filter(
        models.CategoryReturnBenchmark.category == listing.category
    ).all()
    
    benchmarks_list = [
        {"category": b.category, "gap_type": b.gap_type, "avg_contribution_pct": b.avg_contribution_pct}
        for b in benchmarks
    ]
    
    # Calculate score
    listing_dict = {
        "size_chart": listing.size_chart,
        "photo_count": listing.photo_count,
        "fabric_mentioned": listing.fabric_mentioned,
        "wash_care": listing.wash_care,
        "category": listing.category
    }
    
    score, gaps, predicted_after_fixes = listing_agent.calculate_risk_score(listing_dict, benchmarks_list)
    
    # Store risk score
    risk_record = models.RiskScore(
        listing_id=listing_id,
        score=score,
        gap_details=[{"label": g["label"], "severity": g["severity"], "contribution_pct": g["contribution_pct"], "explanation": g["explanation"]} for g in gaps],
        created_at=datetime.utcnow()
    )
    db.add(risk_record)
    db.commit()
    
    return schemas.RiskScoreResponse(
        listing_id=listing_id,
        score=score,
        gaps=[schemas.RiskGap(**g) for g in gaps],
        predicted_score_after_fixes=predicted_after_fixes,
        timestamp=datetime.utcnow()
    )

@app.post("/listings/{listing_id}/fraud-check", response_model=schemas.FraudCheckResponse)
def check_fraud_risk(listing_id: int, db: Session = Depends(get_db)):
    """Check fraud risk for a listing based on PIN code & COD"""
    listing = db.query(models.Listing).filter(models.Listing.id == listing_id).first()
    if not listing:
        raise HTTPException(status_code=404, detail="Listing not found")
    
    # Get PIN risk data
    pin_risks = {}
    pin_records = db.query(models.PinCodeRisk).all()
    for pin in pin_records:
        pin_risks[pin.pin_code] = {"rto_rate": pin.rto_rate, "fraud_flag": pin.fraud_flag}
    
    result = listing_agent.get_fraud_risk(listing.pin_code, listing.cod_enabled, pin_risks)
    return schemas.FraudCheckResponse(
        pin_code=listing.pin_code,
        rto_rate=result["rto_rate"],
        fraud_flag=result["fraud_flag"],
        risk_level=result["risk_level"],
        message=result["message"]
    )

# ============ Q&A ENDPOINTS ============

@app.get("/qa/pending")
def get_pending_questions(listing_id: int, db: Session = Depends(get_db)):
    """Get ungrouped questions for a listing"""
    questions = db.query(models.BuyerQuestion).filter(
        models.BuyerQuestion.listing_id == listing_id,
        models.BuyerQuestion.cluster_id == None
    ).all()
    return [{"id": q.id, "question_text": q.question_text, "language": q.language, "created_at": q.created_at} for q in questions]

@app.post("/qa/cluster")
def cluster_and_draft(
    listing_id: int,
    db: Session = Depends(get_db)
):
    """Cluster similar questions and draft replies"""
    listing = db.query(models.Listing).filter(models.Listing.id == listing_id).first()
    if not listing:
        raise HTTPException(status_code=404, detail="Listing not found")
    
    # Get all ungrouped questions
    questions = db.query(models.BuyerQuestion).filter(
        models.BuyerQuestion.listing_id == listing_id,
        models.BuyerQuestion.cluster_id == None
    ).all()
    
    if len(questions) < 2:
        return {"clusters": [], "message": "Not enough questions to cluster"}
    
    # Get seller language
    seller = listing.seller
    language = seller.preferred_language
    
    question_dicts = [{"id": q.id, "question_text": q.question_text} for q in questions]
    
    # Cluster
    clusters = qa_agent.cluster_questions(question_dicts, language)
    
    # Process each cluster with 2+ questions
    results = []
    cluster_id = 0
    
    for cluster_indices in clusters:
        if len(cluster_indices) >= 2:
            cluster_id += 1
            cluster_questions = [q for q in questions if q.id in cluster_indices]
            
            # Draft reply and fix
            try:
                draft_reply, listing_fix = qa_agent.draft_reply_and_fix(
                    [{"question_text": q.question_text} for q in cluster_questions],
                    {"title": listing.title, "category": listing.category, "description": listing.description},
                    language
                )
            except Exception as e:
                draft_reply, listing_fix = f"Thank you for your question! We'll get back to you soon.", "Add more product details"
            
            # Update cluster_id for these questions
            for q in cluster_questions:
                q.cluster_id = cluster_id
                db.commit()
            
            # Create QA reply
            qa_reply = models.QAReply(
                question_id=cluster_questions[0].id,
                draft_reply=draft_reply,
                status="pending",
                created_at=datetime.utcnow()
            )
            db.add(qa_reply)
            db.commit()
            
            results.append({
                "cluster_id": cluster_id,
                "question_ids": cluster_indices,
                "draft_reply": draft_reply,
                "listing_fix_suggestion": listing_fix
            })
    
    return {"clusters": results}

@app.post("/qa/approve")
def approve_qa_reply(reply_id: int, db: Session = Depends(get_db)):
    """Approve a Q&A reply and apply listing fix"""
    reply = db.query(models.QAReply).filter(models.QAReply.id == reply_id).first()
    if not reply:
        raise HTTPException(status_code=404, detail="Reply not found")
    
    reply.status = "approved"
    db.commit()
    
    return {"message": "Reply approved", "reply_id": reply_id}

# ============ HEALTH ENDPOINTS ============

@app.post("/health/scan")
def run_health_scan(seller_id: int, db: Session = Depends(get_db)):
    """Run weekly health scan and generate brief"""
    seller = db.query(models.Seller).filter(models.Seller.id == seller_id).first()
    if not seller:
        raise HTTPException(status_code=404, detail="Seller not found")
    
    # Aggregate return stats
    cod_returns = db.query(models.SyntheticReturn).filter(
        models.SyntheticReturn.is_cod == True
    ).count()
    prepaid_returns = db.query(models.SyntheticReturn).filter(
        models.SyntheticReturn.is_cod == False
    ).count()
    
    # Get common return reasons
    returns = db.query(models.SyntheticReturn).all()
    reasons = [r.reason for r in returns]
    reason_counts = {}
    for r in reasons:
        reason_counts[r] = reason_counts.get(r, 0) + 1
    common_reasons = sorted(reason_counts.items(), key=lambda x: x[1], reverse=True)[:3]
    common_reasons = [r[0] for r in common_reasons]
    
    listings_count = db.query(models.Listing).filter(models.Listing.seller_id == seller_id).count()
    
    return_stats = {
        "cod_count": cod_returns,
        "prepaid_count": prepaid_returns,
        "common_reasons": common_reasons,
        "listings_count": listings_count
    }
    
    # Generate brief
    try:
        summary, recommendations = health_agent.generate_health_brief(return_stats, seller.preferred_language)
    except Exception as e:
        summary = "Weekly health scan completed."
        recommendations = [{"title": "Monitor returns", "description": "Keep an eye on return rates."}]
    
    # Store health brief
    brief = models.HealthBrief(
        seller_id=seller_id,
        week_of=datetime.utcnow() - timedelta(days=datetime.utcnow().weekday()),
        summary_text=summary,
        recommendations=recommendations,
        language=seller.preferred_language,
        created_at=datetime.utcnow()
    )
    db.add(brief)
    db.commit()
    db.refresh(brief)
    
    return schemas.HealthBrief(
        id=brief.id,
        seller_id=brief.seller_id,
        week_of=brief.week_of,
        summary_text=brief.summary_text,
        recommendations=brief.recommendations,
        language=brief.language,
        created_at=brief.created_at
    )

@app.get("/health/briefs")
def get_health_briefs(seller_id: int, db: Session = Depends(get_db)):
    """Get all health briefs for a seller"""
    briefs = db.query(models.HealthBrief).filter(models.HealthBrief.seller_id == seller_id).all()
    return briefs

# ============ HEALTH CHECK ============

@app.get("/health")
def health_check():
    return {"status": "healthy", "message": "Shuruaat AI is running!"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
