"""Seed database with demo data for Shuruaat AI."""
import sys
sys.path.insert(0, '.')

from database import SessionLocal, init_db
import models
from datetime import datetime, timedelta

def seed_database():
    # Initialize database tables
    init_db()
    
    db = SessionLocal()
    
    # Clear existing data (optional, for clean seeding)
    # db.query(models.Seller).delete()
    # db.commit()
    
    print("Seeding database...")
    
    # ============ SEED SELLERS ============
    priya = models.Seller(
        name="Priya",
        preferred_language="gu",  # Gujarati
        city="Surat"
    )
    db.add(priya)
    db.commit()
    db.refresh(priya)
    print(f"✓ Created seller: Priya (ID: {priya.id})")
    
    # ============ SEED PIN CODE RISKS ============
    pin_risks = [
        # Surat (high risk zone)
        {"pin_code": "395007", "rto_rate": 18.5, "fraud_flag": True, "notes": "High RTO zone in Surat"},
        {"pin_code": "395001", "rto_rate": 16.2, "fraud_flag": True, "notes": "Central Surat - high fraud"},
        {"pin_code": "395003", "rto_rate": 14.0, "fraud_flag": False, "notes": "Medium risk zone"},
        
        # Delhi (medium-high risk)
        {"pin_code": "110005", "rto_rate": 12.0, "fraud_flag": False, "notes": "Delhi - moderate risk"},
        {"pin_code": "110015", "rto_rate": 10.5, "fraud_flag": False, "notes": "South Delhi - lower risk"},
        {"pin_code": "110001", "rto_rate": 13.8, "fraud_flag": True, "notes": "Central Delhi - high chargeback"},
        
        # Mumbai (medium risk)
        {"pin_code": "400001", "rto_rate": 9.2, "fraud_flag": False, "notes": "South Mumbai - lower risk"},
        {"pin_code": "400614", "rto_rate": 8.5, "fraud_flag": False, "notes": "Mumbai suburbs - stable"},
        {"pin_code": "400043", "rto_rate": 11.0, "fraud_flag": False, "notes": "West Mumbai - moderate"},
        
        # Bangalore (low risk)
        {"pin_code": "560001", "rto_rate": 5.2, "fraud_flag": False, "notes": "Central Bangalore - very safe"},
        {"pin_code": "560034", "rto_rate": 6.0, "fraud_flag": False, "notes": "Indiranagar - safe"},
        {"pin_code": "560066", "rto_rate": 4.8, "fraud_flag": False, "notes": "Whitefield - very safe"},
        
        # Hyderabad (low risk)
        {"pin_code": "500001", "rto_rate": 5.5, "fraud_flag": False, "notes": "Central Hyderabad - safe"},
        {"pin_code": "500034", "rto_rate": 6.2, "fraud_flag": False, "notes": "Banjara Hills - safe"},
        
        # Pune (low-medium risk)
        {"pin_code": "411001", "rto_rate": 7.0, "fraud_flag": False, "notes": "Central Pune - stable"},
        {"pin_code": "411002", "rto_rate": 7.5, "fraud_flag": False, "notes": "East Pune - stable"},
        
        # Chennai (medium risk)
        {"pin_code": "600001", "rto_rate": 9.0, "fraud_flag": False, "notes": "Central Chennai"},
        {"pin_code": "600002", "rto_rate": 10.2, "fraud_flag": False, "notes": "North Chennai"},
        
        # Kolkata (high-medium risk)
        {"pin_code": "700001", "rto_rate": 12.5, "fraud_flag": False, "notes": "Central Kolkata"},
        {"pin_code": "700026", "rto_rate": 11.8, "fraud_flag": False, "notes": "South Kolkata"},
        
        # Tier 2/3 (mixed risk)
        {"pin_code": "201001", "rto_rate": 13.0, "fraud_flag": True, "notes": "Noida - high fraud risk"},
        {"pin_code": "282001", "rto_rate": 15.5, "fraud_flag": True, "notes": "Agra - high risk"},
        {"pin_code": "231001", "rto_rate": 18.0, "fraud_flag": True, "notes": "Varanasi - very high risk"},
        {"pin_code": "360001", "rto_rate": 14.2, "fraud_flag": False, "notes": "Rajkot - medium-high"},
        {"pin_code": "362001", "rto_rate": 12.8, "fraud_flag": False, "notes": "Jamnagar - medium"},
        {"pin_code": "364001", "rto_rate": 11.5, "fraud_flag": False, "notes": "Bhavnagar - medium"},
        {"pin_code": "370001", "rto_rate": 10.0, "fraud_flag": False, "notes": "Ahmedabad - lower"},
        {"pin_code": "371001", "rto_rate": 13.2, "fraud_flag": False, "notes": "Bhuj - medium"},
    ]
    
    for risk_data in pin_risks:
        pin_risk = models.PinCodeRisk(**risk_data)
        db.add(pin_risk)
    db.commit()
    print(f"✓ Seeded {len(pin_risks)} PIN code risk records")
    
    # ============ SEED CATEGORY BENCHMARKS ============
    benchmarks = [
        # Kurti category
        {"category": "Kurti", "gap_type": "missing_size_chart", "avg_contribution_pct": 82.0},
        {"category": "Kurti", "gap_type": "single_photo", "avg_contribution_pct": 55.0},
        {"category": "Kurti", "gap_type": "no_fabric", "avg_contribution_pct": 48.0},
        {"category": "Kurti", "gap_type": "no_wash_care", "avg_contribution_pct": 22.0},
        
        # Saree category
        {"category": "Saree", "gap_type": "missing_size_chart", "avg_contribution_pct": 75.0},
        {"category": "Saree", "gap_type": "single_photo", "avg_contribution_pct": 60.0},
        {"category": "Saree", "gap_type": "no_fabric", "avg_contribution_pct": 52.0},
        {"category": "Saree", "gap_type": "no_wash_care", "avg_contribution_pct": 28.0},
        
        # T-Shirt category
        {"category": "T-Shirt", "gap_type": "missing_size_chart", "avg_contribution_pct": 78.0},
        {"category": "T-Shirt", "gap_type": "single_photo", "avg_contribution_pct": 50.0},
        {"category": "T-Shirt", "gap_type": "no_fabric", "avg_contribution_pct": 45.0},
        {"category": "T-Shirt", "gap_type": "no_wash_care", "avg_contribution_pct": 25.0},
        
        # Jeans category
        {"category": "Jeans", "gap_type": "missing_size_chart", "avg_contribution_pct": 85.0},
        {"category": "Jeans", "gap_type": "single_photo", "avg_contribution_pct": 52.0},
        {"category": "Jeans", "gap_type": "no_fabric", "avg_contribution_pct": 40.0},
        {"category": "Jeans", "gap_type": "no_wash_care", "avg_contribution_pct": 30.0},
        
        # Dress category
        {"category": "Dress", "gap_type": "missing_size_chart", "avg_contribution_pct": 80.0},
        {"category": "Dress", "gap_type": "single_photo", "avg_contribution_pct": 58.0},
        {"category": "Dress", "gap_type": "no_fabric", "avg_contribution_pct": 50.0},
        {"category": "Dress", "gap_type": "no_wash_care", "avg_contribution_pct": 26.0},
        
        # Dupatta category
        {"category": "Dupatta", "gap_type": "missing_size_chart", "avg_contribution_pct": 65.0},
        {"category": "Dupatta", "gap_type": "single_photo", "avg_contribution_pct": 48.0},
        {"category": "Dupatta", "gap_type": "no_fabric", "avg_contribution_pct": 55.0},
        {"category": "Dupatta", "gap_type": "no_wash_care", "avg_contribution_pct": 18.0},
        
        # Lehenga category
        {"category": "Lehenga", "gap_type": "missing_size_chart", "avg_contribution_pct": 88.0},
        {"category": "Lehenga", "gap_type": "single_photo", "avg_contribution_pct": 62.0},
        {"category": "Lehenga", "gap_type": "no_fabric", "avg_contribution_pct": 58.0},
        {"category": "Lehenga", "gap_type": "no_wash_care", "avg_contribution_pct": 32.0},
    ]
    
    for bench_data in benchmarks:
        bench = models.CategoryReturnBenchmark(**bench_data)
        db.add(bench)
    db.commit()
    print(f"✓ Seeded {len(benchmarks)} category benchmarks")
    
    # ============ SEED LISTINGS ============
    listing1 = models.Listing(
        seller_id=priya.id,
        title="Blue Cotton Kurti",
        description="Beautiful blue cotton kurti for casual wear",
        category="Kurti",
        price=499.0,
        size_chart=False,  # Missing!
        photo_count=1,     # Only 1 photo
        fabric_mentioned=False,  # Not mentioned
        wash_care=False,   # Not mentioned
        cod_enabled=True,
        pin_code="395007",  # High-risk zone
        created_at=datetime.utcnow()
    )
    db.add(listing1)
    db.commit()
    db.refresh(listing1)
    print(f"✓ Created listing: Blue Cotton Kurti (ID: {listing1.id})")
    
    listing2 = models.Listing(
        seller_id=priya.id,
        title="Silk Saree",
        description="Traditional silk saree with intricate design",
        category="Saree",
        price=1999.0,
        size_chart=True,
        photo_count=3,
        fabric_mentioned=True,
        wash_care=True,
        cod_enabled=False,
        pin_code="395001",
        created_at=datetime.utcnow()
    )
    db.add(listing2)
    db.commit()
    db.refresh(listing2)
    print(f"✓ Created listing: Silk Saree (ID: {listing2.id})")
    
    # ============ SEED BUYER QUESTIONS ============
    # Questions for listing1 (Blue Cotton Kurti) - cluster test
    questions1 = [
        {"text": "What is the fabric type? Is it 100% cotton?", "language": "gu"},
        {"text": "Is this kurti made of pure cotton?", "language": "gu"},
        {"text": "This kurti, fabric is cotton only?", "language": "gu"},
        {"text": "What size should I order for medium fit?", "language": "en"},
        {"text": "Do you have XL size available?", "language": "en"},
    ]
    
    for q_data in questions1:
        question = models.BuyerQuestion(
            listing_id=listing1.id,
            question_text=q_data["text"],
            language=q_data["language"],
            created_at=datetime.utcnow() - timedelta(days=1)
        )
        db.add(question)
    db.commit()
    print(f"✓ Seeded 5 buyer questions for Listing 1")
    
    # Questions for listing2 (Silk Saree)
    questions2 = [
        {"text": "How many meters is this saree?", "language": "en"},
        {"text": "Is blouse included?", "language": "en"},
        {"text": "What's the care instruction for silk?", "language": "en"},
        {"text": "Does it have running or fall?", "language": "gu"},
    ]
    
    for q_data in questions2:
        question = models.BuyerQuestion(
            listing_id=listing2.id,
            question_text=q_data["text"],
            language=q_data["language"],
            created_at=datetime.utcnow() - timedelta(days=1)
        )
        db.add(question)
    db.commit()
    print(f"✓ Seeded 4 buyer questions for Listing 2")
    
    # ============ SEED SYNTHETIC RETURNS ============
    returns = [
        # COD returns (higher volume)
        {"listing_id": listing1.id, "reason": "wrong_size", "is_cod": True},
        {"listing_id": listing1.id, "reason": "wrong_size", "is_cod": True},
        {"listing_id": listing1.id, "reason": "not_as_described", "is_cod": True},
        {"listing_id": listing1.id, "reason": "wrong_size", "is_cod": True},
        {"listing_id": listing1.id, "reason": "color_mismatch", "is_cod": True},
        
        # Prepaid returns (lower volume)
        {"listing_id": listing2.id, "reason": "damaged", "is_cod": False},
        {"listing_id": listing2.id, "reason": "not_as_described", "is_cod": False},
        
        # More COD returns
        {"listing_id": listing1.id, "reason": "wrong_size", "is_cod": True},
        {"listing_id": listing1.id, "reason": "not_as_described", "is_cod": True},
        {"listing_id": listing2.id, "reason": "damaged", "is_cod": True},
        {"listing_id": listing2.id, "reason": "damaged", "is_cod": False},
        {"listing_id": listing1.id, "reason": "color_mismatch", "is_cod": True},
    ]
    
    for ret_data in returns:
        ret = models.SyntheticReturn(
            listing_id=ret_data["listing_id"],
            reason=ret_data["reason"],
            is_cod=ret_data["is_cod"],
            created_at=datetime.utcnow() - timedelta(days=4)
        )
        db.add(ret)
    db.commit()
    print(f"✓ Seeded {len(returns)} synthetic returns")
    
    # ============ SEED RISK SCORES ============
    # Risk score for listing1 (high risk due to gaps)
    risk1 = models.RiskScore(
        listing_id=listing1.id,
        score=68.0,
        gap_details=[
            {"label": "Missing Size Chart", "severity": "HIGH", "contribution_pct": 82.0, "explanation": "Size chart is a critical field for Kurti category, contributing 82% to return risk."},
            {"label": "Only 1 Photo (Need 2+)", "severity": "HIGH", "contribution_pct": 55.0, "explanation": "Single photo angle limits buyer confidence, contributing 55% to returns."},
            {"label": "Fabric Not Mentioned", "severity": "HIGH", "contribution_pct": 48.0, "explanation": "Fabric type is essential; missing it contributes 48% to return risk."},
        ],
        created_at=datetime.utcnow()
    )
    db.add(risk1)
    db.commit()
    print(f"✓ Seeded risk score for Listing 1")
    
    # Risk score for listing2 (low risk, all gaps filled)
    risk2 = models.RiskScore(
        listing_id=listing2.id,
        score=5.0,
        gap_details=[],
        created_at=datetime.utcnow()
    )
    db.add(risk2)
    db.commit()
    print(f"✓ Seeded risk score for Listing 2")
    
    print("\n✅ Database seeding complete!")
    print(f"\n📊 Demo Data Summary:")
    print(f"  - Sellers: 1 (Priya, Gujarati, Surat)")
    print(f"  - Listings: 2 (1 high-risk, 1 optimized)")
    print(f"  - PIN codes: {len(pin_risks)} (mix of high/medium/low risk zones)")
    print(f"  - Category benchmarks: {len(benchmarks)}")
    print(f"  - Buyer questions: 9")
    print(f"  - Synthetic returns: {len(returns)}")
    print(f"\n🚀 Ready to demo! Start the app and navigate to http://localhost:5173")
    
    db.close()

if __name__ == "__main__":
    seed_database()
