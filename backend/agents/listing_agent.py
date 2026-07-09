from anthropic import Anthropic
import json
from typing import Dict, List, Tuple

client = Anthropic()

LANGUAGE_NAMES = {
    "hi": "Hindi",
    "gu": "Gujarati",
    "ta": "Tamil"
}

def generate_listing(raw_input: str, category: str, language: str = "hi") -> Dict:
    """
    Generate a listing from raw seller input (voice transcription or form data).
    Returns JSON: {title, bullets, size_chart_suggestion, keywords, suggested_price_range}
    """
    lang_name = LANGUAGE_NAMES.get(language, "Hindi")
    
    system_prompt = f"""You are an expert e-commerce listing writer for Indian sellers on Meesho.
Your task is to create compelling, conversion-optimized product listings.
YOU MUST RESPOND ONLY IN {lang_name}. DO NOT USE ENGLISH. ALL OUTPUT MUST BE IN {lang_name} ONLY.

You will receive raw seller input and must output STRICT JSON (no markdown, no extra text, just valid JSON object).
JSON format:
{{
  "title": "[compelling product title in {lang_name}]",
  "bullets": ["[bullet 1]", "[bullet 2]", "[bullet 3]", "[bullet 4]", "[bullet 5]"],
  "size_chart_suggestion": "[brief suggestion about sizing]",
  "keywords": ["[keyword1]", "[keyword2]", "[keyword3]"],
  "suggested_price_range": "[price range suggestion]"
}}

Ensure all text is in {lang_name}. The JSON must be valid and parseable.
"""
    
    user_message = f"""Create a listing for this product in category '{category}'.
Raw seller input: {raw_input}

Respond with ONLY a valid JSON object, no other text. Remember: ONLY {lang_name}, NO ENGLISH."""
    
    message = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        system=system_prompt,
        messages=[{"role": "user", "content": user_message}]
    )
    
    response_text = message.content[0].text.strip()
    
    # Parse JSON - handle potential markdown code fences
    if response_text.startswith("```"):
        response_text = response_text.split("```")[1]
        if response_text.startswith("json"):
            response_text = response_text[4:]
        response_text = response_text.strip()
    
    listing_data = json.loads(response_text)
    return listing_data

def calculate_risk_score(
    listing: Dict,
    benchmarks: List[Dict]
) -> Tuple[float, List[Dict], float]:
    """
    Calculate risk score based on listing fields vs category benchmarks.
    Returns: (risk_score, gaps_list, predicted_score_after_fixes)
    """
    gaps = []
    total_contribution = 0.0
    
    # Define gap checks
    gap_checks = [
        {
            "gap_type": "missing_size_chart",
            "condition": not listing.get("size_chart", False),
            "label": "Missing Size Chart"
        },
        {
            "gap_type": "single_photo",
            "condition": listing.get("photo_count", 1) < 2,
            "label": "Only 1 Photo (Need 2+)"
        },
        {
            "gap_type": "no_fabric",
            "condition": not listing.get("fabric_mentioned", False),
            "label": "Fabric Not Mentioned"
        },
        {
            "gap_type": "no_wash_care",
            "condition": not listing.get("wash_care", False),
            "label": "No Wash Care Details"
        }
    ]
    
    # Build benchmark lookup
    benchmark_map = {}
    for b in benchmarks:
        key = (b["category"], b["gap_type"])
        benchmark_map[key] = b["avg_contribution_pct"]
    
    category = listing.get("category", "default")
    
    # Check each gap
    for check in gap_checks:
        if check["condition"]:
            key = (category, check["gap_type"])
            contribution = benchmark_map.get(key, 50)  # default 50% if not found
            gaps.append({
                "label": check["label"],
                "gap_type": check["gap_type"],
                "severity": "HIGH" if contribution > 60 else "MEDIUM" if contribution > 30 else "LOW",
                "contribution_pct": contribution,
                "explanation": f"{check['label']} contributes {contribution}% to return risk based on {category} benchmarks."
            })
            total_contribution += contribution
    
    # Cap at 100 and apply logarithmic scaling for realism
    risk_score = min(100, total_contribution)
    
    # Predicted score if all gaps are fixed (minimum 5% baseline risk)
    predicted_after_fixes = 5.0
    
    return risk_score, gaps, predicted_after_fixes

def get_fraud_risk(pin_code: str, cod_enabled: bool, pin_risks: Dict[str, Dict]) -> Dict:
    """
    Check fraud risk for a PIN code + COD combination.
    Returns: {risk_level, message, rto_rate, fraud_flag}
    """
    pin_data = pin_risks.get(pin_code, {"rto_rate": 0, "fraud_flag": False})
    rto_rate = pin_data.get("rto_rate", 0)
    fraud_flag = pin_data.get("fraud_flag", False)
    
    if not cod_enabled:
        return {
            "risk_level": "NONE",
            "message": "Prepaid only - minimal fraud risk.",
            "rto_rate": rto_rate,
            "fraud_flag": False
        }
    
    # COD enabled - check risk
    if fraud_flag or rto_rate > 15:
        risk_level = "HIGH"
        message = f"⚠️ HIGH FRAUD RISK: PIN {pin_code} has {rto_rate}% RTO rate + COD enabled. Consider switching to prepaid-only for orders >₹700."
    elif rto_rate > 8:
        risk_level = "MEDIUM"
        message = f"⚠️ MEDIUM RISK: PIN {pin_code} has {rto_rate}% RTO rate with COD. Monitor carefully."
    else:
        risk_level = "LOW"
        message = f"✅ LOW RISK: PIN {pin_code} is relatively safe for COD ({rto_rate}% RTO)."
    
    return {
        "risk_level": risk_level,
        "message": message,
        "rto_rate": rto_rate,
        "fraud_flag": fraud_flag
    }
