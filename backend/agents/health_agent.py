from anthropic import Anthropic
import json
from typing import Dict, List, Tuple

client = Anthropic()

LANGUAGE_NAMES = {
    "hi": "Hindi",
    "gu": "Gujarati",
    "ta": "Tamil"
}

def generate_health_brief(return_stats: Dict, language: str = "hi") -> Tuple[str, List[Dict]]:
    """
    Generate a weekly health brief from aggregated return stats.
    Returns: (summary_text, recommendations)
    """
    lang_name = LANGUAGE_NAMES.get(language, "Hindi")
    
    cod_returns = return_stats.get("cod_count", 0)
    prepaid_returns = return_stats.get("prepaid_count", 0)
    common_reasons = return_stats.get("common_reasons", [])
    listings_count = return_stats.get("listings_count", 0)
    
    system_prompt = f"""You are a business advisor for Indian Meesho sellers.
You MUST RESPOND ONLY IN {lang_name}. NO ENGLISH. ALL OUTPUT MUST BE IN {lang_name} ONLY.

Given return statistics, generate:
1. A brief 2-3 sentence health summary in plain language
2. 1-2 concrete, actionable recommendations with potential ROI

Respond with ONLY a valid JSON object with keys: "summary" and "recommendations" (array of {{"title": "...", "description": "..."}}).
All text MUST BE in {lang_name}."""
    
    user_message = f"""Weekly Return Stats for seller:
- COD Returns: {cod_returns}
- Prepaid Returns: {prepaid_returns}
- Common Return Reasons: {', '.join(common_reasons[:3])}
- Total Listings: {listings_count}

Generate a health brief with recommendations. Respond ONLY with JSON.
Remember: ONLY {lang_name}, NO ENGLISH."""
    
    message = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=512,
        system=system_prompt,
        messages=[{"role": "user", "content": user_message}]
    )
    
    response_text = message.content[0].text.strip()
    
    # Parse JSON
    if response_text.startswith("```"):
        response_text = response_text.split("```")[1]
        if response_text.startswith("json"):
            response_text = response_text[4:]
        response_text = response_text.strip()
    
    data = json.loads(response_text)
    return data.get("summary", ""), data.get("recommendations", [])
