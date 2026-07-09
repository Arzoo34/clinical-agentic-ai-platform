from anthropic import Anthropic
import json
from typing import List, Dict, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

client = Anthropic()

LANGUAGE_NAMES = {
    "hi": "Hindi",
    "gu": "Gujarati",
    "ta": "Tamil"
}

def cluster_questions(questions: List[Dict], language: str = "hi") -> List[List[int]]:
    """
    Cluster similar questions using TF-IDF cosine similarity.
    Returns list of clusters, where each cluster is a list of question IDs.
    Clusters only formed if 2+ similar questions found.
    """
    if len(questions) < 2:
        return [[q["id"]] for q in questions]
    
    texts = [q["question_text"] for q in questions]
    
    try:
        vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 3))
        tfidf_matrix = vectorizer.fit_transform(texts)
        similarity_matrix = cosine_similarity(tfidf_matrix)
        
        # Threshold for clustering
        threshold = 0.6
        
        clusters = []
        clustered_indices = set()
        
        for i in range(len(questions)):
            if i in clustered_indices:
                continue
            
            cluster = [questions[i]["id"]]
            for j in range(i + 1, len(questions)):
                if j not in clustered_indices and similarity_matrix[i][j] > threshold:
                    cluster.append(questions[j]["id"])
                    clustered_indices.add(j)
            
            clusters.append(cluster)
            if len(cluster) > 1:
                clustered_indices.add(i)
        
        return clusters
    except Exception as e:
        # Fallback: each question is its own cluster
        return [[q["id"]] for q in questions]

def draft_reply_and_fix(
    cluster_questions: List[Dict],
    listing: Dict,
    language: str = "hi"
) -> Tuple[str, str]:
    """
    Generate a smart reply to clustered questions and suggest a listing fix.
    Returns: (draft_reply, listing_fix_suggestion)
    """
    lang_name = LANGUAGE_NAMES.get(language, "Hindi")
    
    # Summarize cluster questions
    question_summary = "\n".join([f"- {q['question_text']}" for q in cluster_questions])
    
    system_prompt = f"""You are a helpful e-commerce customer service assistant helping Indian sellers.
You must respond ONLY IN {lang_name}. NO ENGLISH. ALL OUTPUT MUST BE IN {lang_name} ONLY.

Given a cluster of similar buyer questions about a product listing, you will:
1. Draft a concise, helpful reply to the questions
2. Suggest an update to the listing that would prevent similar questions

Respond with ONLY a valid JSON object with keys: "reply" and "listing_fix_suggestion", both in {lang_name}."""
    
    user_message = f"""Product: {listing.get('title', 'Product')}
Category: {listing.get('category', 'General')}
Current Description: {listing.get('description', 'N/A')}

Clustured Buyer Questions:
{question_summary}

Draft a single reply that addresses all these questions, and suggest one listing improvement.
Respond with JSON: {{"reply": "...", "listing_fix_suggestion": "..."}}
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
    return data.get("reply", ""), data.get("listing_fix_suggestion", "")
