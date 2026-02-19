import re

# ============================================================
# LAYER 1: Regex-based Vague / Irrelevant Question Filter
# ============================================================

VAGUE_PATTERNS = [
    # Greetings
    r"^(hi|hello|hey|howdy|sup|yo|greetings|hola|namaste)\b",
    r"(good morning|good evening|good night|good afternoon)",
    
    # Pleasantries / Closers
    r"^(thanks|thank you|ok|okay|bye|goodbye|see you|take care)\b",
    
    # Identity / Personal questions about the AI
    r"how are you",
    r"who (are|made|created|built) you",
    r"what('s| is) your name",
    r"are you (a |an )?(human|robot|ai|bot|real|alive|sentient|gay|straight|male|female|man|woman)",
    r"do you have (feelings|emotions|a body|a soul|consciousness)",
    r"what (can you do|do you do|are you capable of)",
    r"how (were|was) you (made|created|built|trained)",
    r"how can i (make|create|build) (one like you|something like you|an ai|a bot|my own (ai|llm|model))",
    r"tell me about yourself",
    
    # Jokes / Fun / Off-topic
    r"tell me (a joke|something funny|a story|a riddle|a poem)",
    r"(sing|write) (me )?(a song|a poem|a haiku)",
    r"let('s| us) (play|chat|talk|have fun)",
    
    # Generic / Vague
    r"^(test|testing|help|hmm|lol|haha|wow|cool|nice|great)$",
    r"^what$",
    r"^why$",
    r"^how$",
    r"^(yes|no|maybe|idk|dunno)$",
    
    # Personal advice (not research)
    r"^(what should i do|give me advice|help me with my life)",
    
    # Explicit / Inappropriate (using \b word boundaries to avoid matching inside words like "asset", "class")
    r"\b(sex|porn|nude|naked|fuck|shit|damn|dick|penis|vagina|boob|breast)\b",
]

def is_vague_query(query: str) -> bool:
    """
    Returns True if the query matches known vague/irrelevant/personal patterns.
    This is Layer 1 — fast regex check, zero cost.
    """
    query_lower = query.lower().strip()
    
    # Too short to be a real research question
    if len(query_lower) < 5:
        return True
    
    for pattern in VAGUE_PATTERNS:
        if re.search(pattern, query_lower):
            return True
    return False


# ============================================================
# LAYER 2: Relevance Score Gate (post-retrieval)
# ============================================================

def check_relevance_with_scores(vectorstore, query: str, k: int = 8, threshold: float = 1.8):
    """
    Performs similarity search WITH scores using FAISS.
    Returns (relevant_docs, is_relevant).
    
    FAISS uses L2 distance by default:
    - Lower score = MORE similar (0.0 = identical)
    - Higher score = LESS similar
    
    Strategy:
    - We check the BEST (lowest) score among all retrieved chunks.
    - If the best score is below the threshold, the query is considered relevant
      (meaning at least SOME document content is related to the query).
    - threshold=1.8 is lenient enough to allow semantically related queries
      (e.g., "blockchain security" matching docs about "cryptocurrency").
    - Only truly unrelated queries (e.g., "how to cook pasta" on finance docs)
      will score above 1.8 and get rejected.
    
    We return ALL retrieved docs (not just the ones below threshold) so the LLM
    gets full context. The strict prompt handles the rest.
    """
    docs_with_scores = vectorstore.similarity_search_with_score(query, k=k)
    
    if not docs_with_scores:
        return [], False
    
    # Check if the BEST (most relevant) chunk passes the threshold
    best_score = min(score for _, score in docs_with_scores)
    
    if best_score > threshold:
        # Even the most relevant chunk is too far away — truly unrelated query
        return [], False
    
    # Return all docs for full context (the strict prompt prevents hallucination)
    return [doc for doc, score in docs_with_scores], True
