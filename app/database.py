import os
import json
import time
import redis
from dotenv import load_dotenv

load_dotenv()

REDIS_URL = os.getenv("REDIS_URL")
redis_client = None

# Connect to Upstash Redis if URL is provided
if REDIS_URL:
    try:
        redis_client = redis.Redis.from_url(REDIS_URL, decode_responses=True)
        redis_client.ping()
        print("[DATABASE] Connected to Upstash Redis successfully.")
    except Exception as e:
        print(f"[DATABASE] WARNING: Failed to connect to Redis at {REDIS_URL}: {e}")
        print("[DATABASE] Falling back to in-memory structures.")
        redis_client = None
else:
    print("[DATABASE] REDIS_URL not found in environment. Falling back to in-memory structures.")

# In-memory fallbacks for development/offline mode
rate_limit_tracker = {}
feedback_log = {}


def check_rate_limit(session_id: str, max_requests: int = 10, window_seconds: int = 60) -> bool:
    """
    Checks if a session_id has exceeded the rate limit.
    Returns True if the request is ALLOWED, False if BLOCKED.
    """
    now = time.time()
    
    if redis_client:
        try:
            key = f"tiet_rate_limit:{session_id}"
            # Add current timestamp to sorted set (score and value are both timestamp)
            redis_client.zadd(key, {str(now): now})
            # Remove timestamps older than window
            redis_client.zremrangebyscore(key, 0, now - window_seconds)
            # Count elements inside the window
            count = redis_client.zcard(key)
            # Expire key to clean up idle data
            redis_client.expire(key, window_seconds)
            
            return count <= max_requests
        except Exception as e:
            print(f"[DATABASE] Redis Rate Limiter error: {e}. Falling back to in-memory.")
    
    # In-memory fallback
    global rate_limit_tracker
    timestamps = rate_limit_tracker.get(session_id, [])
    timestamps = [t for t in timestamps if now - t < window_seconds]
    
    if len(timestamps) >= max_requests:
        return False
        
    timestamps.append(now)
    rate_limit_tracker[session_id] = timestamps
    return True


def save_feedback(feedback_id: str, session_id: str, question: str, answer: str) -> None:
    """Saves a new question and answer feedback placeholder."""
    data = {
        "session_id": session_id,
        "question": question,
        "answer": answer or "",
        "rating": "",  # will be filled by update_feedback ('up' or 'down')
        "timestamp": time.time()
    }
    
    if redis_client:
        try:
            key = f"tiet_feedback:{feedback_id}"
            redis_client.set(key, json.dumps(data))
            # Optional expiration to save memory on free Redis tiers (e.g. 30 days)
            redis_client.expire(key, 30 * 86400)
            return
        except Exception as e:
            print(f"[DATABASE] Redis Feedback Save error: {e}. Falling back to in-memory.")
            
    global feedback_log
    feedback_log[feedback_id] = data


def update_feedback(feedback_id: str, rating: str) -> bool:
    """Updates the rating ('up' or 'down') for an existing feedback entry."""
    if rating not in ("up", "down"):
        return False
        
    if redis_client:
        try:
            key = f"tiet_feedback:{feedback_id}"
            data_str = redis_client.get(key)
            if data_str:
                data = json.loads(data_str)
                data["rating"] = rating
                redis_client.set(key, json.dumps(data))
                return True
            return False
        except Exception as e:
            print(f"[DATABASE] Redis Feedback Update error: {e}. Falling back to in-memory.")
            
    global feedback_log
    if feedback_id in feedback_log:
        feedback_log[feedback_id]["rating"] = rating
        return True
    return False


def get_feedback_summary() -> dict:
    """Aggregates all feedback scores."""
    if redis_client:
        try:
            # Note: SCAN is preferred over KEYS in production to prevent blocking
            cursor = 0
            keys = []
            while True:
                cursor, scan_keys = redis_client.scan(cursor, match="tiet_feedback:*", count=100)
                keys.extend(scan_keys)
                if cursor == 0:
                    break
                    
            total = len(keys)
            up_count = 0
            down_count = 0
            rated = 0
            
            for k in keys:
                val = redis_client.get(k)
                if val:
                    data = json.loads(val)
                    if data.get("rating") == "up":
                        up_count += 1
                        rated += 1
                    elif data.get("rating") == "down":
                        down_count += 1
                        rated += 1
                        
            return {
                "total_questions": total,
                "total_rated": rated,
                "thumbs_up": up_count,
                "thumbs_down": down_count
            }
        except Exception as e:
            print(f"[DATABASE] Redis Feedback Summary error: {e}. Falling back to in-memory.")
            
    global feedback_log
    total = len(feedback_log)
    rated_items = [f for f in feedback_log.values() if f["rating"] != ""]
    up_count = sum(1 for f in rated_items if f["rating"] == "up")
    down_count = sum(1 for f in rated_items if f["rating"] == "down")
    
    return {
        "total_questions": total,
        "total_rated": len(rated_items),
        "thumbs_up": up_count,
        "thumbs_down": down_count
    }
