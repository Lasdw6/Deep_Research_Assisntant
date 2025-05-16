import time
from collections import defaultdict
import threading

class QueryRateLimiter:
    def __init__(self, max_queries_per_hour: int = 10):
        """
        Initialize rate limiter for queries per hour.
        
        Args:
            max_queries_per_hour: Maximum number of queries allowed per hour
        """
        self.max_queries = max_queries_per_hour
        self.queries = defaultdict(list)  # user_id -> list of timestamps
        self.lock = threading.Lock()
    
    def is_allowed(self, user_id: str) -> bool:
        """
        Check if a user is allowed to make another query.
        
        Args:
            user_id: Unique identifier for the user
            
        Returns:
            bool: True if query is allowed, False if rate limited
        """
        current_time = time.time()
        hour_ago = current_time - 3600  # 1 hour in seconds
        
        with self.lock:
            # Remove queries older than 1 hour
            self.queries[user_id] = [t for t in self.queries[user_id] if t > hour_ago]
            
            # Check if under rate limit
            if len(self.queries[user_id]) < self.max_queries:
                self.queries[user_id].append(current_time)
                return True
            
            return False
    
    def get_remaining_queries(self, user_id: str) -> int:
        """
        Get number of remaining queries for a user in the current hour.
        
        Args:
            user_id: Unique identifier for the user
            
        Returns:
            int: Number of remaining queries
        """
        current_time = time.time()
        hour_ago = current_time - 3600
        
        with self.lock:
            # Remove queries older than 1 hour
            self.queries[user_id] = [t for t in self.queries[user_id] if t > hour_ago]
            
            return self.max_queries - len(self.queries[user_id])
    
    def get_time_until_reset(self, user_id: str) -> float:
        """
        Get time in seconds until the rate limit resets for a user.
        
        Args:
            user_id: Unique identifier for the user
            
        Returns:
            float: Seconds until rate limit reset
        """
        current_time = time.time()
        
        with self.lock:
            if not self.queries[user_id]:
                return 0.0
            
            oldest_query = min(self.queries[user_id])
            reset_time = oldest_query + 3600  # 1 hour in seconds
            
            return max(0.0, reset_time - current_time) 