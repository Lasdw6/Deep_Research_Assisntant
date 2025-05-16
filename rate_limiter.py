import time
from collections import defaultdict
import threading

class QueryRateLimiter:
    def __init__(self, max_queries_per_hour: int = 10):
        """
        Initialize rate limiter for global queries per hour.
        
        Args:
            max_queries_per_hour: Maximum number of queries allowed per hour globally
        """
        self.max_queries = max_queries_per_hour
        self.queries = []  # List of timestamps for all queries
        self.lock = threading.Lock()
    
    def is_allowed(self, _: str = None) -> bool:
        """
        Check if another query is allowed within the hourly limit.
        
        Returns:
            bool: True if query is allowed, False if rate limited
        """
        current_time = time.time()
        hour_ago = current_time - 3600  # 1 hour in seconds
        
        with self.lock:
            # Remove queries older than 1 hour
            self.queries = [t for t in self.queries if t > hour_ago]
            
            # Check if under rate limit
            if len(self.queries) < self.max_queries:
                self.queries.append(current_time)
                return True
            
            return False
    
    def get_remaining_queries(self, _: str = None) -> int:
        """
        Get number of remaining queries in the current hour.
        
        Returns:
            int: Number of remaining queries
        """
        current_time = time.time()
        hour_ago = current_time - 3600
        
        with self.lock:
            # Remove queries older than 1 hour
            self.queries = [t for t in self.queries if t > hour_ago]
            
            return self.max_queries - len(self.queries)
    
    def get_time_until_reset(self, _: str = None) -> float:
        """
        Get time in seconds until the rate limit resets.
        
        Returns:
            float: Seconds until rate limit reset
        """
        current_time = time.time()
        
        with self.lock:
            if not self.queries:
                return 0.0
            
            oldest_query = min(self.queries)
            reset_time = oldest_query + 3600  # 1 hour in seconds
            
            return max(0.0, reset_time - current_time) 