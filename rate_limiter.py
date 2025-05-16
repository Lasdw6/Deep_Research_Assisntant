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
        self.queries = defaultdict(list)  # ip_address -> list of timestamps
        self.lock = threading.Lock()
    
    def is_allowed(self, ip_address: str) -> bool:
        """
        Check if an IP address is allowed to make another query.
        
        Args:
            ip_address: IP address of the client
            
        Returns:
            bool: True if query is allowed, False if rate limited
        """
        current_time = time.time()
        hour_ago = current_time - 3600  # 1 hour in seconds
        
        with self.lock:
            # Remove queries older than 1 hour
            self.queries[ip_address] = [t for t in self.queries[ip_address] if t > hour_ago]
            
            # Check if under rate limit
            if len(self.queries[ip_address]) < self.max_queries:
                self.queries[ip_address].append(current_time)
                return True
            
            return False
    
    def get_remaining_queries(self, ip_address: str) -> int:
        """
        Get number of remaining queries for an IP address in the current hour.
        
        Args:
            ip_address: IP address of the client
            
        Returns:
            int: Number of remaining queries
        """
        current_time = time.time()
        hour_ago = current_time - 3600
        
        with self.lock:
            # Remove queries older than 1 hour
            self.queries[ip_address] = [t for t in self.queries[ip_address] if t > hour_ago]
            
            return self.max_queries - len(self.queries[ip_address])
    
    def get_time_until_reset(self, ip_address: str) -> float:
        """
        Get time in seconds until the rate limit resets for an IP address.
        
        Args:
            ip_address: IP address of the client
            
        Returns:
            float: Seconds until rate limit reset
        """
        current_time = time.time()
        
        with self.lock:
            if not self.queries[ip_address]:
                return 0.0
            
            oldest_query = min(self.queries[ip_address])
            reset_time = oldest_query + 3600  # 1 hour in seconds
            
            return max(0.0, reset_time - current_time) 