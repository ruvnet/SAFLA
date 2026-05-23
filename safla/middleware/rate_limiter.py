"""Rate limiting middleware for SAFLA MCP Server"""

import time
import asyncio
import logging
import json
from typing import Dict, Any, Optional, Tuple
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import hashlib

logger = logging.getLogger(__name__)


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting"""
    # Requests per time window
    requests_per_minute: int = 60
    requests_per_hour: int = 1000
    requests_per_day: int = 10000
    
    # Burst allowance
    burst_size: int = 20
    
    # Method-specific limits
    method_limits: Dict[str, int] = field(default_factory=lambda: {
        "tools/call": 30,  # 30 requests per minute
        "resources/write": 10,  # 10 writes per minute
        "auth/login": 5,  # 5 login attempts per minute
    })
    
    # IP-based limits
    enable_ip_limiting: bool = True
    ip_requests_per_minute: int = 100
    
    # User-based limits (authenticated)
    user_requests_per_minute: int = 120
    user_requests_per_hour: int = 2000
    
    # Penalty for violations
    lockout_duration_seconds: int = 300  # 5 minutes
    max_violations: int = 5


@dataclass
class RateLimitStats:
    """Statistics for rate limiting"""
    requests: deque = field(default_factory=lambda: deque(maxlen=10000))
    violations: int = 0
    last_violation: Optional[datetime] = None
    locked_until: Optional[datetime] = None


class TokenBucket:
    """Token bucket algorithm for rate limiting"""
    
    def __init__(self, capacity: int, refill_rate: float):
        """
        Initialize token bucket.
        
        Args:
            capacity: Maximum number of tokens
            refill_rate: Tokens added per second
        """
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.tokens = capacity
        self.last_refill = time.time()
        self._lock = asyncio.Lock()
    
    async def consume(self, tokens: int = 1) -> bool:
        """
        Try to consume tokens from bucket.
        
        Args:
            tokens: Number of tokens to consume
            
        Returns:
            True if tokens were consumed, False if not enough tokens
        """
        async with self._lock:
            # Refill tokens based on time passed
            now = time.time()
            elapsed = now - self.last_refill
            self.tokens = min(
                self.capacity,
                self.tokens + (elapsed * self.refill_rate)
            )
            self.last_refill = now
            
            # Try to consume tokens
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            return False
    
    @property
    def available_tokens(self) -> float:
        """Get current available tokens"""
        elapsed = time.time() - self.last_refill
        return min(
            self.capacity,
            self.tokens + (elapsed * self.refill_rate)
        )


class RateLimiter:
    """Rate limiting middleware for MCP server"""
    
    def __init__(self, config: Optional[RateLimitConfig] = None):
        """Initialize rate limiter with configuration"""
        self.config = config or RateLimitConfig()
        
        # Stats by identifier (IP, user ID, etc.)
        self.stats: Dict[str, RateLimitStats] = defaultdict(RateLimitStats)
        
        # Token buckets for different rate limits
        self.global_bucket = TokenBucket(
            capacity=self.config.burst_size,
            refill_rate=self.config.requests_per_minute / 60
        )
        
        # Method-specific buckets
        self.method_buckets: Dict[str, TokenBucket] = {}
        for method, limit in self.config.method_limits.items():
            self.method_buckets[method] = TokenBucket(
                capacity=min(limit, self.config.burst_size),
                refill_rate=limit / 60
            )
        
        # Cleanup task
        self._cleanup_task = None
        
        logger.info("Rate limiter initialized")
    
    async def check_rate_limit(
        self,
        request: Dict[str, Any],
        identifier: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if request should be rate limited.
        
        Args:
            request: MCP request
            identifier: Client identifier (e.g., IP address)
            user_id: Authenticated user ID
            
        Returns:
            Tuple of (allowed, error_message)
        """
        method = request.get("method", "")
        
        # Generate identifier if not provided
        if not identifier:
            identifier = self._generate_identifier(request)
        
        # Check if locked out
        stats = self.stats[identifier]
        if stats.locked_until and datetime.now() < stats.locked_until:
            remaining = (stats.locked_until - datetime.now()).total_seconds()
            return False, f"Rate limit exceeded. Locked for {int(remaining)} seconds"
        
        # Check global rate limit
        if not await self.global_bucket.consume():
            await self._record_violation(identifier)
            return False, "Global rate limit exceeded"
        
        # Check method-specific rate limit
        if method in self.method_buckets:
            if not await self.method_buckets[method].consume():
                await self._record_violation(identifier)
                return False, f"Rate limit exceeded for method: {method}"
        
        # Check time-window based limits
        now = datetime.now()
        stats.requests.append(now)
        
        # Count requests in different time windows
        minute_ago = now - timedelta(minutes=1)
        hour_ago = now - timedelta(hours=1)
        day_ago = now - timedelta(days=1)
        
        requests_last_minute = sum(1 for t in stats.requests if t > minute_ago)
        requests_last_hour = sum(1 for t in stats.requests if t > hour_ago)
        requests_last_day = sum(1 for t in stats.requests if t > day_ago)
        
        # Check limits based on authentication
        if user_id:
            # Authenticated user limits
            if requests_last_minute > self.config.user_requests_per_minute:
                await self._record_violation(identifier)
                return False, "User rate limit exceeded (per minute)"
            
            if requests_last_hour > self.config.user_requests_per_hour:
                await self._record_violation(identifier)
                return False, "User rate limit exceeded (per hour)"
        else:
            # Unauthenticated limits
            if requests_last_minute > self.config.requests_per_minute:
                await self._record_violation(identifier)
                return False, "Rate limit exceeded (per minute)"
            
            if requests_last_hour > self.config.requests_per_hour:
                await self._record_violation(identifier)
                return False, "Rate limit exceeded (per hour)"
            
            if requests_last_day > self.config.requests_per_day:
                await self._record_violation(identifier)
                return False, "Rate limit exceeded (per day)"
        
        # IP-based limits (if enabled)
        if self.config.enable_ip_limiting and identifier.startswith("ip:"):
            if requests_last_minute > self.config.ip_requests_per_minute:
                await self._record_violation(identifier)
                return False, "IP rate limit exceeded"
        
        return True, None
    
    async def _record_violation(self, identifier: str) -> None:
        """Record a rate limit violation"""
        stats = self.stats[identifier]
        stats.violations += 1
        stats.last_violation = datetime.now()
        
        # Lock out if too many violations
        if stats.violations >= self.config.max_violations:
            stats.locked_until = datetime.now() + timedelta(
                seconds=self.config.lockout_duration_seconds
            )
            logger.warning(f"Rate limit lockout for {identifier}: {stats.violations} violations")
    
    def _generate_identifier(self, request: Dict[str, Any]) -> str:
        """Generate identifier from request"""
        # Try to extract IP from request context
        params = request.get("params", {})
        headers = params.get("headers", {})
        
        # Look for common IP headers
        ip = (
            headers.get("X-Forwarded-For") or
            headers.get("X-Real-IP") or
            headers.get("Remote-Addr") or
            "unknown"
        )
        
        # If no IP, use request signature
        if ip == "unknown":
            # Create signature from method and params
            sig_data = f"{request.get('method', '')}:{json.dumps(params, sort_keys=True)}"
            sig_hash = hashlib.sha256(sig_data.encode()).hexdigest()[:8]
            return f"sig:{sig_hash}"
        
        return f"ip:{ip}"
    
    async def get_stats(self, identifier: str) -> Dict[str, Any]:
        """Get rate limit statistics for an identifier"""
        stats = self.stats.get(identifier, RateLimitStats())
        now = datetime.now()
        
        # Count recent requests
        minute_ago = now - timedelta(minutes=1)
        hour_ago = now - timedelta(hours=1)
        
        requests_last_minute = sum(1 for t in stats.requests if t > minute_ago)
        requests_last_hour = sum(1 for t in stats.requests if t > hour_ago)
        
        return {
            "identifier": identifier,
            "requests_last_minute": requests_last_minute,
            "requests_last_hour": requests_last_hour,
            "violations": stats.violations,
            "last_violation": stats.last_violation.isoformat() if stats.last_violation else None,
            "locked_until": stats.locked_until.isoformat() if stats.locked_until else None,
            "tokens_available": self.global_bucket.available_tokens
        }
    
    async def reset_violations(self, identifier: str) -> None:
        """Reset violations for an identifier"""
        if identifier in self.stats:
            self.stats[identifier].violations = 0
            self.stats[identifier].locked_until = None
            logger.info(f"Reset violations for {identifier}")
    
    async def start_cleanup_task(self) -> None:
        """Start background cleanup task"""
        if not self._cleanup_task:
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
    
    async def stop_cleanup_task(self) -> None:
        """Stop background cleanup task"""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
            self._cleanup_task = None
    
    async def _cleanup_loop(self) -> None:
        """Periodically clean up old statistics"""
        while True:
            try:
                await asyncio.sleep(3600)  # Run every hour
                
                # Remove old entries
                cutoff = datetime.now() - timedelta(days=1)
                identifiers_to_remove = []
                
                for identifier, stats in self.stats.items():
                    # Remove if no recent requests and no active lockout
                    if (not stats.requests or 
                        max(stats.requests) < cutoff and 
                        (not stats.locked_until or stats.locked_until < datetime.now())):
                        identifiers_to_remove.append(identifier)
                
                for identifier in identifiers_to_remove:
                    del self.stats[identifier]
                
                if identifiers_to_remove:
                    logger.info(f"Cleaned up {len(identifiers_to_remove)} rate limit entries")
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in rate limit cleanup: {e}")


# Convenience function for middleware integration
async def rate_limit_middleware(
    request: Dict[str, Any],
    rate_limiter: RateLimiter,
    identifier: Optional[str] = None,
    user_id: Optional[str] = None
) -> Tuple[bool, Optional[str]]:
    """
    Rate limiting middleware function.
    
    Args:
        request: MCP request
        rate_limiter: RateLimiter instance
        identifier: Client identifier
        user_id: Authenticated user ID
        
    Returns:
        Tuple of (allowed, error_message)
    """
    return await rate_limiter.check_rate_limit(request, identifier, user_id)