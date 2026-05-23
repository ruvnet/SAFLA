"""
Optimized MCP Orchestration System for SAFLA Performance Enhancement
==================================================================

This module provides high-performance MCP (Model Context Protocol) orchestration
with advanced optimization techniques to meet strict performance targets:

- MCP round-trip latency: <5ms for standard operations
- Concurrent request handling: 100+ requests/second
- Connection pooling: Efficient resource management
- Async processing: Non-blocking operations

Optimization Techniques:
1. Connection pooling and reuse for reduced overhead
2. Asynchronous request processing with asyncio
3. Request batching for improved throughput
4. Intelligent caching of MCP responses
5. Circuit breaker pattern for fault tolerance
6. Request prioritization and queuing

Following TDD principles: These optimizations are designed to make
the performance benchmark tests pass.
"""

import asyncio
import time
import logging
import json
from typing import Dict, Any, List, Optional, Union, Callable, Awaitable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import aiohttp
import weakref
from collections import defaultdict, deque
from abc import ABC, abstractmethod
import hashlib
import threading
from concurrent.futures import ThreadPoolExecutor
import queue

logger = logging.getLogger(__name__)


class MCPRequestType(Enum):
    """Types of MCP requests."""
    ANALYZE_CODE = "analyze_code"
    MODIFY_CODE = "modify_code"
    SEARCH_CODE = "search_code"
    VALIDATE_CODE = "validate_code"
    OPTIMIZE_CODE = "optimize_code"
    GENERATE_TESTS = "generate_tests"


class MCPPriority(Enum):
    """Priority levels for MCP requests."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


class CircuitBreakerState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class MCPRequest:
    """MCP request with metadata."""
    request_id: str
    request_type: MCPRequestType
    payload: Dict[str, Any]
    priority: MCPPriority = MCPPriority.NORMAL
    timeout: float = 5.0
    created_at: datetime = field(default_factory=datetime.now)
    callback: Optional[Callable[[Any], None]] = None


@dataclass
class MCPResponse:
    """MCP response with metadata."""
    request_id: str
    success: bool
    data: Any
    error: Optional[str] = None
    latency: float = 0.0
    cached: bool = False
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class MCPMetrics:
    """Performance metrics for MCP operations."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_latency: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    circuit_breaker_trips: int = 0
    concurrent_requests: int = 0
    max_concurrent_requests: int = 0


class MCPCache:
    """
    High-performance cache for MCP responses with TTL and LRU eviction.
    """
    
    def __init__(self, max_size: int = 1000, default_ttl: float = 300.0):
        """Initialize MCP cache."""
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.access_order: deque = deque()
        self.lock = threading.RLock()
        
        logger.info(f"Initialized MCP cache with max_size={max_size}, ttl={default_ttl}s")
    
    def get(self, key: str) -> Optional[Any]:
        """Get cached response if valid."""
        with self.lock:
            if key not in self.cache:
                return None
            
            entry = self.cache[key]
            
            # Check TTL
            if time.time() > entry['expires_at']:
                del self.cache[key]
                if key in self.access_order:
                    self.access_order.remove(key)
                return None
            
            # Update access order for LRU
            if key in self.access_order:
                self.access_order.remove(key)
            self.access_order.append(key)
            
            return entry['data']
    
    def put(self, key: str, data: Any, ttl: Optional[float] = None) -> None:
        """Cache response with TTL."""
        with self.lock:
            # Use default TTL if not specified
            if ttl is None:
                ttl = self.default_ttl
            
            # Evict oldest entries if cache is full
            while len(self.cache) >= self.max_size and self.access_order:
                oldest_key = self.access_order.popleft()
                if oldest_key in self.cache:
                    del self.cache[oldest_key]
            
            # Add new entry
            self.cache[key] = {
                'data': data,
                'expires_at': time.time() + ttl,
                'created_at': time.time()
            }
            
            # Update access order
            if key in self.access_order:
                self.access_order.remove(key)
            self.access_order.append(key)
    
    def invalidate(self, pattern: Optional[str] = None) -> None:
        """Invalidate cache entries matching pattern."""
        with self.lock:
            if pattern is None:
                # Clear all
                self.cache.clear()
                self.access_order.clear()
            else:
                # Remove entries matching pattern
                keys_to_remove = [key for key in self.cache.keys() if pattern in key]
                for key in keys_to_remove:
                    del self.cache[key]
                    if key in self.access_order:
                        self.access_order.remove(key)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            return {
                'size': len(self.cache),
                'max_size': self.max_size,
                'hit_rate': 0.0,  # Would need to track hits/misses
                'oldest_entry': min(
                    (entry['created_at'] for entry in self.cache.values()),
                    default=time.time()
                )
            }


class CircuitBreaker:
    """
    Circuit breaker for MCP operations to handle failures gracefully.
    """
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exception: type = Exception
    ):
        """Initialize circuit breaker."""
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time = None
        self.state = CircuitBreakerState.CLOSED
        self.lock = threading.RLock()
        
        logger.info(f"Initialized circuit breaker with threshold={failure_threshold}, timeout={recovery_timeout}s")
    
    def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        with self.lock:
            if self.state == CircuitBreakerState.OPEN:
                if self._should_attempt_reset():
                    self.state = CircuitBreakerState.HALF_OPEN
                else:
                    raise Exception("Circuit breaker is OPEN")
            
            try:
                result = func(*args, **kwargs)
                self._on_success()
                return result
            
            except self.expected_exception as e:
                self._on_failure()
                raise e
    
    async def acall(self, func: Callable, *args, **kwargs):
        """Execute async function with circuit breaker protection."""
        with self.lock:
            if self.state == CircuitBreakerState.OPEN:
                if self._should_attempt_reset():
                    self.state = CircuitBreakerState.HALF_OPEN
                else:
                    raise Exception("Circuit breaker is OPEN")
        
        try:
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            
            self._on_success()
            return result
        
        except self.expected_exception as e:
            self._on_failure()
            raise e
    
    def _should_attempt_reset(self) -> bool:
        """Check if circuit breaker should attempt reset."""
        return (
            self.last_failure_time is not None and
            time.time() - self.last_failure_time >= self.recovery_timeout
        )
    
    def _on_success(self):
        """Handle successful operation."""
        with self.lock:
            self.failure_count = 0
            self.state = CircuitBreakerState.CLOSED
    
    def _on_failure(self):
        """Handle failed operation."""
        with self.lock:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.state = CircuitBreakerState.OPEN


class MCPConnectionPool:
    """
    Connection pool for MCP operations with automatic scaling.
    """
    
    def __init__(
        self,
        min_connections: int = 5,
        max_connections: int = 50,
        connection_timeout: float = 10.0
    ):
        """Initialize connection pool."""
        self.min_connections = min_connections
        self.max_connections = max_connections
        self.connection_timeout = connection_timeout
        
        self.available_connections: queue.Queue = queue.Queue()
        self.active_connections: weakref.WeakSet = weakref.WeakSet()
        self.total_connections = 0
        self.lock = threading.RLock()
        
        # Initialize minimum connections
        self._initialize_connections()
        
        logger.info(f"Initialized MCP connection pool with min={min_connections}, max={max_connections}")
    
    def _initialize_connections(self):
        """Initialize minimum number of connections."""
        for _ in range(self.min_connections):
            connection = self._create_connection()
            self.available_connections.put(connection)
            self.total_connections += 1
    
    def _create_connection(self) -> aiohttp.ClientSession:
        """Create a new connection."""
        timeout = aiohttp.ClientTimeout(total=self.connection_timeout)
        connector = aiohttp.TCPConnector(
            limit=100,
            limit_per_host=30,
            keepalive_timeout=30,
            enable_cleanup_closed=True
        )
        
        session = aiohttp.ClientSession(
            timeout=timeout,
            connector=connector,
            headers={'Content-Type': 'application/json'}
        )
        
        return session
    
    async def get_connection(self) -> aiohttp.ClientSession:
        """Get connection from pool."""
        try:
            # Try to get available connection
            connection = self.available_connections.get_nowait()
            self.active_connections.add(connection)
            return connection
        
        except queue.Empty:
            # Create new connection if under limit
            with self.lock:
                if self.total_connections < self.max_connections:
                    connection = self._create_connection()
                    self.total_connections += 1
                    self.active_connections.add(connection)
                    return connection
                else:
                    # Wait for available connection
                    connection = self.available_connections.get(timeout=5.0)
                    self.active_connections.add(connection)
                    return connection
    
    def return_connection(self, connection: aiohttp.ClientSession):
        """Return connection to pool."""
        if connection in self.active_connections:
            self.active_connections.discard(connection)
            
            # Only keep connection if not closed and under min limit
            if not connection.closed:
                try:
                    self.available_connections.put_nowait(connection)
                except queue.Full:
                    # Pool is full, close connection
                    asyncio.create_task(connection.close())
                    with self.lock:
                        self.total_connections -= 1
    
    async def close_all(self):
        """Close all connections in pool."""
        # Close available connections
        while not self.available_connections.empty():
            try:
                connection = self.available_connections.get_nowait()
                await connection.close()
            except queue.Empty:
                break
        
        # Close active connections
        for connection in list(self.active_connections):
            await connection.close()
        
        self.total_connections = 0


class OptimizedMCPOrchestrator:
    """
    High-performance MCP orchestrator with advanced optimization techniques.
    
    Designed to meet strict performance targets:
    - MCP round-trip latency: <5ms for standard operations
    - Concurrent request handling: 100+ requests/second
    - Intelligent caching and connection pooling
    """
    
    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        max_concurrent_requests: int = 100,
        request_timeout: float = 5.0,
        cache_size: int = 1000,
        cache_ttl: float = 300.0
    ):
        """Initialize optimized MCP orchestrator."""
        self.base_url = base_url
        self.max_concurrent_requests = max_concurrent_requests
        self.request_timeout = request_timeout
        
        # Initialize components
        self.cache = MCPCache(max_size=cache_size, default_ttl=cache_ttl)
        self.connection_pool = MCPConnectionPool()
        self.circuit_breaker = CircuitBreaker()
        
        # Request management
        self.request_queue: asyncio.Queue = asyncio.Queue()
        self.priority_queues: Dict[MCPPriority, asyncio.Queue] = {
            priority: asyncio.Queue() for priority in MCPPriority
        }
        
        # Performance tracking
        self.metrics = MCPMetrics()
        self.active_requests: Dict[str, MCPRequest] = {}
        self.request_semaphore = asyncio.Semaphore(max_concurrent_requests)
        
        # Background tasks
        self.worker_tasks: List[asyncio.Task] = []
        self.is_running = False
        
        logger.info(f"Initialized OptimizedMCPOrchestrator with base_url={base_url}")
    
    async def start(self):
        """Start the MCP orchestrator."""
        if self.is_running:
            return
        
        self.is_running = True
        
        # Start worker tasks
        num_workers = min(10, self.max_concurrent_requests // 10)
        for i in range(num_workers):
            task = asyncio.create_task(self._worker_loop(f"worker-{i}"))
            self.worker_tasks.append(task)
        
        logger.info(f"Started MCP orchestrator with {num_workers} workers")
    
    async def stop(self):
        """Stop the MCP orchestrator."""
        if not self.is_running:
            return
        
        self.is_running = False
        
        # Cancel worker tasks
        for task in self.worker_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        await asyncio.gather(*self.worker_tasks, return_exceptions=True)
        
        # Close connection pool
        await self.connection_pool.close_all()
        
        logger.info("Stopped MCP orchestrator")
    
    async def execute_request(
        self,
        request_type: MCPRequestType,
        payload: Dict[str, Any],
        priority: MCPPriority = MCPPriority.NORMAL,
        timeout: Optional[float] = None,
        use_cache: bool = True
    ) -> MCPResponse:
        """
        Execute MCP request with optimization.
        
        Returns response with performance metrics.
        """
        start_time = time.perf_counter()
        
        # Generate request ID
        request_id = self._generate_request_id(request_type, payload)
        
        # Check cache first
        if use_cache:
            cache_key = self._generate_cache_key(request_type, payload)
            cached_response = self.cache.get(cache_key)
            
            if cached_response is not None:
                self.metrics.cache_hits += 1
                
                return MCPResponse(
                    request_id=request_id,
                    success=True,
                    data=cached_response,
                    latency=time.perf_counter() - start_time,
                    cached=True
                )
        
        self.metrics.cache_misses += 1
        
        # Create request
        request = MCPRequest(
            request_id=request_id,
            request_type=request_type,
            payload=payload,
            priority=priority,
            timeout=timeout or self.request_timeout
        )
        
        # Execute request
        try:
            response = await self._execute_request_internal(request)
            
            # Cache successful responses
            if use_cache and response.success:
                cache_key = self._generate_cache_key(request_type, payload)
                self.cache.put(cache_key, response.data)
            
            return response
        
        except Exception as e:
            logger.error(f"MCP request failed: {e}")
            
            return MCPResponse(
                request_id=request_id,
                success=False,
                data=None,
                error=str(e),
                latency=time.perf_counter() - start_time
            )
    
    async def batch_execute(
        self,
        requests: List[Dict[str, Any]],
        max_concurrent: Optional[int] = None
    ) -> List[MCPResponse]:
        """Execute multiple MCP requests concurrently."""
        if max_concurrent is None:
            max_concurrent = min(len(requests), self.max_concurrent_requests)
        
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def execute_single(request_data):
            async with semaphore:
                return await self.execute_request(**request_data)
        
        # Execute all requests concurrently
        tasks = [execute_single(req) for req in requests]
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Convert exceptions to error responses
        results = []
        for i, response in enumerate(responses):
            if isinstance(response, Exception):
                results.append(MCPResponse(
                    request_id=f"batch-{i}",
                    success=False,
                    data=None,
                    error=str(response)
                ))
            else:
                results.append(response)
        
        return results
    
    async def _execute_request_internal(self, request: MCPRequest) -> MCPResponse:
        """Execute MCP request with circuit breaker protection."""
        start_time = time.perf_counter()
        
        # Update metrics
        self.metrics.total_requests += 1
        self.metrics.concurrent_requests += 1
        self.metrics.max_concurrent_requests = max(
            self.metrics.max_concurrent_requests,
            self.metrics.concurrent_requests
        )
        
        try:
            # Acquire semaphore for concurrency control
            async with self.request_semaphore:
                # Execute with circuit breaker
                response_data = await self.circuit_breaker.acall(
                    self._make_http_request,
                    request
                )
                
                # Update metrics
                self.metrics.successful_requests += 1
                latency = time.perf_counter() - start_time
                self.metrics.total_latency += latency
                
                return MCPResponse(
                    request_id=request.request_id,
                    success=True,
                    data=response_data,
                    latency=latency
                )
        
        except Exception as e:
            # Update metrics
            self.metrics.failed_requests += 1
            latency = time.perf_counter() - start_time
            self.metrics.total_latency += latency
            
            raise e
        
        finally:
            self.metrics.concurrent_requests -= 1
    
    async def _make_http_request(self, request: MCPRequest) -> Any:
        """Make HTTP request to MCP server."""
        connection = await self.connection_pool.get_connection()
        
        try:
            # Prepare request data
            url = f"{self.base_url}/mcp/{request.request_type.value}"
            
            # Make request with timeout
            async with connection.post(
                url,
                json=request.payload,
                timeout=aiohttp.ClientTimeout(total=request.timeout)
            ) as response:
                
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    raise Exception(f"HTTP {response.status}: {error_text}")
        
        finally:
            self.connection_pool.return_connection(connection)
    
    async def _worker_loop(self, worker_id: str):
        """Background worker loop for processing requests."""
        logger.info(f"Started MCP worker: {worker_id}")
        
        while self.is_running:
            try:
                # Process priority queues in order
                request = None
                
                for priority in sorted(MCPPriority, key=lambda p: p.value, reverse=True):
                    try:
                        request = self.priority_queues[priority].get_nowait()
                        break
                    except asyncio.QueueEmpty:
                        continue
                
                if request is None:
                    # No priority requests, check main queue
                    try:
                        request = await asyncio.wait_for(
                            self.request_queue.get(),
                            timeout=1.0
                        )
                    except asyncio.TimeoutError:
                        continue
                
                # Process request
                if request:
                    await self._process_request(request)
            
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}")
                await asyncio.sleep(0.1)
        
        logger.info(f"Stopped MCP worker: {worker_id}")
    
    async def _process_request(self, request: MCPRequest):
        """Process individual MCP request."""
        try:
            response = await self._execute_request_internal(request)
            
            # Call callback if provided
            if request.callback:
                request.callback(response)
        
        except Exception as e:
            logger.error(f"Failed to process request {request.request_id}: {e}")
            
            # Call callback with error
            if request.callback:
                error_response = MCPResponse(
                    request_id=request.request_id,
                    success=False,
                    data=None,
                    error=str(e)
                )
                request.callback(error_response)
    
    def _generate_request_id(self, request_type: MCPRequestType, payload: Dict[str, Any]) -> str:
        """Generate unique request ID."""
        content = f"{request_type.value}:{json.dumps(payload, sort_keys=True)}"
        hash_obj = hashlib.sha256(content.encode())
        return f"{request_type.value}:{hash_obj.hexdigest()[:8]}"
    
    def _generate_cache_key(self, request_type: MCPRequestType, payload: Dict[str, Any]) -> str:
        """Generate cache key for request."""
        # Create deterministic key from request type and payload
        content = f"{request_type.value}:{json.dumps(payload, sort_keys=True)}"
        hash_obj = hashlib.sha256(content.encode())
        return hash_obj.hexdigest()
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for MCP orchestrator."""
        avg_latency = (
            self.metrics.total_latency / self.metrics.total_requests
            if self.metrics.total_requests > 0 else 0.0
        )
        
        success_rate = (
            self.metrics.successful_requests / self.metrics.total_requests
            if self.metrics.total_requests > 0 else 0.0
        )
        
        cache_hit_rate = (
            self.metrics.cache_hits / (self.metrics.cache_hits + self.metrics.cache_misses)
            if (self.metrics.cache_hits + self.metrics.cache_misses) > 0 else 0.0
        )
        
        return {
            'total_requests': self.metrics.total_requests,
            'successful_requests': self.metrics.successful_requests,
            'failed_requests': self.metrics.failed_requests,
            'success_rate': success_rate,
            'average_latency': avg_latency,
            'cache_hit_rate': cache_hit_rate,
            'concurrent_requests': self.metrics.concurrent_requests,
            'max_concurrent_requests': self.metrics.max_concurrent_requests,
            'circuit_breaker_trips': self.metrics.circuit_breaker_trips
        }
    
    def reset_metrics(self):
        """Reset performance metrics."""
        self.metrics = MCPMetrics()


# Compatibility aliases for backward compatibility with existing code
MCPOrchestrator = OptimizedMCPOrchestrator

# Mock classes for compatibility - these would be replaced with actual implementations
class MCPServer:
    """Compatibility class for MCPServer."""
    def __init__(self, *args, **kwargs):
        pass

class MCPServerStatus:
    """Compatibility class for MCPServerStatus."""
    ONLINE = "online"
    OFFLINE = "offline"
    ERROR = "error"

class MCPServerManager:
    """Compatibility class for MCPServerManager."""
    def __init__(self, *args, **kwargs):
        pass

class Agent:
    """Compatibility class for Agent."""
    def __init__(self, *args, **kwargs):
        pass

class ContextVector:
    """Compatibility class for ContextVector."""
    def __init__(self, *args, **kwargs):
        pass

class ContextSharingEngine:
    """Compatibility class for ContextSharingEngine."""
    def __init__(self, *args, **kwargs):
        pass