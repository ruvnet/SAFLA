"""
State manager for SAFLA MCP server.

This module provides centralized state management for the MCP server,
handling shared state across different handlers and maintaining consistency.
"""

import asyncio
import time
from typing import Any, Dict, Optional, Set
from threading import RLock
import logging
from datetime import datetime
from collections import defaultdict

logger = logging.getLogger(__name__)


class StateManager:
    """
    Centralized state manager for the MCP server.
    
    Provides thread-safe state management with support for:
    - Key-value storage
    - State namespaces
    - State expiration
    - Change notifications
    - State persistence (optional)
    """
    
    def __init__(self, enable_persistence: bool = False, persistence_path: Optional[str] = None):
        """
        Initialize the state manager.
        
        Args:
            enable_persistence: Whether to enable state persistence
            persistence_path: Path for state persistence
        """
        self._state: Dict[str, Dict[str, Any]] = defaultdict(dict)
        self._lock = RLock()
        self._expiration: Dict[str, Dict[str, float]] = defaultdict(dict)
        self._listeners: Dict[str, Set[callable]] = defaultdict(set)
        self._enable_persistence = enable_persistence
        self._persistence_path = persistence_path
        
        # Initialize default namespaces
        self._initialize_namespaces()
        
        # Start cleanup task
        self._cleanup_task = None
        self._start_cleanup_task()
    
    def _initialize_namespaces(self):
        """Initialize default state namespaces."""
        namespaces = [
            "config",
            "benchmarks",
            "sessions",
            "metacognitive",
            "agents",
            "deployment",
            "performance"
        ]
        
        for namespace in namespaces:
            self._state[namespace] = {}
            logger.debug(f"Initialized namespace: {namespace}")
    
    def get(self, key: str, namespace: str = "default", default: Any = None) -> Any:
        """
        Get a value from state.
        
        Args:
            key: State key
            namespace: State namespace
            default: Default value if key not found
            
        Returns:
            State value or default
        """
        with self._lock:
            # Check if expired
            if self._is_expired(namespace, key):
                self._remove_expired(namespace, key)
                return default
            
            return self._state[namespace].get(key, default)
    
    def set(self, key: str, value: Any, namespace: str = "default", 
            ttl: Optional[int] = None) -> None:
        """
        Set a value in state.
        
        Args:
            key: State key
            value: State value
            namespace: State namespace
            ttl: Time to live in seconds (optional)
        """
        with self._lock:
            old_value = self._state[namespace].get(key)
            self._state[namespace][key] = value
            
            # Set expiration if TTL provided
            if ttl:
                self._expiration[namespace][key] = time.time() + ttl
            elif key in self._expiration[namespace]:
                # Remove expiration if no TTL
                del self._expiration[namespace][key]
            
            # Notify listeners
            self._notify_listeners(namespace, key, old_value, value)
            
            # Persist if enabled
            if self._enable_persistence:
                self._persist_state()
    
    def delete(self, key: str, namespace: str = "default") -> bool:
        """
        Delete a key from state.
        
        Args:
            key: State key
            namespace: State namespace
            
        Returns:
            True if key existed and was deleted
        """
        with self._lock:
            if key in self._state[namespace]:
                old_value = self._state[namespace][key]
                del self._state[namespace][key]
                
                # Remove expiration
                if key in self._expiration[namespace]:
                    del self._expiration[namespace][key]
                
                # Notify listeners
                self._notify_listeners(namespace, key, old_value, None)
                
                # Persist if enabled
                if self._enable_persistence:
                    self._persist_state()
                
                return True
            return False
    
    def exists(self, key: str, namespace: str = "default") -> bool:
        """Check if a key exists in state."""
        with self._lock:
            if self._is_expired(namespace, key):
                self._remove_expired(namespace, key)
                return False
            return key in self._state[namespace]
    
    def get_namespace(self, namespace: str) -> Dict[str, Any]:
        """
        Get all values in a namespace.
        
        Args:
            namespace: State namespace
            
        Returns:
            Dictionary of all key-value pairs in namespace
        """
        with self._lock:
            # Remove expired keys first
            self._cleanup_namespace(namespace)
            return dict(self._state[namespace])
    
    def clear_namespace(self, namespace: str) -> None:
        """Clear all values in a namespace."""
        with self._lock:
            self._state[namespace].clear()
            self._expiration[namespace].clear()
            
            # Persist if enabled
            if self._enable_persistence:
                self._persist_state()
    
    def add_listener(self, namespace: str, callback: callable) -> None:
        """
        Add a state change listener.
        
        Args:
            namespace: Namespace to listen to
            callback: Callback function (namespace, key, old_value, new_value)
        """
        self._listeners[namespace].add(callback)
    
    def remove_listener(self, namespace: str, callback: callable) -> None:
        """Remove a state change listener."""
        self._listeners[namespace].discard(callback)
    
    def _is_expired(self, namespace: str, key: str) -> bool:
        """Check if a key has expired."""
        if key in self._expiration[namespace]:
            return time.time() > self._expiration[namespace][key]
        return False
    
    def _remove_expired(self, namespace: str, key: str) -> None:
        """Remove an expired key."""
        if key in self._state[namespace]:
            del self._state[namespace][key]
        if key in self._expiration[namespace]:
            del self._expiration[namespace][key]
    
    def _cleanup_namespace(self, namespace: str) -> None:
        """Remove all expired keys from a namespace."""
        expired_keys = []
        for key in self._expiration[namespace]:
            if self._is_expired(namespace, key):
                expired_keys.append(key)
        
        for key in expired_keys:
            self._remove_expired(namespace, key)
    
    def _notify_listeners(self, namespace: str, key: str, 
                         old_value: Any, new_value: Any) -> None:
        """Notify listeners of state change."""
        for callback in self._listeners[namespace]:
            try:
                callback(namespace, key, old_value, new_value)
            except Exception as e:
                logger.error(f"Error in state listener: {str(e)}")
    
    def _persist_state(self) -> None:
        """Persist state to disk (if enabled)."""
        if not self._persistence_path:
            return
        
        try:
            import json
            from pathlib import Path
            
            # Prepare state for serialization
            state_data = {
                "timestamp": datetime.utcnow().isoformat(),
                "state": dict(self._state),
                "expiration": dict(self._expiration)
            }
            
            # Write to temporary file first
            temp_path = Path(self._persistence_path).with_suffix('.tmp')
            with open(temp_path, 'w') as f:
                json.dump(state_data, f, indent=2, default=str)
            
            # Atomic rename
            temp_path.rename(self._persistence_path)
            
        except Exception as e:
            logger.error(f"Failed to persist state: {str(e)}")
    
    def _start_cleanup_task(self) -> None:
        """Start the background cleanup task."""
        async def cleanup_loop():
            while True:
                try:
                    await asyncio.sleep(60)  # Cleanup every minute
                    with self._lock:
                        for namespace in list(self._state.keys()):
                            self._cleanup_namespace(namespace)
                except Exception as e:
                    logger.error(f"Error in cleanup task: {str(e)}")
        
        # Start cleanup task in background
        try:
            loop = asyncio.get_event_loop()
            self._cleanup_task = loop.create_task(cleanup_loop())
        except RuntimeError:
            # No event loop running yet
            pass
    
    def get_stats(self) -> Dict[str, Any]:
        """Get state manager statistics."""
        with self._lock:
            stats = {
                "namespaces": len(self._state),
                "total_keys": sum(len(ns) for ns in self._state.values()),
                "expiring_keys": sum(len(ns) for ns in self._expiration.values()),
                "listeners": sum(len(ls) for ls in self._listeners.values()),
                "namespace_stats": {}
            }
            
            for namespace in self._state:
                stats["namespace_stats"][namespace] = {
                    "keys": len(self._state[namespace]),
                    "expiring": len(self._expiration[namespace]),
                    "listeners": len(self._listeners[namespace])
                }
            
            return stats