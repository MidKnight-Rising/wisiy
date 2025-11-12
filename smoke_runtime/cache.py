"""
Weight cache module implementing LRU/LFU/FIFO eviction policies.

Manages RAM-based caching of model weights with automatic eviction.
"""

from collections import OrderedDict
from typing import Any, Dict, Optional, Literal
import threading
import torch
import logging

logger = logging.getLogger(__name__)


class WeightCache:
    """RAM-based cache for model weights with configurable eviction policy.
    
    Supports LRU (Least Recently Used), LFU (Least Frequently Used), 
    and FIFO (First In First Out) eviction policies.
    
    Args:
        max_memory_bytes: Maximum memory in bytes for cache
        policy: Eviction policy ("lru", "lfu", "fifo")
    """
    
    def __init__(
        self,
        max_memory_bytes: int,
        policy: Literal["lru", "lfu", "fifo"] = "lru"
    ):
        self.max_memory = max_memory_bytes
        self.policy = policy
        self.current_memory = 0
        
        # Cache storage: key -> (tensor, metadata)
        self._cache: OrderedDict[str, tuple[torch.Tensor, Dict]] = OrderedDict()
        
        # Access tracking for LFU
        self._access_counts: Dict[str, int] = {}
        
        # Thread safety
        self._lock = threading.RLock()
        
        logger.info(
            f"Initialized WeightCache with {max_memory_bytes / (1024**3):.2f}GB "
            f"capacity, policy={policy}"
        )
    
    def get(self, key: str) -> Optional[torch.Tensor]:
        """Retrieve weight from cache.
        
        Args:
            key: Unique identifier for the weight
            
        Returns:
            Cached tensor or None if not found
        """
        with self._lock:
            if key not in self._cache:
                return None
            
            tensor, metadata = self._cache[key]
            
            # Update access tracking based on policy
            if self.policy == "lru":
                # Move to end (most recent)
                self._cache.move_to_end(key)
            elif self.policy == "lfu":
                self._access_counts[key] = self._access_counts.get(key, 0) + 1
            
            logger.debug(f"Cache HIT: {key}")
            return tensor
    
    def put(self, key: str, tensor: torch.Tensor, metadata: Optional[Dict] = None) -> bool:
        """Store weight in cache with automatic eviction if needed.
        
        Args:
            key: Unique identifier for the weight
            tensor: Weight tensor to cache
            metadata: Optional metadata (dtype, shape, etc.)
            
        Returns:
            True if successfully cached, False otherwise
        """
        with self._lock:
            if metadata is None:
                metadata = {}
            
            tensor_size = tensor.element_size() * tensor.nelement()
            
            # Check if tensor is too large for cache
            if tensor_size > self.max_memory:
                logger.warning(
                    f"Tensor {key} ({tensor_size / (1024**2):.2f}MB) "
                    f"exceeds cache capacity"
                )
                return False
            
            # Evict until there's space
            while self.current_memory + tensor_size > self.max_memory:
                if not self._evict_one():
                    logger.error("Failed to evict, cache may be corrupted")
                    return False
            
            # Remove old entry if updating
            if key in self._cache:
                old_tensor, _ = self._cache[key]
                old_size = old_tensor.element_size() * old_tensor.nelement()
                self.current_memory -= old_size
            
            # Add to cache
            self._cache[key] = (tensor, metadata)
            self.current_memory += tensor_size
            
            if self.policy == "lfu":
                self._access_counts[key] = 1
            
            logger.debug(
                f"Cache PUT: {key} ({tensor_size / (1024**2):.2f}MB), "
                f"usage={self.current_memory / self.max_memory * 100:.1f}%"
            )
            
            return True
    
    def _evict_one(self) -> bool:
        """Evict one entry based on the current policy.
        
        Returns:
            True if eviction successful, False if cache is empty
        """
        if not self._cache:
            return False
        
        # Select key to evict based on policy
        if self.policy == "lru":
            # First item is least recently used
            key_to_evict = next(iter(self._cache))
        elif self.policy == "fifo":
            # First item is oldest
            key_to_evict = next(iter(self._cache))
        elif self.policy == "lfu":
            # Find least frequently used
            key_to_evict = min(
                self._cache.keys(),
                key=lambda k: self._access_counts.get(k, 0)
            )
        else:
            key_to_evict = next(iter(self._cache))
        
        # Remove from cache
        tensor, _ = self._cache.pop(key_to_evict)
        tensor_size = tensor.element_size() * tensor.nelement()
        self.current_memory -= tensor_size
        
        if self.policy == "lfu":
            self._access_counts.pop(key_to_evict, None)
        
        logger.debug(
            f"Cache EVICT: {key_to_evict} ({tensor_size / (1024**2):.2f}MB)"
        )
        
        return True
    
    def clear(self):
        """Clear all cached weights."""
        with self._lock:
            self._cache.clear()
            self._access_counts.clear()
            self.current_memory = 0
            logger.info("Cache cleared")
    
    def contains(self, key: str) -> bool:
        """Check if key is in cache.
        
        Args:
            key: Weight identifier
            
        Returns:
            True if key is cached
        """
        with self._lock:
            return key in self._cache
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        with self._lock:
            return {
                "size_bytes": self.current_memory,
                "size_gb": self.current_memory / (1024 ** 3),
                "capacity_bytes": self.max_memory,
                "capacity_gb": self.max_memory / (1024 ** 3),
                "utilization": self.current_memory / self.max_memory,
                "num_entries": len(self._cache),
                "policy": self.policy,
            }
    
    def __len__(self) -> int:
        """Get number of cached entries."""
        with self._lock:
            return len(self._cache)
    
    def __contains__(self, key: str) -> bool:
        """Check if key is in cache."""
        return self.contains(key)
