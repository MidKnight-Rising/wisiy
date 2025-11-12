"""Tests for cache module."""

import pytest
import torch
from smoke_runtime.cache import WeightCache


class TestWeightCache:
    """Test WeightCache class."""
    
    def test_initialization(self):
        """Test cache initialization."""
        cache = WeightCache(
            max_memory_bytes=1024 * 1024,  # 1MB
            policy="lru"
        )
        assert cache.max_memory == 1024 * 1024
        assert cache.current_memory == 0
        assert cache.policy == "lru"
        assert len(cache) == 0
    
    def test_put_and_get(self):
        """Test putting and getting tensors."""
        cache = WeightCache(max_memory_bytes=10 * 1024 * 1024, policy="lru")
        
        tensor = torch.randn(100, 100)
        cache.put("test_key", tensor)
        
        retrieved = cache.get("test_key")
        assert retrieved is not None
        assert torch.equal(retrieved, tensor)
    
    def test_get_nonexistent(self):
        """Test getting non-existent key."""
        cache = WeightCache(max_memory_bytes=1024 * 1024, policy="lru")
        
        result = cache.get("nonexistent")
        assert result is None
    
    def test_contains(self):
        """Test cache membership check."""
        cache = WeightCache(max_memory_bytes=10 * 1024 * 1024, policy="lru")
        
        tensor = torch.randn(10, 10)
        cache.put("key1", tensor)
        
        assert cache.contains("key1")
        assert "key1" in cache
        assert not cache.contains("key2")
        assert "key2" not in cache
    
    def test_eviction_when_full(self):
        """Test automatic eviction when cache is full."""
        # Small cache: 20KB
        cache = WeightCache(max_memory_bytes=20 * 1024, policy="lru")
        
        # Add tensors that will fill the cache
        # Each tensor is ~8KB (2000 floats * 4 bytes)
        for i in range(5):
            tensor = torch.randn(2000)
            cache.put(f"key{i}", tensor)
        
        # First 2 keys should be evicted
        assert not cache.contains("key0")
        assert not cache.contains("key1")
        # Later keys should still be there
        assert cache.contains("key4")
    
    def test_lru_policy(self):
        """Test LRU eviction policy."""
        # Cache that can hold 2 tensors comfortably
        cache = WeightCache(max_memory_bytes=18 * 1024, policy="lru")
        
        # Add 2 tensors (~8KB each)
        for i in range(2):
            tensor = torch.randn(2000)
            cache.put(f"key{i}", tensor)
        
        # Verify both are in cache
        assert cache.contains("key0")
        assert cache.contains("key1")
        
        # Access key0 to make it recently used
        cache.get("key0")
        
        # Add one more tensor to trigger eviction
        tensor = torch.randn(2000)
        cache.put("key2", tensor)
        
        # key0 should still be there (recently accessed)
        assert cache.contains("key0")
        # key1 should be evicted (least recently used)
        assert not cache.contains("key1")
        # key2 should be there (just added)
        assert cache.contains("key2")
    
    def test_lfu_policy(self):
        """Test LFU eviction policy."""
        # Cache that can hold 2 tensors comfortably
        cache = WeightCache(max_memory_bytes=18 * 1024, policy="lfu")
        
        # Add 2 tensors (~8KB each)
        for i in range(2):
            tensor = torch.randn(2000)
            cache.put(f"key{i}", tensor)
        
        # Verify both are in cache
        assert cache.contains("key0")
        assert cache.contains("key1")
        
        # Access key0 multiple times
        for _ in range(5):
            cache.get("key0")
        
        # key1 is never accessed (frequency = 1 from initial put)
        
        # Add one more tensor to trigger eviction
        tensor = torch.randn(2000)
        cache.put("key2", tensor)
        
        # key0 should still be there (most frequently used with 5 accesses)
        assert cache.contains("key0")
        # key1 should be evicted (least frequently used - only initial put)
        assert not cache.contains("key1")
        # key2 should be there (just added)
        assert cache.contains("key2")
    
    def test_clear(self):
        """Test cache clearing."""
        cache = WeightCache(max_memory_bytes=10 * 1024 * 1024, policy="lru")
        
        for i in range(3):
            tensor = torch.randn(100, 100)
            cache.put(f"key{i}", tensor)
        
        assert len(cache) == 3
        assert cache.current_memory > 0
        
        cache.clear()
        
        assert len(cache) == 0
        assert cache.current_memory == 0
    
    def test_update_existing_key(self):
        """Test updating an existing key."""
        cache = WeightCache(max_memory_bytes=10 * 1024 * 1024, policy="lru")
        
        tensor1 = torch.randn(100, 100)
        cache.put("key1", tensor1)
        
        tensor2 = torch.randn(200, 200)
        cache.put("key1", tensor2)
        
        retrieved = cache.get("key1")
        assert torch.equal(retrieved, tensor2)
        assert not torch.equal(retrieved, tensor1)
    
    def test_tensor_too_large(self):
        """Test handling tensor larger than cache capacity."""
        cache = WeightCache(max_memory_bytes=1024, policy="lru")  # Very small
        
        large_tensor = torch.randn(1000, 1000)  # Much larger than cache
        result = cache.put("large", large_tensor)
        
        assert not result  # Should fail
        assert not cache.contains("large")
    
    def test_get_stats(self):
        """Test getting cache statistics."""
        cache = WeightCache(max_memory_bytes=10 * 1024 * 1024, policy="lru")
        
        tensor = torch.randn(100, 100)
        cache.put("key1", tensor)
        
        stats = cache.get_stats()
        
        assert "size_bytes" in stats
        assert "capacity_bytes" in stats
        assert "utilization" in stats
        assert "num_entries" in stats
        assert "policy" in stats
        
        assert stats["num_entries"] == 1
        assert stats["policy"] == "lru"
        assert stats["size_bytes"] > 0
