"""
Configuration module for Smoke Runtime.

Manages memory limits, device mappings, and runtime parameters.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Literal
import torch


@dataclass
class MemoryConfig:
    """Memory configuration for different devices.
    
    Args:
        gpu_memory: Maximum GPU memory to use (e.g., "7.8GB", "8192MB")
        ram_cache: Maximum RAM to use for weight caching (e.g., "2GB")
        disk_cache: Optional disk cache location for intermediate storage
        default_dtype: Default dtype for model weights ("float16", "float32", "bfloat16")
        enable_cpu_offload: Whether to allow CPU offloading for overflow
    """
    gpu_memory: str = "7.8GB"
    ram_cache: str = "2GB"
    disk_cache: Optional[str] = None
    default_dtype: Literal["float16", "float32", "bfloat16"] = "float16"
    enable_cpu_offload: bool = False
    
    def __post_init__(self):
        """Validate and normalize configuration."""
        self.gpu_bytes = self._parse_memory(self.gpu_memory)
        self.ram_bytes = self._parse_memory(self.ram_cache)
        
        # Map dtype string to torch dtype
        dtype_map = {
            "float16": torch.float16,
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
        }
        self.torch_dtype = dtype_map[self.default_dtype]
    
    @staticmethod
    def _parse_memory(memory_str: str) -> int:
        """Parse memory string to bytes.
        
        Args:
            memory_str: Memory string like "7.8GB", "2048MB"
            
        Returns:
            Memory in bytes
        """
        memory_str = memory_str.strip().upper()
        
        units = {
            'KB': 1024,
            'MB': 1024 ** 2,
            'GB': 1024 ** 3,
            'TB': 1024 ** 4,
        }
        
        for unit, multiplier in units.items():
            if memory_str.endswith(unit):
                value = float(memory_str[:-len(unit)])
                return int(value * multiplier)
        
        # Assume bytes if no unit specified
        return int(memory_str)
    
    def get_max_memory_dict(self) -> Dict[str, str]:
        """Get max_memory dict for model loading.
        
        Returns:
            Dict mapping device to memory limit
        """
        max_memory = {
            0: self.gpu_memory,  # GPU 0
        }
        
        if self.enable_cpu_offload:
            max_memory["cpu"] = self.ram_cache
        
        return max_memory


@dataclass
class DeviceConfig:
    """Device mapping configuration.
    
    Args:
        gpu_id: GPU device ID to use
        num_gpus: Number of GPUs available
        force_gpu: Force all operations on GPU (fail if OOM)
    """
    gpu_id: int = 0
    num_gpus: int = 1
    force_gpu: bool = False
    
    @property
    def device(self) -> torch.device:
        """Get primary torch device."""
        if torch.cuda.is_available():
            return torch.device(f"cuda:{self.gpu_id}")
        return torch.device("cpu")
    
    def get_device_map(self, num_layers: int) -> Dict[str, int]:
        """Generate custom device map for model layers.
        
        Args:
            num_layers: Total number of layers in the model
            
        Returns:
            Dict mapping layer names to device IDs
        """
        # This will be implemented based on smoke test predictions
        # For now, return a simple mapping
        device_map = {}
        
        if self.force_gpu or not torch.cuda.is_available():
            return "auto"
        
        # Map all layers to primary GPU initially
        # Smoke runtime will override this dynamically
        for i in range(num_layers):
            device_map[f"layer.{i}"] = self.gpu_id
        
        return device_map


@dataclass
class RuntimeConfig:
    """Runtime behavior configuration.
    
    Args:
        prefetch_layers: Number of layers to prefetch ahead
        cache_policy: Cache eviction policy ("lru", "lfu", "fifo")
        async_prefetch: Enable asynchronous prefetching
        smoke_test_enabled: Enable smoke test simulation
        smoke_test_layers: Number of layers to simulate ahead
        profile_memory: Enable memory profiling and logging
    """
    prefetch_layers: int = 3
    cache_policy: Literal["lru", "lfu", "fifo"] = "lru"
    async_prefetch: bool = True
    smoke_test_enabled: bool = True
    smoke_test_layers: int = 5
    profile_memory: bool = False
    
    # Advanced options
    checkpoint_intermediate: bool = False
    use_gradient_checkpointing: bool = False
    stream_attention: bool = False


@dataclass
class SmokeRuntimeConfig:
    """Complete configuration for Smoke Runtime.
    
    Combines all configuration aspects into a single config object.
    """
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    device: DeviceConfig = field(default_factory=DeviceConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    
    @classmethod
    def from_dict(cls, config_dict: Dict) -> "SmokeRuntimeConfig":
        """Create config from dictionary.
        
        Args:
            config_dict: Configuration dictionary
            
        Returns:
            SmokeRuntimeConfig instance
        """
        return cls(
            memory=MemoryConfig(**config_dict.get("memory", {})),
            device=DeviceConfig(**config_dict.get("device", {})),
            runtime=RuntimeConfig(**config_dict.get("runtime", {})),
        )
    
    def to_dict(self) -> Dict:
        """Convert config to dictionary.
        
        Returns:
            Configuration as dictionary
        """
        return {
            "memory": self.memory.__dict__,
            "device": self.device.__dict__,
            "runtime": self.runtime.__dict__,
        }
