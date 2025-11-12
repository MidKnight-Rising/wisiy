"""
Advanced configuration example for Smoke Runtime.

Shows how to fine-tune runtime parameters for specific use cases.
"""

import torch
from smoke_runtime import (
    SmokeRuntime,
    SmokeRuntimeConfig,
    MemoryConfig,
    DeviceConfig,
    RuntimeConfig
)

def configure_for_inference():
    """Configuration optimized for inference on limited hardware."""
    
    config = SmokeRuntimeConfig()
    
    # Memory configuration
    config.memory = MemoryConfig(
        gpu_memory="7.8GB",           # Leave headroom for CUDA overhead
        ram_cache="4GB",               # Larger RAM cache for better hit rate
        default_dtype="float16",       # Half precision
        enable_cpu_offload=False       # Disable CPU offload for speed
    )
    
    # Device configuration
    config.device = DeviceConfig(
        gpu_id=0,                      # Use first GPU
        num_gpus=1,
        force_gpu=True                 # Fail if can't fit on GPU
    )
    
    # Runtime configuration
    config.runtime = RuntimeConfig(
        prefetch_layers=5,             # Aggressive prefetching
        cache_policy="lru",            # LRU works well for sequential access
        async_prefetch=True,           # Async for better throughput
        smoke_test_enabled=True,       # Enable prediction
        smoke_test_layers=7,           # Look further ahead
        profile_memory=True            # Track memory usage
    )
    
    return config


def configure_for_memory_constrained():
    """Configuration for extremely limited memory (e.g., 4GB GPU)."""
    
    config = SmokeRuntimeConfig()
    
    config.memory = MemoryConfig(
        gpu_memory="3.8GB",            # Very conservative
        ram_cache="1GB",               # Smaller cache
        default_dtype="float16",
        enable_cpu_offload=True        # Allow CPU fallback
    )
    
    config.runtime = RuntimeConfig(
        prefetch_layers=2,             # Conservative prefetch
        cache_policy="lru",
        async_prefetch=False,          # Sync for predictability
        smoke_test_enabled=True,
        smoke_test_layers=3,           # Shorter lookahead
        profile_memory=True
    )
    
    return config


def configure_for_multi_gpu():
    """Configuration for multi-GPU setup."""
    
    config = SmokeRuntimeConfig()
    
    config.memory = MemoryConfig(
        gpu_memory="15GB",             # Per-GPU allocation
        ram_cache="8GB",
        default_dtype="float16"
    )
    
    config.device = DeviceConfig(
        gpu_id=0,
        num_gpus=2,                    # Use 2 GPUs
        force_gpu=False
    )
    
    config.runtime = RuntimeConfig(
        prefetch_layers=8,             # More aggressive with more memory
        cache_policy="lru",
        async_prefetch=True,
        smoke_test_enabled=True,
        smoke_test_layers=10
    )
    
    return config


def main():
    print("=== Inference Configuration ===")
    inference_config = configure_for_inference()
    print(inference_config.to_dict())
    
    print("\n=== Memory Constrained Configuration ===")
    constrained_config = configure_for_memory_constrained()
    print(constrained_config.to_dict())
    
    print("\n=== Multi-GPU Configuration ===")
    multi_gpu_config = configure_for_multi_gpu()
    print(multi_gpu_config.to_dict())
    
    # Example: Use one of the configurations
    runtime = SmokeRuntime(
        model_path="path/to/model",
        config=inference_config
    )
    
    print("\nRuntime initialized successfully!")


if __name__ == "__main__":
    main()
