"""
Basic usage example for Smoke Runtime.

Demonstrates how to set up and use the smoke runtime for
memory-efficient model inference.
"""

import torch
from smoke_runtime import SmokeRuntime, SmokeRuntimeConfig, MemoryConfig

def main():
    # Configure memory limits
    config = SmokeRuntimeConfig()
    config.memory = MemoryConfig(
        gpu_memory="7.8GB",      # Reserve 200MB for system
        ram_cache="2GB",          # RAM cache for weights
        default_dtype="float16"   # Use float16 to save memory
    )
    
    # Optionally configure runtime behavior
    config.runtime.prefetch_layers = 3
    config.runtime.cache_policy = "lru"
    config.runtime.smoke_test_layers = 5
    
    # Initialize runtime
    runtime = SmokeRuntime(
        model_path="path/to/your/model",
        config=config
    )
    
    # Use context manager for automatic start/stop
    with runtime:
        # Create dummy input
        input_ids = torch.randint(0, 1000, (1, 128))
        
        # Run forward pass with predictive loading
        output = runtime.forward(input_ids)
        
        print(f"Output shape: {output.shape}")
        
        # Print statistics
        runtime.print_statistics()


if __name__ == "__main__":
    main()
