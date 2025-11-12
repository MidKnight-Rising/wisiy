# Smoke Runtime Examples

This directory contains example scripts demonstrating various use cases and configurations for Smoke Runtime.

## Basic Examples

### `basic_usage.py`
Simple example showing how to:
- Configure memory limits
- Initialize the runtime
- Run inference with predictive loading
- View statistics

**Usage:**
```bash
python examples/basic_usage.py
```

### `advanced_config.py`
Demonstrates advanced configuration options:
- Different configurations for various hardware setups
- Memory-constrained environments
- Multi-GPU configurations
- Performance tuning parameters

**Usage:**
```bash
python examples/advanced_config.py
```

## Running the Examples

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Modify the `model_path` in examples to point to your model

3. Run any example:
```bash
python examples/basic_usage.py
```

## Configuration Tips

### For Limited GPU Memory (4-8GB)
- Use `float16` dtype
- Set conservative `gpu_memory` limit (e.g., "3.8GB" for 4GB card)
- Reduce `prefetch_layers` to 2-3
- Enable `profile_memory` to track usage

### For Larger GPUs (16GB+)
- Can use `bfloat16` or even `float32` if needed
- Increase `prefetch_layers` to 5-8
- Larger `smoke_test_layers` for better prediction
- Bigger `ram_cache` for higher hit rates

### For Sequential Inference
- Use `cache_policy="lru"` (default)
- Enable `async_prefetch`
- Higher `smoke_test_layers` value

### For Random/Non-sequential Access
- Consider `cache_policy="lfu"`
- May want to disable smoke test
- Focus on larger cache size
