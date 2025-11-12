# Smoke Runtime Architecture

## Overview

Smoke Runtime implements a predictive, memory-efficient architecture for running large-scale models on limited hardware. This document provides detailed technical information about the system design.

## System Architecture

```
┌────────────────────────────────────────────────────────────┐
│                     Smoke Runtime System                   │
└────────────────────────────────────────────────────────────┘
                             │
                             ▼
         ┌───────────────────────────────────┐
         │      SmokeRuntime (Orchestrator)  │
         │  - Coordinates all components     │
         │  - Manages execution flow         │
         │  - Tracks statistics              │
         └───────────────────────────────────┘
                             │
         ┌───────────────────┼───────────────────┐
         │                   │                   │
         ▼                   ▼                   ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│ SmokeTest       │  │  WeightCache    │  │ WeightPrefetcher│
│ Simulator       │  │                 │  │                 │
│                 │  │  - RAM storage  │  │ - Async loading │
│ - Predicts next │  │  - LRU eviction │  │ - Dtype convert │
│   layers        │  │  - Thread-safe  │  │ - Multi-threaded│
│ - Learns        │  │  - Statistics   │  │ - Queue mgmt    │
│   patterns      │  │                 │  │                 │
└─────────────────┘  └─────────────────┘  └─────────────────┘
         │                   │                   │
         └───────────────────┼───────────────────┘
                             ▼
                    ┌─────────────────┐
                    │   GPU Memory    │
                    │  (Execution)    │
                    └─────────────────┘
```

## Component Details

### 1. SmokeRuntime (Orchestrator)

**Purpose**: Main entry point and coordinator for the entire system.

**Responsibilities**:
- Initialize all subsystems
- Coordinate forward pass execution
- Manage component lifecycle
- Collect and report statistics

**Key Methods**:
- `forward()`: Execute model inference
- `start()`: Start prefetcher workers
- `stop()`: Stop workers and cleanup
- `get_statistics()`: Retrieve system stats

**State Management**:
- Tracks current layer being executed
- Maintains execution counters
- Stores configuration

### 2. SmokeTestSimulator

**Purpose**: Predict which weights will be needed for upcoming operations.

**How It Works**:

1. **Layer Analysis**: Parses layer name to determine position in model
2. **Lookahead Prediction**: Predicts next N layers based on sequential execution
3. **Pattern Learning**: Learns from execution history for better predictions
4. **Priority Assignment**: Assigns priority scores (closer layers = higher priority)

**Prediction Algorithm**:
```python
def predict_requirements(current_layer):
    current_idx = parse_layer_index(current_layer)
    requirements = []
    
    for i in range(1, lookahead_layers + 1):
        next_idx = current_idx + i
        priority = 1.0 / i  # Decay with distance
        
        requirement = WeightRequirement(
            layer_name=f"layer.{next_idx}",
            weight_keys=predict_keys(next_idx),
            priority=priority,
            estimated_size=estimate_size(next_idx)
        )
        requirements.append(requirement)
    
    return sorted(requirements, key=lambda r: r.priority)
```

**Pattern Learning**:
- Maintains sliding window of recent execution history (100 entries)
- Builds pattern cache: `layer_A → layer_B`
- Uses patterns to predict non-sequential jumps (branches, skip connections)

**Future Enhancements**:
- ML-based prediction using execution traces
- Architecture-aware prediction (attention patterns, skip connections)
- Batched prediction for parallel execution

### 3. WeightCache

**Purpose**: RAM-based cache for model weights with automatic eviction.

**Memory Management**:
```
┌─────────────────────────────────────┐
│         WeightCache (RAM)           │
├─────────────────────────────────────┤
│ Current: 1.8 GB / Max: 2.0 GB       │
├─────────────────────────────────────┤
│ Entry 1: layer.5.weight  (200 MB)  │
│ Entry 2: layer.6.weight  (200 MB)  │
│ Entry 3: layer.7.weight  (200 MB)  │
│ Entry 4: layer.8.weight  (200 MB)  │
│ ...                                 │
├─────────────────────────────────────┤
│ Eviction Policy: LRU                │
└─────────────────────────────────────┘
```

**Eviction Policies**:

1. **LRU (Least Recently Used)**:
   - Best for sequential access patterns
   - Evicts entries not accessed recently
   - Implemented with OrderedDict

2. **LFU (Least Frequently Used)**:
   - Best for repeated access patterns
   - Evicts entries with lowest access count
   - Tracks access frequency

3. **FIFO (First In First Out)**:
   - Simple queue-based eviction
   - Evicts oldest entries first
   - Lowest overhead

**Thread Safety**:
- Uses `threading.RLock()` for thread-safe operations
- Supports concurrent reads and writes
- Atomic eviction operations

### 4. WeightPrefetcher

**Purpose**: Asynchronous loading of weights from disk to RAM to GPU.

**Pipeline Architecture**:
```
┌──────────┐     ┌──────────┐     ┌──────────┐     ┌──────────┐
│  Disk    │────►│  Queue   │────►│ Workers  │────►│   RAM    │
│ Storage  │     │(Priority)│     │(Threads) │     │  Cache   │
└──────────┘     └──────────┘     └──────────┘     └──────────┘
                                                          │
                                                          ▼
                                                    ┌──────────┐
                                                    │   GPU    │
                                                    │  Memory  │
                                                    └──────────┘
```

**Worker Thread Model**:
- Configurable number of worker threads (default: 2)
- Priority queue for task scheduling
- Concurrent loading of multiple weights

**Task Execution Flow**:
1. Receive `WeightRequirement` from simulator
2. Create `PrefetchTask` with priority
3. Queue task in priority queue
4. Worker thread picks task
5. Load weight from disk to CPU memory
6. Convert dtype if needed (float32 → float16)
7. Store in WeightCache
8. Execute callback (if provided)

**Dtype Conversion**:
```python
if weight_tensor.dtype != self.target_dtype:
    weight_tensor = weight_tensor.to(self.target_dtype)
    stats["dtype_conversions"] += 1
```

**Error Handling**:
- Graceful handling of missing weights
- Retries with exponential backoff
- Logs errors without crashing pipeline

### 5. Configuration System

**Hierarchical Configuration**:
```
SmokeRuntimeConfig
├── MemoryConfig
│   ├── gpu_memory: "7.8GB"
│   ├── ram_cache: "2GB"
│   ├── default_dtype: "float16"
│   └── enable_cpu_offload: False
├── DeviceConfig
│   ├── gpu_id: 0
│   ├── num_gpus: 1
│   └── force_gpu: False
└── RuntimeConfig
    ├── prefetch_layers: 3
    ├── cache_policy: "lru"
    ├── async_prefetch: True
    ├── smoke_test_enabled: True
    └── smoke_test_layers: 5
```

**Memory Parsing**:
- Supports various units: KB, MB, GB, TB
- Automatic byte conversion
- Validation and error checking

## Data Flow

### Forward Pass Execution

```
1. User: runtime.forward(input_ids)
                    │
                    ▼
2. Runtime: Simulate execution order
                    │
                    ▼
3. Simulator: Predict requirements for each layer
                    │
                    ▼
4. Runtime: Queue prefetch tasks
                    │
                    ▼
5. Prefetcher: Load weights asynchronously
                    │
                    ▼
6. Cache: Store in RAM with eviction
                    │
                    ▼
7. Runtime: Execute layer with cached weights
                    │
                    ▼
8. Simulator: Update execution history
                    │
                    ▼
9. Repeat for next layer
```

### Weight Loading Pipeline

```
Disk (Full Model)
    │
    ├─ pytorch_model.bin (42 GB)
    └─ safetensors files
         │
         ▼
    Load to CPU Memory
         │
         ├─ torch.load(map_location="cpu")
         └─ Parse weight dict
              │
              ▼
    Dtype Conversion
              │
              ├─ float32 → float16
              └─ Reduce memory footprint
                   │
                   ▼
    Store in WeightCache (RAM)
                   │
                   ├─ LRU eviction if full
                   └─ Thread-safe storage
                        │
                        ▼
    Transfer to GPU on-demand
                        │
                        ├─ tensor.to(device="cuda:0")
                        └─ Only active layers
                             │
                             ▼
    Execute Layer
                             │
                             └─ Model computation
```

## Performance Characteristics

### Memory Usage

**Disk**: Full model (42 GB for 21B params in float16)
**RAM Cache**: Configurable (default 2 GB)
- Holds ~5-10 layers depending on size
- LRU eviction maintains working set

**GPU Memory**: Configurable (default 7.8 GB for 8 GB GPU)
- Active layer weights (~200-500 MB each)
- Activations and intermediate tensors
- CUDA kernel memory

### Throughput

**Sequential Inference**:
- Near-optimal with good prefetch lookahead
- Minimal stalls if cache hit rate > 80%

**Random Access**:
- Higher cache miss rate
- Benefits from larger cache size
- LFU policy may perform better

### Latency

**First Token**: Higher latency due to initial loading
**Subsequent Tokens**: Lower latency with warmed cache
**Cache Miss**: 100-500ms depending on disk speed

## Comparison with Alternatives

### vs. device_map="auto"

**device_map="auto"** (HuggingFace):
- ❌ Aggressively offloads to CPU/disk
- ❌ No prediction mechanism
- ❌ Fixed device mapping
- ✅ Easy to use
- ✅ Works with any model

**Smoke Runtime**:
- ✅ Predictive prefetching
- ✅ Dynamic weight management
- ✅ Optimized for limited GPU
- ❌ Requires setup
- ❌ Currently model-agnostic (needs integration)

### vs. Model Quantization

**Quantization** (4-bit, 8-bit):
- ✅ Reduces model size permanently
- ✅ Lower memory footprint
- ❌ Quality degradation
- ❌ Requires quantized weights

**Smoke Runtime**:
- ✅ No quality loss (full precision)
- ✅ Works with existing models
- ✅ Flexible memory/speed tradeoff
- ❌ More complex system

### vs. Streaming/Pipelining

**Model Streaming**:
- ✅ Layer-by-layer loading
- ❌ No prediction
- ❌ Sequential only

**Smoke Runtime**:
- ✅ Predictive loading
- ✅ Parallel prefetch
- ✅ Pattern learning
- ✅ Better cache utilization

## Future Optimizations

### Adaptive Prediction
- ML model to predict execution patterns
- Learn from inference traces
- Adapt to different input types

### Quantization Integration
- Support 4-bit/8-bit quantized weights
- Dynamic quantization during transfer
- Mixed precision execution

### Distributed Inference
- Multi-GPU coordination
- Pipeline parallelism
- Tensor parallelism support

### Advanced Caching
- Compression in cache
- Smart eviction with execution graphs
- Persistent cache across runs

### Streaming Attention
- Implement streamed attention mechanisms
- Reduce active memory footprint
- Support for very long sequences

## Debugging and Monitoring

### Statistics Collection

```python
stats = runtime.get_statistics()

# Runtime stats
stats["runtime"]["forward_passes"]
stats["runtime"]["layers_executed"]

# Cache stats
stats["cache"]["utilization"]
stats["cache"]["num_entries"]

# Prefetcher stats
stats["prefetcher"]["cache_hits"]
stats["prefetcher"]["cache_misses"]
stats["prefetcher"]["total_gb_loaded"]

# Simulator stats
stats["simulator"]["patterns_learned"]
```

### Memory Profiling

Enable memory profiling in config:
```python
config.runtime.profile_memory = True
```

Logs memory usage at each step:
- GPU memory allocated
- Cache memory used
- Eviction events
- Transfer times

## References

- Original problem statement and requirements
- PyTorch memory management documentation
- HuggingFace Accelerate library
- Model parallelism papers and techniques
