# Smoke Runtime

**Predictive Memory-Efficient Runtime for Large-Scale Model Inference**

Smoke Runtime enables running large-scale models (e.g., 21B+ parameters) on limited GPU hardware through intelligent predictive weight loading and just-in-time execution.

## 🚀 Key Features

- **Predictive Prefetching**: Anticipates which weights will be needed next during inference
- **Dynamic Memory Management**: Intelligently manages weights across disk → RAM → GPU
- **Dtype Normalization**: Automatic dtype conversion (float32 → float16/bfloat16)
- **Flexible Caching**: LRU/LFU/FIFO eviction policies for optimal memory utilization
- **Async Pipeline**: Overlaps weight loading with computation for better throughput
- **Memory Profiling**: Built-in statistics and monitoring

## 📋 Problem Statement

Running large models (21B+ parameters, ~42GB in float16) on consumer GPUs (8GB VRAM) faces several challenges:

1. **Memory Overflow**: Full model cannot fit in GPU memory
2. **Inefficient Offloading**: Automatic device mapping (`device_map="auto"`) offloads too aggressively to CPU/disk
3. **No Prediction**: Traditional approaches lack mechanisms to predict which weights are needed next
4. **Dtype Mismatches**: Runtime errors from dtype incompatibilities between layers

## 💡 Solution: Smoke Test Architecture

Smoke Runtime implements a parallel "smoke test" mechanism that:

1. **Simulates** upcoming inference steps in RAM
2. **Predicts** which model weights will be needed
3. **Prefetches** weights from disk to RAM to GPU
4. **Executes** inference using just-in-time loaded weights
5. **Evicts** unused weights based on LRU/LFU policies

```
┌──────────────────────────────────────────────────────────┐
│                    Smoke Runtime Pipeline                 │
├──────────────────────────────────────────────────────────┤
│                                                           │
│  Disk Storage  ──►  RAM Cache  ──►  GPU Memory           │
│  (Full Model)      (Predicted)     (Active)              │
│                                                           │
│  ┌─────────┐      ┌──────────┐    ┌──────────┐          │
│  │ 42GB    │      │ 2GB      │    │ 8GB      │          │
│  │ float16 │ ───► │ LRU      │ ──►│ Active   │          │
│  │ Weights │      │ Cache    │    │ Layers   │          │
│  └─────────┘      └──────────┘    └──────────┘          │
│       ▲                ▲               │                 │
│       │                │               │                 │
│       └────────────────┴───────────────┘                 │
│              Smoke Test Predictor                        │
│         (Simulates next N layers)                        │
└──────────────────────────────────────────────────────────┘
```

## 🔧 Installation

### From Source

```bash
git clone https://github.com/MidKnight-Rising/wisiy.git
cd wisiy
pip install -e .
```

### Dependencies

```bash
pip install -r requirements.txt
```

**Requirements:**
- Python >= 3.8
- PyTorch >= 2.0.0
- NumPy >= 1.20.0

## 📖 Quick Start

### Basic Usage

```python
import torch
from smoke_runtime import SmokeRuntime, MemoryConfig

# Configure memory limits
config = MemoryConfig(
    gpu_memory="7.8GB",      # Leave headroom for CUDA
    ram_cache="2GB",          # RAM cache size
    default_dtype="float16"   # Target dtype
)

# Initialize runtime
runtime = SmokeRuntime(
    model_path="path/to/your/model",
    config=config
)

# Run inference with automatic weight management
with runtime:
    input_ids = torch.randint(0, 1000, (1, 128))
    output = runtime.forward(input_ids)
    
    # View statistics
    runtime.print_statistics()
```

### Advanced Configuration

```python
from smoke_runtime import SmokeRuntimeConfig, RuntimeConfig

config = SmokeRuntimeConfig()

# Memory settings
config.memory.gpu_memory = "7.8GB"
config.memory.ram_cache = "4GB"
config.memory.default_dtype = "float16"

# Runtime behavior
config.runtime.prefetch_layers = 5       # Prefetch 5 layers ahead
config.runtime.cache_policy = "lru"      # LRU eviction
config.runtime.smoke_test_layers = 7     # Simulate 7 layers ahead
config.runtime.async_prefetch = True     # Async loading

runtime = SmokeRuntime("path/to/model", config)
```

## 🏗️ Architecture

### Core Components

#### 1. **WeightCache**
- LRU-based RAM cache for model weights
- Automatic eviction when memory limit reached
- Thread-safe operations
- Supports LRU, LFU, and FIFO policies

#### 2. **SmokeTestSimulator**
- Lightweight forward pass simulation
- Predicts next N layers/operations
- Learns execution patterns over time
- Priority-based weight requirements

#### 3. **WeightPrefetcher**
- Async weight loading pipeline
- Disk → RAM → GPU transfers
- Automatic dtype conversion
- Multi-threaded prefetch workers

#### 4. **SmokeRuntime**
- Main orchestrator
- Coordinates all components
- Manages execution flow
- Provides statistics and monitoring

### Execution Flow

```
1. User calls runtime.forward(input_ids)
2. Simulator predicts next N layers needed
3. Prefetcher queues weight loading tasks
4. Worker threads load weights (disk → RAM)
5. Weights transferred to GPU on-demand
6. Layer executes with loaded weights
7. Unused weights evicted from cache
8. Repeat for next layer
```

## 📊 Configuration Guide

### Memory-Constrained Setup (4-8GB GPU)

```python
config = MemoryConfig(
    gpu_memory="3.8GB",           # Conservative limit
    ram_cache="1GB",              # Small cache
    default_dtype="float16",      # Half precision
    enable_cpu_offload=False      # Avoid CPU slowdown
)
```

### High-Performance Setup (16GB+ GPU)

```python
config = MemoryConfig(
    gpu_memory="15GB",            # Use most of GPU
    ram_cache="8GB",              # Large cache
    default_dtype="float16",
    enable_cpu_offload=False
)

config.runtime.prefetch_layers = 8      # Aggressive prefetch
config.runtime.smoke_test_layers = 10   # Deep lookahead
```

### Multi-GPU Setup

```python
config.device.num_gpus = 2
config.device.gpu_id = 0
config.memory.gpu_memory = "15GB"  # Per-GPU allocation
```

## 🧪 Testing

Run tests with pytest:

```bash
# Install dev dependencies
pip install -r requirements-dev.txt

# Run all tests
pytest

# Run with coverage
pytest --cov=smoke_runtime --cov-report=html

# Run specific test file
pytest tests/test_cache.py -v
```

## 📁 Project Structure

```
wisiy/
├── smoke_runtime/          # Core package
│   ├── __init__.py        # Package exports
│   ├── cache.py           # WeightCache implementation
│   ├── config.py          # Configuration classes
│   ├── prefetcher.py      # WeightPrefetcher
│   ├── runtime.py         # Main SmokeRuntime
│   └── simulator.py       # SmokeTestSimulator
├── tests/                 # Test suite
│   ├── test_cache.py
│   ├── test_config.py
│   └── test_simulator.py
├── examples/              # Usage examples
│   ├── basic_usage.py
│   ├── advanced_config.py
│   └── README.md
├── docs/                  # Documentation
├── requirements.txt       # Dependencies
├── requirements-dev.txt   # Dev dependencies
├── setup.py              # Package setup
├── pyproject.toml        # Build config
└── README.md             # This file
```

## 🎯 Roadmap

### Phase 1: Foundation (Current)
- [x] Core architecture implementation
- [x] Basic caching and prefetching
- [x] Configuration system
- [x] Unit tests
- [x] Documentation

### Phase 2: Integration (Next)
- [ ] Integration with HuggingFace Transformers
- [ ] Support for different model architectures
- [ ] Real model loading and execution
- [ ] Benchmark suite

### Phase 3: Optimization
- [ ] Adaptive smoke test (learns from patterns)
- [ ] Quantization support (4-bit, 8-bit)
- [ ] Streamed attention mechanisms
- [ ] Layer-wise gradient checkpointing
- [ ] Multi-GPU optimization

### Phase 4: Advanced Features
- [ ] Dynamic batch sizing
- [ ] Mixed precision training support
- [ ] Distributed inference
- [ ] Cloud storage backends (S3, GCS)
- [ ] Web UI for monitoring

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

### Development Setup

```bash
# Clone repository
git clone https://github.com/MidKnight-Rising/wisiy.git
cd wisiy

# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest

# Format code
black smoke_runtime/ tests/ examples/

# Lint code
flake8 smoke_runtime/ tests/ examples/
```

## 📄 License

MIT License - see LICENSE file for details

## 🙏 Acknowledgments

- Inspired by challenges in running large language models on consumer hardware
- Built with PyTorch for flexibility and performance
- Designed for the ML community working with limited resources

## 📚 Further Reading

- [Examples](examples/README.md) - Detailed usage examples
- [Architecture Deep Dive](docs/ARCHITECTURE.md) - Technical details
- [Configuration Guide](docs/CONFIGURATION.md) - All config options
- [API Reference](docs/API.md) - Complete API documentation

## 🐛 Issues & Support

Found a bug or have a feature request? Please open an issue on GitHub:
https://github.com/MidKnight-Rising/wisiy/issues

---

**Made with ❤️ for the ML community**
