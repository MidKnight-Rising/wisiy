# Getting Started with Smoke Runtime

This guide will help you get started with developing and using Smoke Runtime.

## For Developers

### Initial Setup

1. **Clone the repository**
```bash
git clone https://github.com/MidKnight-Rising/wisiy.git
cd wisiy
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install in development mode**
```bash
pip install -e ".[dev]"
```

4. **Verify installation**
```bash
pytest
```

You should see 38 tests passing.

### Development Workflow

1. **Create a feature branch**
```bash
git checkout -b feature/your-feature
```

2. **Make changes and test**
```bash
# Make your changes
pytest tests/

# Format code
black smoke_runtime/ tests/ examples/

# Lint
flake8 smoke_runtime/ tests/ examples/
```

3. **Commit and push**
```bash
git add .
git commit -m "Description of changes"
git push origin feature/your-feature
```

4. **Create Pull Request**

## For Users

### Installation

```bash
pip install git+https://github.com/MidKnight-Rising/wisiy.git
```

### Basic Usage

```python
from smoke_runtime import SmokeRuntime, MemoryConfig

# Configure for your hardware
config = MemoryConfig(
    gpu_memory="7.8GB",
    ram_cache="2GB",
    default_dtype="float16"
)

# Initialize runtime
runtime = SmokeRuntime(
    model_path="path/to/model",
    config=config
)

# Run inference
with runtime:
    import torch
    input_ids = torch.randint(0, 1000, (1, 128))
    output = runtime.forward(input_ids)
```

### Configuration Examples

See [examples/](../examples/) directory for:
- `basic_usage.py` - Simple example
- `advanced_config.py` - Advanced configurations for different hardware setups

## Current Status

### What Works Now ✅

- Configuration system (memory limits, device selection, runtime parameters)
- WeightCache with LRU/LFU/FIFO eviction policies
- SmokeTestSimulator for predicting next layers
- WeightPrefetcher architecture (async loading pipeline)
- SmokeRuntime orchestrator (coordinates all components)
- Comprehensive test suite

### What's Coming Next 🚧

**Phase 2: Model Integration**
- HuggingFace Transformers integration
- Real model loading and execution
- Support for popular architectures (GPT, BERT, LLaMA, etc.)
- End-to-end inference examples

### What's Not Ready Yet ⚠️

- **Actual model execution**: The current implementation is a framework. Layer execution (`_execute_layer`) is a placeholder.
- **Weight file loading**: Loading from disk needs integration with specific model formats.
- **GPU transfer**: Automatic GPU transfer needs testing with real models.

## Architecture Overview

```
SmokeRuntime (Orchestrator)
├── SmokeTestSimulator (Predicts next layers)
├── WeightCache (RAM cache with LRU/LFU/FIFO)
├── WeightPrefetcher (Async loading pipeline)
└── Configuration (Memory, Device, Runtime settings)
```

**Execution Flow:**
1. User calls `runtime.forward(input)`
2. Simulator predicts next N layers needed
3. Prefetcher loads predicted weights (disk→RAM→GPU)
4. Runtime executes layers with cached weights
5. Cache evicts unused weights automatically

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed technical information.

## Next Steps

### For Contributors

Priority areas (see [ROADMAP.md](ROADMAP.md)):

1. **HuggingFace Integration** (High Priority)
   - Adapter for loading HF models
   - Custom device_map for Accelerate
   - Dtype conversion integration

2. **Real Model Testing**
   - Test with GPT-2 (small model)
   - Validate memory management
   - Benchmark performance

3. **Documentation**
   - API reference
   - Tutorial notebooks
   - Video walkthrough

4. **Performance**
   - Profile hot paths
   - Optimize cache operations
   - Reduce overhead

### For Users

**Current recommendation**: Wait for Phase 2 (Model Integration) to complete before using in production. The foundation is solid, but model execution integration is needed.

**Can help with**:
- Testing the framework design
- Providing feedback on API
- Contributing to documentation
- Suggesting features

## Resources

- **Documentation**: [docs/](../docs/)
- **Examples**: [examples/](../examples/)
- **Tests**: [tests/](../tests/)
- **Issues**: [GitHub Issues](https://github.com/MidKnight-Rising/wisiy/issues)
- **Discussions**: [GitHub Discussions](https://github.com/MidKnight-Rising/wisiy/discussions)

## Common Tasks

### Run specific tests
```bash
pytest tests/test_cache.py -v
```

### Run tests with coverage
```bash
pytest --cov=smoke_runtime --cov-report=html
```

### Format code
```bash
black smoke_runtime/ tests/ examples/
```

### Type checking
```bash
mypy smoke_runtime/
```

### View test coverage report
```bash
pytest --cov=smoke_runtime --cov-report=html
open htmlcov/index.html  # On Mac/Linux
start htmlcov/index.html  # On Windows
```

## Troubleshooting

### Import errors
Make sure you installed in development mode:
```bash
pip install -e ".[dev]"
```

### Test failures
Verify your Python version is 3.8+:
```bash
python --version
```

### Memory errors during tests
Tests use small tensors and should work on any machine. If you see memory errors, check available RAM.

## Questions?

- Check [CONTRIBUTING.md](../CONTRIBUTING.md) for development guidelines
- Open a [GitHub Discussion](https://github.com/MidKnight-Rising/wisiy/discussions) for questions
- File a [GitHub Issue](https://github.com/MidKnight-Rising/wisiy/issues) for bugs

---

**Status**: Foundation Complete ✅ | Ready for Phase 2 Development 🚀
