"""Tests for configuration module."""

import pytest
import torch
from smoke_runtime.config import (
    MemoryConfig,
    DeviceConfig,
    RuntimeConfig,
    SmokeRuntimeConfig
)


class TestMemoryConfig:
    """Test MemoryConfig class."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = MemoryConfig()
        assert config.gpu_memory == "7.8GB"
        assert config.ram_cache == "2GB"
        assert config.default_dtype == "float16"
        assert config.torch_dtype == torch.float16
    
    def test_memory_parsing_gb(self):
        """Test parsing GB memory strings."""
        config = MemoryConfig(gpu_memory="4.5GB")
        assert config.gpu_bytes == int(4.5 * 1024 ** 3)
    
    def test_memory_parsing_mb(self):
        """Test parsing MB memory strings."""
        config = MemoryConfig(gpu_memory="2048MB")
        assert config.gpu_bytes == 2048 * 1024 ** 2
    
    def test_dtype_mapping(self):
        """Test dtype string to torch dtype mapping."""
        config_fp16 = MemoryConfig(default_dtype="float16")
        assert config_fp16.torch_dtype == torch.float16
        
        config_fp32 = MemoryConfig(default_dtype="float32")
        assert config_fp32.torch_dtype == torch.float32
        
        config_bf16 = MemoryConfig(default_dtype="bfloat16")
        assert config_bf16.torch_dtype == torch.bfloat16
    
    def test_max_memory_dict(self):
        """Test max_memory dict generation."""
        config = MemoryConfig(
            gpu_memory="8GB",
            ram_cache="2GB",
            enable_cpu_offload=True
        )
        max_mem = config.get_max_memory_dict()
        assert max_mem[0] == "8GB"
        assert max_mem["cpu"] == "2GB"
    
    def test_max_memory_dict_no_offload(self):
        """Test max_memory dict without CPU offload."""
        config = MemoryConfig(enable_cpu_offload=False)
        max_mem = config.get_max_memory_dict()
        assert 0 in max_mem
        assert "cpu" not in max_mem


class TestDeviceConfig:
    """Test DeviceConfig class."""
    
    def test_default_config(self):
        """Test default device configuration."""
        config = DeviceConfig()
        assert config.gpu_id == 0
        assert config.num_gpus == 1
        assert not config.force_gpu
    
    def test_device_property(self):
        """Test device property."""
        config = DeviceConfig(gpu_id=0)
        device = config.device
        assert isinstance(device, torch.device)
        # Will be cuda:0 if CUDA available, else cpu
    
    def test_device_map_generation(self):
        """Test device map generation."""
        config = DeviceConfig(gpu_id=0, force_gpu=False)
        device_map = config.get_device_map(num_layers=10)
        
        # Should return auto or dict
        assert device_map is not None


class TestRuntimeConfig:
    """Test RuntimeConfig class."""
    
    def test_default_config(self):
        """Test default runtime configuration."""
        config = RuntimeConfig()
        assert config.prefetch_layers == 3
        assert config.cache_policy == "lru"
        assert config.async_prefetch
        assert config.smoke_test_enabled
        assert config.smoke_test_layers == 5


class TestSmokeRuntimeConfig:
    """Test SmokeRuntimeConfig class."""
    
    def test_default_config(self):
        """Test default combined configuration."""
        config = SmokeRuntimeConfig()
        assert isinstance(config.memory, MemoryConfig)
        assert isinstance(config.device, DeviceConfig)
        assert isinstance(config.runtime, RuntimeConfig)
    
    def test_from_dict(self):
        """Test creating config from dictionary."""
        config_dict = {
            "memory": {
                "gpu_memory": "4GB",
                "ram_cache": "1GB"
            },
            "device": {
                "gpu_id": 1
            },
            "runtime": {
                "prefetch_layers": 5
            }
        }
        
        config = SmokeRuntimeConfig.from_dict(config_dict)
        assert config.memory.gpu_memory == "4GB"
        assert config.device.gpu_id == 1
        assert config.runtime.prefetch_layers == 5
    
    def test_to_dict(self):
        """Test converting config to dictionary."""
        config = SmokeRuntimeConfig()
        config_dict = config.to_dict()
        
        assert "memory" in config_dict
        assert "device" in config_dict
        assert "runtime" in config_dict
        assert isinstance(config_dict, dict)
