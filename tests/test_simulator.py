"""Tests for simulator module."""

import pytest
import torch
from smoke_runtime.simulator import SmokeTestSimulator, WeightRequirement


class TestWeightRequirement:
    """Test WeightRequirement dataclass."""
    
    def test_creation(self):
        """Test creating weight requirement."""
        req = WeightRequirement(
            layer_name="layer.5",
            weight_keys=["weight", "bias"],
            priority=0.8,
            estimated_size=1024
        )
        
        assert req.layer_name == "layer.5"
        assert req.weight_keys == ["weight", "bias"]
        assert req.priority == 0.8
        assert req.estimated_size == 1024
    
    def test_repr(self):
        """Test string representation."""
        req = WeightRequirement(
            layer_name="layer.5",
            weight_keys=["weight"],
            priority=1.0,
            estimated_size=1024
        )
        
        repr_str = repr(req)
        assert "layer.5" in repr_str
        assert "priority" in repr_str


class TestSmokeTestSimulator:
    """Test SmokeTestSimulator class."""
    
    def test_initialization(self):
        """Test simulator initialization."""
        simulator = SmokeTestSimulator(lookahead_layers=5)
        
        assert simulator.lookahead_layers == 5
        assert len(simulator.execution_history) == 0
        assert len(simulator.pattern_cache) == 0
    
    def test_parse_layer_index(self):
        """Test parsing layer indices from names."""
        simulator = SmokeTestSimulator()
        
        assert simulator._parse_layer_index("layer.5") == 5
        assert simulator._parse_layer_index("layers.10") == 10
        assert simulator._parse_layer_index("block.3") == 3
        assert simulator._parse_layer_index("blocks.7") == 7
    
    def test_parse_layer_index_invalid(self):
        """Test parsing invalid layer names."""
        simulator = SmokeTestSimulator()
        
        with pytest.raises(ValueError):
            simulator._parse_layer_index("invalid_name")
    
    def test_predict_requirements(self):
        """Test predicting weight requirements."""
        simulator = SmokeTestSimulator(lookahead_layers=3)
        
        requirements = simulator.predict_requirements("layer.5")
        
        assert len(requirements) == 3
        assert all(isinstance(req, WeightRequirement) for req in requirements)
        
        # Check that priorities decrease with distance
        assert requirements[0].priority > requirements[1].priority
        assert requirements[1].priority > requirements[2].priority
        
        # Check predicted layer names
        assert requirements[0].layer_name == "layer.6"
        assert requirements[1].layer_name == "layer.7"
        assert requirements[2].layer_name == "layer.8"
    
    def test_predict_weight_keys_attention(self):
        """Test predicting weight keys for attention layers."""
        simulator = SmokeTestSimulator()
        
        keys = simulator._predict_weight_keys("layer.attention")
        
        assert "weight" in keys
        assert "q_proj" in keys
        assert "k_proj" in keys
        assert "v_proj" in keys
        assert "o_proj" in keys
    
    def test_predict_weight_keys_standard(self):
        """Test predicting weight keys for standard layers."""
        simulator = SmokeTestSimulator()
        
        keys = simulator._predict_weight_keys("layer.mlp")
        
        assert "weight" in keys
        assert "bias" in keys
    
    def test_update_execution_history(self):
        """Test updating execution history."""
        simulator = SmokeTestSimulator()
        
        simulator.update_execution_history("layer.0")
        simulator.update_execution_history("layer.1")
        
        assert len(simulator.execution_history) == 2
        assert simulator.execution_history[0] == "layer.0"
        assert simulator.execution_history[1] == "layer.1"
    
    def test_execution_history_max_length(self):
        """Test execution history doesn't grow indefinitely."""
        simulator = SmokeTestSimulator()
        
        # Add more than max_history items
        for i in range(150):
            simulator.update_execution_history(f"layer.{i}")
        
        # Should be capped at 100
        assert len(simulator.execution_history) <= 100
    
    def test_pattern_learning(self):
        """Test pattern learning from execution history."""
        simulator = SmokeTestSimulator()
        
        # Create a pattern: layer.0 -> layer.1
        simulator.update_execution_history("layer.0")
        simulator.update_execution_history("layer.1")
        
        # Check pattern was learned
        assert "layer.0" in simulator.pattern_cache
        assert "layer.1" in simulator.pattern_cache["layer.0"]
    
    def test_simulate_forward_pass(self):
        """Test simulating forward pass."""
        model_config = {"num_layers": 10}
        simulator = SmokeTestSimulator(model_config=model_config)
        
        input_ids = torch.randn(1, 128)
        execution_order = simulator.simulate_forward_pass(
            input_ids,
            start_layer=0,
            end_layer=5
        )
        
        assert len(execution_order) == 5
        assert execution_order[0] == "layer.0"
        assert execution_order[4] == "layer.4"
    
    def test_simulate_forward_pass_all_layers(self):
        """Test simulating forward pass over all layers."""
        model_config = {"num_layers": 8}
        simulator = SmokeTestSimulator(model_config=model_config)
        
        input_ids = torch.randn(1, 128)
        execution_order = simulator.simulate_forward_pass(input_ids)
        
        assert len(execution_order) == 8
    
    def test_get_statistics(self):
        """Test getting simulator statistics."""
        simulator = SmokeTestSimulator(lookahead_layers=5)
        
        simulator.update_execution_history("layer.0")
        simulator.update_execution_history("layer.1")
        
        stats = simulator.get_statistics()
        
        assert "execution_history_length" in stats
        assert "patterns_learned" in stats
        assert "lookahead_layers" in stats
        
        assert stats["execution_history_length"] == 2
        assert stats["lookahead_layers"] == 5
