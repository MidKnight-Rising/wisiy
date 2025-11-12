"""
Main Smoke Runtime orchestrator.

Coordinates smoke test simulation, weight prefetching, and execution.
"""

from typing import Optional, Dict, Any, List
import torch
import logging
from pathlib import Path

from .config import SmokeRuntimeConfig, MemoryConfig, DeviceConfig, RuntimeConfig
from .cache import WeightCache
from .simulator import SmokeTestSimulator
from .prefetcher import WeightPrefetcher

logger = logging.getLogger(__name__)


class SmokeRuntime:
    """Main orchestrator for predictive memory-efficient model inference.
    
    Coordinates smoke test simulations, predictive weight loading,
    and just-in-time execution for large models on limited hardware.
    
    Args:
        model_path: Path to model weights
        config: Runtime configuration
    
    Example:
        >>> config = SmokeRuntimeConfig()
        >>> runtime = SmokeRuntime("path/to/model", config)
        >>> with runtime:
        >>>     output = runtime.forward(input_ids)
    """
    
    def __init__(
        self,
        model_path: str,
        config: Optional[SmokeRuntimeConfig] = None
    ):
        self.model_path = Path(model_path)
        self.config = config or SmokeRuntimeConfig()
        
        # Initialize components
        self.cache = self._init_cache()
        self.simulator = self._init_simulator()
        self.prefetcher = self._init_prefetcher()
        
        # Runtime state
        self.current_layer: Optional[str] = None
        self.is_running = False
        
        # Statistics
        self.stats = {
            "forward_passes": 0,
            "layers_executed": 0,
            "predictions_made": 0,
        }
        
        logger.info(
            f"Initialized SmokeRuntime for {model_path}\n"
            f"GPU Memory: {self.config.memory.gpu_memory}\n"
            f"RAM Cache: {self.config.memory.ram_cache}\n"
            f"Target dtype: {self.config.memory.default_dtype}"
        )
    
    def _init_cache(self) -> WeightCache:
        """Initialize weight cache."""
        return WeightCache(
            max_memory_bytes=self.config.memory.ram_bytes,
            policy=self.config.runtime.cache_policy
        )
    
    def _init_simulator(self) -> SmokeTestSimulator:
        """Initialize smoke test simulator."""
        # Load model config if available
        model_config = self._load_model_config()
        
        return SmokeTestSimulator(
            model_config=model_config,
            lookahead_layers=self.config.runtime.smoke_test_layers
        )
    
    def _init_prefetcher(self) -> WeightPrefetcher:
        """Initialize weight prefetcher."""
        num_workers = 2 if self.config.runtime.async_prefetch else 1
        
        return WeightPrefetcher(
            model_path=str(self.model_path),
            cache=self.cache,
            target_device=self.config.device.device,
            target_dtype=self.config.memory.torch_dtype,
            num_workers=num_workers
        )
    
    def _load_model_config(self) -> Dict[str, Any]:
        """Load model configuration from disk.
        
        Returns:
            Model configuration dict
        """
        config_path = self.model_path / "config.json"
        
        if config_path.exists():
            try:
                import json
                with open(config_path) as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Could not load model config: {e}")
        
        return {}
    
    def start(self):
        """Start the runtime (start prefetcher workers)."""
        if self.is_running:
            logger.warning("Runtime already started")
            return
        
        self.prefetcher.start()
        self.is_running = True
        logger.info("Smoke Runtime started")
    
    def stop(self):
        """Stop the runtime (stop prefetcher workers)."""
        if not self.is_running:
            return
        
        self.prefetcher.stop()
        self.is_running = False
        logger.info("Smoke Runtime stopped")
    
    def forward(
        self,
        input_ids: torch.Tensor,
        start_layer: int = 0,
        end_layer: Optional[int] = None,
        **kwargs
    ) -> torch.Tensor:
        """Execute forward pass with predictive weight loading.
        
        Args:
            input_ids: Input tensor
            start_layer: Starting layer index
            end_layer: Ending layer index (None = all layers)
            **kwargs: Additional forward pass arguments
            
        Returns:
            Output tensor
        """
        if not self.is_running:
            raise RuntimeError("Runtime not started. Call start() or use context manager.")
        
        self.stats["forward_passes"] += 1
        
        # Simulate forward pass to get execution order
        execution_order = self.simulator.simulate_forward_pass(
            input_ids, start_layer, end_layer
        )
        
        logger.info(
            f"Executing forward pass over {len(execution_order)} layers"
        )
        
        # Execute layers with predictive prefetching
        hidden_states = input_ids
        
        for layer_name in execution_order:
            # Run smoke test to predict next requirements
            if self.config.runtime.smoke_test_enabled:
                requirements = self.simulator.predict_requirements(
                    layer_name,
                    input_shape=hidden_states.shape
                )
                
                self.stats["predictions_made"] += len(requirements)
                
                # Queue prefetch for predicted weights
                self.prefetcher.prefetch_batch(requirements)
            
            # Execute current layer
            hidden_states = self._execute_layer(layer_name, hidden_states, **kwargs)
            
            # Update execution history
            self.simulator.update_execution_history(layer_name)
            self.current_layer = layer_name
            self.stats["layers_executed"] += 1
        
        logger.info("Forward pass completed")
        return hidden_states
    
    def _execute_layer(
        self,
        layer_name: str,
        input_tensor: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """Execute a single layer.
        
        This is a placeholder that should be replaced with actual
        layer execution logic based on the model architecture.
        
        Args:
            layer_name: Name of layer to execute
            input_tensor: Input to the layer
            **kwargs: Additional arguments
            
        Returns:
            Layer output tensor
        """
        # Get layer weights from prefetcher
        weight = self.prefetcher.get_weight(layer_name, "weight")
        
        if weight is None:
            logger.warning(f"Weight not available for {layer_name}, skipping")
            return input_tensor
        
        # Placeholder: In real implementation, this would call the actual
        # layer forward method with loaded weights
        # For now, just return input (identity operation)
        
        logger.debug(f"Executed layer: {layer_name}")
        return input_tensor
    
    def load_layer_weights(
        self,
        layer_name: str,
        wait: bool = True
    ) -> bool:
        """Explicitly load weights for a specific layer.
        
        Args:
            layer_name: Name of layer to load
            wait: Whether to wait for load to complete
            
        Returns:
            True if weights are loaded/available
        """
        # Create requirement for this layer
        from .simulator import WeightRequirement
        
        req = WeightRequirement(
            layer_name=layer_name,
            weight_keys=self.simulator._predict_weight_keys(layer_name),
            priority=1.0,
            estimated_size=0
        )
        
        self.prefetcher.prefetch(req)
        
        if wait:
            self.prefetcher.wait_for_completion()
        
        return self.cache.contains(f"{layer_name}.weight")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive runtime statistics.
        
        Returns:
            Dictionary with all component statistics
        """
        return {
            "runtime": self.stats,
            "cache": self.cache.get_stats(),
            "prefetcher": self.prefetcher.get_stats(),
            "simulator": self.simulator.get_statistics(),
        }
    
    def print_statistics(self):
        """Print formatted statistics."""
        stats = self.get_statistics()
        
        print("\n=== Smoke Runtime Statistics ===")
        print(f"\nRuntime:")
        for key, value in stats["runtime"].items():
            print(f"  {key}: {value}")
        
        print(f"\nCache:")
        for key, value in stats["cache"].items():
            print(f"  {key}: {value}")
        
        print(f"\nPrefetcher:")
        for key, value in stats["prefetcher"].items():
            print(f"  {key}: {value}")
        
        print(f"\nSimulator:")
        for key, value in stats["simulator"].items():
            print(f"  {key}: {value}")
        print()
    
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
