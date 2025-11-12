"""
Weight prefetcher for async disk→RAM→GPU transfers.

Manages the pipeline of loading weights from disk, caching in RAM,
and transferring to GPU memory with dtype normalization.
"""

import asyncio
import threading
from typing import Optional, Dict, Any, Callable
from pathlib import Path
import torch
import logging
from queue import Queue, PriorityQueue
from dataclasses import dataclass, field

from .cache import WeightCache
from .simulator import WeightRequirement

logger = logging.getLogger(__name__)


@dataclass(order=True)
class PrefetchTask:
    """Represents a weight prefetch task.
    
    Attributes:
        priority: Task priority (higher = more urgent)
        requirement: Weight requirement to fulfill
        callback: Optional callback when task completes
    """
    priority: float
    requirement: WeightRequirement = field(compare=False)
    callback: Optional[Callable] = field(default=None, compare=False)


class WeightPrefetcher:
    """Manages asynchronous weight loading and transfer pipeline.
    
    Orchestrates the movement of weights from disk → RAM cache → GPU
    with automatic dtype conversion and memory management.
    
    Args:
        model_path: Path to model weights on disk
        cache: WeightCache instance for RAM caching
        target_device: Target GPU device
        target_dtype: Target dtype for weights
        num_workers: Number of prefetch worker threads
    """
    
    def __init__(
        self,
        model_path: str,
        cache: WeightCache,
        target_device: torch.device,
        target_dtype: torch.dtype = torch.float16,
        num_workers: int = 2
    ):
        self.model_path = Path(model_path)
        self.cache = cache
        self.target_device = target_device
        self.target_dtype = target_dtype
        self.num_workers = num_workers
        
        # Task queue for prefetch requests
        self.task_queue: PriorityQueue[PrefetchTask] = PriorityQueue()
        
        # Track in-progress and completed tasks
        self.in_progress: set[str] = set()
        self.completed: set[str] = set()
        
        # Worker threads
        self.workers: list[threading.Thread] = []
        self.stop_event = threading.Event()
        
        # Statistics
        self.stats = {
            "weights_loaded": 0,
            "total_bytes_loaded": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "dtype_conversions": 0,
        }
        
        self._lock = threading.Lock()
        
        logger.info(
            f"Initialized WeightPrefetcher with {num_workers} workers, "
            f"target={target_device}, dtype={target_dtype}"
        )
    
    def start(self):
        """Start prefetch worker threads."""
        if self.workers:
            logger.warning("Workers already started")
            return
        
        for i in range(self.num_workers):
            worker = threading.Thread(
                target=self._worker_loop,
                name=f"Prefetcher-{i}",
                daemon=True
            )
            worker.start()
            self.workers.append(worker)
        
        logger.info(f"Started {self.num_workers} prefetch workers")
    
    def stop(self):
        """Stop prefetch worker threads."""
        self.stop_event.set()
        
        for worker in self.workers:
            worker.join(timeout=5.0)
        
        self.workers.clear()
        logger.info("Stopped prefetch workers")
    
    def prefetch(
        self,
        requirement: WeightRequirement,
        callback: Optional[Callable] = None
    ):
        """Queue a weight prefetch request.
        
        Args:
            requirement: Weight requirement to prefetch
            callback: Optional callback when prefetch completes
        """
        task = PrefetchTask(
            priority=-requirement.priority,  # Negative for max-heap behavior
            requirement=requirement,
            callback=callback
        )
        
        self.task_queue.put(task)
        logger.debug(f"Queued prefetch: {requirement.layer_name}")
    
    def prefetch_batch(
        self,
        requirements: list[WeightRequirement],
        callback: Optional[Callable] = None
    ):
        """Queue multiple prefetch requests.
        
        Args:
            requirements: List of weight requirements
            callback: Optional callback for each completion
        """
        for req in requirements:
            self.prefetch(req, callback)
    
    def _worker_loop(self):
        """Worker thread main loop."""
        while not self.stop_event.is_set():
            try:
                # Get task with timeout to allow checking stop_event
                task = self.task_queue.get(timeout=1.0)
            except:
                continue
            
            try:
                self._execute_prefetch(task)
            except Exception as e:
                logger.error(f"Prefetch error: {e}", exc_info=True)
            finally:
                self.task_queue.task_done()
    
    def _execute_prefetch(self, task: PrefetchTask):
        """Execute a single prefetch task.
        
        Args:
            task: PrefetchTask to execute
        """
        req = task.requirement
        layer_name = req.layer_name
        
        # Check if already in progress or completed
        with self._lock:
            if layer_name in self.in_progress or layer_name in self.completed:
                logger.debug(f"Skipping {layer_name} (already processed)")
                return
            self.in_progress.add(layer_name)
        
        try:
            # For each weight key in the requirement
            for weight_key in req.weight_keys:
                full_key = f"{layer_name}.{weight_key}"
                
                # Check if already in cache
                if self.cache.contains(full_key):
                    with self._lock:
                        self.stats["cache_hits"] += 1
                    logger.debug(f"Cache hit: {full_key}")
                    continue
                
                with self._lock:
                    self.stats["cache_misses"] += 1
                
                # Load from disk
                weight_tensor = self._load_from_disk(layer_name, weight_key)
                
                if weight_tensor is not None:
                    # Convert dtype if needed
                    if weight_tensor.dtype != self.target_dtype:
                        weight_tensor = weight_tensor.to(self.target_dtype)
                        with self._lock:
                            self.stats["dtype_conversions"] += 1
                    
                    # Store in cache
                    metadata = {
                        "dtype": str(weight_tensor.dtype),
                        "shape": list(weight_tensor.shape),
                        "device": str(weight_tensor.device),
                    }
                    
                    success = self.cache.put(full_key, weight_tensor, metadata)
                    
                    if success:
                        with self._lock:
                            self.stats["weights_loaded"] += 1
                            self.stats["total_bytes_loaded"] += (
                                weight_tensor.element_size() * weight_tensor.nelement()
                            )
                        logger.debug(f"Loaded and cached: {full_key}")
                    else:
                        logger.warning(f"Failed to cache: {full_key}")
            
            # Mark as completed
            with self._lock:
                self.in_progress.remove(layer_name)
                self.completed.add(layer_name)
            
            # Execute callback if provided
            if task.callback:
                task.callback(layer_name)
            
        except Exception as e:
            logger.error(f"Error prefetching {layer_name}: {e}", exc_info=True)
            with self._lock:
                self.in_progress.discard(layer_name)
    
    def _load_from_disk(
        self,
        layer_name: str,
        weight_key: str
    ) -> Optional[torch.Tensor]:
        """Load weight tensor from disk.
        
        Args:
            layer_name: Layer identifier
            weight_key: Weight key within layer
            
        Returns:
            Loaded tensor or None if not found
        """
        # Construct file path (adjust based on actual model format)
        # This assumes PyTorch model format with safetensors or pickle
        
        # Try different file patterns
        patterns = [
            self.model_path / f"{layer_name}.{weight_key}.pt",
            self.model_path / f"{layer_name}.pt",
            self.model_path / "pytorch_model.bin",
        ]
        
        for path in patterns:
            if path.exists():
                try:
                    # Load tensor
                    if path.suffix == ".pt":
                        state_dict = torch.load(
                            path,
                            map_location="cpu"  # Load to CPU first
                        )
                        
                        # Extract specific weight if in dict
                        if isinstance(state_dict, dict):
                            full_key = f"{layer_name}.{weight_key}"
                            if full_key in state_dict:
                                return state_dict[full_key]
                            elif weight_key in state_dict:
                                return state_dict[weight_key]
                        else:
                            return state_dict
                    
                except Exception as e:
                    logger.warning(f"Failed to load from {path}: {e}")
                    continue
        
        logger.warning(f"Could not find weight: {layer_name}.{weight_key}")
        return None
    
    def get_weight(
        self,
        layer_name: str,
        weight_key: str,
        device: Optional[torch.device] = None
    ) -> Optional[torch.Tensor]:
        """Get weight from cache and transfer to device if needed.
        
        Args:
            layer_name: Layer identifier
            weight_key: Weight key within layer
            device: Target device (defaults to self.target_device)
            
        Returns:
            Weight tensor on target device or None if not available
        """
        full_key = f"{layer_name}.{weight_key}"
        
        # Get from cache
        tensor = self.cache.get(full_key)
        
        if tensor is None:
            return None
        
        # Transfer to device if needed
        target = device or self.target_device
        if tensor.device != target:
            tensor = tensor.to(target)
        
        return tensor
    
    def wait_for_completion(self, timeout: Optional[float] = None):
        """Wait for all pending prefetch tasks to complete.
        
        Args:
            timeout: Maximum time to wait in seconds
        """
        self.task_queue.join()
        logger.debug("All prefetch tasks completed")
    
    def clear_completed(self):
        """Clear completed task tracking."""
        with self._lock:
            self.completed.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get prefetcher statistics.
        
        Returns:
            Dictionary with prefetch statistics
        """
        with self._lock:
            return {
                **self.stats,
                "queue_size": self.task_queue.qsize(),
                "in_progress": len(self.in_progress),
                "completed": len(self.completed),
                "total_gb_loaded": self.stats["total_bytes_loaded"] / (1024 ** 3),
            }
    
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
