"""
Methods module - Defines training methods for PINN experiments.

This module provides an extensible framework for different collocation point
selection and mesh refinement strategies.

Available methods:
    - AdaptiveMethod: Residual-based adaptive mesh refinement
    - AdaptiveEntropyBalancedMethod: Entropy-balanced adaptive mesh sampling
    - AdaptiveHaltonBaseMethod: Adaptive sampling with Halton coverage backbone
    - AdaptivePowerTemperedMethod: Power-tempered adaptive mesh sampling
    - RandomMethod: Uniform random point sampling (baseline)
    - HaltonMethod: Halton low-discrepancy sequence sampling
    - SobolMethod: Sobol low-discrepancy sequence sampling
    - RADMethod: Residual-based Adaptive Distribution (Wu et al. 2022)
    
To add a new competitive method:
    1. Create a new file in this directory (e.g., uniform_refinement.py)
    2. Subclass TrainingMethod from base.py
    3. Implement required methods: get_collocation_points, refine_mesh, name
    4. Register in this __init__.py

Example:
    from methods import get_method
    
    method = get_method("adaptive")
    points = method.get_collocation_points(mesh, model, iteration)
"""

from .base import TrainingMethod
from .adaptive import AdaptiveMethod
from .adaptive_entropy_balanced import AdaptiveEntropyBalancedMethod
from .adaptive_halton_base import AdaptiveHaltonBaseMethod
from .adaptive_persistent import AdaptivePersistentMethod
from .adaptive_power_tempered import (
    AdaptivePowerTemperedBeta25Method,
    AdaptivePowerTemperedBeta30Method,
    AdaptivePowerTemperedFloor15Method,
    AdaptivePowerTemperedFloor25Method,
    AdaptivePowerTemperedMethod,
)
from .hybrid_anchor import AdaptiveHybridAnchorMethod
from .random import RandomMethod
from .quasi_random import HaltonMethod, SobolMethod
from .rad import RADMethod

# Registry of available methods for CLI/config selection
METHOD_REGISTRY = {
    "adaptive": AdaptiveMethod,
    "adaptive_entropy_balanced": AdaptiveEntropyBalancedMethod,
    "adaptive_halton_base": AdaptiveHaltonBaseMethod,
    "adaptive_persistent": AdaptivePersistentMethod,
    "adaptive_power_tempered": AdaptivePowerTemperedMethod,
    "adaptive_power_tempered_beta25": AdaptivePowerTemperedBeta25Method,
    "adaptive_power_tempered_beta30": AdaptivePowerTemperedBeta30Method,
    "adaptive_power_tempered_floor15": AdaptivePowerTemperedFloor15Method,
    "adaptive_power_tempered_floor25": AdaptivePowerTemperedFloor25Method,
    "adaptive_hybrid_anchor": AdaptiveHybridAnchorMethod,
    "random": RandomMethod,
    "halton": HaltonMethod,
    "sobol": SobolMethod,
    "rad": RADMethod,
}


def get_method(name: str, **kwargs) -> TrainingMethod:
    """Get a method instance by name.

    Args:
        name: Method name (e.g., 'adaptive', 'random')
        **kwargs: Additional arguments passed to method constructor

    Returns:
        TrainingMethod instance

    Raises:
        ValueError: If method name not found
    """
    if name not in METHOD_REGISTRY:
        available = ", ".join(METHOD_REGISTRY.keys())
        raise ValueError(f"Unknown method '{name}'. Available: {available}")
    return METHOD_REGISTRY[name](**kwargs)


def list_methods() -> list:
    """List available method names."""
    return list(METHOD_REGISTRY.keys())


def register_method(name: str, method_class: type):
    """Register a new method class.

    Args:
        name: Method name for CLI/config
        method_class: Class that subclasses TrainingMethod
    """
    METHOD_REGISTRY[name] = method_class
