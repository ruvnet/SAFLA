"""
SAFLA - Self-Aware Feedback Loop Algorithm

A sophisticated AI/ML system implementing autonomous learning and adaptation with
comprehensive safety mechanisms, hybrid memory architecture, and meta-cognitive capabilities.

This package provides:
- Hybrid Memory Architecture with vector, episodic, semantic, and working memory
- Meta-Cognitive Engine for self-awareness and adaptive learning
- MCP Orchestration for distributed agent coordination
- Safety & Validation Framework with comprehensive constraints
- Delta Evaluation System for quantifying improvements
"""

__version__ = "0.1.4"
__author__ = "SAFLA Development Team"
__email__ = "ruv@ruv.net"
__license__ = "MIT"
__url__ = "https://github.com/ruvnet/SAFLA"

# Core component imports
from safla.core.hybrid_memory import HybridMemoryArchitecture
from safla.core.meta_cognitive_engine import MetaCognitiveEngine
from safla.core.mcp_orchestration import OptimizedMCPOrchestrator as MCPOrchestrator
from safla.core.safety_validation import OptimizedSafetyValidator as SafetyValidationFramework
from safla.core.delta_evaluation import OptimizedDeltaEvaluator as DeltaEvaluator

# Utility imports
from safla.utils.config import SAFLAConfig
from safla.utils.logging import get_logger

# Exception imports
from safla.exceptions import (
    SAFLAError,
    MemoryError,
    SafetyViolationError,
    MCPError,
    ConfigurationError
)

__all__ = [
    # Core Components
    "HybridMemoryArchitecture",
    "MetaCognitiveEngine",
    "MCPOrchestrator",
    "SafetyValidationFramework",
    "DeltaEvaluator",
    
    # Utilities
    "SAFLAConfig",
    "get_logger",
    
    # Exceptions
    "SAFLAError",
    "MemoryError",
    "SafetyViolationError",
    "MCPError",
    "ConfigurationError",
    
    # Metadata
    "__version__",
    "__author__",
    "__email__",
    "__license__",
    "__url__"
]

# Package-level configuration
def get_version():
    """Get the current version of SAFLA."""
    return __version__

def get_info():
    """Get package information."""
    return {
        "name": "safla",
        "version": __version__,
        "author": __author__,
        "email": __email__,
        "license": __license__,
        "url": __url__,
        "description": "Self-Aware Feedback Loop Algorithm"
    }