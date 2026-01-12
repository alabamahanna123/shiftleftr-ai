"""
Shift Left AI: Design-first hallucination management for enterprise AI systems.

Core modules:
- preflight: Pre-generation assessment of task feasibility
- graduated: Confidence-banded responses with uncertainty tracking
- design: Reliability contracts and architecture recommendations
- orchestration: Compositional reliability for multi-step workflows
- connectors: Platform-specific integrations (Domo, etc.)
"""

from shiftleft.preflight.assessment import PreflightAssessment, PreflightResult
from shiftleft.graduated.response import GraduatedResponse, ConfidenceBand
from shiftleft.design.contracts import ReliabilityContract, ContractBuilder
from shiftleft.orchestration.chain import ReliabilityChain, ChainBuilder

__version__ = "0.1.0"
__author__ = "Lee Griffiths"

__all__ = [
    # Core modules
    "PreflightAssessment",
    "PreflightResult",
    "GraduatedResponse",
    "ConfidenceBand",
    "ReliabilityContract",
    "ContractBuilder",
    "ReliabilityChain",
    "ChainBuilder",
]

# Lazy import for connectors to avoid dependency issues
def __getattr__(name):
    if name == "domo":
        from shiftleft.connectors import domo
        return domo
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
