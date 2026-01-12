"""
Shift Left AI: Design Constraints Module

Tooling for defining reliability requirements BEFORE building AI systems.
This is the architectural layer - embedding hallucination management into
system design rather than adding it as runtime detection.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional, Any
import json


class TaskCategory(Enum):
    """Categories of AI tasks with different reliability profiles."""
    FACTUAL_RETRIEVAL = "factual_retrieval"
    SUMMARISATION = "summarisation"
    ANALYSIS = "analysis"
    GENERATION = "generation"
    CLASSIFICATION = "classification"
    EXTRACTION = "extraction"
    TRANSLATION = "translation"
    CODE_GENERATION = "code_generation"
    REASONING = "reasoning"
    CONVERSATION = "conversation"


class DecompositionStrategy(Enum):
    """Strategies for breaking down complex tasks."""
    NONE = "none"  # Atomic task
    CLAIM_BY_CLAIM = "claim_by_claim"  # Verify each claim independently
    PARALLEL = "parallel"  # Independent subtasks
    SEQUENTIAL = "sequential"  # Dependent chain
    HIERARCHICAL = "hierarchical"  # Tree structure
    ITERATIVE = "iterative"  # Refinement loops


class EscalationTrigger(Enum):
    """Conditions that trigger escalation to human review."""
    RELIABILITY_BELOW_THRESHOLD = "reliability_below_threshold"
    CONFIDENCE_BAND_LOW = "confidence_band_low"
    CONFIDENCE_BAND_INSUFFICIENT = "confidence_band_insufficient"
    HIGH_STAKES_DECISION = "high_stakes_decision"
    NOVEL_QUERY = "novel_query"
    CONFLICTING_SOURCES = "conflicting_sources"
    USER_REQUEST = "user_request"


@dataclass
class SourceRequirement:
    """Specification for required information sources."""
    source_type: str  # "authoritative", "secondary", "user_provided", etc.
    minimum_count: int = 1
    freshness_days: Optional[int] = None  # Max age in days
    required_coverage: float = 0.8  # Minimum coverage required
    verification_method: str = "none"  # "none", "cross_reference", "expert_review"
    fallback_allowed: bool = True


@dataclass
class MonitoringRequirement:
    """Specification for runtime monitoring."""
    metric: str
    threshold: float
    action: str  # "log", "alert", "escalate", "block"
    window_size: Optional[int] = None  # For rolling metrics


@dataclass
class ArchitectureRecommendation:
    """Specific architectural recommendation."""
    component: str
    recommendation: str
    rationale: str
    priority: str  # "required", "recommended", "optional"
    implementation_notes: Optional[str] = None


@dataclass
class ReliabilityContract:
    """
    A formal specification of reliability requirements for an AI task.
    
    This is the core "shift left" artifact - defining requirements
    before building, not testing after deployment.
    """
    
    # Identity
    contract_id: str
    task_type: TaskCategory
    description: str
    
    # Reliability targets
    max_hallucination_rate: float  # e.g., 0.05 for 5%
    min_confidence_band: str  # "high", "moderate", "low"
    required_source_coverage: float  # 0-1
    
    # Decomposition
    decomposition_strategy: DecompositionStrategy
    max_chain_length: int = 5  # For sequential/hierarchical
    
    # Sources
    source_requirements: List[SourceRequirement] = field(default_factory=list)
    
    # Monitoring
    monitoring_requirements: List[MonitoringRequirement] = field(default_factory=list)
    
    # Escalation
    escalation_triggers: List[EscalationTrigger] = field(default_factory=list)
    escalation_target: str = "human_review"  # Where to escalate
    
    # Generated recommendations
    architecture_recommendations: List[ArchitectureRecommendation] = field(default_factory=list)
    
    # Metadata
    created_at: Optional[str] = None
    version: str = "1.0"
    owner: Optional[str] = None
    
    def __post_init__(self):
        """Generate recommendations based on requirements."""
        if not self.architecture_recommendations:
            self.architecture_recommendations = self._generate_recommendations()
        if not self.monitoring_requirements:
            self.monitoring_requirements = self._generate_monitoring()
        if not self.escalation_triggers:
            self.escalation_triggers = self._generate_escalation_triggers()
    
    def _generate_recommendations(self) -> List[ArchitectureRecommendation]:
        """Generate architecture recommendations from requirements."""
        recommendations = []
        
        # Hallucination rate recommendations
        if self.max_hallucination_rate <= 0.02:
            recommendations.append(ArchitectureRecommendation(
                component="grounding",
                recommendation="Implement continuous grounding with citation tracking",
                rationale=f"Target hallucination rate ({self.max_hallucination_rate:.0%}) requires strong grounding",
                priority="required",
                implementation_notes="Use retrieval-augmented generation with inline citations"
            ))
            recommendations.append(ArchitectureRecommendation(
                component="validation",
                recommendation="Add post-generation fact-checking layer",
                rationale="Sub-2% hallucination requires redundant validation",
                priority="required"
            ))
        elif self.max_hallucination_rate <= 0.05:
            recommendations.append(ArchitectureRecommendation(
                component="grounding",
                recommendation="Implement source-based generation with provenance tracking",
                rationale=f"Target hallucination rate ({self.max_hallucination_rate:.0%}) requires grounding",
                priority="required"
            ))
        elif self.max_hallucination_rate <= 0.10:
            recommendations.append(ArchitectureRecommendation(
                component="validation",
                recommendation="Implement confidence scoring with graduated responses",
                rationale="Moderate hallucination tolerance can use confidence bands",
                priority="recommended"
            ))
        
        # Source coverage recommendations
        if self.required_source_coverage >= 0.95:
            recommendations.append(ArchitectureRecommendation(
                component="retrieval",
                recommendation="Multi-source retrieval with coverage verification",
                rationale=f"High coverage requirement ({self.required_source_coverage:.0%})",
                priority="required",
                implementation_notes="Verify source coverage before generation"
            ))
        
        # Decomposition recommendations
        if self.decomposition_strategy == DecompositionStrategy.CLAIM_BY_CLAIM:
            recommendations.append(ArchitectureRecommendation(
                component="decomposition",
                recommendation="Implement claim extraction and per-claim verification",
                rationale="Claim-by-claim strategy requires granular tracking",
                priority="required",
                implementation_notes="Extract claims, verify each, track provenance"
            ))
        elif self.decomposition_strategy == DecompositionStrategy.SEQUENTIAL:
            recommendations.append(ArchitectureRecommendation(
                component="orchestration",
                recommendation="Implement reliability chain with handoff validation",
                rationale="Sequential tasks need reliability propagation",
                priority="required",
                implementation_notes="Validate at each step, track cumulative reliability"
            ))
        
        # Task-specific recommendations
        if self.task_type == TaskCategory.FACTUAL_RETRIEVAL:
            recommendations.append(ArchitectureRecommendation(
                component="retrieval",
                recommendation="Use authoritative source prioritisation",
                rationale="Factual tasks benefit from authoritative sources",
                priority="recommended"
            ))
        elif self.task_type == TaskCategory.SUMMARISATION:
            recommendations.append(ArchitectureRecommendation(
                component="validation",
                recommendation="Implement faithfulness checking against source",
                rationale="Summaries must be faithful to source material",
                priority="required"
            ))
        elif self.task_type == TaskCategory.CODE_GENERATION:
            recommendations.append(ArchitectureRecommendation(
                component="validation",
                recommendation="Add syntax validation and test generation",
                rationale="Code outputs can be programmatically verified",
                priority="required"
            ))
        elif self.task_type == TaskCategory.REASONING:
            recommendations.append(ArchitectureRecommendation(
                component="decomposition",
                recommendation="Implement chain-of-thought with step verification",
                rationale="Reasoning tasks benefit from explicit step tracking",
                priority="recommended"
            ))
        
        return recommendations
    
    def _generate_monitoring(self) -> List[MonitoringRequirement]:
        """Generate monitoring requirements from targets."""
        monitoring = []
        
        # Core hallucination monitoring
        monitoring.append(MonitoringRequirement(
            metric="hallucination_rate_rolling",
            threshold=self.max_hallucination_rate * 1.5,  # Alert before breach
            action="alert",
            window_size=100
        ))
        
        monitoring.append(MonitoringRequirement(
            metric="hallucination_rate_rolling",
            threshold=self.max_hallucination_rate * 2.0,
            action="escalate",
            window_size=100
        ))
        
        # Confidence band monitoring
        band_thresholds = {
            "high": 0.80,
            "moderate": 0.60,
            "low": 0.40,
        }
        if self.min_confidence_band in band_thresholds:
            monitoring.append(MonitoringRequirement(
                metric="high_confidence_rate",
                threshold=band_thresholds[self.min_confidence_band],
                action="alert",
                window_size=50
            ))
        
        # Source coverage monitoring
        if self.required_source_coverage > 0.8:
            monitoring.append(MonitoringRequirement(
                metric="source_coverage_rate",
                threshold=self.required_source_coverage * 0.9,
                action="alert"
            ))
        
        # ISR monitoring (EDFL-specific)
        monitoring.append(MonitoringRequirement(
            metric="isr_below_1_rate",
            threshold=0.20,  # Alert if >20% of queries have ISR < 1
            action="alert",
            window_size=100
        ))
        
        return monitoring
    
    def _generate_escalation_triggers(self) -> List[EscalationTrigger]:
        """Generate escalation triggers based on requirements."""
        triggers = []
        
        # Always include low/insufficient confidence escalation
        if self.min_confidence_band == "high":
            triggers.append(EscalationTrigger.CONFIDENCE_BAND_LOW)
        triggers.append(EscalationTrigger.CONFIDENCE_BAND_INSUFFICIENT)
        
        # Reliability threshold escalation
        triggers.append(EscalationTrigger.RELIABILITY_BELOW_THRESHOLD)
        
        # Conflicting sources for high-reliability tasks
        if self.max_hallucination_rate <= 0.05:
            triggers.append(EscalationTrigger.CONFLICTING_SOURCES)
        
        # User request always available
        triggers.append(EscalationTrigger.USER_REQUEST)
        
        return triggers
    
    def validate(self) -> List[str]:
        """Validate the contract for internal consistency."""
        issues = []
        
        # Check hallucination rate is achievable
        if self.max_hallucination_rate < 0.01:
            issues.append("Hallucination rate below 1% may not be achievable with current technology")
        
        # Check source requirements match coverage
        if self.required_source_coverage > 0.9 and not self.source_requirements:
            issues.append("High source coverage requires explicit source requirements")
        
        # Check decomposition matches task type
        if self.task_type == TaskCategory.GENERATION and self.decomposition_strategy == DecompositionStrategy.NONE:
            issues.append("Generative tasks benefit from decomposition strategy")
        
        # Check chain length for sequential
        if self.decomposition_strategy == DecompositionStrategy.SEQUENTIAL and self.max_chain_length > 10:
            issues.append("Long sequential chains accumulate reliability loss")
        
        return issues
    
    def to_json(self) -> str:
        """Serialise contract to JSON."""
        return json.dumps({
            "contract_id": self.contract_id,
            "task_type": self.task_type.value,
            "description": self.description,
            "max_hallucination_rate": self.max_hallucination_rate,
            "min_confidence_band": self.min_confidence_band,
            "required_source_coverage": self.required_source_coverage,
            "decomposition_strategy": self.decomposition_strategy.value,
            "max_chain_length": self.max_chain_length,
            "source_requirements": [
                {
                    "source_type": s.source_type,
                    "minimum_count": s.minimum_count,
                    "freshness_days": s.freshness_days,
                    "required_coverage": s.required_coverage,
                    "verification_method": s.verification_method,
                    "fallback_allowed": s.fallback_allowed,
                }
                for s in self.source_requirements
            ],
            "monitoring_requirements": [
                {
                    "metric": m.metric,
                    "threshold": m.threshold,
                    "action": m.action,
                    "window_size": m.window_size,
                }
                for m in self.monitoring_requirements
            ],
            "escalation_triggers": [t.value for t in self.escalation_triggers],
            "escalation_target": self.escalation_target,
            "architecture_recommendations": [
                {
                    "component": r.component,
                    "recommendation": r.recommendation,
                    "rationale": r.rationale,
                    "priority": r.priority,
                    "implementation_notes": r.implementation_notes,
                }
                for r in self.architecture_recommendations
            ],
            "version": self.version,
            "owner": self.owner,
        }, indent=2)
    
    @classmethod
    def from_json(cls, json_str: str) -> "ReliabilityContract":
        """Deserialise contract from JSON."""
        data = json.loads(json_str)
        
        return cls(
            contract_id=data["contract_id"],
            task_type=TaskCategory(data["task_type"]),
            description=data["description"],
            max_hallucination_rate=data["max_hallucination_rate"],
            min_confidence_band=data["min_confidence_band"],
            required_source_coverage=data["required_source_coverage"],
            decomposition_strategy=DecompositionStrategy(data["decomposition_strategy"]),
            max_chain_length=data.get("max_chain_length", 5),
            source_requirements=[
                SourceRequirement(**s) for s in data.get("source_requirements", [])
            ],
            monitoring_requirements=[
                MonitoringRequirement(**m) for m in data.get("monitoring_requirements", [])
            ],
            escalation_triggers=[
                EscalationTrigger(t) for t in data.get("escalation_triggers", [])
            ],
            escalation_target=data.get("escalation_target", "human_review"),
            architecture_recommendations=[
                ArchitectureRecommendation(**r) for r in data.get("architecture_recommendations", [])
            ],
            version=data.get("version", "1.0"),
            owner=data.get("owner"),
        )


class ContractBuilder:
    """
    Builder for creating reliability contracts with sensible defaults.
    """
    
    @staticmethod
    def for_factual_qa(
        contract_id: str,
        description: str = "Factual question answering",
        max_hallucination_rate: float = 0.05
    ) -> ReliabilityContract:
        """Create contract for factual Q&A tasks."""
        return ReliabilityContract(
            contract_id=contract_id,
            task_type=TaskCategory.FACTUAL_RETRIEVAL,
            description=description,
            max_hallucination_rate=max_hallucination_rate,
            min_confidence_band="moderate",
            required_source_coverage=0.85,
            decomposition_strategy=DecompositionStrategy.NONE,
            source_requirements=[
                SourceRequirement(
                    source_type="authoritative",
                    minimum_count=1,
                    freshness_days=365,
                    required_coverage=0.7
                ),
                SourceRequirement(
                    source_type="secondary",
                    minimum_count=1,
                    required_coverage=0.5,
                    verification_method="cross_reference"
                )
            ]
        )
    
    @staticmethod
    def for_summarisation(
        contract_id: str,
        description: str = "Document summarisation",
        max_hallucination_rate: float = 0.03
    ) -> ReliabilityContract:
        """Create contract for summarisation tasks."""
        return ReliabilityContract(
            contract_id=contract_id,
            task_type=TaskCategory.SUMMARISATION,
            description=description,
            max_hallucination_rate=max_hallucination_rate,
            min_confidence_band="high",
            required_source_coverage=0.95,  # Summary must cover source
            decomposition_strategy=DecompositionStrategy.CLAIM_BY_CLAIM,
            source_requirements=[
                SourceRequirement(
                    source_type="user_provided",
                    minimum_count=1,
                    required_coverage=1.0,  # Must have the source document
                    verification_method="none"
                )
            ]
        )
    
    @staticmethod
    def for_analysis(
        contract_id: str,
        description: str = "Analytical task",
        max_hallucination_rate: float = 0.10
    ) -> ReliabilityContract:
        """Create contract for analytical tasks."""
        return ReliabilityContract(
            contract_id=contract_id,
            task_type=TaskCategory.ANALYSIS,
            description=description,
            max_hallucination_rate=max_hallucination_rate,
            min_confidence_band="moderate",
            required_source_coverage=0.80,
            decomposition_strategy=DecompositionStrategy.HIERARCHICAL,
            max_chain_length=4,
            source_requirements=[
                SourceRequirement(
                    source_type="authoritative",
                    minimum_count=2,
                    required_coverage=0.6,
                    verification_method="cross_reference"
                )
            ]
        )
    
    @staticmethod
    def for_high_stakes(
        contract_id: str,
        task_type: TaskCategory,
        description: str
    ) -> ReliabilityContract:
        """Create contract for high-stakes decisions."""
        return ReliabilityContract(
            contract_id=contract_id,
            task_type=task_type,
            description=description,
            max_hallucination_rate=0.01,  # Very strict
            min_confidence_band="high",
            required_source_coverage=0.95,
            decomposition_strategy=DecompositionStrategy.CLAIM_BY_CLAIM,
            source_requirements=[
                SourceRequirement(
                    source_type="authoritative",
                    minimum_count=2,
                    freshness_days=30,
                    required_coverage=0.8,
                    verification_method="expert_review"
                )
            ],
            escalation_triggers=[
                EscalationTrigger.RELIABILITY_BELOW_THRESHOLD,
                EscalationTrigger.CONFIDENCE_BAND_LOW,
                EscalationTrigger.CONFIDENCE_BAND_INSUFFICIENT,
                EscalationTrigger.HIGH_STAKES_DECISION,
                EscalationTrigger.CONFLICTING_SOURCES,
                EscalationTrigger.NOVEL_QUERY,
                EscalationTrigger.USER_REQUEST,
            ]
        )
