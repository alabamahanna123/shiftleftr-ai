"""
Shift Left AI: Graduated Response Module

Replace binary ANSWER/REFUSE with confidence-banded outputs that give
users actionable information about uncertainty.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict, Any


class ConfidenceBand(Enum):
    """Confidence bands for graduated responses."""
    HIGH = "high_confidence"
    MODERATE = "moderate_confidence"
    LOW = "low_confidence"
    INSUFFICIENT = "insufficient"
    
    @property
    def description(self) -> str:
        descriptions = {
            "high_confidence": "High confidence in accuracy. Suitable for direct use.",
            "moderate_confidence": "Moderate confidence. Verify critical details before use.",
            "low_confidence": "Low confidence. Use as starting point only.",
            "insufficient": "Insufficient information to provide reliable answer.",
        }
        return descriptions[self.value]
    
    @property
    def reliability_range(self) -> tuple[float, float]:
        """Returns (min, max) reliability for this band."""
        ranges = {
            "high_confidence": (0.95, 1.0),
            "moderate_confidence": (0.85, 0.95),
            "low_confidence": (0.70, 0.85),
            "insufficient": (0.0, 0.70),
        }
        return ranges[self.value]


@dataclass
class UncertaintyFlag:
    """Specific uncertainty indicator within a response."""
    claim: str
    uncertainty_type: str  # "factual", "temporal", "source", "reasoning"
    severity: str  # "minor", "moderate", "major"
    mitigation: str  # How to address this uncertainty


@dataclass
class VerificationStep:
    """Recommended verification action for the user."""
    action: str
    rationale: str
    priority: str  # "required", "recommended", "optional"
    resources: List[str] = field(default_factory=list)


@dataclass
class GraduatedResponse:
    """
    A response with explicit confidence banding and uncertainty tracking.
    
    Unlike binary ANSWER/REFUSE, this provides:
    - Clear confidence band (high/moderate/low/insufficient)
    - Specific uncertainty flags within the response
    - Actionable verification steps
    - Transparent metrics for audit
    """
    
    # Core response
    confidence_band: ConfidenceBand
    answer: Optional[str]
    
    # Uncertainty detail
    uncertainty_flags: List[UncertaintyFlag]
    verification_steps: List[VerificationStep]
    
    # Transparency
    reliability_estimate: float
    isr: float  # Information Sufficiency Ratio from EDFL
    roh_bound: float  # Risk of Hallucination bound
    
    # Provenance
    sources_used: List[str]
    reasoning_trace: Optional[str]
    
    # Metadata
    meta: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_edfl_metrics(
        cls,
        metrics: Any,  # ItemMetrics from hallbayes
        answer: Optional[str] = None,
        sources: Optional[List[str]] = None,
        custom_thresholds: Optional[Dict[str, float]] = None
    ) -> "GraduatedResponse":
        """
        Create a GraduatedResponse from EDFL/HallBayes metrics.
        
        Default thresholds:
        - HIGH: ISR > 2.0, RoH < 0.02
        - MODERATE: ISR > 1.0, RoH < 0.10
        - LOW: ISR > 0.5, RoH < 0.25
        - INSUFFICIENT: otherwise
        """
        thresholds = custom_thresholds or {
            "high_isr": 2.0,
            "high_roh": 0.02,
            "moderate_isr": 1.0,
            "moderate_roh": 0.10,
            "low_isr": 0.5,
            "low_roh": 0.25,
        }
        
        # Extract metrics
        isr = getattr(metrics, "isr", 0.0)
        roh_bound = getattr(metrics, "roh_bound", 1.0)
        decision_answer = getattr(metrics, "decision_answer", False)
        
        # Determine confidence band
        if isr >= thresholds["high_isr"] and roh_bound <= thresholds["high_roh"]:
            band = ConfidenceBand.HIGH
        elif isr >= thresholds["moderate_isr"] and roh_bound <= thresholds["moderate_roh"]:
            band = ConfidenceBand.MODERATE
        elif isr >= thresholds["low_isr"] and roh_bound <= thresholds["low_roh"]:
            band = ConfidenceBand.LOW
        else:
            band = ConfidenceBand.INSUFFICIENT
        
        # Generate uncertainty flags
        flags = cls._generate_uncertainty_flags(metrics, band)
        
        # Generate verification steps
        verification = cls._generate_verification_steps(band, flags)
        
        # Calculate reliability estimate
        reliability = 1.0 - roh_bound
        
        return cls(
            confidence_band=band,
            answer=answer if band != ConfidenceBand.INSUFFICIENT else None,
            uncertainty_flags=flags,
            verification_steps=verification,
            reliability_estimate=reliability,
            isr=isr,
            roh_bound=roh_bound,
            sources_used=sources or [],
            reasoning_trace=getattr(metrics, "rationale", None),
            meta={
                "original_decision": decision_answer,
                "q_conservative": getattr(metrics, "q_conservative", None),
                "q_avg": getattr(metrics, "q_avg", None),
                "b2t": getattr(metrics, "b2t", None),
            }
        )
    
    @classmethod
    def from_preflight(
        cls,
        preflight_result: Any,  # PreflightResult
        answer: Optional[str] = None
    ) -> "GraduatedResponse":
        """Create a GraduatedResponse from pre-flight assessment."""
        
        # Map preflight to confidence band
        if preflight_result.estimated_reliability >= 0.95:
            band = ConfidenceBand.HIGH
        elif preflight_result.estimated_reliability >= 0.85:
            band = ConfidenceBand.MODERATE
        elif preflight_result.estimated_reliability >= 0.70:
            band = ConfidenceBand.LOW
        else:
            band = ConfidenceBand.INSUFFICIENT
        
        # Convert gaps to uncertainty flags
        flags = []
        for gap in preflight_result.information_gaps:
            flags.append(UncertaintyFlag(
                claim="Overall response",
                uncertainty_type="source",
                severity="moderate" if band in [ConfidenceBand.HIGH, ConfidenceBand.MODERATE] else "major",
                mitigation=gap
            ))
        
        # Convert risks to flags
        for risk in preflight_result.risk_factors:
            flags.append(UncertaintyFlag(
                claim="Overall response",
                uncertainty_type="source",
                severity="minor" if "may" in risk.lower() else "moderate",
                mitigation=risk
            ))
        
        # Generate verification steps
        verification = []
        for required_source in preflight_result.required_sources:
            verification.append(VerificationStep(
                action=f"Obtain: {required_source}",
                rationale="Identified as missing during pre-flight assessment",
                priority="recommended"
            ))
        
        return cls(
            confidence_band=band,
            answer=answer if preflight_result.is_feasible else None,
            uncertainty_flags=flags,
            verification_steps=verification,
            reliability_estimate=preflight_result.estimated_reliability,
            isr=0.0,  # Not applicable for preflight
            roh_bound=1.0 - preflight_result.estimated_reliability,
            sources_used=[s.source_id for s in preflight_result.sources_assessed],
            reasoning_trace=preflight_result.feasibility_rationale,
            meta={
                "preflight_recommendation": preflight_result.recommendation,
                "task_complexity": preflight_result.complexity.value,
                "confidence_in_estimate": preflight_result.confidence_in_estimate,
            }
        )
    
    @staticmethod
    def _generate_uncertainty_flags(
        metrics: Any, 
        band: ConfidenceBand
    ) -> List[UncertaintyFlag]:
        """Generate specific uncertainty flags from metrics."""
        flags = []
        
        # Check for low ISR
        isr = getattr(metrics, "isr", 0.0)
        if isr < 1.0:
            flags.append(UncertaintyFlag(
                claim="Overall response",
                uncertainty_type="source",
                severity="major" if isr < 0.5 else "moderate",
                mitigation="Information sufficiency below threshold. Consider additional sources."
            ))
        
        # Check for high RoH
        roh = getattr(metrics, "roh_bound", 1.0)
        if roh > 0.10:
            flags.append(UncertaintyFlag(
                claim="Overall response",
                uncertainty_type="factual",
                severity="major" if roh > 0.25 else "moderate",
                mitigation=f"Hallucination risk bound is {roh:.0%}. Verify key claims independently."
            ))
        
        # Check for low prior
        q_lo = getattr(metrics, "q_conservative", None)
        if q_lo is not None and q_lo < 0.3:
            flags.append(UncertaintyFlag(
                claim="Overall response",
                uncertainty_type="reasoning",
                severity="moderate",
                mitigation="Model showed low baseline confidence. May be outside training distribution."
            ))
        
        return flags
    
    @staticmethod
    def _generate_verification_steps(
        band: ConfidenceBand,
        flags: List[UncertaintyFlag]
    ) -> List[VerificationStep]:
        """Generate verification steps based on band and flags."""
        steps = []
        
        if band == ConfidenceBand.HIGH:
            steps.append(VerificationStep(
                action="Spot-check one key claim",
                rationale="Standard verification even for high-confidence responses",
                priority="optional"
            ))
        
        elif band == ConfidenceBand.MODERATE:
            steps.append(VerificationStep(
                action="Verify all specific facts (dates, numbers, names)",
                rationale="Moderate confidence suggests potential for minor errors",
                priority="recommended"
            ))
            steps.append(VerificationStep(
                action="Cross-reference with authoritative source",
                rationale="Second source confirmation increases reliability",
                priority="recommended"
            ))
        
        elif band == ConfidenceBand.LOW:
            steps.append(VerificationStep(
                action="Treat as draft only - verify all claims before use",
                rationale="Low confidence indicates significant uncertainty",
                priority="required"
            ))
            steps.append(VerificationStep(
                action="Seek expert review if high-stakes decision",
                rationale="Human expertise needed for important decisions",
                priority="required"
            ))
            steps.append(VerificationStep(
                action="Consider alternative information sources",
                rationale="Original sources may be insufficient",
                priority="recommended"
            ))
        
        else:  # INSUFFICIENT
            steps.append(VerificationStep(
                action="Do not use this response - information insufficient",
                rationale="Reliability below acceptable threshold",
                priority="required"
            ))
            steps.append(VerificationStep(
                action="Gather additional sources before re-attempting",
                rationale="Task requires more information to answer reliably",
                priority="required"
            ))
        
        # Add flag-specific steps
        for flag in flags:
            if flag.severity == "major":
                steps.append(VerificationStep(
                    action=flag.mitigation,
                    rationale=f"Addressing {flag.uncertainty_type} uncertainty",
                    priority="required" if flag.severity == "major" else "recommended"
                ))
        
        return steps
    
    def to_user_message(self) -> str:
        """Format the graduated response for end-user consumption."""
        lines = []
        
        # Confidence indicator
        band_emoji = {
            ConfidenceBand.HIGH: "✓",
            ConfidenceBand.MODERATE: "~",
            ConfidenceBand.LOW: "?",
            ConfidenceBand.INSUFFICIENT: "✗",
        }
        
        lines.append(f"[{band_emoji[self.confidence_band]} {self.confidence_band.value.replace('_', ' ').title()}]")
        lines.append("")
        
        # Answer or explanation
        if self.answer:
            lines.append(self.answer)
        else:
            lines.append("Unable to provide a reliable answer with available information.")
        
        lines.append("")
        
        # Uncertainty summary (only if moderate or low)
        if self.confidence_band in [ConfidenceBand.MODERATE, ConfidenceBand.LOW]:
            major_flags = [f for f in self.uncertainty_flags if f.severity == "major"]
            if major_flags:
                lines.append("Key uncertainties:")
                for flag in major_flags[:3]:  # Limit to 3
                    lines.append(f"  • {flag.mitigation}")
                lines.append("")
        
        # Required actions (only for low/insufficient)
        if self.confidence_band in [ConfidenceBand.LOW, ConfidenceBand.INSUFFICIENT]:
            required_steps = [s for s in self.verification_steps if s.priority == "required"]
            if required_steps:
                lines.append("Required before use:")
                for step in required_steps[:3]:
                    lines.append(f"  • {step.action}")
        
        return "\n".join(lines)
    
    def to_audit_record(self) -> Dict[str, Any]:
        """Format for audit logging."""
        return {
            "confidence_band": self.confidence_band.value,
            "reliability_estimate": self.reliability_estimate,
            "isr": self.isr,
            "roh_bound": self.roh_bound,
            "uncertainty_count": len(self.uncertainty_flags),
            "major_uncertainties": len([f for f in self.uncertainty_flags if f.severity == "major"]),
            "verification_required": len([s for s in self.verification_steps if s.priority == "required"]),
            "sources_used": self.sources_used,
            "meta": self.meta,
        }


class GraduatedResponseBuilder:
    """
    Builder for creating graduated responses with custom configuration.
    """
    
    def __init__(self):
        self._thresholds = {
            "high_isr": 2.0,
            "high_roh": 0.02,
            "moderate_isr": 1.0,
            "moderate_roh": 0.10,
            "low_isr": 0.5,
            "low_roh": 0.25,
        }
        self._custom_flags: List[UncertaintyFlag] = []
        self._custom_steps: List[VerificationStep] = []
    
    def with_thresholds(
        self,
        high_isr: float = 2.0,
        high_roh: float = 0.02,
        moderate_isr: float = 1.0,
        moderate_roh: float = 0.10,
        low_isr: float = 0.5,
        low_roh: float = 0.25
    ) -> "GraduatedResponseBuilder":
        """Set custom thresholds for confidence bands."""
        self._thresholds = {
            "high_isr": high_isr,
            "high_roh": high_roh,
            "moderate_isr": moderate_isr,
            "moderate_roh": moderate_roh,
            "low_isr": low_isr,
            "low_roh": low_roh,
        }
        return self
    
    def add_uncertainty_flag(
        self,
        claim: str,
        uncertainty_type: str,
        severity: str,
        mitigation: str
    ) -> "GraduatedResponseBuilder":
        """Add a custom uncertainty flag."""
        self._custom_flags.append(UncertaintyFlag(
            claim=claim,
            uncertainty_type=uncertainty_type,
            severity=severity,
            mitigation=mitigation
        ))
        return self
    
    def add_verification_step(
        self,
        action: str,
        rationale: str,
        priority: str = "recommended",
        resources: Optional[List[str]] = None
    ) -> "GraduatedResponseBuilder":
        """Add a custom verification step."""
        self._custom_steps.append(VerificationStep(
            action=action,
            rationale=rationale,
            priority=priority,
            resources=resources or []
        ))
        return self
    
    def build_from_metrics(
        self,
        metrics: Any,
        answer: Optional[str] = None,
        sources: Optional[List[str]] = None
    ) -> GraduatedResponse:
        """Build graduated response from EDFL metrics."""
        response = GraduatedResponse.from_edfl_metrics(
            metrics=metrics,
            answer=answer,
            sources=sources,
            custom_thresholds=self._thresholds
        )
        
        # Add custom flags and steps
        response.uncertainty_flags.extend(self._custom_flags)
        response.verification_steps.extend(self._custom_steps)
        
        return response
