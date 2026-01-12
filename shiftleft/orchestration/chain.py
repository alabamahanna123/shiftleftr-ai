"""
Shift Left AI: Orchestration Module

Propagate uncertainty through multi-step agent workflows.
Track reliability across agent boundaries and identify weak links.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional, Any, Callable
from datetime import datetime
import uuid


class StepStatus(Enum):
    """Status of a step in the reliability chain."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class HandoffValidation(Enum):
    """Validation strategy at step handoffs."""
    NONE = "none"  # No validation
    SCHEMA = "schema"  # Validate output schema
    CONFIDENCE = "confidence"  # Check confidence threshold
    SEMANTIC = "semantic"  # Semantic consistency check
    FULL = "full"  # All validations


@dataclass
class StepResult:
    """Result of executing a step in the chain."""
    step_id: str
    status: StepStatus
    output: Any
    reliability: float
    confidence_band: str
    execution_time_ms: int
    uncertainty_flags: List[str] = field(default_factory=list)
    error: Optional[str] = None


@dataclass
class ChainStep:
    """
    A single step in a reliability chain.
    
    Each step has:
    - An agent/function to execute
    - Reliability estimate for that step
    - Validation requirements
    - Fallback behaviour
    """
    
    step_id: str
    name: str
    agent: Callable  # Function that takes input and returns output
    estimated_reliability: float  # 0-1
    
    # Configuration
    handoff_validation: HandoffValidation = HandoffValidation.CONFIDENCE
    confidence_threshold: float = 0.70  # Minimum to proceed
    retry_count: int = 1
    timeout_ms: int = 30000
    
    # Fallback
    fallback_agent: Optional[Callable] = None
    fallback_reliability: float = 0.50
    
    # Dependencies
    depends_on: List[str] = field(default_factory=list)
    
    # Metadata
    description: Optional[str] = None


@dataclass
class ChainResult:
    """Result of executing a full reliability chain."""
    
    chain_id: str
    status: StepStatus
    
    # Step results
    step_results: List[StepResult]
    
    # Reliability metrics
    end_to_end_reliability: float
    weakest_link: Optional[str]  # Step ID of lowest reliability
    reliability_by_step: Dict[str, float]
    
    # Output
    final_output: Any
    final_confidence_band: str
    
    # Timing
    total_execution_time_ms: int
    
    # Recommendations
    intervention_points: List[str]  # Steps where intervention would help most
    
    def to_audit_record(self) -> Dict[str, Any]:
        """Format for audit logging."""
        return {
            "chain_id": self.chain_id,
            "status": self.status.value,
            "end_to_end_reliability": self.end_to_end_reliability,
            "weakest_link": self.weakest_link,
            "step_count": len(self.step_results),
            "failed_steps": len([s for s in self.step_results if s.status == StepStatus.FAILED]),
            "total_execution_time_ms": self.total_execution_time_ms,
            "intervention_points": self.intervention_points,
        }


class ReliabilityChain:
    """
    A chain of AI steps with reliability tracking.
    
    Calculates end-to-end reliability bounds, identifies weak links,
    and suggests intervention points.
    """
    
    def __init__(
        self,
        chain_id: Optional[str] = None,
        name: str = "Unnamed Chain",
        global_reliability_target: float = 0.90
    ):
        self.chain_id = chain_id or str(uuid.uuid4())[:8]
        self.name = name
        self.global_reliability_target = global_reliability_target
        self.steps: List[ChainStep] = []
        self._step_index: Dict[str, int] = {}
    
    def add_step(
        self,
        name: str,
        agent: Callable,
        reliability: float = 0.90,
        handoff_validation: HandoffValidation = HandoffValidation.CONFIDENCE,
        confidence_threshold: float = 0.70,
        depends_on: Optional[List[str]] = None,
        fallback_agent: Optional[Callable] = None,
        description: Optional[str] = None
    ) -> "ReliabilityChain":
        """Add a step to the chain."""
        step_id = f"step_{len(self.steps)}_{name}"
        
        step = ChainStep(
            step_id=step_id,
            name=name,
            agent=agent,
            estimated_reliability=reliability,
            handoff_validation=handoff_validation,
            confidence_threshold=confidence_threshold,
            depends_on=depends_on or [],
            fallback_agent=fallback_agent,
            description=description
        )
        
        self.steps.append(step)
        self._step_index[step_id] = len(self.steps) - 1
        
        return self
    
    def calculate_theoretical_reliability(self) -> Dict[str, Any]:
        """
        Calculate theoretical end-to-end reliability before execution.
        
        Uses pessimistic (multiplicative) model for sequential steps.
        """
        if not self.steps:
            return {
                "end_to_end": 1.0,
                "by_step": {},
                "weakest_link": None,
                "meets_target": True
            }
        
        # Sequential reliability multiplication
        cumulative = 1.0
        by_step = {}
        weakest = None
        weakest_reliability = 1.0
        
        for step in self.steps:
            cumulative *= step.estimated_reliability
            by_step[step.step_id] = {
                "step_reliability": step.estimated_reliability,
                "cumulative_after": cumulative
            }
            
            if step.estimated_reliability < weakest_reliability:
                weakest_reliability = step.estimated_reliability
                weakest = step.step_id
        
        return {
            "end_to_end": cumulative,
            "by_step": by_step,
            "weakest_link": weakest,
            "weakest_link_reliability": weakest_reliability,
            "meets_target": cumulative >= self.global_reliability_target,
            "gap_to_target": max(0, self.global_reliability_target - cumulative)
        }
    
    def identify_intervention_points(self) -> List[Dict[str, Any]]:
        """
        Identify where interventions would most improve reliability.
        
        Returns steps ranked by impact of improvement.
        """
        if not self.steps:
            return []
        
        current_e2e = self.calculate_theoretical_reliability()["end_to_end"]
        interventions = []
        
        for step in self.steps:
            # Calculate impact of improving this step to 0.99
            improved_reliability = 0.99
            improvement_factor = improved_reliability / step.estimated_reliability
            new_e2e = current_e2e * improvement_factor
            
            impact = new_e2e - current_e2e
            
            interventions.append({
                "step_id": step.step_id,
                "step_name": step.name,
                "current_reliability": step.estimated_reliability,
                "impact_if_improved": impact,
                "new_e2e_if_improved": new_e2e,
                "recommendation": self._get_intervention_recommendation(step)
            })
        
        # Sort by impact
        interventions.sort(key=lambda x: x["impact_if_improved"], reverse=True)
        
        return interventions
    
    def _get_intervention_recommendation(self, step: ChainStep) -> str:
        """Get specific recommendation for improving a step."""
        if step.estimated_reliability < 0.80:
            return "Consider adding validation layer or using more reliable agent"
        elif step.estimated_reliability < 0.90:
            return "Add redundant verification or cross-checking"
        elif step.estimated_reliability < 0.95:
            return "Fine-tune agent or add targeted guardrails"
        else:
            return "Already high reliability - focus elsewhere"
    
    async def execute(
        self, 
        initial_input: Any,
        context: Optional[Dict[str, Any]] = None
    ) -> ChainResult:
        """
        Execute the full chain with reliability tracking.
        
        Note: This is an async method for real execution.
        For synchronous execution, use execute_sync().
        """
        import time
        
        start_time = time.time()
        step_results: List[StepResult] = []
        current_input = initial_input
        current_reliability = 1.0
        
        for step in self.steps:
            step_start = time.time()
            
            try:
                # Execute the step
                output = await self._execute_step(step, current_input, context)
                
                # Assess reliability (would integrate with EDFL here)
                step_reliability = self._assess_step_reliability(step, output)
                current_reliability *= step_reliability
                
                # Determine confidence band
                confidence_band = self._reliability_to_band(current_reliability)
                
                # Check handoff validation
                if not self._validate_handoff(step, output, confidence_band):
                    if step.fallback_agent:
                        output = await step.fallback_agent(current_input)
                        step_reliability = step.fallback_reliability
                        current_reliability = current_reliability / step.estimated_reliability * step_reliability
                    else:
                        raise ValueError(f"Handoff validation failed for {step.name}")
                
                step_results.append(StepResult(
                    step_id=step.step_id,
                    status=StepStatus.COMPLETED,
                    output=output,
                    reliability=step_reliability,
                    confidence_band=confidence_band,
                    execution_time_ms=int((time.time() - step_start) * 1000)
                ))
                
                current_input = output
                
            except Exception as e:
                step_results.append(StepResult(
                    step_id=step.step_id,
                    status=StepStatus.FAILED,
                    output=None,
                    reliability=0.0,
                    confidence_band="insufficient",
                    execution_time_ms=int((time.time() - step_start) * 1000),
                    error=str(e)
                ))
                
                # Chain fails if any step fails without fallback
                return ChainResult(
                    chain_id=self.chain_id,
                    status=StepStatus.FAILED,
                    step_results=step_results,
                    end_to_end_reliability=0.0,
                    weakest_link=step.step_id,
                    reliability_by_step={s.step_id: s.reliability for s in step_results},
                    final_output=None,
                    final_confidence_band="insufficient",
                    total_execution_time_ms=int((time.time() - start_time) * 1000),
                    intervention_points=[step.step_id]
                )
        
        # Identify weakest link
        weakest = min(step_results, key=lambda s: s.reliability) if step_results else None
        
        # Identify intervention points
        intervention_points = [
            s.step_id for s in step_results 
            if s.reliability < 0.90
        ]
        
        return ChainResult(
            chain_id=self.chain_id,
            status=StepStatus.COMPLETED,
            step_results=step_results,
            end_to_end_reliability=current_reliability,
            weakest_link=weakest.step_id if weakest else None,
            reliability_by_step={s.step_id: s.reliability for s in step_results},
            final_output=current_input,
            final_confidence_band=self._reliability_to_band(current_reliability),
            total_execution_time_ms=int((time.time() - start_time) * 1000),
            intervention_points=intervention_points
        )
    
    def execute_sync(
        self, 
        initial_input: Any,
        context: Optional[Dict[str, Any]] = None
    ) -> ChainResult:
        """
        Synchronous execution wrapper.
        """
        import asyncio
        return asyncio.run(self.execute(initial_input, context))
    
    async def _execute_step(
        self, 
        step: ChainStep, 
        input_data: Any,
        context: Optional[Dict[str, Any]]
    ) -> Any:
        """Execute a single step."""
        import asyncio
        
        # Handle both sync and async agents
        if asyncio.iscoroutinefunction(step.agent):
            return await step.agent(input_data, context)
        else:
            return step.agent(input_data, context)
    
    def _assess_step_reliability(self, step: ChainStep, output: Any) -> float:
        """
        Assess actual reliability of a step execution.
        
        In a full implementation, this would integrate with EDFL metrics.
        """
        # Placeholder - would use actual EDFL assessment
        # For now, use the estimated reliability with some variance
        import random
        variance = random.uniform(-0.05, 0.05)
        return max(0.0, min(1.0, step.estimated_reliability + variance))
    
    def _validate_handoff(
        self, 
        step: ChainStep, 
        output: Any,
        confidence_band: str
    ) -> bool:
        """Validate output before passing to next step."""
        
        if step.handoff_validation == HandoffValidation.NONE:
            return True
        
        if step.handoff_validation == HandoffValidation.CONFIDENCE:
            band_values = {
                "high_confidence": 4,
                "moderate_confidence": 3,
                "low_confidence": 2,
                "insufficient": 1
            }
            threshold_band = self._reliability_to_band(step.confidence_threshold)
            return band_values.get(confidence_band, 0) >= band_values.get(threshold_band, 0)
        
        if step.handoff_validation == HandoffValidation.SCHEMA:
            # Would validate output schema
            return output is not None
        
        if step.handoff_validation == HandoffValidation.SEMANTIC:
            # Would check semantic consistency
            return True
        
        if step.handoff_validation == HandoffValidation.FULL:
            # All validations
            return output is not None
        
        return True
    
    def _reliability_to_band(self, reliability: float) -> str:
        """Convert reliability score to confidence band."""
        if reliability >= 0.95:
            return "high_confidence"
        elif reliability >= 0.85:
            return "moderate_confidence"
        elif reliability >= 0.70:
            return "low_confidence"
        else:
            return "insufficient"
    
    def visualise(self) -> str:
        """Generate a text visualisation of the chain."""
        if not self.steps:
            return "Empty chain"
        
        lines = [f"Chain: {self.name} (target: {self.global_reliability_target:.0%})"]
        lines.append("=" * 60)
        
        theoretical = self.calculate_theoretical_reliability()
        cumulative = 1.0
        
        for i, step in enumerate(self.steps):
            cumulative *= step.estimated_reliability
            
            indicator = "→" if i < len(self.steps) - 1 else "⬤"
            weak_marker = " ⚠️" if step.step_id == theoretical["weakest_link"] else ""
            
            lines.append(
                f"{indicator} {step.name}: {step.estimated_reliability:.0%} "
                f"(cumulative: {cumulative:.0%}){weak_marker}"
            )
        
        lines.append("=" * 60)
        lines.append(f"End-to-end reliability: {theoretical['end_to_end']:.1%}")
        lines.append(f"Meets target: {'✓' if theoretical['meets_target'] else '✗'}")
        
        if not theoretical["meets_target"]:
            lines.append(f"Gap to target: {theoretical['gap_to_target']:.1%}")
        
        return "\n".join(lines)


class ChainBuilder:
    """
    Fluent builder for creating reliability chains.
    """
    
    def __init__(self, name: str = "Chain"):
        self._chain = ReliabilityChain(name=name)
    
    def with_target(self, reliability_target: float) -> "ChainBuilder":
        """Set the global reliability target."""
        self._chain.global_reliability_target = reliability_target
        return self
    
    def add(
        self,
        name: str,
        agent: Callable,
        reliability: float = 0.90,
        **kwargs
    ) -> "ChainBuilder":
        """Add a step to the chain."""
        self._chain.add_step(name, agent, reliability, **kwargs)
        return self
    
    def build(self) -> ReliabilityChain:
        """Build and return the chain."""
        return self._chain
