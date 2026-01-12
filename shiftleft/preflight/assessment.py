"""
Shift Left AI: Pre-flight Assessment Module

Assess whether a task has sufficient information and appropriate decomposition
BEFORE any generation occurs. This is the core "shift left" principle.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional, Any
import re


class TaskComplexity(Enum):
    """Classification of task complexity for reliability estimation."""
    SIMPLE_FACTUAL = "simple_factual"  # Single fact retrieval
    MULTI_FACT = "multi_fact"  # Multiple facts, no reasoning
    COMPARATIVE = "comparative"  # Comparison requiring multiple sources
    ANALYTICAL = "analytical"  # Requires reasoning over evidence
    GENERATIVE = "generative"  # Creative or open-ended
    PROCEDURAL = "procedural"  # Step-by-step instructions


class SourceType(Enum):
    """Classification of information sources."""
    AUTHORITATIVE = "authoritative"  # Primary sources, official documents
    SECONDARY = "secondary"  # Reputable secondary sources
    USER_PROVIDED = "user_provided"  # Documents uploaded by user
    MODEL_KNOWLEDGE = "model_knowledge"  # LLM parametric knowledge
    RETRIEVAL = "retrieval"  # RAG or search results
    UNVERIFIED = "unverified"  # Unknown provenance


@dataclass
class SourceAssessment:
    """Assessment of a single information source."""
    source_id: str
    source_type: SourceType
    estimated_coverage: float  # 0-1: how much of the task this source covers
    freshness_score: float  # 0-1: how current the information is
    reliability_estimate: float  # 0-1: estimated reliability
    gaps: List[str] = field(default_factory=list)  # Identified information gaps


@dataclass
class DecompositionStrategy:
    """Recommended approach for breaking down a complex task."""
    strategy_name: str
    steps: List[str]
    per_step_reliability: List[float]
    combined_reliability: float
    rationale: str


@dataclass
class PreflightResult:
    """Result of pre-flight assessment."""
    task: str
    complexity: TaskComplexity
    reliability_target: float
    
    # Source analysis
    sources_assessed: List[SourceAssessment]
    total_coverage: float
    information_gaps: List[str]
    
    # Feasibility
    is_feasible: bool
    feasibility_rationale: str
    
    # Recommendations
    recommendation: str
    decomposition: Optional[DecompositionStrategy]
    required_sources: List[str]
    risk_factors: List[str]
    
    # Metrics
    estimated_reliability: float
    confidence_in_estimate: float


class PreflightAssessment:
    """
    Pre-flight assessment for AI generation tasks.
    
    Evaluates whether a task has sufficient information and appropriate
    structure to meet reliability requirements BEFORE any generation occurs.
    """
    
    def __init__(
        self,
        task: str,
        sources: Optional[List[str]] = None,
        source_metadata: Optional[Dict[str, Dict[str, Any]]] = None,
        reliability_target: float = 0.95,
        allow_model_knowledge: bool = True
    ):
        self.task = task
        self.sources = sources or []
        self.source_metadata = source_metadata or {}
        self.reliability_target = reliability_target
        self.allow_model_knowledge = allow_model_knowledge
    
    def run(self) -> PreflightResult:
        """Execute the pre-flight assessment."""
        # 1. Classify task complexity
        complexity = self._classify_complexity()
        
        # 2. Assess available sources
        source_assessments = self._assess_sources()
        
        # 3. Calculate coverage and identify gaps
        total_coverage = self._calculate_coverage(source_assessments)
        gaps = self._identify_gaps(complexity, source_assessments)
        
        # 4. Estimate reliability
        estimated_reliability = self._estimate_reliability(
            complexity, source_assessments, total_coverage
        )
        
        # 5. Determine feasibility
        is_feasible, feasibility_rationale = self._assess_feasibility(
            estimated_reliability, gaps
        )
        
        # 6. Generate recommendations
        recommendation = self._generate_recommendation(
            complexity, is_feasible, estimated_reliability, gaps
        )
        
        # 7. Suggest decomposition if beneficial
        decomposition = self._suggest_decomposition(complexity, gaps)
        
        # 8. Identify required additional sources
        required_sources = self._identify_required_sources(gaps)
        
        # 9. Flag risk factors
        risk_factors = self._identify_risks(complexity, source_assessments, gaps)
        
        return PreflightResult(
            task=self.task,
            complexity=complexity,
            reliability_target=self.reliability_target,
            sources_assessed=source_assessments,
            total_coverage=total_coverage,
            information_gaps=gaps,
            is_feasible=is_feasible,
            feasibility_rationale=feasibility_rationale,
            recommendation=recommendation,
            decomposition=decomposition,
            required_sources=required_sources,
            risk_factors=risk_factors,
            estimated_reliability=estimated_reliability,
            confidence_in_estimate=self._estimate_confidence(source_assessments)
        )
    
    def _classify_complexity(self) -> TaskComplexity:
        """Classify the task complexity based on linguistic patterns."""
        task_lower = self.task.lower()
        
        # Simple factual indicators
        factual_patterns = [
            r"^who (is|was|are|were)\b",
            r"^what (is|was|are|were)\b",
            r"^when (did|was|is)\b",
            r"^where (is|was|are|were)\b",
            r"capital of",
            r"born in",
            r"died in",
        ]
        
        # Comparative indicators
        comparative_patterns = [
            r"\bcompare\b",
            r"\bversus\b",
            r"\bvs\.?\b",
            r"\bdifference between\b",
            r"\bwhich is (better|worse|more|less)\b",
        ]
        
        # Analytical indicators
        analytical_patterns = [
            r"\bwhy\b",
            r"\bhow does\b",
            r"\bexplain\b",
            r"\banalyse\b",
            r"\banalyze\b",
            r"\bevaluate\b",
            r"\bassess\b",
        ]
        
        # Generative indicators
        generative_patterns = [
            r"\bwrite\b",
            r"\bcreate\b",
            r"\bgenerate\b",
            r"\bcompose\b",
            r"\bdraft\b",
            r"\bstory\b",
            r"\bpoem\b",
        ]
        
        # Procedural indicators
        procedural_patterns = [
            r"\bhow to\b",
            r"\bsteps to\b",
            r"\bguide\b",
            r"\binstructions\b",
            r"\brecipe\b",
            r"\btutorial\b",
        ]
        
        # Multi-fact indicators
        multi_fact_patterns = [
            r"\band\b.*\band\b",
            r"\blist\b",
            r"\ball\b",
            r"\bmultiple\b",
            r"\bseveral\b",
        ]
        
        # Check patterns in order of specificity
        for pattern in generative_patterns:
            if re.search(pattern, task_lower):
                return TaskComplexity.GENERATIVE
        
        for pattern in procedural_patterns:
            if re.search(pattern, task_lower):
                return TaskComplexity.PROCEDURAL
        
        for pattern in analytical_patterns:
            if re.search(pattern, task_lower):
                return TaskComplexity.ANALYTICAL
        
        for pattern in comparative_patterns:
            if re.search(pattern, task_lower):
                return TaskComplexity.COMPARATIVE
        
        for pattern in multi_fact_patterns:
            if re.search(pattern, task_lower):
                return TaskComplexity.MULTI_FACT
        
        for pattern in factual_patterns:
            if re.search(pattern, task_lower):
                return TaskComplexity.SIMPLE_FACTUAL
        
        # Default to analytical for ambiguous cases
        return TaskComplexity.ANALYTICAL
    
    def _assess_sources(self) -> List[SourceAssessment]:
        """Assess each available source."""
        assessments = []
        
        for source in self.sources:
            metadata = self.source_metadata.get(source, {})
            
            # Determine source type
            source_type = self._classify_source_type(source, metadata)
            
            # Estimate coverage (placeholder - would need NLP analysis)
            coverage = metadata.get("coverage", 0.5)
            
            # Assess freshness
            freshness = self._assess_freshness(metadata)
            
            # Estimate reliability based on source type
            reliability = self._source_type_reliability(source_type)
            
            assessments.append(SourceAssessment(
                source_id=source,
                source_type=source_type,
                estimated_coverage=coverage,
                freshness_score=freshness,
                reliability_estimate=reliability,
                gaps=[]
            ))
        
        # If no external sources and model knowledge is allowed
        if not assessments and self.allow_model_knowledge:
            assessments.append(SourceAssessment(
                source_id="model_parametric_knowledge",
                source_type=SourceType.MODEL_KNOWLEDGE,
                estimated_coverage=0.7,  # Conservative estimate
                freshness_score=0.5,  # Knowledge cutoff considerations
                reliability_estimate=0.85,  # Models hallucinate ~10-15%
                gaps=["May be outdated", "Cannot verify without external source"]
            ))
        
        return assessments
    
    def _classify_source_type(self, source: str, metadata: Dict) -> SourceType:
        """Classify the type of a source."""
        if "type" in metadata:
            return SourceType(metadata["type"])
        
        # Heuristics based on source name/path
        source_lower = source.lower()
        
        if any(ext in source_lower for ext in [".gov", ".edu", "official"]):
            return SourceType.AUTHORITATIVE
        elif any(ext in source_lower for ext in ["wikipedia", "news", "article"]):
            return SourceType.SECONDARY
        elif any(ext in source_lower for ext in ["upload", "user", "my_"]):
            return SourceType.USER_PROVIDED
        elif any(ext in source_lower for ext in ["search", "retrieved", "rag"]):
            return SourceType.RETRIEVAL
        
        return SourceType.UNVERIFIED
    
    def _assess_freshness(self, metadata: Dict) -> float:
        """Assess how current a source is."""
        if "last_updated" in metadata:
            # Would calculate based on date difference
            return metadata.get("freshness", 0.8)
        if "year" in metadata:
            # Older sources get lower freshness scores
            return max(0.3, 1.0 - (2026 - metadata["year"]) * 0.1)
        return 0.5  # Unknown freshness
    
    def _source_type_reliability(self, source_type: SourceType) -> float:
        """Base reliability estimates by source type."""
        reliability_map = {
            SourceType.AUTHORITATIVE: 0.95,
            SourceType.SECONDARY: 0.85,
            SourceType.USER_PROVIDED: 0.80,
            SourceType.MODEL_KNOWLEDGE: 0.85,
            SourceType.RETRIEVAL: 0.75,
            SourceType.UNVERIFIED: 0.50,
        }
        return reliability_map.get(source_type, 0.50)
    
    def _calculate_coverage(self, assessments: List[SourceAssessment]) -> float:
        """Calculate total information coverage from all sources."""
        if not assessments:
            return 0.0
        
        # Simple union model: coverage grows with diminishing returns
        total = 0.0
        for assessment in assessments:
            remaining = 1.0 - total
            total += assessment.estimated_coverage * remaining
        
        return min(total, 1.0)
    
    def _identify_gaps(
        self, 
        complexity: TaskComplexity, 
        assessments: List[SourceAssessment]
    ) -> List[str]:
        """Identify information gaps based on task requirements."""
        gaps = []
        
        # Complexity-specific gap analysis
        if complexity == TaskComplexity.COMPARATIVE:
            # Need multiple perspectives
            if len(assessments) < 2:
                gaps.append("Comparative task requires multiple sources for fair comparison")
        
        if complexity == TaskComplexity.ANALYTICAL:
            # Need authoritative sources
            has_authoritative = any(
                a.source_type == SourceType.AUTHORITATIVE for a in assessments
            )
            if not has_authoritative:
                gaps.append("Analytical task benefits from authoritative primary sources")
        
        # Check for freshness gaps on time-sensitive queries
        task_lower = self.task.lower()
        time_sensitive_patterns = ["current", "latest", "now", "today", "recent"]
        is_time_sensitive = any(p in task_lower for p in time_sensitive_patterns)
        
        if is_time_sensitive:
            avg_freshness = sum(a.freshness_score for a in assessments) / max(len(assessments), 1)
            if avg_freshness < 0.7:
                gaps.append("Time-sensitive query requires more recent sources")
        
        # Check for coverage gaps
        total_coverage = self._calculate_coverage(assessments)
        if total_coverage < 0.8:
            gaps.append(f"Source coverage ({total_coverage:.0%}) may be insufficient")
        
        # Aggregate source-level gaps
        for assessment in assessments:
            gaps.extend(assessment.gaps)
        
        return gaps
    
    def _estimate_reliability(
        self,
        complexity: TaskComplexity,
        assessments: List[SourceAssessment],
        coverage: float
    ) -> float:
        """Estimate achievable reliability for this task."""
        if not assessments:
            return 0.3  # Very low reliability without sources
        
        # Base reliability from sources
        source_reliability = sum(
            a.reliability_estimate * a.estimated_coverage 
            for a in assessments
        ) / max(sum(a.estimated_coverage for a in assessments), 0.01)
        
        # Complexity penalty
        complexity_factors = {
            TaskComplexity.SIMPLE_FACTUAL: 1.0,
            TaskComplexity.MULTI_FACT: 0.95,
            TaskComplexity.COMPARATIVE: 0.90,
            TaskComplexity.ANALYTICAL: 0.85,
            TaskComplexity.PROCEDURAL: 0.90,
            TaskComplexity.GENERATIVE: 0.70,  # Hardest to verify
        }
        complexity_factor = complexity_factors.get(complexity, 0.80)
        
        # Coverage factor
        coverage_factor = 0.5 + 0.5 * coverage  # Scale from 0.5 to 1.0
        
        # Combined estimate
        estimated = source_reliability * complexity_factor * coverage_factor
        
        return min(max(estimated, 0.0), 1.0)
    
    def _assess_feasibility(
        self, 
        estimated_reliability: float,
        gaps: List[str]
    ) -> tuple[bool, str]:
        """Determine if the task is feasible at target reliability."""
        
        # Clear feasibility
        if estimated_reliability >= self.reliability_target:
            return True, f"Estimated reliability ({estimated_reliability:.0%}) meets target ({self.reliability_target:.0%})"
        
        # Close to target
        margin = self.reliability_target - estimated_reliability
        if margin < 0.10:
            return True, f"Estimated reliability ({estimated_reliability:.0%}) is close to target. Proceed with caution."
        
        # Significant gap
        if margin < 0.25:
            return False, f"Reliability gap ({margin:.0%}) is significant. Consider adding sources or decomposing task."
        
        # Large gap
        return False, f"Reliability gap ({margin:.0%}) is too large. Task requires restructuring."
    
    def _generate_recommendation(
        self,
        complexity: TaskComplexity,
        is_feasible: bool,
        estimated_reliability: float,
        gaps: List[str]
    ) -> str:
        """Generate actionable recommendation."""
        
        if is_feasible and estimated_reliability >= 0.95:
            return "PROCEED: Task has sufficient information for high-confidence response."
        
        if is_feasible and estimated_reliability >= 0.85:
            return "PROCEED WITH CAVEATS: Include uncertainty indicators in response."
        
        if is_feasible:
            return "PROCEED WITH CAUTION: Use graduated response with clear confidence bands."
        
        if len(gaps) > 0:
            gap_text = "; ".join(gaps[:3])
            return f"RESTRUCTURE: Address information gaps before proceeding. Key gaps: {gap_text}"
        
        return "DEFER: Task reliability requirements cannot be met with available information."
    
    def _suggest_decomposition(
        self, 
        complexity: TaskComplexity,
        gaps: List[str]
    ) -> Optional[DecompositionStrategy]:
        """Suggest task decomposition if beneficial."""
        
        # Simple tasks don't benefit from decomposition
        if complexity == TaskComplexity.SIMPLE_FACTUAL:
            return None
        
        # Decomposition strategies by complexity
        if complexity == TaskComplexity.COMPARATIVE:
            return DecompositionStrategy(
                strategy_name="parallel_comparison",
                steps=[
                    "Gather facts about first entity",
                    "Gather facts about second entity",
                    "Identify comparison dimensions",
                    "Compare on each dimension",
                    "Synthesise findings"
                ],
                per_step_reliability=[0.95, 0.95, 0.90, 0.90, 0.85],
                combined_reliability=0.95 * 0.95 * 0.90 * 0.90 * 0.85,
                rationale="Parallel fact-gathering allows independent verification before synthesis"
            )
        
        if complexity == TaskComplexity.ANALYTICAL:
            return DecompositionStrategy(
                strategy_name="claim_by_claim",
                steps=[
                    "Extract key claims to verify",
                    "Verify each claim against sources",
                    "Assess claim interdependencies",
                    "Build supported conclusion"
                ],
                per_step_reliability=[0.95, 0.90, 0.85, 0.90],
                combined_reliability=0.95 * 0.90 * 0.85 * 0.90,
                rationale="Claim-by-claim verification enables granular confidence tracking"
            )
        
        if complexity == TaskComplexity.MULTI_FACT:
            return DecompositionStrategy(
                strategy_name="enumerate_and_verify",
                steps=[
                    "Enumerate required facts",
                    "Retrieve each fact independently",
                    "Cross-check against multiple sources",
                    "Compile verified facts"
                ],
                per_step_reliability=[0.98, 0.90, 0.85, 0.95],
                combined_reliability=0.98 * 0.90 * 0.85 * 0.95,
                rationale="Independent fact retrieval prevents cascade failures"
            )
        
        return None
    
    def _identify_required_sources(self, gaps: List[str]) -> List[str]:
        """Identify additional sources needed to close gaps."""
        required = []
        
        for gap in gaps:
            gap_lower = gap.lower()
            if "authoritative" in gap_lower:
                required.append("Primary/official source for the topic")
            if "recent" in gap_lower or "current" in gap_lower:
                required.append("Web search for current information")
            if "multiple" in gap_lower or "comparison" in gap_lower:
                required.append("Additional perspective sources")
            if "coverage" in gap_lower:
                required.append("Broader topical sources")
        
        return list(set(required))  # Deduplicate
    
    def _identify_risks(
        self,
        complexity: TaskComplexity,
        assessments: List[SourceAssessment],
        gaps: List[str]
    ) -> List[str]:
        """Identify risk factors that could affect reliability."""
        risks = []
        
        # Complexity risks
        if complexity in [TaskComplexity.GENERATIVE, TaskComplexity.ANALYTICAL]:
            risks.append("Complex task type increases hallucination risk")
        
        # Source risks
        unverified_count = sum(
            1 for a in assessments if a.source_type == SourceType.UNVERIFIED
        )
        if unverified_count > 0:
            risks.append(f"{unverified_count} source(s) with unverified provenance")
        
        model_only = (
            len(assessments) == 1 and 
            assessments[0].source_type == SourceType.MODEL_KNOWLEDGE
        )
        if model_only:
            risks.append("Relying solely on model parametric knowledge")
        
        # Freshness risks
        stale_count = sum(1 for a in assessments if a.freshness_score < 0.5)
        if stale_count > 0:
            risks.append(f"{stale_count} source(s) may be outdated")
        
        # Gap count
        if len(gaps) > 3:
            risks.append(f"Multiple information gaps ({len(gaps)}) increase uncertainty")
        
        return risks
    
    def _estimate_confidence(self, assessments: List[SourceAssessment]) -> float:
        """Estimate confidence in our reliability estimate."""
        if not assessments:
            return 0.3
        
        # More sources = more confidence in estimate
        source_factor = min(len(assessments) / 3, 1.0)
        
        # Higher coverage = more confidence
        coverage = self._calculate_coverage(assessments)
        coverage_factor = coverage
        
        # Authoritative sources increase confidence
        authoritative_count = sum(
            1 for a in assessments if a.source_type == SourceType.AUTHORITATIVE
        )
        auth_factor = min(authoritative_count / 2, 1.0)
        
        # Weighted combination
        confidence = 0.4 * source_factor + 0.4 * coverage_factor + 0.2 * auth_factor
        
        return min(max(confidence, 0.2), 0.95)
