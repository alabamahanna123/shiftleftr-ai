"""
Shift Left AI: Domo Connector

Integration with Domo's AI ecosystem:
- PyDomo SDK for data operations
- Agent Catalyst for workflow AI tasks
- DomoGPT as LLM backend
- Reliability tracking for Domo workflows

This connector enables shift-left hallucination management for
Domo-based AI deployments.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Union
from enum import Enum
import json
import logging
from datetime import datetime

# Shift Left AI imports
from shiftleft.preflight import PreflightAssessment, PreflightResult
from shiftleft.graduated import GraduatedResponse, ConfidenceBand
from shiftleft.design import ReliabilityContract, TaskCategory
from shiftleft.orchestration import ReliabilityChain, ChainResult, StepStatus


logger = logging.getLogger(__name__)


class DomoDataSourceType(Enum):
    """Types of data sources in Domo."""
    DATASET = "dataset"
    FILESET = "fileset"
    DATAFLOW = "dataflow"
    CARD = "card"
    SEMANTIC_LAYER = "semantic_layer"


@dataclass
class DomoDataSource:
    """Reference to a Domo data source."""
    source_type: DomoDataSourceType
    source_id: str
    name: str
    description: Optional[str] = None
    last_updated: Optional[datetime] = None
    row_count: Optional[int] = None
    schema: Optional[Dict[str, str]] = None


@dataclass
class DomoAgentConfig:
    """Configuration for a Domo AI agent."""
    agent_name: str
    llm_model: str = "domogpt"  # or external model
    instructions: str = ""
    knowledge_sources: List[DomoDataSource] = field(default_factory=list)
    tools: List[str] = field(default_factory=list)
    temperature: float = 0.3
    max_tokens: int = 2048


@dataclass
class DomoWorkflowStep:
    """A step in a Domo workflow."""
    step_id: str
    step_type: str  # "ai_agent_task", "data_transform", "action", etc.
    name: str
    config: Dict[str, Any] = field(default_factory=dict)
    inputs: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)


class DomoClient:
    """
    Client for interacting with Domo APIs.
    
    Wraps PyDomo and adds reliability tracking.
    """
    
    def __init__(
        self,
        client_id: str,
        client_secret: str,
        api_host: str = "api.domo.com",
        instance_domain: Optional[str] = None
    ):
        self.client_id = client_id
        self.client_secret = client_secret
        self.api_host = api_host
        self.instance_domain = instance_domain
        self._domo = None
        self._authenticated = False
    
    def connect(self) -> bool:
        """Establish connection to Domo."""
        try:
            from pydomo import Domo
            self._domo = Domo(
                self.client_id,
                self.client_secret,
                api_host=self.api_host
            )
            self._authenticated = True
            logger.info("Connected to Domo successfully")
            return True
        except ImportError:
            logger.error("PyDomo not installed. Run: pip install pydomo")
            return False
        except Exception as e:
            logger.error(f"Failed to connect to Domo: {e}")
            return False
    
    @property
    def is_connected(self) -> bool:
        return self._authenticated and self._domo is not None
    
    def get_dataset(self, dataset_id: str) -> Optional[DomoDataSource]:
        """Get metadata for a Domo dataset."""
        if not self.is_connected:
            raise ConnectionError("Not connected to Domo")
        
        try:
            ds_info = self._domo.datasets.get(dataset_id)
            return DomoDataSource(
                source_type=DomoDataSourceType.DATASET,
                source_id=dataset_id,
                name=ds_info.get("name", ""),
                description=ds_info.get("description"),
                row_count=ds_info.get("rows"),
                schema={
                    col["name"]: col["type"] 
                    for col in ds_info.get("schema", {}).get("columns", [])
                }
            )
        except Exception as e:
            logger.error(f"Failed to get dataset {dataset_id}: {e}")
            return None
    
    def query_dataset(
        self, 
        dataset_id: str, 
        sql: str
    ) -> Optional[List[Dict[str, Any]]]:
        """Execute SQL query against a Domo dataset."""
        if not self.is_connected:
            raise ConnectionError("Not connected to Domo")
        
        try:
            result = self._domo.ds_query(dataset_id, sql)
            return result
        except Exception as e:
            logger.error(f"Failed to query dataset {dataset_id}: {e}")
            return None
    
    def list_datasets(self, limit: int = 50) -> List[DomoDataSource]:
        """List available datasets."""
        if not self.is_connected:
            raise ConnectionError("Not connected to Domo")
        
        try:
            datasets = self._domo.datasets.list(limit=limit)
            return [
                DomoDataSource(
                    source_type=DomoDataSourceType.DATASET,
                    source_id=ds.get("id", ""),
                    name=ds.get("name", ""),
                    description=ds.get("description"),
                    row_count=ds.get("rows")
                )
                for ds in datasets
            ]
        except Exception as e:
            logger.error(f"Failed to list datasets: {e}")
            return []


class DomoReliabilityTracker:
    """
    Track reliability metrics for Domo AI operations.
    
    Stores metrics to a Domo dataset for governance and monitoring.
    """
    
    def __init__(
        self,
        client: DomoClient,
        metrics_dataset_id: Optional[str] = None
    ):
        self.client = client
        self.metrics_dataset_id = metrics_dataset_id
        self._pending_metrics: List[Dict[str, Any]] = []
    
    def record_response(
        self,
        agent_name: str,
        task_id: str,
        response: GraduatedResponse,
        workflow_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Record a graduated response for tracking."""
        metric = {
            "timestamp": datetime.utcnow().isoformat(),
            "agent_name": agent_name,
            "task_id": task_id,
            "workflow_id": workflow_id or "",
            "confidence_band": response.confidence_band.value,
            "reliability_estimate": response.reliability_estimate,
            "isr": response.isr,
            "roh_bound": response.roh_bound,
            "uncertainty_count": len(response.uncertainty_flags),
            "verification_required": len([
                s for s in response.verification_steps 
                if s.priority == "required"
            ]),
            "sources_count": len(response.sources_used),
            "metadata_json": json.dumps(metadata or {})
        }
        self._pending_metrics.append(metric)
        logger.debug(f"Recorded metric for task {task_id}")
    
    def record_chain_result(
        self,
        workflow_name: str,
        result: ChainResult,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Record reliability chain execution result."""
        metric = {
            "timestamp": datetime.utcnow().isoformat(),
            "workflow_name": workflow_name,
            "chain_id": result.chain_id,
            "status": result.status.value,
            "end_to_end_reliability": result.end_to_end_reliability,
            "weakest_link": result.weakest_link or "",
            "step_count": len(result.step_results),
            "failed_steps": len([
                s for s in result.step_results 
                if s.status == StepStatus.FAILED
            ]),
            "execution_time_ms": result.total_execution_time_ms,
            "intervention_points": ",".join(result.intervention_points),
            "metadata_json": json.dumps(metadata or {})
        }
        self._pending_metrics.append(metric)
        logger.debug(f"Recorded chain result for {workflow_name}")
    
    def flush_to_domo(self) -> bool:
        """Push pending metrics to Domo dataset."""
        if not self._pending_metrics:
            return True
        
        if not self.metrics_dataset_id:
            logger.warning("No metrics dataset configured")
            return False
        
        if not self.client.is_connected:
            logger.error("Domo client not connected")
            return False
        
        try:
            import pandas as pd
            df = pd.DataFrame(self._pending_metrics)
            self.client._domo.ds_update(self.metrics_dataset_id, df)
            self._pending_metrics = []
            logger.info(f"Flushed {len(df)} metrics to Domo")
            return True
        except Exception as e:
            logger.error(f"Failed to flush metrics to Domo: {e}")
            return False
    
    def get_recent_reliability(
        self,
        agent_name: Optional[str] = None,
        hours: int = 24
    ) -> Dict[str, Any]:
        """Get recent reliability statistics."""
        if not self.metrics_dataset_id or not self.client.is_connected:
            # Return from local pending metrics
            relevant = self._pending_metrics
            if agent_name:
                relevant = [m for m in relevant if m.get("agent_name") == agent_name]
            
            if not relevant:
                return {"count": 0}
            
            reliabilities = [m["reliability_estimate"] for m in relevant]
            return {
                "count": len(relevant),
                "mean_reliability": sum(reliabilities) / len(reliabilities),
                "min_reliability": min(reliabilities),
                "max_reliability": max(reliabilities),
                "high_confidence_rate": len([
                    m for m in relevant 
                    if m["confidence_band"] == "high_confidence"
                ]) / len(relevant)
            }
        
        # Query from Domo
        try:
            sql = f"""
                SELECT 
                    COUNT(*) as count,
                    AVG(reliability_estimate) as mean_reliability,
                    MIN(reliability_estimate) as min_reliability,
                    MAX(reliability_estimate) as max_reliability
                FROM table
                WHERE timestamp > DATEADD(hour, -{hours}, GETDATE())
            """
            if agent_name:
                sql += f" AND agent_name = '{agent_name}'"
            
            result = self.client.query_dataset(self.metrics_dataset_id, sql)
            return result[0] if result else {"count": 0}
        except Exception as e:
            logger.error(f"Failed to query reliability metrics: {e}")
            return {"count": 0, "error": str(e)}


class DomoAgentWrapper:
    """
    Wrapper for Domo AI agents with shift-left reliability.
    
    Adds pre-flight assessment, graduated responses, and
    reliability tracking to Domo Agent Catalyst agents.
    """
    
    def __init__(
        self,
        client: DomoClient,
        config: DomoAgentConfig,
        reliability_contract: Optional[ReliabilityContract] = None,
        tracker: Optional[DomoReliabilityTracker] = None
    ):
        self.client = client
        self.config = config
        self.contract = reliability_contract
        self.tracker = tracker
    
    def assess_task(self, task: str) -> PreflightResult:
        """Run pre-flight assessment before agent execution."""
        # Convert Domo knowledge sources to preflight format
        sources = [ks.source_id for ks in self.config.knowledge_sources]
        source_metadata = {
            ks.source_id: {
                "type": "user_provided",
                "coverage": 0.8,  # Conservative estimate
                "name": ks.name
            }
            for ks in self.config.knowledge_sources
        }
        
        # Determine reliability target
        target = 0.95
        if self.contract:
            target = 1.0 - self.contract.max_hallucination_rate
        
        assessment = PreflightAssessment(
            task=task,
            sources=sources,
            source_metadata=source_metadata,
            reliability_target=target
        )
        
        return assessment.run()
    
    async def execute(
        self,
        task: str,
        context: Optional[Dict[str, Any]] = None,
        skip_preflight: bool = False
    ) -> GraduatedResponse:
        """
        Execute agent task with reliability tracking.
        
        Returns a GraduatedResponse with confidence bands.
        """
        task_id = f"{self.config.agent_name}_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
        
        # 1. Pre-flight assessment
        if not skip_preflight:
            preflight = self.assess_task(task)
            if not preflight.is_feasible:
                response = GraduatedResponse.from_preflight(preflight, answer=None)
                if self.tracker:
                    self.tracker.record_response(
                        self.config.agent_name,
                        task_id,
                        response,
                        metadata={"preflight_failed": True}
                    )
                return response
        
        # 2. Execute via Domo (placeholder for actual Agent Catalyst API)
        # In production, this would call the Domo Agent Catalyst API
        answer = await self._call_domo_agent(task, context)
        
        # 3. Assess response reliability
        # In production, integrate with EDFL metrics
        mock_reliability = self._estimate_reliability(task, answer)
        
        # 4. Create graduated response
        response = self._create_graduated_response(answer, mock_reliability)
        
        # 5. Track metrics
        if self.tracker:
            self.tracker.record_response(
                self.config.agent_name,
                task_id,
                response,
                metadata={"task": task}
            )
        
        return response
    
    async def _call_domo_agent(
        self,
        task: str,
        context: Optional[Dict[str, Any]]
    ) -> str:
        """
        Call Domo Agent Catalyst API.
        
        This is a placeholder - actual implementation would use
        Domo's Agent Catalyst API when available.
        """
        # Placeholder for Domo Agent Catalyst integration
        # The actual API would be called here
        logger.info(f"Executing Domo agent task: {task[:50]}...")
        
        # For now, return a placeholder
        # In production, this would call:
        # - DomoGPT endpoint
        # - Or external LLM via Domo's AI framework
        return f"[Domo Agent Response for: {task}]"
    
    def _estimate_reliability(self, task: str, answer: str) -> Dict[str, float]:
        """
        Estimate reliability of the response.
        
        In production, this would integrate with EDFL/HallBayes.
        """
        # Placeholder reliability estimation
        # Would be replaced with actual EDFL metrics
        base_reliability = 0.85
        
        # Adjust based on knowledge source coverage
        if self.config.knowledge_sources:
            base_reliability += 0.05
        
        # Adjust based on answer characteristics
        if len(answer) < 50:
            base_reliability += 0.05  # Short answers often more reliable
        elif len(answer) > 500:
            base_reliability -= 0.05  # Long answers have more room for error
        
        return {
            "isr": 1.2 if base_reliability > 0.85 else 0.8,
            "roh_bound": 1.0 - base_reliability,
            "reliability": base_reliability
        }
    
    def _create_graduated_response(
        self,
        answer: str,
        metrics: Dict[str, float]
    ) -> GraduatedResponse:
        """Create a graduated response from metrics."""
        
        # Determine confidence band
        reliability = metrics["reliability"]
        if reliability >= 0.95:
            band = ConfidenceBand.HIGH
        elif reliability >= 0.85:
            band = ConfidenceBand.MODERATE
        elif reliability >= 0.70:
            band = ConfidenceBand.LOW
        else:
            band = ConfidenceBand.INSUFFICIENT
        
        # Generate verification steps based on band
        from shiftleft.graduated.response import VerificationStep
        verification_steps = []
        
        if band == ConfidenceBand.MODERATE:
            verification_steps.append(VerificationStep(
                action="Cross-reference key facts with Domo dataset",
                rationale="Moderate confidence suggests verification recommended",
                priority="recommended",
                resources=[ks.name for ks in self.config.knowledge_sources]
            ))
        elif band == ConfidenceBand.LOW:
            verification_steps.append(VerificationStep(
                action="Verify all claims against source data",
                rationale="Low confidence requires manual verification",
                priority="required"
            ))
        elif band == ConfidenceBand.INSUFFICIENT:
            verification_steps.append(VerificationStep(
                action="Do not use - add knowledge sources",
                rationale="Insufficient information for reliable answer",
                priority="required"
            ))
        
        return GraduatedResponse(
            confidence_band=band,
            answer=answer if band != ConfidenceBand.INSUFFICIENT else None,
            uncertainty_flags=[],
            verification_steps=verification_steps,
            reliability_estimate=reliability,
            isr=metrics["isr"],
            roh_bound=metrics["roh_bound"],
            sources_used=[ks.name for ks in self.config.knowledge_sources],
            reasoning_trace=None,
            meta={
                "domo_agent": self.config.agent_name,
                "llm_model": self.config.llm_model
            }
        )


class DomoWorkflowWrapper:
    """
    Wrapper for Domo Workflows with reliability tracking.
    
    Integrates with Domo Workflows to track reliability
    through multi-step processes.
    """
    
    def __init__(
        self,
        client: DomoClient,
        workflow_name: str,
        steps: List[DomoWorkflowStep],
        tracker: Optional[DomoReliabilityTracker] = None
    ):
        self.client = client
        self.workflow_name = workflow_name
        self.steps = steps
        self.tracker = tracker
        self._reliability_chain: Optional[ReliabilityChain] = None
    
    def build_reliability_chain(
        self,
        reliability_estimates: Optional[Dict[str, float]] = None
    ) -> ReliabilityChain:
        """
        Build a ReliabilityChain from Domo workflow steps.
        
        Args:
            reliability_estimates: Optional dict of step_id -> reliability
        """
        estimates = reliability_estimates or {}
        
        chain = ReliabilityChain(
            name=self.workflow_name,
            global_reliability_target=0.90
        )
        
        for step in self.steps:
            # Estimate reliability based on step type
            default_reliability = self._default_reliability_for_step(step)
            reliability = estimates.get(step.step_id, default_reliability)
            
            # Create agent function (placeholder)
            async def step_agent(input_data, ctx, step=step):
                return await self._execute_step(step, input_data, ctx)
            
            chain.add_step(
                name=step.name,
                agent=step_agent,
                reliability=reliability,
                description=f"Domo workflow step: {step.step_type}"
            )
        
        self._reliability_chain = chain
        return chain
    
    def _default_reliability_for_step(self, step: DomoWorkflowStep) -> float:
        """Estimate default reliability for a step type."""
        step_type_reliability = {
            "ai_agent_task": 0.85,  # AI tasks have inherent uncertainty
            "data_transform": 0.98,  # Data transforms are deterministic
            "sql_query": 0.98,
            "action": 0.95,  # Actions generally reliable
            "conditional": 0.99,  # Logic branches very reliable
            "webhook": 0.92,  # External calls less reliable
            "email": 0.97,
            "slack": 0.97,
        }
        return step_type_reliability.get(step.step_type, 0.90)
    
    async def _execute_step(
        self,
        step: DomoWorkflowStep,
        input_data: Any,
        context: Optional[Dict[str, Any]]
    ) -> Any:
        """Execute a single workflow step."""
        # Placeholder - would call Domo Workflow API
        logger.info(f"Executing workflow step: {step.name}")
        return {"step": step.name, "status": "completed", "input": input_data}
    
    def analyse(self) -> Dict[str, Any]:
        """Analyse workflow for reliability characteristics."""
        if not self._reliability_chain:
            self.build_reliability_chain()
        
        theoretical = self._reliability_chain.calculate_theoretical_reliability()
        interventions = self._reliability_chain.identify_intervention_points()
        
        return {
            "workflow_name": self.workflow_name,
            "step_count": len(self.steps),
            "end_to_end_reliability": theoretical["end_to_end"],
            "meets_target": theoretical["meets_target"],
            "weakest_link": theoretical["weakest_link"],
            "gap_to_target": theoretical.get("gap_to_target", 0),
            "interventions": interventions[:3],
            "visualisation": self._reliability_chain.visualise()
        }
    
    async def execute(
        self,
        initial_input: Any,
        context: Optional[Dict[str, Any]] = None
    ) -> ChainResult:
        """Execute workflow with reliability tracking."""
        if not self._reliability_chain:
            self.build_reliability_chain()
        
        result = await self._reliability_chain.execute(initial_input, context)
        
        if self.tracker:
            self.tracker.record_chain_result(
                self.workflow_name,
                result,
                metadata={"input_type": type(initial_input).__name__}
            )
        
        return result


# Convenience functions

def create_domo_reliability_dashboard_schema() -> Dict[str, Any]:
    """
    Returns the schema for a Domo dataset to track reliability metrics.
    
    Use this to create the metrics tracking dataset in Domo.
    """
    return {
        "name": "Shift Left AI - Reliability Metrics",
        "description": "Tracks AI agent and workflow reliability metrics for governance",
        "schema": {
            "columns": [
                {"name": "timestamp", "type": "DATETIME"},
                {"name": "agent_name", "type": "STRING"},
                {"name": "task_id", "type": "STRING"},
                {"name": "workflow_id", "type": "STRING"},
                {"name": "confidence_band", "type": "STRING"},
                {"name": "reliability_estimate", "type": "DECIMAL"},
                {"name": "isr", "type": "DECIMAL"},
                {"name": "roh_bound", "type": "DECIMAL"},
                {"name": "uncertainty_count", "type": "LONG"},
                {"name": "verification_required", "type": "LONG"},
                {"name": "sources_count", "type": "LONG"},
                {"name": "metadata_json", "type": "STRING"},
            ]
        }
    }


def create_domo_contract_for_agent(
    agent_name: str,
    task_type: TaskCategory = TaskCategory.ANALYSIS,
    max_hallucination_rate: float = 0.10
) -> ReliabilityContract:
    """
    Create a reliability contract suitable for Domo AI agents.
    """
    from shiftleft.design.contracts import (
        DecompositionStrategy,
        SourceRequirement,
        EscalationTrigger
    )
    
    return ReliabilityContract(
        contract_id=f"domo-agent-{agent_name}",
        task_type=task_type,
        description=f"Reliability contract for Domo agent: {agent_name}",
        max_hallucination_rate=max_hallucination_rate,
        min_confidence_band="moderate",
        required_source_coverage=0.80,
        decomposition_strategy=DecompositionStrategy.NONE,
        source_requirements=[
            SourceRequirement(
                source_type="domo_dataset",
                minimum_count=1,
                required_coverage=0.7,
                verification_method="cross_reference"
            )
        ],
        escalation_triggers=[
            EscalationTrigger.CONFIDENCE_BAND_LOW,
            EscalationTrigger.CONFIDENCE_BAND_INSUFFICIENT,
            EscalationTrigger.RELIABILITY_BELOW_THRESHOLD,
            EscalationTrigger.USER_REQUEST,
        ]
    )
