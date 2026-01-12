"""
Shift Left AI: Domo Governance Dashboard

Generates Domo dashboard configurations for AI reliability governance.
These can be imported into Domo to create monitoring dashboards.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import json


@dataclass
class DomoCard:
    """Configuration for a Domo dashboard card."""
    title: str
    card_type: str  # "bar", "line", "gauge", "table", "kpi"
    dataset_id: str
    description: Optional[str] = None
    sql: Optional[str] = None
    config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DomoDashboard:
    """Configuration for a Domo dashboard."""
    name: str
    description: str
    cards: List[DomoCard] = field(default_factory=list)
    filters: List[Dict[str, Any]] = field(default_factory=list)


class GovernanceDashboardBuilder:
    """
    Builder for Domo AI governance dashboards.
    
    Creates dashboard configurations that can be imported
    into Domo for reliability monitoring.
    """
    
    def __init__(self, metrics_dataset_id: str):
        self.metrics_dataset_id = metrics_dataset_id
        self._cards: List[DomoCard] = []
    
    def add_reliability_kpi(self) -> "GovernanceDashboardBuilder":
        """Add overall reliability KPI card."""
        self._cards.append(DomoCard(
            title="Overall AI Reliability",
            card_type="kpi",
            dataset_id=self.metrics_dataset_id,
            description="Average reliability across all AI operations",
            sql="""
                SELECT 
                    AVG(reliability_estimate) * 100 as reliability_pct
                FROM table
                WHERE timestamp > DATEADD(day, -7, GETDATE())
            """,
            config={
                "format": "percent",
                "goal": 95,
                "comparison_period": "previous_week"
            }
        ))
        return self
    
    def add_confidence_band_distribution(self) -> "GovernanceDashboardBuilder":
        """Add confidence band distribution chart."""
        self._cards.append(DomoCard(
            title="Confidence Band Distribution",
            card_type="bar",
            dataset_id=self.metrics_dataset_id,
            description="Distribution of responses by confidence level",
            sql="""
                SELECT 
                    confidence_band,
                    COUNT(*) as count
                FROM table
                WHERE timestamp > DATEADD(day, -7, GETDATE())
                GROUP BY confidence_band
                ORDER BY 
                    CASE confidence_band
                        WHEN 'high_confidence' THEN 1
                        WHEN 'moderate_confidence' THEN 2
                        WHEN 'low_confidence' THEN 3
                        WHEN 'insufficient' THEN 4
                    END
            """,
            config={
                "colors": {
                    "high_confidence": "#22c55e",
                    "moderate_confidence": "#eab308",
                    "low_confidence": "#f97316",
                    "insufficient": "#ef4444"
                }
            }
        ))
        return self
    
    def add_reliability_trend(self) -> "GovernanceDashboardBuilder":
        """Add reliability trend over time."""
        self._cards.append(DomoCard(
            title="Reliability Trend (7 Days)",
            card_type="line",
            dataset_id=self.metrics_dataset_id,
            description="Daily average reliability over the past week",
            sql="""
                SELECT 
                    CAST(timestamp AS DATE) as date,
                    AVG(reliability_estimate) as avg_reliability,
                    MIN(reliability_estimate) as min_reliability,
                    MAX(reliability_estimate) as max_reliability
                FROM table
                WHERE timestamp > DATEADD(day, -7, GETDATE())
                GROUP BY CAST(timestamp AS DATE)
                ORDER BY date
            """,
            config={
                "target_line": 0.95,
                "target_label": "Target (95%)"
            }
        ))
        return self
    
    def add_agent_performance_table(self) -> "GovernanceDashboardBuilder":
        """Add agent performance comparison table."""
        self._cards.append(DomoCard(
            title="Agent Performance Summary",
            card_type="table",
            dataset_id=self.metrics_dataset_id,
            description="Performance metrics by AI agent",
            sql="""
                SELECT 
                    agent_name,
                    COUNT(*) as total_tasks,
                    AVG(reliability_estimate) as avg_reliability,
                    SUM(CASE WHEN confidence_band = 'high_confidence' THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as high_confidence_pct,
                    SUM(CASE WHEN confidence_band = 'insufficient' THEN 1 ELSE 0 END) as insufficient_count,
                    AVG(isr) as avg_isr
                FROM table
                WHERE timestamp > DATEADD(day, -7, GETDATE())
                GROUP BY agent_name
                ORDER BY avg_reliability DESC
            """,
            config={
                "columns": [
                    {"name": "agent_name", "label": "Agent"},
                    {"name": "total_tasks", "label": "Tasks"},
                    {"name": "avg_reliability", "label": "Avg Reliability", "format": "percent"},
                    {"name": "high_confidence_pct", "label": "High Confidence %", "format": "percent"},
                    {"name": "insufficient_count", "label": "Insufficient"},
                    {"name": "avg_isr", "label": "Avg ISR", "format": "decimal"}
                ],
                "conditional_formatting": [
                    {"column": "avg_reliability", "threshold": 0.90, "color_below": "#ef4444"},
                    {"column": "insufficient_count", "threshold": 5, "color_above": "#f97316"}
                ]
            }
        ))
        return self
    
    def add_hallucination_risk_gauge(self) -> "GovernanceDashboardBuilder":
        """Add hallucination risk gauge."""
        self._cards.append(DomoCard(
            title="Hallucination Risk (RoH)",
            card_type="gauge",
            dataset_id=self.metrics_dataset_id,
            description="Average risk of hallucination bound",
            sql="""
                SELECT 
                    AVG(roh_bound) * 100 as avg_roh_pct
                FROM table
                WHERE timestamp > DATEADD(day, -1, GETDATE())
            """,
            config={
                "min": 0,
                "max": 30,
                "zones": [
                    {"min": 0, "max": 5, "color": "#22c55e", "label": "Low Risk"},
                    {"min": 5, "max": 15, "color": "#eab308", "label": "Moderate Risk"},
                    {"min": 15, "max": 30, "color": "#ef4444", "label": "High Risk"}
                ]
            }
        ))
        return self
    
    def add_verification_required_count(self) -> "GovernanceDashboardBuilder":
        """Add count of responses requiring verification."""
        self._cards.append(DomoCard(
            title="Responses Requiring Verification",
            card_type="kpi",
            dataset_id=self.metrics_dataset_id,
            description="Count of responses with required verification steps",
            sql="""
                SELECT 
                    SUM(CASE WHEN verification_required > 0 THEN 1 ELSE 0 END) as requires_verification,
                    COUNT(*) as total,
                    SUM(CASE WHEN verification_required > 0 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as pct
                FROM table
                WHERE timestamp > DATEADD(day, -1, GETDATE())
            """,
            config={
                "display_field": "pct",
                "format": "percent",
                "goal_direction": "down",
                "goal": 10
            }
        ))
        return self
    
    def add_isr_distribution(self) -> "GovernanceDashboardBuilder":
        """Add Information Sufficiency Ratio distribution."""
        self._cards.append(DomoCard(
            title="Information Sufficiency (ISR)",
            card_type="bar",
            dataset_id=self.metrics_dataset_id,
            description="Distribution of ISR values",
            sql="""
                SELECT 
                    CASE 
                        WHEN isr >= 2.0 THEN 'High (≥2.0)'
                        WHEN isr >= 1.0 THEN 'Sufficient (1.0-2.0)'
                        WHEN isr >= 0.5 THEN 'Low (0.5-1.0)'
                        ELSE 'Insufficient (<0.5)'
                    END as isr_band,
                    COUNT(*) as count
                FROM table
                WHERE timestamp > DATEADD(day, -7, GETDATE())
                GROUP BY 
                    CASE 
                        WHEN isr >= 2.0 THEN 'High (≥2.0)'
                        WHEN isr >= 1.0 THEN 'Sufficient (1.0-2.0)'
                        WHEN isr >= 0.5 THEN 'Low (0.5-1.0)'
                        ELSE 'Insufficient (<0.5)'
                    END
            """,
            config={
                "colors": {
                    "High (≥2.0)": "#22c55e",
                    "Sufficient (1.0-2.0)": "#3b82f6",
                    "Low (0.5-1.0)": "#f97316",
                    "Insufficient (<0.5)": "#ef4444"
                }
            }
        ))
        return self
    
    def add_workflow_reliability(self) -> "GovernanceDashboardBuilder":
        """Add workflow reliability tracking."""
        self._cards.append(DomoCard(
            title="Workflow Reliability",
            card_type="table",
            dataset_id=self.metrics_dataset_id,
            description="End-to-end reliability by workflow",
            sql="""
                SELECT 
                    workflow_name,
                    COUNT(*) as executions,
                    AVG(end_to_end_reliability) as avg_reliability,
                    SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as failures,
                    AVG(execution_time_ms) as avg_time_ms
                FROM table
                WHERE workflow_name IS NOT NULL
                  AND timestamp > DATEADD(day, -7, GETDATE())
                GROUP BY workflow_name
                ORDER BY avg_reliability DESC
            """
        ))
        return self
    
    def add_alerts_section(self) -> "GovernanceDashboardBuilder":
        """Add alerts for reliability issues."""
        self._cards.append(DomoCard(
            title="Reliability Alerts",
            card_type="table",
            dataset_id=self.metrics_dataset_id,
            description="Recent low-reliability responses requiring attention",
            sql="""
                SELECT 
                    timestamp,
                    agent_name,
                    task_id,
                    confidence_band,
                    reliability_estimate,
                    verification_required
                FROM table
                WHERE reliability_estimate < 0.80
                  AND timestamp > DATEADD(day, -1, GETDATE())
                ORDER BY timestamp DESC
                LIMIT 20
            """,
            config={
                "row_highlighting": {
                    "condition": "confidence_band = 'insufficient'",
                    "color": "#fee2e2"
                }
            }
        ))
        return self
    
    def build(self) -> DomoDashboard:
        """Build the complete dashboard configuration."""
        return DomoDashboard(
            name="AI Reliability Governance",
            description="Monitor AI agent and workflow reliability metrics for enterprise governance",
            cards=self._cards,
            filters=[
                {
                    "column": "timestamp",
                    "type": "date_range",
                    "default": "last_7_days"
                },
                {
                    "column": "agent_name",
                    "type": "dropdown",
                    "multi_select": True
                },
                {
                    "column": "confidence_band",
                    "type": "dropdown",
                    "multi_select": True
                }
            ]
        )
    
    def build_standard(self) -> DomoDashboard:
        """Build a standard governance dashboard with all components."""
        return (
            self
            .add_reliability_kpi()
            .add_hallucination_risk_gauge()
            .add_confidence_band_distribution()
            .add_reliability_trend()
            .add_isr_distribution()
            .add_agent_performance_table()
            .add_verification_required_count()
            .add_workflow_reliability()
            .add_alerts_section()
            .build()
        )
    
    def to_json(self, dashboard: DomoDashboard) -> str:
        """Export dashboard configuration to JSON."""
        return json.dumps({
            "name": dashboard.name,
            "description": dashboard.description,
            "cards": [
                {
                    "title": card.title,
                    "card_type": card.card_type,
                    "dataset_id": card.dataset_id,
                    "description": card.description,
                    "sql": card.sql,
                    "config": card.config
                }
                for card in dashboard.cards
            ],
            "filters": dashboard.filters
        }, indent=2)


def generate_governance_dashboard(
    metrics_dataset_id: str,
    output_path: Optional[str] = None
) -> str:
    """
    Generate a complete governance dashboard configuration.
    
    Args:
        metrics_dataset_id: The Domo dataset ID for reliability metrics
        output_path: Optional path to save the JSON configuration
    
    Returns:
        JSON string of the dashboard configuration
    """
    builder = GovernanceDashboardBuilder(metrics_dataset_id)
    dashboard = builder.build_standard()
    json_config = builder.to_json(dashboard)
    
    if output_path:
        with open(output_path, "w") as f:
            f.write(json_config)
    
    return json_config


# Card Beast / Domo App Studio integration
class DomoAppStudioConfig:
    """
    Configuration for Domo App Studio integration.
    
    Generates app configurations that can be deployed
    as custom Domo apps for reliability monitoring.
    """
    
    @staticmethod
    def generate_reliability_app_manifest(
        app_name: str = "AI Reliability Monitor",
        metrics_dataset_id: str = ""
    ) -> Dict[str, Any]:
        """Generate App Studio manifest for reliability monitoring app."""
        return {
            "name": app_name,
            "version": "1.0.0",
            "description": "Monitor and manage AI agent reliability",
            "size": {
                "width": 4,
                "height": 3
            },
            "mapping": [
                {
                    "dataSetId": metrics_dataset_id,
                    "alias": "metrics",
                    "fields": [
                        {"alias": "timestamp", "columnName": "timestamp"},
                        {"alias": "agent_name", "columnName": "agent_name"},
                        {"alias": "confidence_band", "columnName": "confidence_band"},
                        {"alias": "reliability", "columnName": "reliability_estimate"},
                        {"alias": "isr", "columnName": "isr"},
                        {"alias": "roh", "columnName": "roh_bound"}
                    ]
                }
            ],
            "properties": [
                {
                    "name": "reliability_target",
                    "type": "number",
                    "default": 0.95,
                    "label": "Reliability Target"
                },
                {
                    "name": "alert_threshold",
                    "type": "number",
                    "default": 0.80,
                    "label": "Alert Threshold"
                },
                {
                    "name": "time_window_days",
                    "type": "number",
                    "default": 7,
                    "label": "Time Window (Days)"
                }
            ]
        }
