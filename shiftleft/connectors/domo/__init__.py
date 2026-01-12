"""
Shift Left AI: Domo Connector

Integration with Domo's AI and data platform for enterprise
hallucination management and governance.
"""

from shiftleft.connectors.domo.client import (
    DomoClient,
    DomoDataSource,
    DomoDataSourceType,
    DomoAgentConfig,
    DomoAgentWrapper,
    DomoWorkflowStep,
    DomoWorkflowWrapper,
    DomoReliabilityTracker,
    create_domo_reliability_dashboard_schema,
    create_domo_contract_for_agent,
)

from shiftleft.connectors.domo.dashboard import (
    GovernanceDashboardBuilder,
    DomoDashboard,
    DomoCard,
    DomoAppStudioConfig,
    generate_governance_dashboard,
)

__all__ = [
    # Client and data
    "DomoClient",
    "DomoDataSource",
    "DomoDataSourceType",
    
    # Agent integration
    "DomoAgentConfig",
    "DomoAgentWrapper",
    
    # Workflow integration
    "DomoWorkflowStep",
    "DomoWorkflowWrapper",
    
    # Tracking and governance
    "DomoReliabilityTracker",
    "GovernanceDashboardBuilder",
    "DomoDashboard",
    "DomoCard",
    "DomoAppStudioConfig",
    
    # Utilities
    "create_domo_reliability_dashboard_schema",
    "create_domo_contract_for_agent",
    "generate_governance_dashboard",
]
