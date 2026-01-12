"""
Shift Left AI: Domo Integration Example

Demonstrates how to integrate Shift Left AI with Domo for:
1. AI agent reliability tracking
2. Workflow reliability chains
3. Governance dashboards
4. Reliability contracts for Domo agents
"""

import asyncio
from datetime import datetime

# Note: In production, install pydomo with: pip install pydomo


def example_setup_domo_client():
    """
    Example 1: Setting up the Domo client
    """
    print("=" * 60)
    print("EXAMPLE 1: Domo Client Setup")
    print("=" * 60)
    
    from shiftleft.connectors.domo import DomoClient
    
    # In production, use environment variables or secure config
    client = DomoClient(
        client_id="your-client-id",
        client_secret="your-client-secret",
        api_host="api.domo.com",
        instance_domain="your-instance.domo.com"
    )
    
    # Note: connect() requires pydomo installed and valid credentials
    # connected = client.connect()
    
    print("Client configured (not connected - requires credentials)")
    print(f"API Host: {client.api_host}")
    
    return client


def example_create_reliability_contract():
    """
    Example 2: Creating reliability contracts for Domo agents
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Reliability Contract for Domo Agent")
    print("=" * 60)
    
    from shiftleft.connectors.domo import create_domo_contract_for_agent
    from shiftleft.design import TaskCategory
    
    # Create contract for a financial analysis agent
    contract = create_domo_contract_for_agent(
        agent_name="financial_analyst",
        task_type=TaskCategory.ANALYSIS,
        max_hallucination_rate=0.05  # 5% max
    )
    
    print(f"Contract ID: {contract.contract_id}")
    print(f"Task type: {contract.task_type.value}")
    print(f"Max hallucination rate: {contract.max_hallucination_rate:.0%}")
    print(f"Min confidence band: {contract.min_confidence_band}")
    
    print("\nArchitecture recommendations:")
    for rec in contract.architecture_recommendations[:3]:
        print(f"  [{rec.priority}] {rec.component}: {rec.recommendation}")
    
    print("\nEscalation triggers:")
    for trigger in contract.escalation_triggers:
        print(f"  - {trigger.value}")
    
    return contract


def example_configure_domo_agent():
    """
    Example 3: Configuring a Domo AI agent with reliability tracking
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Configure Domo Agent with Reliability")
    print("=" * 60)
    
    from shiftleft.connectors.domo import (
        DomoClient,
        DomoAgentConfig,
        DomoAgentWrapper,
        DomoReliabilityTracker,
        DomoDataSource,
        DomoDataSourceType,
        create_domo_contract_for_agent,
    )
    
    # Setup client (mock for example)
    client = DomoClient(
        client_id="demo",
        client_secret="demo"
    )
    
    # Define knowledge sources
    knowledge_sources = [
        DomoDataSource(
            source_type=DomoDataSourceType.DATASET,
            source_id="abc123",
            name="Q3 Financial Results",
            description="Quarterly financial metrics",
            row_count=10000
        ),
        DomoDataSource(
            source_type=DomoDataSourceType.FILESET,
            source_id="def456",
            name="Financial Reports PDFs",
            description="Supporting documentation"
        )
    ]
    
    # Configure the agent
    agent_config = DomoAgentConfig(
        agent_name="financial_qa_agent",
        llm_model="domogpt",
        instructions="""
            You are a financial analyst assistant. Answer questions about
            company financial performance using the provided datasets.
            Always cite specific data points from the knowledge sources.
            If information is not available, say so clearly.
        """,
        knowledge_sources=knowledge_sources,
        tools=["sql_query", "chart_generator"],
        temperature=0.2,
        max_tokens=1024
    )
    
    # Create reliability contract
    contract = create_domo_contract_for_agent(
        agent_name=agent_config.agent_name,
        max_hallucination_rate=0.05
    )
    
    # Setup tracker (would use real dataset in production)
    tracker = DomoReliabilityTracker(
        client=client,
        metrics_dataset_id="metrics_dataset_id"
    )
    
    # Create wrapped agent
    agent = DomoAgentWrapper(
        client=client,
        config=agent_config,
        reliability_contract=contract,
        tracker=tracker
    )
    
    print(f"Agent configured: {agent_config.agent_name}")
    print(f"LLM model: {agent_config.llm_model}")
    print(f"Knowledge sources: {len(agent_config.knowledge_sources)}")
    print(f"Tools: {agent_config.tools}")
    print(f"Contract target: {1 - contract.max_hallucination_rate:.0%} reliability")
    
    # Run pre-flight assessment
    print("\nPre-flight assessment for sample task:")
    preflight = agent.assess_task("What was our Q3 revenue growth?")
    print(f"  Complexity: {preflight.complexity.value}")
    print(f"  Estimated reliability: {preflight.estimated_reliability:.0%}")
    print(f"  Feasible: {preflight.is_feasible}")
    print(f"  Recommendation: {preflight.recommendation}")
    
    return agent


def example_workflow_reliability():
    """
    Example 4: Tracking reliability through Domo workflows
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Domo Workflow Reliability Tracking")
    print("=" * 60)
    
    from shiftleft.connectors.domo import (
        DomoClient,
        DomoWorkflowStep,
        DomoWorkflowWrapper,
        DomoReliabilityTracker,
    )
    
    # Setup client (mock)
    client = DomoClient(client_id="demo", client_secret="demo")
    
    # Define workflow steps
    steps = [
        DomoWorkflowStep(
            step_id="step_1",
            step_type="data_transform",
            name="Extract Financial Data",
            config={"dataset_id": "abc123", "filters": ["Q3", "2025"]},
            outputs=["financial_data"]
        ),
        DomoWorkflowStep(
            step_id="step_2",
            step_type="ai_agent_task",
            name="Analyse Trends",
            config={"agent": "trend_analyser", "prompt_template": "analyse_quarterly"},
            inputs=["financial_data"],
            outputs=["trend_analysis"]
        ),
        DomoWorkflowStep(
            step_id="step_3",
            step_type="ai_agent_task",
            name="Generate Summary",
            config={"agent": "summariser", "max_tokens": 500},
            inputs=["trend_analysis"],
            outputs=["summary"]
        ),
        DomoWorkflowStep(
            step_id="step_4",
            step_type="action",
            name="Send to Slack",
            config={"channel": "#finance-updates"},
            inputs=["summary"]
        )
    ]
    
    # Create workflow wrapper
    workflow = DomoWorkflowWrapper(
        client=client,
        workflow_name="Q3 Financial Summary Pipeline",
        steps=steps,
        tracker=DomoReliabilityTracker(client, "metrics_ds")
    )
    
    # Build reliability chain with custom estimates for AI steps
    workflow.build_reliability_chain(
        reliability_estimates={
            "step_2": 0.88,  # AI analysis step
            "step_3": 0.85,  # AI summary step
        }
    )
    
    # Analyse before execution
    analysis = workflow.analyse()
    
    print(f"Workflow: {analysis['workflow_name']}")
    print(f"Steps: {analysis['step_count']}")
    print(f"End-to-end reliability: {analysis['end_to_end_reliability']:.1%}")
    print(f"Meets 90% target: {analysis['meets_target']}")
    print(f"Weakest link: {analysis['weakest_link']}")
    
    if analysis['gap_to_target'] > 0:
        print(f"Gap to target: {analysis['gap_to_target']:.1%}")
    
    print("\nTop intervention recommendations:")
    for intervention in analysis['interventions']:
        print(f"  {intervention['step_name']}: "
              f"Current {intervention['current_reliability']:.0%}, "
              f"Impact if improved: +{intervention['impact_if_improved']:.1%}")
    
    print("\nChain visualisation:")
    print(analysis['visualisation'])
    
    return workflow


def example_governance_dashboard():
    """
    Example 5: Creating a governance dashboard
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 5: Governance Dashboard Generation")
    print("=" * 60)
    
    from shiftleft.connectors.domo import (
        GovernanceDashboardBuilder,
        generate_governance_dashboard,
        create_domo_reliability_dashboard_schema,
    )
    
    # First, get the schema for the metrics dataset
    schema = create_domo_reliability_dashboard_schema()
    print("Metrics Dataset Schema:")
    print(f"  Name: {schema['name']}")
    print(f"  Columns: {len(schema['schema']['columns'])}")
    for col in schema['schema']['columns'][:5]:
        print(f"    - {col['name']} ({col['type']})")
    print("    ...")
    
    # Generate dashboard
    metrics_dataset_id = "your_metrics_dataset_id"
    
    builder = GovernanceDashboardBuilder(metrics_dataset_id)
    dashboard = (
        builder
        .add_reliability_kpi()
        .add_hallucination_risk_gauge()
        .add_confidence_band_distribution()
        .add_reliability_trend()
        .add_agent_performance_table()
        .add_alerts_section()
        .build()
    )
    
    print(f"\nDashboard: {dashboard.name}")
    print(f"Cards: {len(dashboard.cards)}")
    for card in dashboard.cards:
        print(f"  - {card.title} ({card.card_type})")
    
    print(f"\nFilters: {len(dashboard.filters)}")
    for f in dashboard.filters:
        print(f"  - {f['column']} ({f['type']})")
    
    # Export to JSON
    json_config = builder.to_json(dashboard)
    print(f"\nJSON config length: {len(json_config)} characters")
    print("(Save this JSON and import to Domo to create the dashboard)")
    
    return dashboard


def example_full_integration():
    """
    Example 6: Full integration workflow
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 6: Full Integration Workflow")
    print("=" * 60)
    
    from shiftleft.connectors.domo import (
        DomoClient,
        DomoAgentConfig,
        DomoAgentWrapper,
        DomoReliabilityTracker,
        DomoDataSource,
        DomoDataSourceType,
        create_domo_contract_for_agent,
    )
    from shiftleft.graduated import ConfidenceBand
    
    print("Simulating end-to-end reliability tracking...\n")
    
    # 1. Setup
    client = DomoClient(client_id="demo", client_secret="demo")
    tracker = DomoReliabilityTracker(client, "metrics_dataset")
    
    # 2. Configure agent
    config = DomoAgentConfig(
        agent_name="sales_insights",
        llm_model="domogpt",
        knowledge_sources=[
            DomoDataSource(
                source_type=DomoDataSourceType.DATASET,
                source_id="sales_data",
                name="Sales Pipeline Data"
            )
        ]
    )
    
    contract = create_domo_contract_for_agent("sales_insights")
    agent = DomoAgentWrapper(client, config, contract, tracker)
    
    # 3. Simulate multiple tasks
    tasks = [
        "What was our win rate last month?",
        "Which region had the highest growth?",
        "Predict Q4 revenue based on current pipeline",
        "Explain the variance between forecast and actual",
    ]
    
    print("Simulating agent tasks:")
    for i, task in enumerate(tasks, 1):
        preflight = agent.assess_task(task)
        
        # Simulate response with mock reliability
        from shiftleft.graduated.response import GraduatedResponse, VerificationStep
        
        # Mock different confidence levels
        if i == 1:
            band = ConfidenceBand.HIGH
            reliability = 0.96
        elif i == 2:
            band = ConfidenceBand.MODERATE
            reliability = 0.88
        elif i == 3:
            band = ConfidenceBand.LOW
            reliability = 0.72
        else:
            band = ConfidenceBand.MODERATE
            reliability = 0.85
        
        response = GraduatedResponse(
            confidence_band=band,
            answer=f"[Response to: {task[:30]}...]",
            uncertainty_flags=[],
            verification_steps=[],
            reliability_estimate=reliability,
            isr=1.5 if reliability > 0.85 else 0.8,
            roh_bound=1 - reliability,
            sources_used=["Sales Pipeline Data"],
            reasoning_trace=None
        )
        
        tracker.record_response(
            config.agent_name,
            f"task_{i}",
            response,
            metadata={"task": task}
        )
        
        print(f"  Task {i}: {band.value} ({reliability:.0%})")
    
    # 4. Get statistics
    print("\nReliability statistics:")
    stats = tracker.get_recent_reliability()
    print(f"  Total tasks: {stats['count']}")
    print(f"  Mean reliability: {stats['mean_reliability']:.0%}")
    print(f"  Min reliability: {stats['min_reliability']:.0%}")
    print(f"  Max reliability: {stats['max_reliability']:.0%}")
    print(f"  High confidence rate: {stats['high_confidence_rate']:.0%}")
    
    # 5. Check against contract
    meets_contract = stats['mean_reliability'] >= (1 - contract.max_hallucination_rate)
    print(f"\nContract compliance: {'PASS' if meets_contract else 'FAIL'}")
    print(f"  Target: {1 - contract.max_hallucination_rate:.0%}")
    print(f"  Actual: {stats['mean_reliability']:.0%}")


if __name__ == "__main__":
    # Run all examples
    example_setup_domo_client()
    example_create_reliability_contract()
    example_configure_domo_agent()
    example_workflow_reliability()
    example_governance_dashboard()
    example_full_integration()
    
    print("\n" + "=" * 60)
    print("Examples complete.")
    print("=" * 60)
    print("\nTo use in production:")
    print("1. Install pydomo: pip install pydomo")
    print("2. Configure Domo API credentials")
    print("3. Create metrics dataset using provided schema")
    print("4. Import governance dashboard JSON to Domo")
    print("5. Connect agents and workflows to reliability tracking")
