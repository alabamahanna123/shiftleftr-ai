"""
Shift Left AI: Complete Example

This example demonstrates the full shift-left workflow:
1. Define reliability contract (design phase)
2. Run pre-flight assessment (before generation)
3. Execute with graduated response (during/after generation)
4. Track reliability through agent chain (multi-step)
"""

from shiftleft import (
    PreflightAssessment,
    GraduatedResponse,
    ReliabilityContract,
    ContractBuilder,
    ReliabilityChain,
    ChainBuilder,
)


def example_design_phase():
    """
    Example 1: Design Phase
    
    Before building anything, define your reliability requirements.
    """
    print("=" * 60)
    print("PHASE 1: DESIGN - Define Reliability Contract")
    print("=" * 60)
    
    # Use the builder for common patterns
    contract = ContractBuilder.for_factual_qa(
        contract_id="qa-financial-001",
        description="Answer questions about Q3 financial results",
        max_hallucination_rate=0.05
    )
    
    # Validate the contract
    issues = contract.validate()
    if issues:
        print(f"Contract issues: {issues}")
    else:
        print("Contract valid")
    
    # View generated recommendations
    print(f"\nReliability target: {1 - contract.max_hallucination_rate:.0%}")
    print(f"Decomposition strategy: {contract.decomposition_strategy.value}")
    print(f"\nArchitecture recommendations:")
    for rec in contract.architecture_recommendations:
        print(f"  [{rec.priority.upper()}] {rec.component}: {rec.recommendation}")
    
    print(f"\nMonitoring requirements:")
    for mon in contract.monitoring_requirements[:3]:
        print(f"  - {mon.metric} {mon.action} at {mon.threshold}")
    
    return contract


def example_preflight_phase():
    """
    Example 2: Pre-flight Phase
    
    Before generating any response, assess feasibility.
    """
    print("\n" + "=" * 60)
    print("PHASE 2: PRE-FLIGHT - Assess Before Generation")
    print("=" * 60)
    
    # Simple factual query with no external sources
    assessment = PreflightAssessment(
        task="Who won the 2019 Nobel Prize in Physics?",
        sources=[],  # No external sources provided
        reliability_target=0.95
    )
    
    result = assessment.run()
    
    print(f"\nTask: {result.task}")
    print(f"Complexity: {result.complexity.value}")
    print(f"Estimated reliability: {result.estimated_reliability:.0%}")
    print(f"Target: {result.reliability_target:.0%}")
    print(f"Feasible: {result.is_feasible}")
    print(f"\nRecommendation: {result.recommendation}")
    
    if result.information_gaps:
        print(f"\nInformation gaps:")
        for gap in result.information_gaps[:3]:
            print(f"  - {gap}")
    
    if result.risk_factors:
        print(f"\nRisk factors:")
        for risk in result.risk_factors[:3]:
            print(f"  - {risk}")
    
    # Query with sources
    print("\n--- With external sources ---")
    
    assessment_with_sources = PreflightAssessment(
        task="Summarise the Q3 financial results",
        sources=["quarterly_report.pdf"],
        source_metadata={
            "quarterly_report.pdf": {
                "type": "user_provided",
                "coverage": 0.95,
                "year": 2025
            }
        },
        reliability_target=0.95
    )
    
    result2 = assessment_with_sources.run()
    print(f"With sources - Estimated reliability: {result2.estimated_reliability:.0%}")
    print(f"Feasible: {result2.is_feasible}")
    print(f"Recommendation: {result2.recommendation}")
    
    return result, result2


def example_graduated_response():
    """
    Example 3: Graduated Response
    
    Create responses with explicit confidence bands.
    """
    print("\n" + "=" * 60)
    print("PHASE 3: GRADUATED RESPONSE - Confidence-Banded Output")
    print("=" * 60)
    
    # Simulate EDFL metrics (in real use, these come from hallbayes)
    class MockMetrics:
        isr = 1.5
        roh_bound = 0.08
        decision_answer = True
        q_conservative = 0.4
        q_avg = 0.6
        b2t = 1.2
        rationale = "ISR above threshold, proceeding with moderate confidence"
    
    metrics = MockMetrics()
    
    response = GraduatedResponse.from_edfl_metrics(
        metrics=metrics,
        answer="The 2019 Nobel Prize in Physics was awarded to James Peebles (1/2) for theoretical discoveries in physical cosmology, and Michel Mayor and Didier Queloz (1/2) for the discovery of an exoplanet orbiting a solar-type star.",
        sources=["Nobel Prize official announcement"]
    )
    
    print(f"\nConfidence band: {response.confidence_band.value}")
    print(f"Reliability estimate: {response.reliability_estimate:.0%}")
    print(f"ISR: {response.isr:.2f}")
    print(f"RoH bound: {response.roh_bound:.0%}")
    
    print(f"\nUncertainty flags: {len(response.uncertainty_flags)}")
    for flag in response.uncertainty_flags[:2]:
        print(f"  - [{flag.severity}] {flag.uncertainty_type}: {flag.mitigation}")
    
    print(f"\nVerification steps:")
    for step in response.verification_steps[:3]:
        print(f"  - [{step.priority}] {step.action}")
    
    # User-facing message
    print("\n--- User-facing output ---")
    print(response.to_user_message())
    
    return response


def example_reliability_chain():
    """
    Example 4: Reliability Chain
    
    Track reliability through multi-step agent workflows.
    """
    print("\n" + "=" * 60)
    print("PHASE 4: RELIABILITY CHAIN - Multi-Step Tracking")
    print("=" * 60)
    
    # Define mock agents
    def retriever(input_data, context=None):
        return {"documents": ["doc1", "doc2"], "query": input_data}
    
    def summariser(input_data, context=None):
        return {"summary": f"Summary of {len(input_data.get('documents', []))} documents"}
    
    def validator(input_data, context=None):
        return {"validated": True, "summary": input_data.get("summary")}
    
    # Build the chain
    chain = (
        ChainBuilder("Financial Report Pipeline")
        .with_target(0.90)
        .add("retrieve", retriever, reliability=0.95)
        .add("summarise", summariser, reliability=0.88)
        .add("validate", validator, reliability=0.92)
        .build()
    )
    
    # Analyse before execution
    print("\nChain visualisation:")
    print(chain.visualise())
    
    # Identify intervention points
    print("\nIntervention analysis:")
    interventions = chain.identify_intervention_points()
    for i, intervention in enumerate(interventions[:3], 1):
        print(f"  {i}. {intervention['step_name']}: "
              f"Current {intervention['current_reliability']:.0%}, "
              f"Impact if improved: +{intervention['impact_if_improved']:.1%}")
        print(f"     Recommendation: {intervention['recommendation']}")
    
    return chain


def example_full_workflow():
    """
    Full workflow combining all phases.
    """
    print("\n" + "=" * 60)
    print("FULL WORKFLOW: Design → Pre-flight → Execute → Track")
    print("=" * 60)
    
    # 1. Design: Define contract
    contract = ContractBuilder.for_summarisation(
        contract_id="summary-001",
        description="Summarise uploaded document",
        max_hallucination_rate=0.03
    )
    print(f"\n1. Contract defined: target {1 - contract.max_hallucination_rate:.0%} reliability")
    
    # 2. Pre-flight: Assess feasibility
    preflight = PreflightAssessment(
        task="Summarise the attached quarterly report",
        sources=["quarterly_report.pdf"],
        source_metadata={
            "quarterly_report.pdf": {
                "type": "user_provided",
                "coverage": 1.0,
                "year": 2025
            }
        },
        reliability_target=1 - contract.max_hallucination_rate
    )
    
    result = preflight.run()
    print(f"2. Pre-flight: {'PASS' if result.is_feasible else 'FAIL'} "
          f"(estimated {result.estimated_reliability:.0%})")
    
    if not result.is_feasible:
        print(f"   Recommendation: {result.recommendation}")
        return
    
    # 3. Execute with graduated response
    # (In real use, this would call the LLM and get EDFL metrics)
    class MockMetrics:
        isr = 2.1
        roh_bound = 0.02
        decision_answer = True
        q_conservative = 0.5
        q_avg = 0.7
        b2t = 1.0
        rationale = "High ISR, proceeding with high confidence"
    
    response = GraduatedResponse.from_edfl_metrics(
        metrics=MockMetrics(),
        answer="The Q3 report shows revenue growth of 15% year-over-year...",
        sources=["quarterly_report.pdf"]
    )
    print(f"3. Response: {response.confidence_band.value} "
          f"(reliability {response.reliability_estimate:.0%})")
    
    # 4. Audit record
    audit = response.to_audit_record()
    print(f"4. Audit: {audit['uncertainty_count']} uncertainties, "
          f"{audit['verification_required']} required verifications")
    
    # Check against contract
    meets_contract = response.reliability_estimate >= (1 - contract.max_hallucination_rate)
    print(f"\nContract compliance: {'PASS' if meets_contract else 'FAIL'}")


if __name__ == "__main__":
    # Run all examples
    example_design_phase()
    example_preflight_phase()
    example_graduated_response()
    example_reliability_chain()
    example_full_workflow()
    
    print("\n" + "=" * 60)
    print("Examples complete. See README.md for more details.")
    print("=" * 60)
