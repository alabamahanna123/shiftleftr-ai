# Shift Left AI

**Design-first hallucination management for enterprise AI systems.**

Just as DevSecOps transformed security from a gate at the end of the pipeline to a principle embedded throughout development, Shift Left AI transforms hallucination management from detection to design.

## The Problem

Current approaches to AI hallucination follow a familiar pattern: deploy first, detect problems later. Even the best detection frameworks (like EDFL) operate at runtime, after the system has already been built. This is where security was twenty years ago.

The latest benchmarks tell the story. Flagship reasoning models from every major provider exceed 10% hallucination rates on straightforward factual tasks. Guardian agents can reduce this to under 1%, but only by catching errors after generation. We are building increasingly sophisticated ambulances at the bottom of the cliff.

## The Shift Left Approach

This toolkit extends the EDFL framework (Chlon et al., 2025) with a design-first philosophy:

### Five Maturity Levels

1. **Hope** — Deploy and trust outputs (where most organisations start)
2. **Detection** — Flag hallucinations after generation (Vectara HHEM, SelfCheckGPT)
3. **Correction** — Guardian agents fix errors in real-time (Vectara VHC)
4. **Prediction** — Assess risk before generation (EDFL/HallBayes)
5. **Design** — Embed hallucination management in system architecture from the start

Shift Left AI focuses on Level 5, while remaining compatible with Levels 3 and 4.

## Key Extensions

### 1. Pre-flight Checks (`shiftleft.preflight`)

Before any generation, assess whether the task decomposition and information sources are sufficient for the reliability requirements.

```python
from shiftleft.preflight import PreflightAssessment

assessment = PreflightAssessment(
    task="Summarise Q3 financial results",
    sources=["quarterly_report.pdf", "earnings_call_transcript.txt"],
    reliability_target=0.95
)

result = assessment.run()
# Returns: source_coverage, information_gaps, recommended_decomposition
```

### 2. Graduated Response Layer (`shiftleft.graduated`)

Replace binary ANSWER/REFUSE with confidence-banded outputs that give users actionable information.

```python
from shiftleft.graduated import GraduatedResponse

response = GraduatedResponse.from_edfl_metrics(metrics)
# Returns one of:
#   HIGH_CONFIDENCE (ISR > 2.0, RoH < 0.02)
#   MODERATE_CONFIDENCE (ISR > 1.0, RoH < 0.10)
#   LOW_CONFIDENCE (ISR > 0.5, RoH < 0.25)
#   INSUFFICIENT (ISR < 0.5)

# Each band includes:
#   - The answer (if any)
#   - Specific uncertainty flags
#   - Recommended verification steps
```

### 3. Design Constraints (`shiftleft.design`)

Tooling that helps architects define reliability requirements before building.

```python
from shiftleft.design import ReliabilityContract

contract = ReliabilityContract(
    task_type="financial_summary",
    max_hallucination_rate=0.05,
    required_source_coverage=0.90,
    decomposition_strategy="claim_by_claim"
)

# Generates:
#   - Architecture recommendations
#   - Required grounding sources
#   - Monitoring requirements
#   - Escalation thresholds
```

### 4. Compositional Reliability (`shiftleft.orchestration`)

Propagate uncertainty through multi-step agent workflows.

```python
from shiftleft.orchestration import ReliabilityChain

chain = ReliabilityChain()
chain.add_step("retrieve", retriever_agent, reliability=0.95)
chain.add_step("summarise", summariser_agent, reliability=0.90)
chain.add_step("validate", validator_agent, reliability=0.98)

# Calculates end-to-end reliability bounds
# Identifies weakest links
# Suggests intervention points
```

## Installation

```bash
# Clone this repository
git clone https://github.com/YOUR_USERNAME/shiftleft-ai.git
cd shiftleft-ai

# Install dependencies
pip install -e .

# For EDFL integration (optional but recommended)
pip install hallbayes
```

## Quick Start

```python
from shiftleft import ShiftLeftPlanner
from shiftleft.preflight import PreflightAssessment
from shiftleft.graduated import GraduatedResponse

# 1. Pre-flight assessment
preflight = PreflightAssessment(
    task="Who won the 2019 Nobel Prize in Physics?",
    sources=[],  # No external sources for this query
    reliability_target=0.95
)

assessment = preflight.run()
print(f"Recommended approach: {assessment.recommendation}")

# 2. Run with graduated response
planner = ShiftLeftPlanner(
    backend="anthropic",  # or "openai", "ollama", etc.
    model="claude-sonnet-4-20250514"
)

result = planner.run(
    prompt="Who won the 2019 Nobel Prize in Physics?",
    reliability_target=0.95
)

print(f"Confidence band: {result.confidence_band}")
print(f"Answer: {result.answer}")
print(f"Verification steps: {result.verification_steps}")
```

## Philosophy

The goal is not to build better detection. The goal is to build systems where hallucination management is so deeply embedded that detection becomes less necessary.

This means:
- **Information sufficiency as a design constraint**, not a runtime check
- **Graduated responses** that give users actionable uncertainty, not binary gates
- **Compositional reliability** that tracks confidence across agent boundaries
- **Graceful degradation** instead of silent failure or blanket refusal

## Compatibility

Shift Left AI is built on top of the EDFL framework from HallBayes. It supports the same backend providers:
- OpenAI (GPT-4o, GPT-5, etc.)
- Anthropic (Claude Sonnet 4, Claude Opus 4.5)
- Hugging Face (local, TGI, Inference API)
- Ollama (local models)
- OpenRouter (100+ models via unified API)

### Domo Integration

Shift Left AI includes a dedicated connector for Domo's AI platform:

```python
from shiftleft.connectors.domo import (
    DomoClient,
    DomoAgentConfig,
    DomoAgentWrapper,
    DomoWorkflowWrapper,
    DomoReliabilityTracker,
    GovernanceDashboardBuilder,
)

# Connect to Domo
client = DomoClient(
    client_id="your-client-id",
    client_secret="your-client-secret"
)
client.connect()

# Wrap a Domo AI agent with reliability tracking
agent = DomoAgentWrapper(
    client=client,
    config=DomoAgentConfig(
        agent_name="financial_qa",
        llm_model="domogpt",
        knowledge_sources=[...]
    ),
    tracker=DomoReliabilityTracker(client, metrics_dataset_id)
)

# Generate governance dashboard
dashboard = GovernanceDashboardBuilder(metrics_dataset_id).build_standard()
```

The Domo connector provides:
- Integration with PyDomo SDK for data operations
- Wrappers for Agent Catalyst AI agents
- Reliability tracking for Domo Workflows
- Pre-built governance dashboard configurations
- Metrics dataset schema for tracking

Install with: `pip install shiftleft-ai[domo]`

## Research Foundation

This toolkit builds on:
- Chlon et al. (2025), "Predictable Compression Failures" (arXiv:2509.11208)
- Vectara Hallucination Leaderboard research
- DevSecOps maturity models

## Licence

MIT — see LICENCE file for details.

## Contributing

This is an early-stage project. Contributions welcome, particularly:
- Additional pre-flight assessment heuristics
- Integration with agent orchestration frameworks (LangGraph, CrewAI)
- Real-world validation datasets
- Documentation and examples

## Attribution

Developed by Lee Griffiths, extending research by Hassana Labs.
