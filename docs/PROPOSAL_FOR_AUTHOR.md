# Shift Left AI: Extending EDFL with Design-First Hallucination Management

**A proposed extension to the HallBayes toolkit**

---

## Summary

I've built an extension to your EDFL framework that shifts hallucination management from runtime detection to design-time architecture. The core insight comes from DevSecOps: just as security moved from a gate at the end of the pipeline to a principle embedded throughout development, hallucination management needs the same transformation.

Your EDFL work provides the mathematical foundation for Level 4 (Prediction) in a five-level maturity model. This extension adds Level 5 (Design) and makes the framework practical for enterprise AI deployments.

---

## The Problem Your Work Solves (and What's Missing)

EDFL elegantly answers: "Given this prompt, should we answer or refuse?"

What enterprises also need:
- "Before I build this agent, what reliability can I achieve?"
- "Which step in my workflow is the weakest link?"
- "How do I give users actionable uncertainty, not just binary decisions?"
- "How do I track reliability across my AI estate for governance?"

---

## What I've Built

### 1. Pre-flight Assessment (`shiftleft.preflight`)

Evaluates task feasibility *before* any generation:

```python
from shiftleft import PreflightAssessment

assessment = PreflightAssessment(
    task="Summarise Q3 financial results",
    sources=["quarterly_report.pdf"],
    reliability_target=0.95
)

result = assessment.run()
# Returns: complexity classification, information gaps,
# feasibility assessment, decomposition recommendations
```

This catches problems before you waste tokens on generation that will fail ISR thresholds anyway.

### 2. Graduated Response Layer (`shiftleft.graduated`)

Replaces binary ANSWER/REFUSE with confidence bands:

```python
from shiftleft import GraduatedResponse

response = GraduatedResponse.from_edfl_metrics(metrics)
# Returns one of:
#   HIGH_CONFIDENCE (ISR > 2.0, RoH < 0.02)
#   MODERATE_CONFIDENCE (ISR > 1.0, RoH < 0.10)
#   LOW_CONFIDENCE (ISR > 0.5, RoH < 0.25)
#   INSUFFICIENT (ISR < 0.5)
#
# Each includes: the answer, specific uncertainty flags,
# and verification steps for the user
```

This gives users actionable information rather than silent refusal.

### 3. Reliability Contracts (`shiftleft.design`)

Define requirements *before* building:

```python
from shiftleft import ContractBuilder

contract = ContractBuilder.for_summarisation(
    contract_id="summary-001",
    max_hallucination_rate=0.03
)

# Auto-generates:
# - Architecture recommendations
# - Monitoring requirements  
# - Escalation triggers
# - Source coverage requirements
```

This is the "shift left" core: embedding reliability requirements in system design.

### 4. Compositional Reliability (`shiftleft.orchestration`)

Track reliability through multi-step agent workflows:

```python
from shiftleft import ChainBuilder

chain = (
    ChainBuilder("Financial Report Pipeline")
    .with_target(0.90)
    .add("retrieve", retriever_agent, reliability=0.95)
    .add("summarise", summariser_agent, reliability=0.88)
    .add("validate", validator_agent, reliability=0.92)
    .build()
)

# Output:
# Chain: Financial Report Pipeline (target: 90%)
# → retrieve: 95% (cumulative: 95%)
# → summarise: 88% (cumulative: 84%) ⚠️ [weakest link]
# → validate: 92% (cumulative: 77%)
# End-to-end reliability: 76.9%
# Meets target: ✗
```

This identifies where interventions would have the most impact.

### 5. Domo Connector (`shiftleft.connectors.domo`)

Enterprise integration for Domo's AI platform:

- Wraps Agent Catalyst agents with reliability tracking
- Tracks reliability through Domo Workflows
- Pushes metrics to Domo datasets for governance
- Generates dashboard configurations for monitoring

---

## How It Extends EDFL

| EDFL Provides | Shift Left AI Adds |
|---------------|-------------------|
| ISR calculation | Pre-flight ISR estimation before generation |
| Binary ANSWER/REFUSE | Four confidence bands with verification steps |
| Per-prompt assessment | End-to-end reliability through agent chains |
| Mathematical bounds | Architecture recommendations from bounds |
| Runtime decision | Design-time contracts |

The extension is designed to wrap your existing `OpenAIPlanner` and `ItemMetrics`:

```python
# Your existing EDFL flow
metrics = planner.run(items, h_star=0.05)

# Shift Left AI enhancement
response = GraduatedResponse.from_edfl_metrics(
    metrics=metrics[0],
    answer=generated_answer,
    sources=["doc1.pdf", "doc2.pdf"]
)

print(response.to_user_message())
# [~ Moderate Confidence]
# 
# [Answer here]
#
# Key uncertainties:
#   • Information sufficiency below threshold
#
# Recommended before use:
#   • Verify specific facts against source
```

---

## Technical Details

- **Pure Python 3.9+**, no additional dependencies for core modules
- **Optional integrations**: pydomo for Domo, hallbayes for EDFL metrics
- **Fully typed** with dataclasses throughout
- **MIT licensed** (same as HallBayes)

Tested and working:
```
All imports: OK
Preflight assessment: OK
Graduated response: OK  
Reliability contract: OK
Reliability chain: OK
Domo connector: OK
Dashboard builder: OK

=== ALL TESTS PASSED ===
```

---

## Proposed Integration

**Option A: Submodule**
Add as `hallbayes/shiftleft/` directory, import via `from hallbayes.shiftleft import ...`

**Option B: Separate package**
Publish as `shiftleft-ai` with `hallbayes` as optional dependency

**Option C: Feature merge**
Cherry-pick specific features (graduated responses, chain analysis) into core HallBayes

---

## Why This Matters

Your Vectara benchmarks show flagship models exceeding 10% hallucination rates. Guardian agents can reduce this to under 1%, but only by catching errors after generation.

The shift-left approach means:
- Catching infeasible tasks before wasting compute
- Giving users confidence bands instead of binary gates
- Tracking reliability across complex workflows
- Building governance dashboards for enterprise deployment

This is the difference between "ambulance at the bottom of the cliff" and "fence at the top."

---

## About Me

Lee Griffiths, Director of Strategic Partnerships at Domo Europe. I'm working on AI agent governance dashboards and commercialising AI solutions for enterprise. Your EDFL paper directly addresses problems I'm seeing in production deployments where customers need more than "answer or refuse."

Happy to discuss further or submit a PR if you're interested.

---

## Links

- HallBayes repo: https://github.com/leochlon/hallbayes
- EDFL paper: arXiv:2509.11208
- Package: [attached zip file]
