# SOTA Evaluation Comparison

## How SOTA Labs Evaluate LLMs for Agentic Work

### 1. SWE-bench (Princeton/OpenAI)
- **What**: 2,294 real GitHub issues from 12 Python repos
- **How**: Model receives issue description + repo context, must produce a patch
- **Score**: % of issues resolved (patch passes unit tests)
- **SOTA**: 79.2% on Verified (GPT-5.4, Claude Opus 4.6), 77.8% on Pro (Claude Mythos)
- **Limitation**: Python-only, requires cloud API, no build-system tasks

### 2. RULER (NVIDIA)
- **What**: Synthetic long-context benchmark with 13 task types
- **How**: Needle-in-a-haystack, multi-hop tracing, aggregation, QA
- **Score**: Task accuracy at varying context lengths (4K-128K+)
- **SOTA**: >95% at 32K for frontier models
- **Limitation**: Synthetic tasks, no real-world grounding, no tool use

### 3. METR Time Horizon
- **What**: Measures how long an AI agent can work autonomously
- **How**: Agent given tasks of increasing complexity, measured by max duration
- **Score**: Maximum autonomous task duration (minutes)
- **SOTA**: 160 minutes
- **Limitation**: Expensive to run, requires full environment access

### 4. BinaryAudit
- **What**: Security-focused backdoor detection in compiled binaries
- **How**: Model analyzes binary code for planted vulnerabilities
- **Score**: Detection rate
- **SOTA**: 49%
- **Limitation**: Narrow security domain

### 5. OTelBench
- **What**: Observability instrumentation across 11 programming languages
- **How**: Model must add correct telemetry to codebases
- **Score**: Instrumentation correctness
- **SOTA**: 29%
- **Limitation**: Specific to observability

## Our Approach: Build Engineering Agentic Evaluation

### Design Philosophy

We differ from SOTA evaluations in several key ways:

| Aspect | SOTA Cloud Evals | Our On-Device Eval |
|--------|-----------------|-------------------|
| **Execution** | Cloud API (GPT-5, Claude, Gemini) | On-device via LiteRT-LM |
| **Model size** | 100B-1T+ parameters | ~4B params (4-bit quantized) |
| **Memory** | Unlimited cloud VRAM | ~3.5GB model footprint |
| **Context** | 128K-1M tokens | Up to 32K tokens |
| **Domain** | General software eng | Build-system orchestration |
| **Realism** | Real GitHub issues | Real Chromium build scenarios |
| **Cost** | $0.01-$0.10 per eval | Free (runs locally) |

### Why Chromium Build as Eval Domain

Compiling Chromium is uniquely suited for evaluating agentic capabilities:

1. **Scale**: 52,891 compilation units, 100M+ lines of code
2. **Complexity**: Cross-platform (6 OS), multiple backends (CPU/GPU/NPU)
3. **Toolchain depth**: depot_tools -> gclient -> gn -> ninja -> clang/gcc
4. **Error diversity**: Missing deps, OOM, disk space, linker errors, ABI issues
5. **Real-world relevance**: Major browsers (Chrome, Edge, Brave) all compile Chromium
6. **Context demands**: Error logs easily reach 10K-100K tokens
7. **Multi-step reasoning**: 50+ interdependent build steps
8. **Tool use**: Shell commands, file edits, config management

### Scoring Methodology

**Harness 1 (Planning)**: Weighted combination of:
- Step completeness (35%): Are all required build steps present?
- Dependency ordering (25%): Are steps in correct order?
- Technical accuracy (25%): Are commands syntactically correct?
- Bonus coverage (15%): Does the plan address practical concerns?

**Harness 2 (Error Diagnosis)**: Weighted combination of:
- Root cause identification (40%): Correct diagnosis of build failure
- Fix quality (35%): Appropriate and effective remediation steps
- Evidence citation (25%): References to specific log lines/configs

**Harness 3 (Tool Use)**: Weighted combination of:
- Action correctness (30%): Appropriate tool invocations
- Reasoning quality (30%): Sound diagnostic reasoning
- Context retention (25%): References to prior turns
- Efficiency (15%): Fewest turns to resolution

### Expected Performance Ranges

For a 4B parameter quantized model running on-device:
- **Planning**: 0.4-0.7 (smaller models miss nuanced build steps)
- **Error Diagnosis**: 0.3-0.6 (requires understanding of build systems)
- **Tool Use**: 0.2-0.5 (multi-turn coherence is challenging at 4B scale)

For reference, we'd expect frontier cloud models (GPT-5.4, Claude Opus 4.6):
- **Planning**: 0.8-0.95
- **Error Diagnosis**: 0.7-0.9
- **Tool Use**: 0.6-0.85
