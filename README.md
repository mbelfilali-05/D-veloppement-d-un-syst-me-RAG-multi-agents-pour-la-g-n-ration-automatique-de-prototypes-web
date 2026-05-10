<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/LangChain-🦜-green?style=for-the-badge" alt="LangChain"/>
  <img src="https://img.shields.io/badge/LangGraph-Orchestration-orange?style=for-the-badge" alt="LangGraph"/>
  <img src="https://img.shields.io/badge/ChromaDB-Vector_Store-purple?style=for-the-badge" alt="ChromaDB"/>
  <img src="https://img.shields.io/badge/Streamlit-Interface-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit"/>
  <img src="https://img.shields.io/badge/OpenAI-GPT--4o-412991?style=for-the-badge&logo=openai&logoColor=white" alt="OpenAI"/>
</p>

# 🏗️ RAG Multi-Agent System — PDF Specs to HTML Prototypes

> **Transform any PDF specification document into a fully navigable, multi-page HTML prototype — automatically.**

A multi-agent system powered by Retrieval-Augmented Generation (RAG) that reads a PDF *cahier des charges* (specification document), understands its structure, and generates a complete, interactive single-page HTML application with navigation, forms, data tables, and responsive design — all without writing a single line of frontend code.

Built during an internship at **Sofrecom Services Maroc** (Orange Group subsidiary) · ENSAM Rabat · 2025–2026

---

## 🎯 The Problem

Turning a PDF specification into a web prototype is traditionally a manual, time-consuming process that requires a designer to interpret requirements, a developer to code the UI, and multiple rounds of review. For early-stage validation, this overhead often kills ideas before they're even tested.

**What if an AI could read your spec and build the prototype for you?**

---

## ✨ What It Does

```
📄 PDF Specification  ──▶  🤖 6 Specialized AI Agents  ──▶  🌐 Navigable HTML Prototype
```

Upload a PDF spec, and the system will:

1. **Parse & vectorize** the document into a semantic search index
2. **Analyze** the spec to extract every page, component, form, and data table
3. **Design** a visual identity (palette, typography, layout) from the spec's context
4. **Generate** the HTML shell (navbar, footer, global structure)
5. **Build each view** individually with dense, realistic content
6. **Review** the output with a hybrid 3-layer evaluation system
7. **Auto-fix** detected issues through targeted retry
8. **Deliver** the final prototype in a Streamlit interface with live preview

---

## 🏛️ Architecture — The Journey Through 3 Iterations

The system didn't start at 6 agents. It evolved through three major architectural iterations, each teaching critical lessons about LLM systems design.

### V1 — Linear Pipeline (3 agents)

```
PDF → CRAgent → CoderAgent → ExecutorAgent → HTML
```

A simple sequential pipeline. The `CoderAgent` had to generate the entire HTML in a single LLM call — often hitting output token limits and producing shallow, incomplete prototypes.

**Lesson learned:** Monolithic generation doesn't scale. A single agent can't hold the full complexity of a multi-page app in one generation pass.

### V2 — Feedback Loop (4 agents)

```
PDF → CRAgent → CoderAgent ⇄ ReviewerAgent → ExecutorAgent → HTML
```

Added a `ReviewerAgent` to critique the HTML and send feedback for re-generation. The reviewer scored 4.7/5 on a prototype that had *visible placeholder text in the navbar and broken icons*. 

**Lesson learned:** The LLM-as-Judge pattern, applied naively, is dangerously optimistic. LLMs evaluate semantics, not visual reality. You need mechanical safeguards.

### V3 — Final Architecture (6 agents) ✅

```
PDF → CRAgent → DesignAgent → ShellAgent → ViewAgent(×N) → Assembler → ReviewerAgent V2 → ExecutorAgent
                                                                              ↕
                                                                        Targeted Retry
```

The architecture that works. Six agents with **single responsibility**, orchestrated by a LangGraph state graph. Each agent does one thing well. The Assembler is pure Python (zero LLM tokens). The ReviewerAgent V2 uses a 3-layer evaluation that the LLM *cannot* game.

---

## 🤖 The 6 Agents

| Agent | Role | LLM? | Key Innovation |
|-------|------|------|----------------|
| **CRAgent** | Extracts a structured summary from the PDF via RAG | ✅ GPT-4o | Multi-thematic retrieval (4 queries × k=5) with deduplication |
| **DesignAgent** | Derives visual config (palette, fonts, layout) from the summary | ✅ GPT-4o | Outputs a parseable JSON design system |
| **ShellAgent** | Generates the HTML skeleton (navbar, footer, empty view containers) | ✅ GPT-4o | Produces the navigable SPA structure with Alpine.js routing |
| **ViewAgent** | Generates dense content for 1–2 views per call | ✅ GPT-4o | Batched generation avoids context window saturation |
| **Assembler** | Merges shell + views into a single HTML file | ❌ Pure Python | Zero tokens, zero hallucination — deterministic string injection |
| **ReviewerAgent V2** | Evaluates quality with 3-layer hybrid analysis | ✅ GPT-4o | Mechanical checks feed into the LLM as confirmed facts |
| **ExecutorAgent** | Validates structure, repairs if needed, injects resize script | ✅ GPT-4o (repair only) | Auto-height for Streamlit iframe via postMessage |

---

## 🔍 The 3-Layer Review System

The ReviewerAgent V2 is the methodological heart of the project. It solves the fundamental flaw of V1: *an LLM cannot reliably detect visual defects in its own code*.

```
┌─────────────────────────────────────────────────────────┐
│  Layer 1 — Mechanical Verification (Python, 0 tokens)   │
│  ➤ Unresolved placeholders: [Nom exact du CDC]          │
│  ➤ Broken icon text: "shopping_cart" rendered as string  │
│  ➤ Tables with < 4 rows, forms with < 3 fields          │
│  ➤ Generic data: "Lorem ipsum", "John Doe"              │
│  ➤ Missing Alpine.js x-data / x-show                    │
├─────────────────────────────────────────────────────────┤
│  Layer 2 — Conformity Check (Python, 0 tokens)          │
│  ➤ Summary views vs HTML views (fuzzy matching)          │
│  ➤ Missing CDC features (language selector, FAQ, etc.)   │
│  ➤ Currency mismatch (MAD specified but € in HTML)       │
├─────────────────────────────────────────────────────────┤
│  Layer 3 — Semantic Evaluation (GPT-4o)                 │
│  ➤ Receives Layer 1+2 issues as CONFIRMED facts         │
│  ➤ Cannot ignore or override mechanical findings        │
│  ➤ Evaluates tone, visual quality, UX coherence         │
│  ➤ Scores: completude, densité, données, interactivité  │
└─────────────────────────────────────────────────────────┘
              ↓
        Hybrid Verdict: score ≥ 4.0 AND 0 high-severity issues → "good"
                         otherwise → targeted retry (only broken views)
```

**Why this matters beyond this project:** The pattern of layering deterministic checks *before* LLM evaluation — and injecting the results as immutable context — is a reusable technique for any system using the LLM-as-Judge pattern.

---

## 🧪 The RAG Experiment Framework

The quality of the final prototype depends entirely on the quality of the initial summary. To optimize the CRAgent, a systematic experiment framework was built to test **25 configurations** across three axes:

| Axis | Variations |
|------|-----------|
| **Prompt template** | V1 (basic) → V2 (structured `##`) → V4 (with login example) → V5 (full enrichment) |
| **Retrieval strategy** | Single query vs. multi-thematic (4 targeted queries) |
| **k parameter** | 3, 4, 5, 8, 12 chunks per query |

Each configuration was evaluated by a **GPT-4o-mini judge** (temperature 0) on 5 weighted criteria:

| Criterion | Weight | Question |
|-----------|--------|----------|
| Coverage | 25% | Are all pages from the spec identified? |
| Structure | 15% | Is the summary parseable by a code generator? |
| Precision | 30% | Are UI components described with enough detail? |
| Fidelity | 20% | Does it stick to the spec without hallucination? |
| Exploitability | 10% | Can a prototype be generated without manual edits? |

**Winner:** Multi-thematic strategy, k=5, Prompt V5 — the best quality/cost tradeoff among top performers.

The framework's three phases (generation → evaluation → reporting) are fully decoupled: you can re-evaluate without regenerating, and regenerate the HTML report without re-evaluating.

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| **LLM** | OpenAI GPT-4o (agents) · GPT-4o-mini (evaluation judge) |
| **Orchestration** | LangGraph (StateGraph with conditional edges & cycles) |
| **RAG Framework** | LangChain (LCEL chains, prompt templates, output parsers) |
| **Vector Store** | ChromaDB (embedded, persistent, with deduplication index) |
| **PDF Parsing** | PyMuPDFLoader via LangChain |
| **Text Splitting** | RecursiveCharacterTextSplitter (600 chars, 150 overlap) |
| **Embeddings** | OpenAI text-embedding-ada-002 |
| **Frontend Stack** | Tailwind CSS · Alpine.js · Flowbite |
| **Interface** | Streamlit (3-step wizard with live iframe preview) |
| **Language** | Python 3.10+ |

---

## 📁 Project Structure

```
.
├── core/
│   ├── pdf_loader.py          # PDF ingestion & chunking (PyMuPDFLoader + RecursiveCharacterTextSplitter)
│   ├── vector_store.py        # ChromaDB wrapper (create / load / retrieve)
│   ├── llm_config.py          # OpenAI model & embedding configuration
│   └── token_tracker.py       # Token usage tracking across agents
│
├── agents/
│   ├── base_agent.py          # Abstract base class (run + _build_chain contract)
│   ├── cr_agent.py            # CRAgent — PDF analysis via multi-thematic RAG
│   ├── design_agent.py        # DesignAgent — visual config extraction (JSON)
│   ├── shell_agent.py         # ShellAgent — HTML skeleton generation
│   ├── view_agent.py          # ViewAgent — dense per-view content generation
│   ├── assembler.py           # Assembler — pure Python shell+views merge
│   ├── reviewer_agent.py      # ReviewerAgent V2 — 3-layer hybrid evaluation
│   ├── executor_agent.py      # ExecutorAgent — validation, repair & resize
│   └── coder_agent.py         # (Legacy V1) Monolithic HTML generation
│
├── graph/
│   └── workflow.py            # LangGraph StateGraph orchestration + dual API
│
├── experiments/
│   ├── configs.py             # 25 RAG configurations (prompt × strategy × k)
│   ├── models.py              # Dataclasses: RAGConfig, ChunkInfo, ExperimentResult
│   ├── run_experiments.py     # Batch execution of all configs on same PDF
│   ├── evaluate.py            # LLM-as-Judge evaluation (GPT-4o-mini, temp=0)
│   └── generate_report.py     # HTML comparative report generation
│
├── interface/
│   └── app.py                 # Streamlit UI (3-step wizard with live preview)
│
└── results/                   # Generated experiment outputs (JSON + HTML reports)
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- An OpenAI API key with access to GPT-4o

### Installation

```bash
git clone https://github.com/<your-username>/rag-multi-agent-prototype.git
cd rag-multi-agent-prototype

pip install -r requirements.txt
```

### Environment Setup

```bash
# Create a .env file
echo "OPENAI_API_KEY=sk-your-key-here" > .env
```

### Run the App

```bash
streamlit run interface/app.py
```

Then:
1. **Upload** your PDF specification
2. **Review** the AI-generated structured summary (editable!)
3. **Generate** → watch 6 agents work in real-time → get your HTML prototype

### Run the Experiment Framework

```bash
# Step 1: Run all 25 configurations
python -m experiments.run_experiments --pdf path/to/spec.pdf

# Step 2: Evaluate summaries with LLM judge
python -m experiments.evaluate --pdf path/to/spec.pdf

# Step 3: Generate comparative HTML report
python -m experiments.generate_report
```

---

## 📊 Key Results

- **25 RAG configurations** systematically tested and evaluated
- **3 architectural iterations** with documented failure analysis
- **6-agent final architecture** producing navigable multi-page HTML prototypes
- **3-layer evaluation** catching defects invisible to pure LLM judges
- **Targeted retry** mechanism regenerating only broken views (not the entire prototype)
- The final prototype includes 8+ views with forms, data tables, product cards, dashboards, and functional Alpine.js navigation

---

## 📚 Key References

- Lewis et al. (2020) — *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks* (NeurIPS)
- Madaan et al. (2023) — *Self-Refine: Iterative Refinement with Self-Feedback* (NeurIPS)
- Liu et al. (2023) — *Lost in the Middle: How Language Models Use Long Contexts* (TACL)
- Zheng et al. (2023) — *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena* (NeurIPS)
- Shinn et al. (2023) — *Reflexion: Language Agents with Verbal Reinforcement Learning* (NeurIPS)

---

## 👤 Author

**Mohamed Belfilali Mimoun**  
Engineering student — AI, Data Science & Digital Health  
Université Mohamed V / ENSAM Rabat  

Internship supervisor: **Basma Chouara** — Sofrecom Maroc (Orange Group)  
Academic supervisor: **Amal Tmiri**

---

## 📝 License

This project was developed as part of an academic internship. Please contact the author for usage permissions.

---

<p align="center">
  <i>Built with 🧠 LLMs, 🔗 LangGraph, and a healthy distrust of LLM self-evaluation.</i>
</p>
