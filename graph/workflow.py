# graph/workflow.py
#
# Orchestration LangGraph — architecture 6 agents
#
# Pipeline :
#   START → analyze → design → shell → views → assemble → review → [retry?] → validate → END
#
# Le retry est CIBLÉ : seules les vues problématiques sont régénérées,
# pas tout le HTML.

from typing import TypedDict, List, Optional, Dict
from langgraph.graph import StateGraph, END

from core.pdf_loader import PDFLoader
from core.vector_store import VectorStore
from agents.cr_agent import CRAgent
from agents.design_agent import DesignAgent
from agents.shell_agent import ShellAgent
from agents.view_agent import ViewAgent
from agents.assembler import Assembler
from agents.reviewer_agent import ReviewerAgent
from agents.executor_agent import ExecutorAgent


# ════════════════════════════════════════════════════════════════════
#  AgentState — état partagé entre tous les nœuds
# ════════════════════════════════════════════════════════════════════

class AgentState(TypedDict, total=False):
    # Pipeline de base
    pdf_path:        str
    summary:         str
    final_result:    str
    render_height:   int
    status:          str
    errors:          List[str]

    # Architecture 6 agents
    design_config:   Optional[dict]       # JSON du DesignAgent
    shell_html:      str                  # Shell du ShellAgent
    views_html:      Dict[str, str]       # {view_id: html} du ViewAgent
    html_code:       str                  # HTML assemblé final

    # Cycle de review
    review_feedback: Optional[dict]
    quality_score:   float
    verdict:         str
    iteration:       int
    max_iterations:  int


def initial_state(pdf_path: str = "") -> AgentState:
    return {
        "pdf_path":        pdf_path,
        "summary":         "",
        "final_result":    "",
        "render_height":   900,
        "errors":          [],
        "design_config":   None,
        "shell_html":      "",
        "views_html":      {},
        "html_code":       "",
        "review_feedback": None,
        "quality_score":   0.0,
        "verdict":         "",
        "iteration":       0,
        "max_iterations":  2,
    }


# ════════════════════════════════════════════════════════════════════
#  Nœuds du graphe
# ════════════════════════════════════════════════════════════════════

def analyze_node(state: AgentState) -> AgentState:
    """Nœud 1 : PDFLoader → VectorStore → CRAgent → summary"""
    pdf_path = state.get("pdf_path", "")
    if not pdf_path:
        return {**state, "final_result": "ERROR",
                "errors": state.get("errors", []) + ["pdf_path manquant"]}

    loader = PDFLoader()
    chunks = loader.load(pdf_path)

    vs = VectorStore()
    if not vs.load():
        vs.create(chunks)

    cr_agent = CRAgent(vector_store=vs)
    state = cr_agent.run(state)

    if state.get("status") == "error" or not state.get("summary"):
        state["final_result"] = "ERROR"
    return state


def design_node(state: AgentState) -> AgentState:
    """Nœud 2 : DesignAgent → design_config (JSON)"""
    agent = DesignAgent()
    return agent.run(state)


def shell_node(state: AgentState) -> AgentState:
    """Nœud 3 : ShellAgent → shell_html"""
    agent = ShellAgent()
    return agent.run(state)


def views_node(state: AgentState) -> AgentState:
    """Nœud 4 : ViewAgent → views_html (dict)"""
    agent = ViewAgent()
    state = agent.run(state)
    state["iteration"] = state.get("iteration", 0) + 1
    return state


def assemble_node(state: AgentState) -> AgentState:
    """Nœud 5 : Assembler → html_code (Python pur)"""
    assembler = Assembler()
    return assembler.run(state)


def review_node(state: AgentState) -> AgentState:
    """Nœud 6 : ReviewerAgent → review_feedback + quality_score + verdict"""
    agent = ReviewerAgent()
    return agent.run(state)


def targeted_retry_node(state: AgentState) -> AgentState:
    """
    Nœud de retry ciblé : ne régénère QUE les vues problématiques
    identifiées par le ReviewerAgent, puis réassemble.
    """
    feedback = state.get("review_feedback", {})
    design_config = state.get("design_config", {})
    summary = state.get("summary", "")
    views_html = dict(state.get("views_html", {}))  # copie

    if not feedback or not design_config:
        return state

    # Identifier les vues à régénérer
    views_to_retry = set()

    # 1. Vues manquantes
    missing = feedback.get("missing_views", [])
    for mv in missing:
        views_to_retry.add(mv)

    # 2. Vues avec issues high
    issues = feedback.get("issues", [])
    for issue in issues:
        if issue.get("severity") == "high":
            vue_name = issue.get("vue", "")
            if vue_name and vue_name != "global":
                views_to_retry.add(vue_name)

    # 3. Si aucune vue spécifique identifiée mais score bas,
    #    régénérer les 2 premières vues (landing + catalog généralement)
    if not views_to_retry and state.get("quality_score", 0) < 3.5:
        all_views = design_config.get("views", [])
        for v in all_views[:2]:
            views_to_retry.add(v["id"])

    if not views_to_retry:
        print(f"[targeted_retry] Aucune vue à régénérer identifiée")
        return state

    print(f"[targeted_retry] 🔄 Régénération ciblée de {len(views_to_retry)} vue(s) : "
          f"{', '.join(views_to_retry)}")

    # Régénérer chaque vue ciblée
    view_agent = ViewAgent()
    all_views = design_config.get("views", [])

    for view_id_or_name in views_to_retry:
        # Trouver la view_info correspondante
        view_info = None
        for v in all_views:
            if v["id"] == view_id_or_name or v["name"] == view_id_or_name:
                view_info = v
                break

        if not view_info:
            print(f"[targeted_retry]   ⚠️ Vue '{view_id_or_name}' non trouvée dans design_config")
            continue

        try:
            new_html = view_agent.generate_single_view(view_info, design_config, summary)
            views_html[view_info["id"]] = new_html
            lines = new_html.count("\n") if new_html else 0
            print(f"[targeted_retry]   ✅ {view_info['name']} régénérée ({lines} lignes)")
        except Exception as e:
            print(f"[targeted_retry]   ❌ {view_info['name']} : {e}")

    # Incrémenter l'itération
    state["iteration"] = state.get("iteration", 0) + 1
    state["views_html"] = views_html

    # Réassembler
    assembler = Assembler()
    state = assembler.run(state)

    return state


def validate_node(state: AgentState) -> AgentState:
    """Nœud final : ExecutorAgent → validation + resize"""
    agent = ExecutorAgent()
    return agent.run(state)


# ════════════════════════════════════════════════════════════════════
#  Routage conditionnel
# ════════════════════════════════════════════════════════════════════

def route_after_analyze(state: AgentState) -> str:
    if state.get("final_result") == "ERROR":
        return "end"
    return "design"


def route_after_design(state: AgentState) -> str:
    if not state.get("design_config"):
        return "end"
    return "shell"


def route_after_review(state: AgentState) -> str:
    """
    Après review : retry ciblé ou validation finale.
    """
    verdict = state.get("verdict", "insufficient")
    iteration = state.get("iteration", 0)
    max_iter = state.get("max_iterations", 2)

    if verdict == "good":
        print(f"[workflow] ✅ Verdict 'good' (score {state.get('quality_score', 0)}/5) "
              f"— passage à validate")
        return "validate"

    if iteration > max_iter:
        print(f"[workflow] ⏹️  Budget épuisé ({iteration - 1} retries) "
              f"— score {state.get('quality_score', 0)}/5 — passage à validate")
        return "validate"

    print(f"[workflow] 🔄 Retry ciblé (score {state.get('quality_score', 0)}/5, "
          f"itération {iteration})")
    return "retry"


# ════════════════════════════════════════════════════════════════════
#  Construction du graphe
# ════════════════════════════════════════════════════════════════════

def build_graph():
    """
    Construit le StateGraph avec les 6 agents + retry ciblé.

    START → analyze → design → shell → views → assemble → review
                                                            │
                                                  ┌────────┤
                                                  ▼        │
                                              [verdict?]   │
                                                  │        │
                                      good/max ──►validate │
                                                  │        │
                                      insufficient ──► targeted_retry ──► review
    """
    graph = StateGraph(AgentState)

    graph.add_node("analyze",   analyze_node)
    graph.add_node("design",    design_node)
    graph.add_node("shell",     shell_node)
    graph.add_node("views",     views_node)
    graph.add_node("assemble",  assemble_node)
    graph.add_node("review",    review_node)
    graph.add_node("retry",     targeted_retry_node)
    graph.add_node("validate",  validate_node)

    graph.set_entry_point("analyze")

    graph.add_conditional_edges("analyze", route_after_analyze,
                                {"design": "design", "end": END})
    graph.add_conditional_edges("design", route_after_design,
                                {"shell": "shell", "end": END})
    graph.add_edge("shell", "views")
    graph.add_edge("views", "assemble")
    graph.add_edge("assemble", "review")

    graph.add_conditional_edges("review", route_after_review,
                                {"validate": "validate", "retry": "retry"})

    graph.add_edge("retry", "review")  # après retry → re-review
    graph.add_edge("validate", END)

    return graph.compile()


_compiled_graph = None

def get_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph


# ════════════════════════════════════════════════════════════════════
#  API 1 — Pipeline complet
# ════════════════════════════════════════════════════════════════════

def run_pipeline(pdf_path: str, max_iterations: int = 2) -> dict:
    graph = get_graph()
    state = initial_state(pdf_path)
    state["max_iterations"] = max_iterations
    return graph.invoke(state)


# ════════════════════════════════════════════════════════════════════
#  API 2 — Étapes individuelles (pour Streamlit)
# ════════════════════════════════════════════════════════════════════

def analyze_pdf(pdf_path: str) -> dict:
    return analyze_node(initial_state(pdf_path))


def run_generation_with_review(
    summary: str,
    state: dict = None,
    max_iterations: int = 2,
    on_step_callback=None,
) -> dict:
    """
    Exécute design → shell → views → assemble → review → [retry] → validate
    avec callback pour feedback temps réel Streamlit.

    Args:
        summary: description structurée du CRAgent
        state: état existant (optionnel)
        max_iterations: retries max
        on_step_callback: fonction(step_name: str, state: dict) appelée après chaque étape
    """
    if state is None:
        state = initial_state()
    state = {
        **state,
        "summary":         summary,
        "max_iterations":  max_iterations,
        "iteration":       0,
        "design_config":   None,
        "shell_html":      "",
        "views_html":      {},
        "html_code":       "",
        "review_feedback": None,
        "quality_score":   0.0,
        "verdict":         "",
        "errors":          [],
    }

    def notify(step):
        if on_step_callback:
            on_step_callback(step, state)

    # Étape 1 : Design
    state = design_node(state)
    notify("design")
    if not state.get("design_config"):
        return state

    # Étape 2 : Shell
    state = shell_node(state)
    notify("shell")
    if not state.get("shell_html"):
        return state

    # Étape 3 : Views
    state = views_node(state)
    notify("views")

    # Étape 4 : Assemble
    state = assemble_node(state)
    notify("assemble")
    if not state.get("html_code"):
        return state

    # Boucle review → retry ciblé
    while True:
        # Review
        state = review_node(state)
        notify("review")

        # Décision
        next_step = route_after_review(state)
        if next_step == "validate":
            break

        # Retry ciblé
        state = targeted_retry_node(state)
        notify("retry")

    # Validation finale
    state = validate_node(state)
    notify("validate")

    return state


def draw_graph_mermaid() -> str:
    return get_graph().get_graph().draw_mermaid()