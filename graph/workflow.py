# graph/workflow.py
#
# Orchestration LangGraph du pipeline RAG multi-agents.
#
# Le graphe complet : START → analyze_node → generate_node → validate_node → END
# Avec routage conditionnel : si une étape échoue, on saute directement à END.
#
# L'interface Streamlit peut :
#  - soit invoquer le graphe entier via run_pipeline()
#  - soit appeler les étapes individuellement via analyze_pdf(), generate_html(),
#    validate_html() pour permettre l'édition intermédiaire du summary

from typing import TypedDict, List
from langgraph.graph import StateGraph, END

from core.pdf_loader import PDFLoader
from core.vector_store import VectorStore
from agents.cr_agent import CRAgent
from agents.coder_agent import CoderAgent
from agents.executor_agent import ExecutorAgent


# ════════════════════════════════════════════════════════════════════
#  Définition typée de l'AgentState partagé entre les nœuds
# ════════════════════════════════════════════════════════════════════

class AgentState(TypedDict, total=False):
    """État partagé entre tous les nœuds du graphe."""
    pdf_path:       str
    summary:        str
    html_code:      str
    final_result:   str        # "OK" | "ERROR" | ""
    render_height:  int
    status:         str        # "error" éventuellement posé par un nœud
    errors:         List[str]


def initial_state(pdf_path: str = "") -> AgentState:
    """Retourne un AgentState vierge."""
    return {
        "pdf_path":      pdf_path,
        "summary":       "",
        "html_code":     "",
        "final_result":  "",
        "render_height": 900,
        "errors":        [],
    }


# ════════════════════════════════════════════════════════════════════
#  Nœuds du graphe
#  Chaque nœud prend un AgentState et retourne un AgentState modifié.
# ════════════════════════════════════════════════════════════════════

def analyze_node(state: AgentState) -> AgentState:
    """
    Nœud 1 : PDFLoader → VectorStore → CRAgent
    Remplit state["summary"] à partir de state["pdf_path"].
    """
    pdf_path = state.get("pdf_path", "")
    if not pdf_path:
        return {
            **state,
            "final_result": "ERROR",
            "errors": state.get("errors", []) + ["pdf_path manquant dans l'état"],
        }

    loader = PDFLoader()
    chunks = loader.load(pdf_path)

    vector_store = VectorStore()
    if not vector_store.load():
        vector_store.create(chunks)

    cr_agent = CRAgent(vector_store=vector_store)
    state = cr_agent.run(state)

    if state.get("status") == "error" or not state.get("summary"):
        state["final_result"] = "ERROR"

    return state


def generate_node(state: AgentState) -> AgentState:
    """
    Nœud 2 : CoderAgent
    Remplit state["html_code"] à partir de state["summary"].
    """
    coder_agent = CoderAgent()
    state = coder_agent.run(state)

    if not state.get("html_code"):
        state["final_result"] = "ERROR"

    return state


def validate_node(state: AgentState) -> AgentState:
    """
    Nœud 3 : ExecutorAgent
    Valide state["html_code"], injecte le resize script, calcule render_height.
    """
    executor_agent = ExecutorAgent()
    state = executor_agent.run(state)
    return state


# ════════════════════════════════════════════════════════════════════
#  Routage conditionnel — saute à END si une étape a échoué
# ════════════════════════════════════════════════════════════════════

def route_after_analyze(state: AgentState) -> str:
    """Après analyze : continue vers generate, ou END si erreur."""
    if state.get("final_result") == "ERROR":
        return "end"
    return "generate"


def route_after_generate(state: AgentState) -> str:
    """Après generate : continue vers validate, ou END si erreur."""
    if state.get("final_result") == "ERROR":
        return "end"
    return "validate"


# ════════════════════════════════════════════════════════════════════
#  Construction du StateGraph LangGraph
# ════════════════════════════════════════════════════════════════════

def build_graph():
    """
    Construit et compile le StateGraph complet.

    Topologie :
        START → analyze → [ok?] → generate → [ok?] → validate → END
                       ↓ error              ↓ error
                      END                  END
    """
    graph = StateGraph(AgentState)

    # Ajout des nœuds
    graph.add_node("analyze",  analyze_node)
    graph.add_node("generate", generate_node)
    graph.add_node("validate", validate_node)

    # Point d'entrée
    graph.set_entry_point("analyze")

    # Routage conditionnel après analyze et generate
    graph.add_conditional_edges(
        "analyze",
        route_after_analyze,
        {"generate": "generate", "end": END},
    )
    graph.add_conditional_edges(
        "generate",
        route_after_generate,
        {"validate": "validate", "end": END},
    )

    # validate est toujours terminal
    graph.add_edge("validate", END)

    return graph.compile()


# Instance compilée réutilisable (évite de reconstruire à chaque appel)
_compiled_graph = None


def get_graph():
    """Retourne le graphe compilé (lazy init)."""
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph


# ════════════════════════════════════════════════════════════════════
#  API 1 — Pipeline complet via LangGraph (pour tests CLI)
# ════════════════════════════════════════════════════════════════════

def run_pipeline(pdf_path: str) -> dict:
    """
    Exécute le pipeline complet via le StateGraph LangGraph.
    Utilisé pour les tests en ligne de commande.
    """
    graph = get_graph()
    final_state = graph.invoke(initial_state(pdf_path))
    return final_state


# ════════════════════════════════════════════════════════════════════
#  API 2 — Étapes individuelles (pour Streamlit)
#
#  Ces fonctions réutilisent les MÊMES fonctions de nœud que le graphe.
#  Aucune duplication de logique — elles sont juste des façades qui
#  permettent à l'interface d'appeler chaque étape séparément.
# ════════════════════════════════════════════════════════════════════

def analyze_pdf(pdf_path: str) -> dict:
    """Étape 1 isolée — analyse du CDC."""
    return analyze_node(initial_state(pdf_path))


def generate_html(summary: str, state: dict = None) -> dict:
    """Étape 2 isolée — génération HTML."""
    if state is None:
        state = initial_state()
    state = {**state, "summary": summary, "html_code": "", "errors": []}
    return generate_node(state)


def validate_html(html_code: str, state: dict = None) -> dict:
    """Étape 3 isolée — validation et préparation."""
    if state is None:
        state = initial_state()
    state = {**state, "html_code": html_code}
    return validate_node(state)


# ════════════════════════════════════════════════════════════════════
#  Utilitaire — visualisation du graphe (pour rapport / debug)
# ════════════════════════════════════════════════════════════════════

def draw_graph_mermaid() -> str:
    """
    Retourne une représentation Mermaid du graphe,
    utilisable dans un markdown ou st.markdown().
    """
    return get_graph().get_graph().draw_mermaid()