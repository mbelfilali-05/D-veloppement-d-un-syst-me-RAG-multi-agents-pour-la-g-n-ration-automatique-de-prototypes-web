# graph/workflow.py
#
# Orchestration LangGraph du pipeline RAG multi-agents — avec cycle de review.
#
# Topologie :
#
#   START
#     │
#     ▼
#   analyze_node (CRAgent)
#     │
#     ├─ [erreur] ──► END
#     ▼
#   generate_node (CoderAgent) ◄──────┐
#     │                                │
#     ▼                                │
#   review_node (ReviewerAgent)       │
#     │                                │
#     ├─ [erreur] ──► validate_node ──┤
#     ├─ [verdict good] ────► validate_node
#     ├─ [iteration >= max] ► validate_node
#     └─ [verdict insuffisant + retries disponibles] ─┘  (retour vers generate)
#     ▼
#   validate_node (ExecutorAgent)
#     │
#     ▼
#   END
#
# L'interface Streamlit peut :
#  - soit invoquer le graphe entier via run_pipeline()
#  - soit appeler les étapes individuellement via analyze_pdf(), generate_html(),
#    review_html(), validate_html() pour permettre l'édition intermédiaire du summary
#    et le feedback temps réel sur chaque itération du cycle

from typing import TypedDict, List, Optional
from langgraph.graph import StateGraph, END

from core.pdf_loader import PDFLoader
from core.vector_store import VectorStore
from agents.cr_agent import CRAgent
from agents.coder_agent import CoderAgent
from agents.reviewer_agent import ReviewerAgent
from agents.executor_agent import ExecutorAgent


# ════════════════════════════════════════════════════════════════════
#  Définition typée de l'AgentState partagé entre les nœuds
# ════════════════════════════════════════════════════════════════════

class AgentState(TypedDict, total=False):
    """État partagé entre tous les nœuds du graphe."""
    # Champs de base
    pdf_path:        str
    summary:         str
    html_code:       str
    final_result:    str        # "OK" | "ERROR" | ""
    render_height:   int
    status:          str        # "error" éventuellement posé par un nœud
    errors:          List[str]

    # Champs du cycle de review (Axe 2)
    review_feedback: Optional[dict]   # JSON structuré retourné par ReviewerAgent
    quality_score:   float             # Score global pondéré (0.0 à 5.0)
    verdict:         str               # "good" | "insufficient"
    iteration:       int               # Compteur d'itérations generate ↔ review
    max_iterations:  int               # Limite supérieure du cycle (défaut 2 retries)


def initial_state(pdf_path: str = "") -> AgentState:
    """Retourne un AgentState vierge, avec compteurs du cycle initialisés."""
    return {
        "pdf_path":        pdf_path,
        "summary":         "",
        "html_code":       "",
        "final_result":    "",
        "render_height":   900,
        "errors":          [],
        # Cycle de review
        "review_feedback": None,
        "quality_score":   0.0,
        "verdict":         "",
        "iteration":       0,
        "max_iterations":  2,   # 2 retries max (3 générations au total)
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

    Au premier passage : génération initiale.
    Au retry (iteration >= 1) : génération corrective avec le feedback
    du ReviewerAgent injecté dans le prompt.

    Le compteur 'iteration' est incrémenté à chaque passage
    (géré en interne par CoderAgent.run()).
    """
    coder_agent = CoderAgent()
    state = coder_agent.run(state)

    if not state.get("html_code"):
        state["final_result"] = "ERROR"

    return state


def review_node(state: AgentState) -> AgentState:
    """
    Nœud 3 (NOUVEAU) : ReviewerAgent
    Évalue state["html_code"] et enrichit l'état avec :
      - review_feedback (JSON structuré)
      - quality_score (0.0 à 5.0)
      - verdict ("good" | "insufficient")
    """
    reviewer_agent = ReviewerAgent()
    state = reviewer_agent.run(state)
    return state


def validate_node(state: AgentState) -> AgentState:
    """
    Nœud 4 : ExecutorAgent
    Valide state["html_code"], injecte le resize script, calcule render_height.
    Terminus du pipeline.
    """
    executor_agent = ExecutorAgent()
    state = executor_agent.run(state)
    return state


# ════════════════════════════════════════════════════════════════════
#  Routage conditionnel
# ════════════════════════════════════════════════════════════════════

def route_after_analyze(state: AgentState) -> str:
    """Après analyze : continue vers generate, ou END si erreur."""
    if state.get("final_result") == "ERROR":
        return "end"
    return "generate"


def route_after_generate(state: AgentState) -> str:
    """Après generate : continue vers review, ou END si erreur critique."""
    if state.get("final_result") == "ERROR":
        return "end"
    return "review"


def route_after_review(state: AgentState) -> str:
    """
    Après review : décide si on retry (generate) ou si on valide (validate).

    Conditions de sortie du cycle (passage à validate) :
      - verdict == "good" (qualité suffisante atteinte)
      - iteration >= max_iterations (budget de retries épuisé)
      - erreur lors du review (le feedback est inutilisable)

    Sinon : retour vers generate avec le feedback pour un retry corrigé.
    """
    verdict = state.get("verdict", "insufficient")
    iteration = state.get("iteration", 0)
    max_iter = state.get("max_iterations", 2)

    # Sortie : qualité atteinte
    if verdict == "good":
        print(f"[workflow] ✅ Verdict 'good' atteint à l'itération {iteration} "
              f"(score {state.get('quality_score', 0)}/5) — passage à validate")
        return "validate"

    # Sortie : budget de retries épuisé
    # NB : iteration est incrémenté APRÈS generate_node, donc si iteration == max_iter+1
    # ça signifie qu'on a fait max_iter+1 générations au total (1 initiale + max_iter retries)
    if iteration > max_iter:
        print(f"[workflow] ⏹️  Budget de retries épuisé ({iteration - 1} retries effectués) "
              f"— score final {state.get('quality_score', 0)}/5 — passage à validate")
        return "validate"

    # Retry : qualité insuffisante, budget disponible
    retries_left = max_iter - (iteration - 1)
    print(f"[workflow] 🔄 Verdict 'insufficient' (score {state.get('quality_score', 0)}/5) "
          f"— retry {iteration} ({retries_left} restant{'s' if retries_left > 1 else ''})")
    return "generate"


# ════════════════════════════════════════════════════════════════════
#  Construction du StateGraph LangGraph
# ════════════════════════════════════════════════════════════════════

def build_graph():
    """
    Construit et compile le StateGraph complet avec le cycle de review.

    Topologie :
        START
          │
          ▼
        analyze ─[erreur]──► END
          │
          ▼
        generate ◄──────────┐
          │                 │
          ▼                 │ retry (si verdict=insufficient et budget dispo)
        review ─────────────┤
          │                 │
          ├─[good]──────────► validate ─► END
          └─[max iter]──────►
    """
    graph = StateGraph(AgentState)

    # Ajout des nœuds
    graph.add_node("analyze",  analyze_node)
    graph.add_node("generate", generate_node)
    graph.add_node("review",   review_node)      # NOUVEAU
    graph.add_node("validate", validate_node)

    # Point d'entrée
    graph.set_entry_point("analyze")

    # Routage conditionnel après analyze
    graph.add_conditional_edges(
        "analyze",
        route_after_analyze,
        {"generate": "generate", "end": END},
    )

    # Routage conditionnel après generate
    graph.add_conditional_edges(
        "generate",
        route_after_generate,
        {"review": "review", "end": END},
    )

    # Routage conditionnel après review (c'est ici le cycle)
    graph.add_conditional_edges(
        "review",
        route_after_review,
        {
            "generate": "generate",   # retry (cycle)
            "validate": "validate",   # sortie (qualité OK ou budget épuisé)
        },
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

def run_pipeline(pdf_path: str, max_iterations: int = 2) -> dict:
    """
    Exécute le pipeline complet via le StateGraph LangGraph.
    Utilisé pour les tests en ligne de commande.

    Args:
        pdf_path: Chemin vers le PDF du cahier des charges
        max_iterations: Nombre maximum de retries du cycle (défaut 2)
    """
    graph = get_graph()
    state = initial_state(pdf_path)
    state["max_iterations"] = max_iterations
    final_state = graph.invoke(state)
    return final_state


# ════════════════════════════════════════════════════════════════════
#  API 2 — Étapes individuelles (pour Streamlit avec feedback temps réel)
#
#  Ces fonctions réutilisent les MÊMES fonctions de nœud que le graphe.
#  Aucune duplication de logique — elles sont juste des façades qui
#  permettent à l'interface d'appeler chaque étape séparément
#  pour afficher le feedback en temps réel entre chaque itération.
# ════════════════════════════════════════════════════════════════════

def analyze_pdf(pdf_path: str) -> dict:
    """Étape 1 isolée — analyse du CDC."""
    return analyze_node(initial_state(pdf_path))


def generate_html(summary: str, state: dict = None) -> dict:
    """
    Étape 2 isolée — génération HTML.
    Si state contient 'review_feedback', c'est une génération corrective.
    """
    if state is None:
        state = initial_state()
    state = {**state, "summary": summary, "html_code": "", "errors": []}
    return generate_node(state)


def review_html(state: dict) -> dict:
    """
    Étape 3 isolée — review du HTML.
    Requiert state avec 'html_code' et 'summary' non vides.
    """
    return review_node(state)


def validate_html(html_code: str, state: dict = None) -> dict:
    """Étape 4 isolée — validation et préparation."""
    if state is None:
        state = initial_state()
    state = {**state, "html_code": html_code}
    return validate_node(state)


def run_generation_with_review(
    summary: str,
    state: dict = None,
    max_iterations: int = 2,
    on_iteration_callback = None,
) -> dict:
    """
    Exécute le cycle generate → review → (retry ou sortie) sans invoquer
    tout le graphe LangGraph. Utile pour Streamlit qui veut afficher
    chaque itération en temps réel via st.status.

    Args:
        summary: Description structurée produite par CRAgent
        state: AgentState existant à enrichir (optionnel)
        max_iterations: Nombre max de retries
        on_iteration_callback: Fonction appelée après chaque itération
            Signature: callback(iteration: int, state: dict) -> None
            Utile pour afficher l'avancement en temps réel dans Streamlit

    Returns:
        État final après le cycle (incluant html_code et review_feedback)
    """
    if state is None:
        state = initial_state()
    state = {
        **state,
        "summary":        summary,
        "max_iterations": max_iterations,
        "iteration":      0,
        "html_code":      "",
        "review_feedback": None,
        "verdict":        "",
        "quality_score":  0.0,
        "errors":         [],
    }

    while True:
        # Génération (initiale ou corrective)
        state = generate_node(state)
        if state.get("final_result") == "ERROR":
            if on_iteration_callback:
                on_iteration_callback(state.get("iteration", 0), state)
            return state

        # Review
        state = review_node(state)

        # Callback pour Streamlit
        if on_iteration_callback:
            on_iteration_callback(state.get("iteration", 0), state)

        # Décision : sortir ou retry ?
        next_step = route_after_review(state)
        if next_step == "validate":
            break
        # Sinon : boucle continue (next_step == "generate")

    return state


# ════════════════════════════════════════════════════════════════════
#  Utilitaire — visualisation du graphe (pour rapport / debug)
# ════════════════════════════════════════════════════════════════════

def draw_graph_mermaid() -> str:
    """
    Retourne une représentation Mermaid du graphe,
    utilisable dans un markdown ou st.markdown().
    """
    return get_graph().get_graph().draw_mermaid()