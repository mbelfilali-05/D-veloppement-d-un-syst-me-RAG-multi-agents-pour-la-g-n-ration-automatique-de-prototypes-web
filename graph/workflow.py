# graph/workflow.py
#
# Pipeline séquentiel : PDFLoader → VectorStore → CRAgent → CoderAgent → ExecutorAgent
# Chaque étape vérifie le statut avant de continuer.

from core.pdf_loader import PDFLoader
from core.vector_store import VectorStore
from agents.cr_agent import CRAgent
from agents.coder_agent import CoderAgent
from agents.executor_agent import ExecutorAgent


def run_pipeline(pdf_path: str) -> dict:
    """
    Exécute le pipeline complet depuis un PDF jusqu'au prototype HTML.

    Args:
        pdf_path: Chemin vers le PDF du cahier des charges

    Returns:
        AgentState final avec 'html_code', 'final_result', 'render_height', 'errors'
    """
    state: dict = {
        "pdf_path": pdf_path,
        "summary": "",
        "html_code": "",
        "final_result": "",
        "render_height": 800,
        "errors": [],
    }

    # ── 1. Chargement et vectorisation du PDF ─────────────────────────────
    print("\n[1/4] Chargement du PDF et vectorisation...")
    loader = PDFLoader()
    chunks = loader.load(pdf_path)

    vector_store = VectorStore()
    already_exists = vector_store.load()
    if not already_exists:
        vector_store.create(chunks)

    # ── 2. CRAgent — analyse du cahier des charges ────────────────────────
    print("\n[2/4] Analyse du cahier des charges (CRAgent)...")
    cr_agent = CRAgent(vector_store=vector_store)
    state = cr_agent.run(state)

    if state.get("status") == "error" or not state.get("summary"):
        state["final_result"] = "ERROR"
        print(f"\n❌ Pipeline interrompu après CRAgent : {state['errors']}")
        return state

    # ── 3. CoderAgent — génération du prototype HTML ──────────────────────
    print("\n[3/4] Génération du prototype HTML (CoderAgent)...")
    coder_agent = CoderAgent()
    state = coder_agent.run(state)

    if not state.get("html_code"):
        state["final_result"] = "ERROR"
        print(f"\n❌ Pipeline interrompu après CoderAgent : {state['errors']}")
        return state

    # ── 4. ExecutorAgent — validation et préparation du rendu ─────────────
    print("\n[4/4] Validation du HTML (ExecutorAgent)...")
    executor_agent = ExecutorAgent()
    state = executor_agent.run(state)

    return state
