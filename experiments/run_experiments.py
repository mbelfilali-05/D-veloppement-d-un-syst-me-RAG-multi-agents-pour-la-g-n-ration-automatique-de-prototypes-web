# experiments/run_experiments.py
#
# Usage :
#   python -m experiments.run_experiments --pdf path/to/doc.pdf
#   python -m experiments.run_experiments --pdf path/to/doc.pdf --only baseline_single_k8 multi_thematique_k3
#   python -m experiments.run_experiments --pdf path/to/doc.pdf --skip single_k4
 
import argparse
import json
import time
from datetime import datetime
from pathlib import Path
 
from core.vector_store import VectorStore
from core.pdf_loader import PDFLoader   # ← ajuste si ton fichier s'appelle autrement
from utils.token_tracker import TokenTracker
from agents.cr_agent import CRAgent
from experiments.configs import EXPERIMENTS
from experiments.models import ChunkInfo, ExperimentResult, RAGConfig
 
 
RESULTS_DIR = Path(__file__).parent / "results"
INDIVIDUAL_DIR = RESULTS_DIR / "individual"
 
 
# ─────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────
 
def _ensure_dirs():
    RESULTS_DIR.mkdir(exist_ok=True)
    INDIVIDUAL_DIR.mkdir(exist_ok=True)
 
 
def _retrieve_chunks(vector_store: VectorStore, config: RAGConfig) -> list:
    """
    Stratégie single  → 1 requête, k chunks.
    Stratégie multi   → N requêtes × k chunks chacune,
                        dédupliqués par (page, chunk_index).
    Retourne une liste de Documents triés chronologiquement.
    """
    retriever = vector_store.get_retriever(k=config.k)
 
    if config.strategy == "single":
        docs = retriever.invoke(config.queries[0])
 
    elif config.strategy == "multi":
        seen = {}  # clé = (page, chunk_index) → doc
        for query in config.queries:
            for doc in retriever.invoke(query):
                key = (
                    doc.metadata.get("page", 0),
                    doc.metadata.get("chunk_index", 0),
                )
                if key not in seen:
                    seen[key] = doc
        docs = list(seen.values())
 
    else:
        raise ValueError(f"Stratégie inconnue : '{config.strategy}' — utilise 'single' ou 'multi'")
 
    # Tri chronologique : page d'abord, puis chunk_index
    docs_sorted = sorted(
        docs,
        key=lambda d: (
            d.metadata.get("page", 0),
            d.metadata.get("chunk_index", 0),
        ),
    )
    return docs_sorted
 
 
def _build_context(docs: list) -> str:
    """Concatène les chunks en un seul bloc de contexte."""
    return "\n\n---\n\n".join(
        f"[Page {doc.metadata.get('page', '?')}]\n{doc.page_content}"
        for doc in docs
    )
 
 
def _docs_to_chunk_infos(docs: list) -> list[ChunkInfo]:
    return [
        ChunkInfo(
            page=doc.metadata.get("page", 0),
            chunk_index=doc.metadata.get("chunk_index", 0),
            preview=doc.page_content[:120].replace("\n", " "),
        )
        for doc in docs
    ]
 
 
# ─────────────────────────────────────────────
#  CŒUR : exécution d'une config
# ─────────────────────────────────────────────
 
def run_single_experiment(
    config: RAGConfig,
    vector_store: VectorStore,
) -> ExperimentResult:
    """
    Exécute une RAGConfig et retourne un ExperimentResult.
    Gère le retrieval (single + multi-query avec dédup),
    puis délègue la génération à CRAgent.chain.invoke().
    """
    print(f"\n{'─'*55}")
    print(f"  ▶  {config.name}")
    print(f"     {config.description}")
    print(f"     stratégie={config.strategy}  k={config.k}  "
          f"requêtes={len(config.queries)}")
    print(f"{'─'*55}")
 
    start = time.time()
 
    try:
        # 1. Retrieval
        docs = _retrieve_chunks(vector_store, config)
        if not docs:
            raise ValueError("Aucun chunk récupéré depuis ChromaDB.")
 
        context = _build_context(docs)
        chunk_infos = _docs_to_chunk_infos(docs)
 
        print(f"  ✔ {len(docs)} chunks récupérés "
              f"(pages : {sorted(set(c.page for c in chunk_infos))})")
 
        # 2. Instancie CRAgent avec la config courante.
        #    Le retrieval est déjà fait — on réutilise sa chain construite
        #    dans _build_chain() avec le bon prompt_template.
        agent = CRAgent(
            vector_store=vector_store,
            retrieval_k=config.k,
            retrieval_query=config.queries[0],
            prompt_template=config.prompt_template,
        )
 
        # 3. Invocation de la chain + tracking tokens
        with TokenTracker(config.name) as tracker:
            summary = agent.chain.invoke({"context": context})
        tracker.report()
 
        duration = time.time() - start
        print(f"  ✔ Summary généré — {len(summary.split())} mots  "
              f"({duration:.1f}s)")
 
        return ExperimentResult(
            config_name=config.name,
            config=config,
            summary=summary,
            chunks=chunk_infos,
            tokens_prompt=tracker.prompt_tokens,
            tokens_completion=tracker.completion_tokens,
            tokens_total=tracker.total_tokens,
            duration_seconds=duration,
        )
 
    except Exception as e:
        duration = time.time() - start
        error_msg = str(e)
        print(f"  ✘ Erreur : {error_msg}")
        return ExperimentResult(
            config_name=config.name,
            config=config,
            summary="",
            chunks=[],
            tokens_prompt=0,
            tokens_completion=0,
            tokens_total=0,
            duration_seconds=duration,
            error=error_msg,
        )
 
 
# ─────────────────────────────────────────────
#  SAUVEGARDE
# ─────────────────────────────────────────────
 
def _save_individual(result: ExperimentResult):
    """Sauvegarde un résultat dans results/individual/<name>.json"""
    path = INDIVIDUAL_DIR / f"{result.config_name}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result.to_dict(), f, ensure_ascii=False, indent=2)
    print(f"  💾 Sauvegardé → {path.relative_to(Path(__file__).parent.parent)}")
 
 
def _save_all(results: list[ExperimentResult]):
    """Sauvegarde tous les résultats dans results/all_results.json"""
    path = RESULTS_DIR / "all_results.json"
    payload = {
        "run_timestamp": datetime.now().isoformat(),
        "total_experiments": len(results),
        "results": [r.to_dict() for r in results],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"\n📦 all_results.json sauvegardé ({len(results)} expériences)")
 
 
# ─────────────────────────────────────────────
#  POINT D'ENTRÉE
# ─────────────────────────────────────────────
 
def main():
    parser = argparse.ArgumentParser(
        description="Lance les expériences RAG sur un PDF."
    )
    parser.add_argument(
        "--pdf", required=True,
        help="Chemin vers le PDF du cahier des charges"
    )
    parser.add_argument(
        "--only", nargs="+", default=None,
        metavar="NOM",
        help="N'exécute que ces configs (par nom)"
    )
    parser.add_argument(
        "--skip", nargs="+", default=None,
        metavar="NOM",
        help="Ignore ces configs (par nom)"
    )
    args = parser.parse_args()
 
    # Filtre les configs selon --only / --skip
    configs = EXPERIMENTS
    if args.only:
        configs = [c for c in configs if c.name in args.only]
        unknown = set(args.only) - {c.name for c in configs}
        if unknown:
            print(f"⚠️  Configs introuvables : {unknown}")
    if args.skip:
        configs = [c for c in configs if c.name not in args.skip]
 
    if not configs:
        print("❌ Aucune configuration à exécuter.")
        return
 
    print(f"\n🚀 {len(configs)} expérience(s) à lancer sur : {args.pdf}")
 
    # Initialise le VectorStore une seule fois pour toutes les configs
    vector_store = VectorStore()
 
    # Essaie de charger une base existante en premier
    # (évite de re-vectoriser si le PDF n'a pas changé)
    print("\n⏳ Chargement du VectorStore...")
    already_exists = vector_store.load()
 
    if not already_exists:
        # Aucune base existante → charge et vectorise le PDF
        print(f"   Aucune base trouvée — vectorisation de {args.pdf}...")
        loader = PDFLoader()            # chunk_size et chunk_overlap par défaut
        chunks = loader.load(args.pdf)  # retourne une List[Document]
        vector_store.create(chunks)
 
    print("✅ VectorStore prêt\n")
 
    _ensure_dirs()
 
    results = []
    for config in configs:
        result = run_single_experiment(config, vector_store)
        results.append(result)
        _save_individual(result)
 
    _save_all(results)
 
    # Résumé terminal
    print("\n" + "═" * 55)
    print(f"  RÉSUMÉ — {len(results)} expériences")
    print("═" * 55)
    ok  = [r for r in results if not r.error]
    err = [r for r in results if r.error]
    print(f"  ✔ Succès : {len(ok)}   ✘ Erreurs : {len(err)}")
    print(f"\n  {'NOM':<30} {'MOTS':>6} {'TOKENS':>8} {'DURÉE':>7}")
    print(f"  {'─'*30} {'─'*6} {'─'*8} {'─'*7}")
    for r in results:
        status = "✘" if r.error else " "
        print(f"  {status} {r.config_name:<28} "
              f"{r.summary_word_count:>6} "
              f"{r.tokens_total:>8} "
              f"{r.duration_seconds:>6.1f}s")
    print()
 
 
if __name__ == "__main__":
    main()