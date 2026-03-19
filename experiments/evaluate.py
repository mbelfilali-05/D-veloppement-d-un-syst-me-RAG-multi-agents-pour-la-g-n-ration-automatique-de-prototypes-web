# experiments/evaluate.py
#
# Évalue automatiquement chaque ExperimentResult avec un LLM juge.
# Le LLM attribue un score sur 5 critères, avec une justification courte.
#
# Usage :
#   python -m experiments.evaluate
#   python -m experiments.evaluate --only baseline_single_k8 multi_thematique_k3
#   python -m experiments.evaluate --input results/all_results.json
 
import argparse
import json
from pathlib import Path
 
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
 
from core.llm_config import get_llm
from experiments.models import ExperimentResult
 
 
RESULTS_DIR  = Path(__file__).parent / "results"
EVAL_OUTPUT  = RESULTS_DIR / "evaluations.json"
 
 
# ─────────────────────────────────────────────
#  PROMPT D'ÉVALUATION
# ─────────────────────────────────────────────
# Le LLM juge reçoit le summary d'une expérience et répond en JSON pur.
 
EVAL_PROMPT = """
Tu es un évaluateur expert en qualité de documentation technique.
 
On t'a demandé d'analyser un cahier des charges (CDC) pour en extraire
une description structurée de toutes les pages/vues d'une application web.
 
Voici le résultat produit par un système RAG :
 
---
{summary}
---
 
Évalue ce résultat selon les 5 critères suivants.
Pour chaque critère, attribue un score de 1 à 5 ET une justification en 1 phrase.
 
CRITÈRES :
1. couverture        — Toutes les pages/vues importantes sont-elles mentionnées ?
2. structure         — La réponse est-elle bien organisée et lisible ?
3. precision         — Les informations sont-elles précises et cohérentes avec un CDC type ?
4. completude_ui     — Les composants UI (formulaires, tableaux, boutons, etc.) sont-ils détaillés ?
5. absence_invention — Le système évite-t-il d'inventer des informations non présentes dans un CDC ?
 
Réponds UNIQUEMENT avec un objet JSON valide, sans markdown, sans texte avant ou après.
Format exact :
{{
  "couverture":        {{"score": <1-5>, "justification": "<phrase>"}},
  "structure":         {{"score": <1-5>, "justification": "<phrase>"}},
  "precision":         {{"score": <1-5>, "justification": "<phrase>"}},
  "completude_ui":     {{"score": <1-5>, "justification": "<phrase>"}},
  "absence_invention": {{"score": <1-5>, "justification": "<phrase>"}},
  "score_global":      <moyenne arrondie à 1 décimale>,
  "commentaire":       "<synthèse générale en 2 phrases max>"
}}
"""
 
 
# ─────────────────────────────────────────────
#  CŒUR : évaluation d'un résultat
# ─────────────────────────────────────────────
 
def evaluate_result(result: ExperimentResult, chain) -> dict:
    """
    Soumet le summary d'un ExperimentResult au LLM juge.
    Retourne un dict avec les scores et justifications.
    En cas d'erreur de parsing JSON, retourne un dict d'erreur.
    """
    if result.error or not result.summary:
        return {
            "config_name": result.config_name,
            "skipped": True,
            "reason": result.error or "summary vide",
        }
 
    print(f"  ⚖️  Évaluation de '{result.config_name}'...")
 
    try:
        raw = chain.invoke({"summary": result.summary})
 
        # Nettoyage défensif : supprime d'éventuels blocs ```json
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("```")[1]
            if cleaned.startswith("json"):
                cleaned = cleaned[4:]
        cleaned = cleaned.strip()
 
        scores = json.loads(cleaned)
        scores["config_name"] = result.config_name
        scores["skipped"] = False
 
        print(f"     score global : {scores.get('score_global', '?')}/5")
        return scores
 
    except json.JSONDecodeError as e:
        print(f"  ✘ Parsing JSON échoué pour '{result.config_name}' : {e}")
        return {
            "config_name": result.config_name,
            "skipped": True,
            "reason": f"JSONDecodeError : {e}",
            "raw_response": raw,
        }
    except Exception as e:
        print(f"  ✘ Erreur pour '{result.config_name}' : {e}")
        return {
            "config_name": result.config_name,
            "skipped": True,
            "reason": str(e),
        }
 
 
# ─────────────────────────────────────────────
#  CHARGEMENT DES RÉSULTATS
# ─────────────────────────────────────────────
 
def load_results(path: Path) -> list[ExperimentResult]:
    """Charge les ExperimentResult depuis all_results.json."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
 
    results = [ExperimentResult.from_dict(r) for r in data["results"]]
    print(f"✅ {len(results)} résultats chargés depuis {path.name}")
    return results
 
 
# ─────────────────────────────────────────────
#  RAPPORT TERMINAL
# ─────────────────────────────────────────────
 
CRITERIA = ["couverture", "structure", "precision", "completude_ui", "absence_invention"]
CRITERIA_LABELS = {
    "couverture":        "Couverture",
    "structure":         "Structure",
    "precision":         "Précision",
    "completude_ui":     "Complétude UI",
    "absence_invention": "Pas d'invention",
}
 
 
def print_summary(evaluations: list[dict]):
    """Affiche un tableau comparatif dans le terminal."""
    scored = [e for e in evaluations if not e.get("skipped")]
    if not scored:
        print("Aucun résultat évalué.")
        return
 
    col_w = 16
    header = f"  {'CONFIG':<28}" + "".join(
        f"{CRITERIA_LABELS[c]:>{col_w}}" for c in CRITERIA
    ) + f"  {'GLOBAL':>8}"
    print("\n" + "═" * len(header))
    print(header)
    print("─" * len(header))
 
    # Tri par score global décroissant
    scored_sorted = sorted(scored, key=lambda e: e.get("score_global", 0), reverse=True)
 
    for e in scored_sorted:
        row = f"  {e['config_name']:<28}"
        for c in CRITERIA:
            score = e.get(c, {}).get("score", "?")
            row += f"{score:>{col_w}}"
        row += f"  {e.get('score_global', '?'):>8}"
        print(row)
 
    print("═" * len(header))
    print()
 
 
# ─────────────────────────────────────────────
#  POINT D'ENTRÉE
# ─────────────────────────────────────────────
 
def main():
    parser = argparse.ArgumentParser(
        description="Évalue automatiquement les résultats RAG avec un LLM juge."
    )
    parser.add_argument(
        "--input", default=str(RESULTS_DIR / "all_results.json"),
        help="Chemin vers all_results.json (défaut : results/all_results.json)"
    )
    parser.add_argument(
        "--only", nargs="+", default=None,
        metavar="NOM",
        help="N'évalue que ces configs (par nom)"
    )
    args = parser.parse_args()
 
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"❌ Fichier introuvable : {input_path}")
        print("   Lance d'abord : python -m experiments.run_experiments --pdf <pdf>")
        return
 
    results = load_results(input_path)
 
    if args.only:
        results = [r for r in results if r.config_name in args.only]
        print(f"   Filtre --only : {len(results)} config(s) sélectionnée(s)")
 
    if not results:
        print("❌ Aucun résultat à évaluer.")
        return
 
    # Le LLM juge tourne à température 0 pour des scores reproductibles
    llm   = get_llm(temperature=0.0)
    prompt = ChatPromptTemplate.from_template(EVAL_PROMPT)
    chain  = prompt | llm | StrOutputParser()
 
    print(f"\n⚖️  Évaluation de {len(results)} résultat(s)...\n")
    evaluations = [evaluate_result(r, chain) for r in results]
 
    # Sauvegarde
    RESULTS_DIR.mkdir(exist_ok=True)
    with open(EVAL_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(evaluations, f, ensure_ascii=False, indent=2)
    print(f"\n💾 Scores sauvegardés → {EVAL_OUTPUT.relative_to(Path(__file__).parent.parent)}")
 
    # Tableau terminal
    print_summary(evaluations)
 
 
if __name__ == "__main__":
    main()