# experiments/evaluate.py
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


RESULTS_DIR = Path(__file__).parent / "results"
EVAL_OUTPUT = RESULTS_DIR / "evaluations.json"


# ─────────────────────────────────────────────
#  CRITÈRES — alignés avec le prompt ci-dessous
# ─────────────────────────────────────────────

CRITERIA = ["couverture", "structure", "precision_ui", "fidelite", "exploitabilite"]
CRITERIA_LABELS = {
    "couverture":     "Couverture",
    "structure":      "Structure",
    "precision_ui":   "Précision UI",
    "fidelite":       "Fidélité",
    "exploitabilite": "Exploitabilité",
}
WEIGHTS = {
    "couverture":     0.25,
    "structure":      0.15,
    "precision_ui":   0.30,
    "fidelite":       0.20,
    "exploitabilite": 0.10,
}


# ─────────────────────────────────────────────
#  PROMPT D'ÉVALUATION
# ─────────────────────────────────────────────

EVAL_PROMPT = """
Tu es un expert senior en conception d'interfaces web et en évaluation de systèmes RAG.
Tu évalues la qualité d'un résumé produit automatiquement à partir d'un cahier des charges (CDC).

Tu disposes de deux éléments :
1. Le CDC original (source de vérité)
2. Le résumé généré par le système RAG (ce que tu évalues)

---
CDC ORIGINAL :
{context}
---

---
RÉSUMÉ GÉNÉRÉ PAR LE SYSTÈME RAG :
{summary}
---

ÉVALUE selon les 5 critères ci-dessous.
Pour chaque critère : un score entier de 1 à 5, et une justification factuelle en 1 phrase
(cite un exemple concret du résumé — page manquante, composant inventé, etc.)

RUBRIQUES DE SCORING :

1. couverture — Combien de pages/vues du CDC sont identifiées dans le résumé ?
   1 = moins de 40% des pages mentionnées
   2 = 40-60% des pages mentionnées
   3 = 60-80% des pages mentionnées
   4 = 80-95% des pages mentionnées
   5 = 100% des pages mentionnées, rien de manquant

2. structure — Le résumé est-il organisé de façon exploitable par un générateur de code ?
   1 = texte libre, aucune structure
   2 = structure partielle, incohérente
   3 = structure présente mais inégale selon les pages
   4 = format cohérent sur toutes les pages, quelques lacunes mineures
   5 = format parfaitement cohérent, immédiatement parsable

3. precision_ui — Les composants UI sont-ils décrits avec assez de détail pour générer du code ?
   1 = composants absents ou trop vagues ("interface utilisateur")
   2 = composants nommés mais sans détail ("un formulaire")
   3 = composants décrits avec type mais sans données ("formulaire de connexion")
   4 = composants décrits avec type et données ("formulaire : email + mot de passe")
   5 = composants décrits avec type, données, interactions et états

4. fidelite — Le résumé se limite-t-il à ce qui est dans le CDC, sans invention ?
   1 = nombreuses pages ou fonctionnalités inventées absentes du CDC
   2 = quelques inventions significatives
   3 = inventions mineures ou hypothèses non signalées
   4 = très fidèle, au plus 1-2 ajouts mineurs signalés [À PRÉCISER]
   5 = parfaitement fidèle, zéro invention, incertitudes explicitement signalées

5. exploitabilite — Ce résumé peut-il être utilisé directement pour générer un prototype HTML ?
   1 = inutilisable tel quel
   2 = nécessite une réécriture majeure
   3 = utilisable avec des corrections importantes
   4 = utilisable avec des ajustements mineurs
   5 = utilisable directement sans modification

PONDÉRATION pour le score global :
- couverture      : 25%
- structure       : 15%
- precision_ui    : 30%
- fidelite        : 20%
- exploitabilite  : 10%

Réponds UNIQUEMENT avec un objet JSON valide.
Aucun markdown, aucun texte avant ou après, aucune balise ```.

{{
  "couverture":       {{"score": <1-5>, "justification": "<exemple factuel>"}},
  "structure":        {{"score": <1-5>, "justification": "<exemple factuel>"}},
  "precision_ui":     {{"score": <1-5>, "justification": "<exemple factuel>"}},
  "fidelite":         {{"score": <1-5>, "justification": "<exemple factuel>"}},
  "exploitabilite":   {{"score": <1-5>, "justification": "<exemple factuel>"}},
  "score_global":     <score pondéré arrondi à 1 décimale>,
  "pages_manquantes": ["<page du CDC absente du résumé>"],
  "inventions":       ["<élément du résumé absent du CDC>"],
  "commentaire":      "<2 phrases : point fort principal + amélioration prioritaire>"
}}
"""


# ─────────────────────────────────────────────
#  CŒUR : évaluation d'un résultat
# ─────────────────────────────────────────────

def evaluate_result(result: ExperimentResult, chain) -> dict:
    """
    Soumet le context + summary au LLM juge.
    Retourne un dict avec les scores et justifications.
    """
    if result.error or not result.summary:
        return {
            "config_name": result.config_name,
            "skipped": True,
            "reason": result.error or "summary vide",
        }

    if not result.context:
        return {
            "config_name": result.config_name,
            "skipped": True,
            "reason": (
                "context RAG manquant — relance run_experiments.py "
                "pour regénérer les résultats avec le context stocké"
            ),
        }

    print(f"  ⚖️  Évaluation de '{result.config_name}'...")
    raw = ""

    try:
        raw = chain.invoke({
            "context": result.context,   # CDC original (chunks RAG)
            "summary": result.summary,   # résumé généré à évaluer
        })

        # Nettoyage défensif : supprime d'éventuels blocs ```json
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("```")[1]
            if cleaned.startswith("json"):
                cleaned = cleaned[4:]
        cleaned = cleaned.strip()

        scores = json.loads(cleaned)

        # Recalcule le score global avec la pondération définie ici
        # (ne fait pas confiance au calcul du LLM)
        weighted = sum(
            scores.get(c, {}).get("score", 0) * w
            for c, w in WEIGHTS.items()
        )
        scores["score_global"] = round(weighted, 1)
        scores["config_name"]  = result.config_name
        scores["skipped"]      = False

        print(f"     score global : {scores['score_global']}/5")
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
#  CHARGEMENT
# ─────────────────────────────────────────────

def load_results(path: Path) -> list[ExperimentResult]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    results = [ExperimentResult.from_dict(r) for r in data["results"]]
    print(f"✅ {len(results)} résultats chargés depuis {path.name}")
    return results


# ─────────────────────────────────────────────
#  RAPPORT TERMINAL
# ─────────────────────────────────────────────

def print_summary(evaluations: list[dict]):
    scored  = [e for e in evaluations if not e.get("skipped")]
    skipped = [e for e in evaluations if e.get("skipped")]

    if skipped:
        print(f"\n⚠️  {len(skipped)} résultat(s) ignoré(s) :")
        for e in skipped:
            print(f"   - {e['config_name']} : {e.get('reason', '?')}")

    if not scored:
        print("Aucun résultat évalué.")
        return

    col_w = 15
    header = f"  {'CONFIG':<28}" + "".join(
        f"{CRITERIA_LABELS[c]:>{col_w}}" for c in CRITERIA
    ) + f"  {'GLOBAL':>8}"

    print("\n" + "═" * len(header))
    print(header)
    print("─" * len(header))

    for e in sorted(scored, key=lambda e: e.get("score_global", 0), reverse=True):
        row = f"  {e['config_name']:<28}"
        for c in CRITERIA:
            score = e.get(c, {}).get("score", "?")
            row += f"{str(score):>{col_w}}"
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
        help="Chemin vers all_results.json"
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

    llm    = get_llm(temperature=0.0)
    prompt = ChatPromptTemplate.from_template(EVAL_PROMPT)
    chain  = prompt | llm | StrOutputParser()

    print(f"\n⚖️  Évaluation de {len(results)} résultat(s)...\n")
    evaluations = [evaluate_result(r, chain) for r in results]

    RESULTS_DIR.mkdir(exist_ok=True)
    with open(EVAL_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(evaluations, f, ensure_ascii=False, indent=2)
    print(f"\n💾 Scores sauvegardés → {EVAL_OUTPUT.relative_to(Path(__file__).parent.parent)}")

    print_summary(evaluations)
    


if __name__ == "__main__":
    main()