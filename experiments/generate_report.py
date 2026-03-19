# experiments/generate_report.py
#
# Usage :
#   python -m experiments.generate_report
#   python -m experiments.generate_report --output results/mon_rapport.html
 
import argparse
import json
from pathlib import Path
from datetime import datetime
 
from experiments.models import ExperimentResult
 
 
RESULTS_DIR  = Path(__file__).parent / "results"
ALL_RESULTS  = RESULTS_DIR / "all_results.json"
EVALUATIONS  = RESULTS_DIR / "evaluations.json"
DEFAULT_OUT  = RESULTS_DIR / "report.html"
 
CRITERIA = ["couverture", "structure", "precision", "completude_ui", "absence_invention"]
CRITERIA_LABELS = {
    "couverture":        "Couverture",
    "structure":         "Structure",
    "precision":         "Précision",
    "completude_ui":     "Complétude UI",
    "absence_invention": "Sans invention",
}
 
 
# ─────────────────────────────────────────────
#  CHARGEMENT
# ─────────────────────────────────────────────
 
def load_results() -> list[ExperimentResult]:
    if not ALL_RESULTS.exists():
        raise FileNotFoundError(
            f"{ALL_RESULTS} introuvable.\n"
            "Lance d'abord : python -m experiments.run_experiments --pdf <pdf>"
        )
    with open(ALL_RESULTS, encoding="utf-8") as f:
        data = json.load(f)
    return [ExperimentResult.from_dict(r) for r in data["results"]]
 
 
def load_evaluations() -> dict:
    """Retourne un dict config_name → scores (peut être vide si evaluate pas encore lancé)."""
    if not EVALUATIONS.exists():
        return {}
    with open(EVALUATIONS, encoding="utf-8") as f:
        evals = json.load(f)
    return {e["config_name"]: e for e in evals if not e.get("skipped")}
 
 
# ─────────────────────────────────────────────
#  CONSTRUCTION HTML
# ─────────────────────────────────────────────
 
def _score_color(score) -> str:
    """Couleur de fond selon le score (1-5)."""
    if score is None:
        return "#f1f0eb"
    if score >= 4.5: return "#c8f7c5"
    if score >= 3.5: return "#eaf7c5"
    if score >= 2.5: return "#fff3c4"
    if score >= 1.5: return "#ffddb3"
    return "#ffc5c5"
 
 
def _badge(score) -> str:
    if score is None:
        return '<span style="color:#999">—</span>'
    color = _score_color(score)
    return (f'<span style="background:{color};padding:2px 8px;'
            f'border-radius:12px;font-weight:600">{score}</span>')
 
 
def _render_summary_card(result: ExperimentResult, eval_data: dict | None) -> str:
    """Génère la carte HTML d'un résultat (config + summary + scores)."""
 
    # En-tête de la carte
    error_banner = ""
    if result.error:
        error_banner = (
            f'<div style="background:#ffc5c5;padding:8px 12px;'
            f'border-radius:6px;margin-bottom:12px">'
            f'❌ Erreur : {result.error}</div>'
        )
 
    # Métadonnées
    meta_items = [
        ("Stratégie",  result.config.strategy),
        ("k",          str(result.config.k)),
        ("Requêtes",   str(len(result.config.queries))),
        ("Chunks",     str(result.chunks_retrieved)),
        ("Pages",      str(result.unique_pages)),
        ("Mots",       str(result.summary_word_count)),
        ("Tokens",     str(result.tokens_total)),
        ("Durée",      f"{result.duration_seconds:.1f}s"),
    ]
    meta_html = "".join(
        f'<span style="margin-right:16px;font-size:13px">'
        f'<span style="color:#888">{k}</span> '
        f'<strong>{v}</strong></span>'
        for k, v in meta_items
    )
 
    # Requêtes utilisées
    queries_html = "".join(
        f'<li style="font-size:13px;color:#555;margin-bottom:4px">{q}</li>'
        for q in result.config.queries
    )
 
    # Scores LLM (si disponibles)
    scores_html = ""
    if eval_data:
        global_score = eval_data.get("score_global")
        commentaire  = eval_data.get("commentaire", "")
        scores_html = f"""
        <div style="margin:16px 0;padding:12px;background:#f8f8f6;border-radius:8px">
            <div style="margin-bottom:8px;font-weight:600">
                Score global : {_badge(global_score)}
            </div>
            <div style="display:flex;flex-wrap:wrap;gap:8px;margin-bottom:10px">
        """
        for c in CRITERIA:
            sc = eval_data.get(c, {}).get("score")
            just = eval_data.get(c, {}).get("justification", "")
            scores_html += (
                f'<span title="{just}" style="background:{_score_color(sc)};'
                f'padding:3px 10px;border-radius:12px;font-size:13px;cursor:help">'
                f'{CRITERIA_LABELS[c]} {_badge(sc)}</span>'
            )
        scores_html += f"""
            </div>
            <p style="font-size:13px;color:#555;margin:0;font-style:italic">
                {commentaire}
            </p>
        </div>
        """
    else:
        scores_html = (
            '<p style="font-size:13px;color:#aaa;font-style:italic">'
            'Scores non disponibles — lance evaluate.py</p>'
        )
 
    # Summary (pré-formaté)
    summary_body = result.summary.replace("<", "&lt;").replace(">", "&gt;")
    summary_html = (
        f'<pre style="white-space:pre-wrap;font-family:inherit;'
        f'font-size:13px;line-height:1.7;background:#fafaf8;'
        f'padding:16px;border-radius:8px;max-height:500px;overflow-y:auto;'
        f'border:1px solid #eee">{summary_body}</pre>'
        if summary_body else
        '<p style="color:#aaa;font-style:italic">Aucun summary généré.</p>'
    )
 
    # Prompt utilisé (collapsible)
    prompt_escaped = result.config.prompt_template.replace("<","&lt;").replace(">","&gt;")
    prompt_html = (
        f'<details style="margin-top:12px">'
        f'<summary style="cursor:pointer;font-size:13px;color:#888">Voir le prompt template</summary>'
        f'<pre style="white-space:pre-wrap;font-size:12px;color:#666;'
        f'background:#f4f4f2;padding:12px;border-radius:6px;margin-top:8px">'
        f'{prompt_escaped}</pre></details>'
    )
 
    return f"""
    <div style="background:#fff;border:1px solid #e8e6e0;border-radius:12px;
                padding:24px;margin-bottom:32px">
 
        <div style="display:flex;align-items:center;justify-content:space-between;
                    margin-bottom:8px">
            <h2 style="margin:0;font-size:18px">{result.config_name}</h2>
        </div>
 
        <p style="color:#777;font-size:14px;margin:0 0 12px">
            {result.config.description}
        </p>
 
        {error_banner}
 
        <div style="margin-bottom:12px">{meta_html}</div>
 
        <details style="margin-bottom:12px">
            <summary style="cursor:pointer;font-size:13px;color:#888">
                Voir les requêtes ({len(result.config.queries)})
            </summary>
            <ul style="margin-top:8px;padding-left:20px">{queries_html}</ul>
        </details>
 
        {scores_html}
 
        <h3 style="font-size:15px;margin:16px 0 8px">Summary généré</h3>
        {summary_html}
        {prompt_html}
    </div>
    """
 
 
def _render_comparison_table(
    results: list[ExperimentResult],
    evaluations: dict
) -> str:
    """Tableau de comparaison rapide en haut du rapport."""
 
    header_cells = "".join(
        f'<th style="padding:10px 14px;text-align:left;'
        f'border-bottom:2px solid #e0ddd5">{h}</th>'
        for h in ["Config", "Stratégie", "k", "Chunks", "Pages", "Mots",
                  "Tokens", "Durée", "Score global"]
    )
 
    rows = ""
    for r in results:
        ev = evaluations.get(r.config_name)
        global_score = ev.get("score_global") if ev else None
        bg = "#fff8f0" if r.error else "#fff"
        rows += f"""
        <tr style="background:{bg};border-bottom:1px solid #f0ede5">
            <td style="padding:8px 14px;font-weight:600">
                <a href="#{r.config_name}" style="text-decoration:none;color:#333">
                    {r.config_name}
                </a>
            </td>
            <td style="padding:8px 14px">{r.config.strategy}</td>
            <td style="padding:8px 14px;text-align:center">{r.config.k}</td>
            <td style="padding:8px 14px;text-align:center">{r.chunks_retrieved}</td>
            <td style="padding:8px 14px;font-size:12px">{r.unique_pages}</td>
            <td style="padding:8px 14px;text-align:center">{r.summary_word_count}</td>
            <td style="padding:8px 14px;text-align:center">{r.tokens_total}</td>
            <td style="padding:8px 14px;text-align:center">{r.duration_seconds:.1f}s</td>
            <td style="padding:8px 14px;text-align:center">{_badge(global_score)}</td>
        </tr>
        """
 
    return f"""
    <table style="width:100%;border-collapse:collapse;background:#fff;
                  border:1px solid #e8e6e0;border-radius:12px;overflow:hidden;
                  margin-bottom:48px">
        <thead><tr style="background:#f8f8f6">{header_cells}</tr></thead>
        <tbody>{rows}</tbody>
    </table>
    """
 
 
def build_html(
    results: list[ExperimentResult],
    evaluations: dict,
    timestamp: str,
) -> str:
    """Assemble le rapport HTML complet."""
 
    table_html = _render_comparison_table(results, evaluations)
 
    cards_html = ""
    for result in results:
        ev = evaluations.get(result.config_name)
        cards_html += f'<div id="{result.config_name}">'
        cards_html += _render_summary_card(result, ev)
        cards_html += "</div>"
 
    has_evals = bool(evaluations)
    eval_notice = "" if has_evals else """
    <div style="background:#fff3c4;padding:12px 20px;border-radius:8px;
                margin-bottom:24px;font-size:14px">
        ⚠️ Scores non disponibles.
        Lance <code>python -m experiments.evaluate</code> puis régénère ce rapport.
    </div>
    """
 
    return f"""<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RAG Experiment Report</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            background: #f5f3ee;
            color: #2c2c2a;
            margin: 0;
            padding: 32px 16px;
            line-height: 1.6;
        }}
        .container {{ max-width: 960px; margin: 0 auto; }}
        h1 {{ font-size: 26px; font-weight: 600; margin-bottom: 4px; }}
        .subtitle {{ color: #888; font-size: 14px; margin-bottom: 32px; }}
        h2 {{ font-size: 18px; font-weight: 600; }}
        details summary::-webkit-details-marker {{ color: #aaa; }}
    </style>
</head>
<body>
<div class="container">
 
    <h1>RAG Experiment Report</h1>
    <p class="subtitle">
        Généré le {datetime.now().strftime("%d/%m/%Y à %H:%M")} —
        {len(results)} expériences —
        données du {timestamp}
    </p>
 
    {eval_notice}
 
    <h2 style="margin-bottom:16px">Tableau comparatif</h2>
    {table_html}
 
    <h2 style="margin-bottom:24px">Détail par expérience</h2>
    {cards_html}
 
</div>
</body>
</html>"""
 
 
# ─────────────────────────────────────────────
#  POINT D'ENTRÉE
# ─────────────────────────────────────────────
 
def main():
    parser = argparse.ArgumentParser(
        description="Génère un rapport HTML comparatif des expériences RAG."
    )
    parser.add_argument(
        "--output", default=str(DEFAULT_OUT),
        help=f"Chemin du fichier HTML de sortie (défaut : {DEFAULT_OUT})"
    )
    args = parser.parse_args()
 
    print("📂 Chargement des résultats...")
    results = load_results()
    print(f"   ✔ {len(results)} expériences chargées")
 
    print("📊 Chargement des scores...")
    evaluations = load_evaluations()
    if evaluations:
        print(f"   ✔ {len(evaluations)} évaluations chargées")
    else:
        print("   ⚠️  Aucun score trouvé (evaluate.py pas encore lancé)")
 
    # Timestamp du run
    with open(ALL_RESULTS, encoding="utf-8") as f:
        run_ts = json.load(f).get("run_timestamp", "inconnu")
 
    print("🔨 Génération du rapport HTML...")
    html = build_html(results, evaluations, run_ts)
 
    out_path = Path(args.output)
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
 
    print(f"\n✅ Rapport généré → {out_path}")
    print(f"   Ouvre-le dans ton navigateur : file://{out_path.resolve()}")
 
 
if __name__ == "__main__":
    main()
 