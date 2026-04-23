# interface/app.py
#
# Interface Streamlit — orchestration du pipeline avec cycle de review.
# Axe 2 : feedback temps réel de chaque itération coder ↔ reviewer,
# expander JSON du dernier rapport de review, régénération ciblée.
#
# Lancement : streamlit run interface/app.py

import sys
import json
import tempfile
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

# Permet d'importer les modules du projet depuis le dossier parent
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from graph.workflow import (
    analyze_pdf,
    run_generation_with_review,
    validate_html,
)


# ════════════════════════════════════════════════════════════════════
#  Configuration de la page
# ════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Générateur de prototypes web",
    page_icon="🎨",
    layout="wide",
)

st.title("🎨 Générateur de prototypes web")
st.caption(
    "Upload un cahier des charges PDF → analyse RAG → génération avec cycle de review → prototype HTML navigable"
)


# ════════════════════════════════════════════════════════════════════
#  Initialisation du session_state
# ════════════════════════════════════════════════════════════════════

def init_state():
    defaults = {
        "pdf_path":        None,
        "pdf_name":        None,
        "summary":         "",
        "html_code":       "",
        "render_height":   900,
        "step":            "upload",  # upload | analyzed | generated
        "errors":          [],
        # Axe 2 — cycle de review
        "review_feedback": None,
        "quality_score":   0.0,
        "verdict":         "",
        "iteration":       0,
        "max_iterations":  2,
        "iterations_log":  [],   # liste des snapshots à chaque itération (pour affichage)
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_state()


# ════════════════════════════════════════════════════════════════════
#  Helpers
# ════════════════════════════════════════════════════════════════════

def reset_all():
    """Remet l'application dans son état initial."""
    for k in list(st.session_state.keys()):
        del st.session_state[k]
    init_state()


def save_uploaded_pdf(uploaded_file) -> str:
    """Sauvegarde le PDF uploadé dans un fichier temporaire et retourne son chemin."""
    suffix = Path(uploaded_file.name).suffix or ".pdf"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded_file.read())
    tmp.close()
    return tmp.name


def score_color(score: float) -> str:
    """Retourne une couleur hex selon le score (pour affichage visuel)."""
    if score >= 4.5:
        return "#065F46"   # vert foncé
    if score >= 4.0:
        return "#16A34A"   # vert
    if score >= 3.0:
        return "#D97706"   # orange
    if score >= 2.0:
        return "#DC2626"   # rouge
    return "#7F1D1D"       # rouge foncé


def severity_icon(severity: str) -> str:
    """Icône visuelle selon la sévérité de l'issue."""
    return {"high": "🔴", "medium": "🟠", "low": "🟡"}.get(severity, "⚪")


def run_cycle_with_streamlit_feedback(summary: str, max_iter: int = 2) -> dict:
    """
    Lance le cycle generate ↔ review avec affichage temps réel dans Streamlit.
    Utilise st.status pour afficher chaque itération au fur et à mesure.
    """
    iterations_log = []

    with st.status("Cycle de génération et review...", expanded=True) as status:

        def on_iteration(iteration: int, state: dict):
            """Callback appelé après chaque itération coder + review."""
            score = state.get("quality_score", 0)
            verdict = state.get("verdict", "insufficient")
            feedback = state.get("review_feedback") or {}

            # Snapshot pour l'historique
            iterations_log.append({
                "iteration": iteration,
                "score":     score,
                "verdict":   verdict,
                "feedback":  feedback,
            })

            # Affichage dans Streamlit
            if verdict == "good":
                st.markdown(
                    f"✅ **Itération {iteration}** — score **{score}/5** — verdict **good**"
                )
            else:
                # Compter les issues high
                issues = feedback.get("issues", []) if isinstance(feedback, dict) else []
                high_count = sum(1 for i in issues if i.get("severity") == "high")
                high_label = (
                    f" ({high_count} issue{'s' if high_count > 1 else ''} bloquante{'s' if high_count > 1 else ''})"
                    if high_count > 0 else ""
                )

                if iteration <= max_iter:
                    st.markdown(
                        f"🔄 **Itération {iteration}** — score **{score}/5**{high_label} — retry..."
                    )
                else:
                    st.markdown(
                        f"⏹️ **Itération {iteration}** — score **{score}/5**{high_label} — budget épuisé"
                    )

        # Lancement du cycle avec callback
        st.write("🎨 Démarrage du cycle de génération avec review...")
        final_state = run_generation_with_review(
            summary=summary,
            max_iterations=max_iter,
            on_iteration_callback=on_iteration,
        )

        # Étape finale : validation via ExecutorAgent
        if not final_state.get("errors") and final_state.get("html_code"):
            st.write("🔧 Validation du HTML (ExecutorAgent)...")
            final_state = validate_html(final_state["html_code"], final_state)

        # Statut final
        if final_state.get("final_result") == "ERROR":
            status.update(label="❌ Échec du cycle", state="error")
        else:
            final_score = final_state.get("quality_score", 0)
            final_verdict = final_state.get("verdict", "?")
            total_iter = final_state.get("iteration", 1)
            status.update(
                label=f"✅ Cycle terminé — {total_iter} génération{'s' if total_iter > 1 else ''} — score final {final_score}/5 ({final_verdict})",
                state="complete",
            )

    # Sauvegarde de l'historique dans le state global
    st.session_state.iterations_log = iterations_log

    return final_state


def display_review_report(feedback: dict):
    """Affiche le rapport de review de manière structurée et lisible."""
    if not feedback or not isinstance(feedback, dict):
        st.info("Aucun rapport de review disponible.")
        return

    # Erreur de parsing JSON
    if feedback.get("error"):
        st.error(f"Erreur de review : {feedback.get('error')}")
        return

    # Score et verdict en en-tête
    score = feedback.get("score_global", 0)
    verdict = feedback.get("verdict", "?")
    verdict_label = "✅ Qualité suffisante" if verdict == "good" else "⚠️ Qualité insuffisante"

    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown(
            f"<div style='font-size:2rem; font-weight:bold; color:{score_color(score)};'>"
            f"{score}/5</div>",
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(f"**{verdict_label}**")
        commentaire = feedback.get("commentaire", "")
        if commentaire:
            st.caption(commentaire)

    st.divider()

    # Scores par critère
    st.markdown("**Scores par critère**")
    criteria = feedback.get("criteria", {})
    if criteria:
        for key, data in criteria.items():
            crit_score = data.get("score", "?")
            justif = data.get("justification", "")
            st.markdown(
                f"- **{key.capitalize()}** : `{crit_score}/5` — {justif}"
            )
    else:
        st.caption("Pas de détail par critère disponible.")

    # Issues
    issues = feedback.get("issues", [])
    if issues:
        st.markdown("**Issues identifiées**")
        for issue in issues:
            icon = severity_icon(issue.get("severity", "low"))
            vue = issue.get("vue", "?")
            desc = issue.get("description", "")
            sugg = issue.get("suggestion", "")
            st.markdown(
                f"{icon} **[{issue.get('severity', '?').upper()}]** "
                f"Vue « {vue} » : {desc}  \n"
                f"    → **Action** : {sugg}"
            )

    # Vues manquantes
    missing = feedback.get("missing_views", [])
    if missing:
        st.markdown("**Vues manquantes**")
        st.markdown(", ".join(f"`{v}`" for v in missing))

    # Points forts
    strengths = feedback.get("strengths", [])
    if strengths:
        st.markdown("**Points forts préservés**")
        for s in strengths:
            st.markdown(f"✓ {s}")


# ════════════════════════════════════════════════════════════════════
#  SIDEBAR — état du pipeline + actions + config cycle
# ════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.subheader("Pipeline")

    steps = [
        ("upload",    "1. Upload du PDF"),
        ("analyzed",  "2. Analyse du CDC"),
        ("generated", "3. Cycle + prototype"),
    ]
    current_idx = {"upload": 0, "analyzed": 1, "generated": 2}[st.session_state.step]

    for i, (key, label) in enumerate(steps):
        if i < current_idx:
            st.markdown(f"✅ {label}")
        elif i == current_idx:
            st.markdown(f"🔵 **{label}**")
        else:
            st.markdown(f"⚪ {label}")

    st.divider()

    # Configuration du cycle
    st.subheader("Configuration du cycle")
    st.session_state.max_iterations = st.slider(
        "Retries maximum",
        min_value=0,
        max_value=3,
        value=st.session_state.max_iterations,
        help="0 = pas de cycle de review, 2 = recommandé",
    )
    st.caption(f"Soit jusqu'à {st.session_state.max_iterations + 1} génération(s) au total.")

    st.divider()

    if st.session_state.pdf_name:
        st.caption(f"📄 {st.session_state.pdf_name}")

    if st.button("🔄 Recommencer depuis zéro", use_container_width=True):
        reset_all()
        st.rerun()


# ════════════════════════════════════════════════════════════════════
#  ÉTAPE 1 — Upload du PDF
# ════════════════════════════════════════════════════════════════════

if st.session_state.step == "upload":
    st.subheader("Étape 1 — Upload du cahier des charges")

    uploaded = st.file_uploader(
        "Dépose ton PDF ici",
        type=["pdf"],
        help="Le PDF sera analysé pour identifier toutes les pages à prototyper",
    )

    if uploaded is not None:
        st.session_state.pdf_name = uploaded.name

        col1, col2 = st.columns([1, 3])
        with col1:
            launch = st.button("🚀 Lancer l'analyse", type="primary", use_container_width=True)

        if launch:
            pdf_path = save_uploaded_pdf(uploaded)
            st.session_state.pdf_path = pdf_path

            with st.status("Analyse en cours...", expanded=True) as status:
                st.write("📄 Chargement et vectorisation du PDF...")
                st.write("🔍 Récupération des chunks pertinents (RAG)...")
                st.write("🧠 Analyse par CRAgent...")

                state = analyze_pdf(pdf_path)

                if state.get("final_result") == "ERROR":
                    status.update(label="❌ Échec de l'analyse", state="error")
                    st.session_state.errors = state.get("errors", [])
                    st.error("\n".join(state.get("errors", ["Erreur inconnue"])))
                    st.stop()

                st.session_state.summary = state["summary"]
                st.session_state.step = "analyzed"
                status.update(label="✅ Analyse terminée", state="complete")

            st.rerun()


# ════════════════════════════════════════════════════════════════════
#  ÉTAPE 2 — Visualisation / édition du summary + lancement du cycle
# ════════════════════════════════════════════════════════════════════

elif st.session_state.step == "analyzed":
    st.subheader("Étape 2 — Description des pages identifiées")
    st.caption(
        "Vérifie la description générée par l'analyse RAG. "
        "Tu peux l'éditer avant de lancer la génération du prototype."
    )

    edited_summary = st.text_area(
        "Description (éditable)",
        value=st.session_state.summary,
        height=400,
        key="summary_editor",
    )

    col1, col2, col3 = st.columns([2, 2, 3])

    with col1:
        if st.button("🎨 Générer le prototype", type="primary", use_container_width=True):
            st.session_state.summary = edited_summary

            final_state = run_cycle_with_streamlit_feedback(
                summary=edited_summary,
                max_iter=st.session_state.max_iterations,
            )

            if final_state.get("final_result") == "ERROR":
                st.error("\n".join(final_state.get("errors", ["Erreur inconnue"])))
                st.stop()

            # Sauvegarde dans le session_state
            st.session_state.html_code       = final_state["html_code"]
            st.session_state.render_height   = final_state.get("render_height", 900)
            st.session_state.review_feedback = final_state.get("review_feedback")
            st.session_state.quality_score   = final_state.get("quality_score", 0.0)
            st.session_state.verdict         = final_state.get("verdict", "")
            st.session_state.iteration       = final_state.get("iteration", 1)
            st.session_state.step            = "generated"
            st.rerun()

    with col2:
        if st.button("← Retour à l'upload", use_container_width=True):
            st.session_state.step = "upload"
            st.rerun()


# ════════════════════════════════════════════════════════════════════
#  ÉTAPE 3 — Affichage du prototype + rapport de review + actions
# ════════════════════════════════════════════════════════════════════

elif st.session_state.step == "generated":
    st.subheader("Étape 3 — Prototype généré")

    # ── Bandeau de statut : score final + itérations ─────────────────
    total_iter = st.session_state.iteration
    score = st.session_state.quality_score
    verdict = st.session_state.verdict

    col_score, col_iter, col_verdict = st.columns(3)

    with col_score:
        st.metric(
            "Score final",
            f"{score}/5",
            delta=None,
            help="Score global pondéré produit par le ReviewerAgent",
        )

    with col_iter:
        st.metric(
            "Générations",
            f"{total_iter}",
            delta=f"{total_iter - 1} retry{'s' if total_iter - 1 > 1 else ''}" if total_iter > 1 else None,
        )

    with col_verdict:
        verdict_icon = "✅" if verdict == "good" else "⚠️"
        verdict_label = "Qualité suffisante" if verdict == "good" else "Qualité insuffisante"
        st.metric("Verdict", f"{verdict_icon} {verdict_label}")

    st.divider()

    # ── Actions principales ──────────────────────────────────────────
    col1, col2, col3 = st.columns([2, 2, 3])

    with col1:
        st.download_button(
            "⬇️ Télécharger le HTML",
            data=st.session_state.html_code,
            file_name=f"prototype_{st.session_state.pdf_name.replace('.pdf', '')}.html",
            mime="text/html",
            use_container_width=True,
        )

    with col2:
        if st.button("🔄 Régénérer (nouveau cycle)", use_container_width=True):
            final_state = run_cycle_with_streamlit_feedback(
                summary=st.session_state.summary,
                max_iter=st.session_state.max_iterations,
            )

            if final_state.get("final_result") == "ERROR":
                st.error("\n".join(final_state.get("errors", ["Erreur inconnue"])))
                st.stop()

            st.session_state.html_code       = final_state["html_code"]
            st.session_state.render_height   = final_state.get("render_height", 900)
            st.session_state.review_feedback = final_state.get("review_feedback")
            st.session_state.quality_score   = final_state.get("quality_score", 0.0)
            st.session_state.verdict         = final_state.get("verdict", "")
            st.session_state.iteration       = final_state.get("iteration", 1)
            st.rerun()

    with col3:
        if st.button("✏️ Éditer la description et régénérer", use_container_width=True):
            st.session_state.step = "analyzed"
            st.rerun()

    st.divider()

    # ── Rapport de review (expander par défaut fermé) ───────────────
    if st.session_state.review_feedback:
        with st.expander("📋 Rapport de review détaillé", expanded=False):
            display_review_report(st.session_state.review_feedback)

    # ── Historique des itérations (si plus d'une itération) ─────────
    if len(st.session_state.iterations_log) > 1:
        with st.expander(
            f"📊 Historique des {len(st.session_state.iterations_log)} itérations",
            expanded=False,
        ):
            for log in st.session_state.iterations_log:
                iter_num = log["iteration"]
                iter_score = log["score"]
                iter_verdict = log["verdict"]
                icon = "✅" if iter_verdict == "good" else "🔄"

                st.markdown(
                    f"{icon} **Itération {iter_num}** — score "
                    f"<span style='color:{score_color(iter_score)}; font-weight:bold;'>{iter_score}/5</span> "
                    f"— verdict {iter_verdict}",
                    unsafe_allow_html=True,
                )
                fb = log.get("feedback") or {}
                if isinstance(fb, dict):
                    issues = fb.get("issues", [])
                    if issues:
                        st.caption(f"   {len(issues)} issue(s) identifiée(s)")

    # ── JSON brut (expander debug) ──────────────────────────────────
    if st.session_state.review_feedback:
        with st.expander("🔧 JSON brut du dernier review (debug)", expanded=False):
            st.json(st.session_state.review_feedback)

    st.divider()

    # ── Aperçu du prototype dans une iframe ─────────────────────────
    st.markdown("### Aperçu du prototype")
    st.caption(f"Hauteur de rendu : {st.session_state.render_height}px")
    components.html(
        st.session_state.html_code,
        height=st.session_state.render_height,
        scrolling=True,
    )

    # ── Infos techniques (optionnel) ────────────────────────────────
    with st.expander("ℹ️ Infos techniques"):
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("Taille du HTML", f"{len(st.session_state.html_code):,} chars")
        with col_b:
            st.metric("Nombre de lignes", st.session_state.html_code.count("\n"))
        with col_c:
            st.metric("Hauteur iframe", f"{st.session_state.render_height} px")