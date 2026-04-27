# interface/app.py
#
# Interface Streamlit — architecture 6 agents
# Pipeline : analyze → design → shell → views → assemble → review → [retry] → validate
#
# Lancement : streamlit run interface/app.py

import sys
import json
import tempfile
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from graph.workflow import (
    analyze_pdf,
    run_generation_with_review,
)


# ════════════════════════════════════════════════════════════════════
#  Configuration
# ════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Générateur de prototypes web",
    page_icon="🎨",
    layout="wide",
)

st.title("🎨 Générateur de prototypes web")
st.caption("Upload un PDF → analyse RAG → 6 agents spécialisés → prototype HTML navigable")


# ════════════════════════════════════════════════════════════════════
#  Session state
# ════════════════════════════════════════════════════════════════════

def init_state():
    defaults = {
        "pdf_path":        None,
        "pdf_name":        None,
        "summary":         "",
        "html_code":       "",
        "render_height":   900,
        "step":            "upload",
        "errors":          [],
        "design_config":   None,
        "review_feedback": None,
        "quality_score":   0.0,
        "verdict":         "",
        "iteration":       0,
        "max_iterations":  2,
        "step_log":        [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# ════════════════════════════════════════════════════════════════════
#  Helpers
# ════════════════════════════════════════════════════════════════════

def reset_all():
    for k in list(st.session_state.keys()):
        del st.session_state[k]
    init_state()


def save_uploaded_pdf(uploaded_file) -> str:
    suffix = Path(uploaded_file.name).suffix or ".pdf"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded_file.read())
    tmp.close()
    return tmp.name


def score_color(score: float) -> str:
    if score >= 4.5: return "#065F46"
    if score >= 4.0: return "#16A34A"
    if score >= 3.0: return "#D97706"
    if score >= 2.0: return "#DC2626"
    return "#7F1D1D"


def severity_icon(sev: str) -> str:
    return {"high": "🔴", "medium": "🟠", "low": "🟡"}.get(sev, "⚪")


STEP_LABELS = {
    "design":   "🎨 Extraction du design system",
    "shell":    "🏗️ Génération du squelette HTML",
    "views":    "📄 Génération des vues (par batch)",
    "assemble": "🔧 Assemblage du HTML",
    "review":   "⚖️ Évaluation par le ReviewerAgent",
    "retry":    "🔄 Retry ciblé des vues problématiques",
    "validate": "✅ Validation finale (ExecutorAgent)",
}


def run_with_feedback(summary: str, max_iter: int) -> dict:
    """Lance le pipeline 6 agents avec feedback temps réel."""
    step_log = []

    with st.status("Pipeline 6 agents en cours...", expanded=True) as status:

        def on_step(step_name: str, state: dict):
            label = STEP_LABELS.get(step_name, step_name)

            if step_name == "design":
                config = state.get("design_config", {})
                if config:
                    name = config.get("project_name", "?")
                    palette = config.get("palette", {}).get("primary", "?")
                    views_count = len(config.get("views", []))
                    st.markdown(f"{label} — **{name}** | palette `{palette}` | {views_count} vues")
                else:
                    st.markdown(f"{label} — ❌ échec")

            elif step_name == "shell":
                shell = state.get("shell_html", "")
                if shell:
                    st.markdown(f"{label} — {len(shell):,} caractères")
                else:
                    st.markdown(f"{label} — ❌ échec")

            elif step_name == "views":
                views = state.get("views_html", {})
                st.markdown(f"{label} — **{len(views)} vues** générées")

            elif step_name == "assemble":
                html = state.get("html_code", "")
                st.markdown(f"{label} — {len(html):,} caractères / {html.count(chr(10))} lignes")

            elif step_name == "review":
                score = state.get("quality_score", 0)
                verdict = state.get("verdict", "?")
                feedback = state.get("review_feedback", {})
                issues = feedback.get("issues", []) if isinstance(feedback, dict) else []
                high_count = sum(1 for i in issues if i.get("severity") == "high")
                pre_count = feedback.get("pre_check_count", 0) if isinstance(feedback, dict) else 0

                color = score_color(score)
                verdict_icon = "✅" if verdict == "good" else "⚠️"

                st.markdown(
                    f"{label} — score "
                    f"<span style='color:{color}; font-weight:bold'>{score}/5</span> "
                    f"{verdict_icon} {verdict}"
                    f"{f' — {high_count} issue(s) high, {pre_count} défaut(s) mécaniques' if high_count > 0 else ''}",
                    unsafe_allow_html=True,
                )

                step_log.append({
                    "iteration": state.get("iteration", 0),
                    "score": score,
                    "verdict": verdict,
                    "issues_count": len(issues),
                    "high_count": high_count,
                })

            elif step_name == "retry":
                feedback = state.get("review_feedback", {})
                missing = feedback.get("missing_views", []) if isinstance(feedback, dict) else []
                issues = feedback.get("issues", []) if isinstance(feedback, dict) else []
                high_vues = [i.get("vue", "?") for i in issues if i.get("severity") == "high" and i.get("vue") != "global"]
                targets = list(set(missing + high_vues))
                st.markdown(f"{label} — vues ciblées : {', '.join(targets) if targets else 'auto'}")

            elif step_name == "validate":
                st.markdown(f"{label}")

        # Lancement
        st.write("🚀 Démarrage du pipeline 6 agents...")
        final_state = run_generation_with_review(
            summary=summary,
            max_iterations=max_iter,
            on_step_callback=on_step,
        )

        # Statut final
        if final_state.get("final_result") == "ERROR":
            status.update(label="❌ Échec du pipeline", state="error")
        else:
            score = final_state.get("quality_score", 0)
            verdict = final_state.get("verdict", "?")
            iteration = final_state.get("iteration", 1)
            status.update(
                label=f"✅ Terminé — {iteration} itération(s) — score {score}/5 ({verdict})",
                state="complete",
            )

    st.session_state.step_log = step_log
    return final_state


def display_review_report(feedback: dict):
    """Affiche le rapport de review structuré."""
    if not feedback or not isinstance(feedback, dict):
        st.info("Aucun rapport de review disponible.")
        return

    if feedback.get("error"):
        st.error(f"Erreur de review : {feedback.get('error')}")
        return

    # Score et verdict
    score = feedback.get("score_global", 0)
    verdict = feedback.get("verdict", "?")
    verdict_label = "✅ Qualité suffisante" if verdict == "good" else "⚠️ Qualité insuffisante"

    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown(
            f"<div style='font-size:2.5rem; font-weight:bold; color:{score_color(score)}'>"
            f"{score}/5</div>",
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(f"**{verdict_label}**")
        pre = feedback.get("pre_check_count", 0)
        llm = feedback.get("llm_issue_count", 0)
        st.caption(f"Défauts mécaniques : {pre} | Issues sémantiques : {llm}")
        commentaire = feedback.get("commentaire", "")
        if commentaire:
            st.caption(commentaire)

    st.divider()

    # Scores par critère
    criteria = feedback.get("criteria", {})
    if criteria:
        st.markdown("**Scores par critère**")
        cols = st.columns(5)
        for i, (key, data) in enumerate(criteria.items()):
            with cols[i % 5]:
                crit_score = data.get("score", "?")
                st.metric(key.capitalize(), f"{crit_score}/5")
                st.caption(data.get("justification", "")[:80])

    # Issues
    issues = feedback.get("issues", [])
    if issues:
        st.divider()
        st.markdown(f"**{len(issues)} issue(s) identifiée(s)**")
        for issue in issues:
            icon = severity_icon(issue.get("severity", "low"))
            source = issue.get("source", "?")
            vue = issue.get("vue", "?")
            desc = issue.get("description", "")
            sugg = issue.get("suggestion", "")
            st.markdown(
                f"{icon} **[{source}]** Vue « {vue} » — {desc}  \n"
                f"→ {sugg}"
            )

    # Points forts
    strengths = feedback.get("strengths", [])
    if strengths:
        st.divider()
        st.markdown("**Points forts**")
        for s in strengths:
            st.markdown(f"✓ {s}")


# ════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.subheader("Pipeline 6 agents")

    steps = [
        ("upload",    "1. Upload PDF"),
        ("analyzed",  "2. Analyse RAG"),
        ("generated", "3. Génération + Review"),
    ]
    idx = {"upload": 0, "analyzed": 1, "generated": 2}[st.session_state.step]
    for i, (_, label) in enumerate(steps):
        if i < idx:   st.markdown(f"✅ {label}")
        elif i == idx: st.markdown(f"🔵 **{label}**")
        else:          st.markdown(f"⚪ {label}")

    st.divider()

    st.subheader("Configuration")
    st.session_state.max_iterations = st.slider(
        "Retries max", 0, 3, st.session_state.max_iterations,
        help="0 = pas de retry, 2 = recommandé",
    )
    st.caption(f"Jusqu'à {st.session_state.max_iterations + 1} itération(s)")

    st.divider()
    if st.session_state.pdf_name:
        st.caption(f"📄 {st.session_state.pdf_name}")

    # Design config aperçu
    if st.session_state.design_config:
        dc = st.session_state.design_config
        st.caption(f"🎨 {dc.get('project_name', '?')}")
        st.caption(f"Palette : {dc.get('palette', {}).get('primary', '?')}")
        st.caption(f"Vues : {len(dc.get('views', []))}")

    if st.button("🔄 Recommencer", use_container_width=True):
        reset_all()
        st.rerun()


# ════════════════════════════════════════════════════════════════════
#  ÉTAPE 1 — Upload
# ════════════════════════════════════════════════════════════════════

if st.session_state.step == "upload":
    st.subheader("Étape 1 — Upload du cahier des charges")

    uploaded = st.file_uploader("Dépose ton PDF ici", type=["pdf"])

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
                st.write("🔍 Récupération des chunks pertinents...")
                st.write("🧠 Analyse par CRAgent...")

                state = analyze_pdf(pdf_path)

                if state.get("final_result") == "ERROR":
                    status.update(label="❌ Échec", state="error")
                    st.error("\n".join(state.get("errors", ["Erreur inconnue"])))
                    st.stop()

                st.session_state.summary = state["summary"]
                st.session_state.step = "analyzed"
                status.update(label="✅ Analyse terminée", state="complete")

            st.rerun()


# ════════════════════════════════════════════════════════════════════
#  ÉTAPE 2 — Summary éditable + lancement
# ════════════════════════════════════════════════════════════════════

elif st.session_state.step == "analyzed":
    st.subheader("Étape 2 — Description des pages")
    st.caption("Vérifie et édite la description avant de lancer les 6 agents.")

    edited = st.text_area(
        "Description (éditable)", value=st.session_state.summary, height=400,
    )

    col1, col2 = st.columns([2, 3])
    with col1:
        if st.button("🎨 Générer le prototype", type="primary", use_container_width=True):
            st.session_state.summary = edited

            final_state = run_with_feedback(edited, st.session_state.max_iterations)

            if final_state.get("final_result") == "ERROR":
                st.error("\n".join(final_state.get("errors", ["Erreur"])))
                st.stop()

            st.session_state.html_code       = final_state.get("html_code", "")
            st.session_state.render_height   = final_state.get("render_height", 900)
            st.session_state.review_feedback = final_state.get("review_feedback")
            st.session_state.quality_score   = final_state.get("quality_score", 0)
            st.session_state.verdict         = final_state.get("verdict", "")
            st.session_state.iteration       = final_state.get("iteration", 1)
            st.session_state.design_config   = final_state.get("design_config")
            st.session_state.step            = "generated"
            st.rerun()

    with col2:
        if st.button("← Retour", use_container_width=True):
            st.session_state.step = "upload"
            st.rerun()


# ════════════════════════════════════════════════════════════════════
#  ÉTAPE 3 — Résultat
# ════════════════════════════════════════════════════════════════════

elif st.session_state.step == "generated":
    st.subheader("Étape 3 — Prototype généré")

    # Métriques
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Score final", f"{st.session_state.quality_score}/5")
    with col2:
        st.metric("Itérations", st.session_state.iteration)
    with col3:
        v = st.session_state.verdict
        st.metric("Verdict", f"{'✅' if v == 'good' else '⚠️'} {v}")

    st.divider()

    # Actions
    col1, col2, col3 = st.columns([2, 2, 3])
    with col1:
        st.download_button(
            "⬇️ Télécharger HTML",
            data=st.session_state.html_code,
            file_name=f"prototype_{st.session_state.pdf_name.replace('.pdf', '')}.html",
            mime="text/html",
            use_container_width=True,
        )
    with col2:
        if st.button("🔄 Régénérer", use_container_width=True):
            final_state = run_with_feedback(
                st.session_state.summary, st.session_state.max_iterations,
            )
            if final_state.get("final_result") != "ERROR":
                st.session_state.html_code       = final_state.get("html_code", "")
                st.session_state.render_height   = final_state.get("render_height", 900)
                st.session_state.review_feedback = final_state.get("review_feedback")
                st.session_state.quality_score   = final_state.get("quality_score", 0)
                st.session_state.verdict         = final_state.get("verdict", "")
                st.session_state.iteration       = final_state.get("iteration", 1)
                st.rerun()
    with col3:
        if st.button("✏️ Éditer le summary", use_container_width=True):
            st.session_state.step = "analyzed"
            st.rerun()

    st.divider()

    # Rapport de review
    if st.session_state.review_feedback:
        with st.expander("📋 Rapport de review", expanded=False):
            display_review_report(st.session_state.review_feedback)

    # Historique itérations
    if len(st.session_state.step_log) > 1:
        with st.expander(f"📊 Historique ({len(st.session_state.step_log)} itérations)"):
            for log in st.session_state.step_log:
                icon = "✅" if log["verdict"] == "good" else "🔄"
                st.markdown(
                    f"{icon} Itération {log['iteration']} — "
                    f"<span style='color:{score_color(log['score'])};font-weight:bold'>"
                    f"{log['score']}/5</span> — {log['issues_count']} issues "
                    f"({log['high_count']} high)",
                    unsafe_allow_html=True,
                )

    # JSON debug
    if st.session_state.review_feedback:
        with st.expander("🔧 JSON brut (debug)", expanded=False):
            st.json(st.session_state.review_feedback)

    # Design config
    if st.session_state.design_config:
        with st.expander("🎨 Design config (debug)", expanded=False):
            st.json(st.session_state.design_config)

    st.divider()

    # Aperçu
    st.markdown("### Aperçu du prototype")
    components.html(
        st.session_state.html_code,
        height=st.session_state.render_height,
        scrolling=True,
    )

    with st.expander("ℹ️ Infos techniques"):
        c1, c2, c3 = st.columns(3)
        with c1: st.metric("Taille HTML", f"{len(st.session_state.html_code):,} chars")
        with c2: st.metric("Lignes", st.session_state.html_code.count("\n"))
        with c3: st.metric("Hauteur", f"{st.session_state.render_height}px")