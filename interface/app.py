# interface/app.py
#
# Interface Streamlit — orchestration des 3 étapes du pipeline
# avec feedback temps réel, édition du summary, et régénération ciblée.
#
# Lancement : streamlit run interface/app.py

import sys
import tempfile
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

# Permet d'importer les modules du projet depuis le dossier parent
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from graph.workflow import analyze_pdf, generate_html, validate_html


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
    "Upload un cahier des charges PDF → analyse RAG → prototype HTML navigable"
)


# ════════════════════════════════════════════════════════════════════
#  Initialisation du session_state
#  On y stocke l'état de chaque étape pour permettre navigation et édition
# ════════════════════════════════════════════════════════════════════

def init_state():
    defaults = {
        "pdf_path":       None,   # chemin temporaire du PDF uploadé
        "pdf_name":       None,   # nom d'origine pour affichage
        "summary":        "",     # sortie du CRAgent (éditable)
        "html_code":      "",     # sortie du CoderAgent + ExecutorAgent
        "render_height":  900,    # hauteur iframe calculée par ExecutorAgent
        "step":           "upload",  # upload | analyzed | generated
        "errors":         [],
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


# ════════════════════════════════════════════════════════════════════
#  SIDEBAR — état du pipeline + actions
# ════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.subheader("Pipeline")

    steps = [
        ("upload",    "1. Upload du PDF"),
        ("analyzed",  "2. Analyse du CDC"),
        ("generated", "3. Prototype généré"),
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
#  ÉTAPE 2 — Visualisation / édition du summary + génération HTML
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

            with st.status("Génération du prototype...", expanded=True) as status:
                st.write("🎨 CoderAgent : génération du HTML...")
                state_gen = generate_html(edited_summary)

                if state_gen.get("final_result") == "ERROR":
                    status.update(label="❌ Échec de la génération", state="error")
                    st.error("\n".join(state_gen.get("errors", ["Erreur inconnue"])))
                    st.stop()

                st.write("🔧 ExecutorAgent : validation du HTML...")
                state_final = validate_html(state_gen["html_code"], state_gen)

                st.session_state.html_code     = state_final["html_code"]
                st.session_state.render_height = state_final.get("render_height", 900)
                st.session_state.step          = "generated"
                status.update(label="✅ Prototype prêt", state="complete")

            st.rerun()

    with col2:
        if st.button("← Retour à l'upload", use_container_width=True):
            st.session_state.step = "upload"
            st.rerun()


# ════════════════════════════════════════════════════════════════════
#  ÉTAPE 3 — Affichage du prototype + actions
# ════════════════════════════════════════════════════════════════════

elif st.session_state.step == "generated":
    st.subheader("Étape 3 — Prototype généré")

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
        if st.button("🔄 Régénérer (sans refaire l'analyse)", use_container_width=True):
            with st.status("Régénération...", expanded=True) as status:
                st.write("🎨 CoderAgent : nouvelle génération...")
                state_gen = generate_html(st.session_state.summary)

                if state_gen.get("final_result") == "ERROR":
                    status.update(label="❌ Échec", state="error")
                    st.error("\n".join(state_gen.get("errors", ["Erreur inconnue"])))
                    st.stop()

                st.write("🔧 ExecutorAgent : validation...")
                state_final = validate_html(state_gen["html_code"], state_gen)

                st.session_state.html_code     = state_final["html_code"]
                st.session_state.render_height = state_final.get("render_height", 900)
                status.update(label="✅ Nouveau prototype prêt", state="complete")

            st.rerun()

    with col3:
        if st.button("✏️ Éditer la description et régénérer", use_container_width=True):
            st.session_state.step = "analyzed"
            st.rerun()

    st.divider()

    # Aperçu du prototype dans une iframe
    st.caption(f"Aperçu — hauteur de rendu : {st.session_state.render_height}px")
    components.html(
        st.session_state.html_code,
        height=st.session_state.render_height,
        scrolling=True,
    )

    # Infos techniques (optionnel, dans un expander)
    with st.expander("ℹ️ Infos techniques"):
        st.metric("Taille du HTML", f"{len(st.session_state.html_code):,} caractères")
        st.metric("Nombre de lignes", st.session_state.html_code.count("\n"))
        st.metric("Hauteur iframe", f"{st.session_state.render_height} px")