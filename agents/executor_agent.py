# agents/executor_agent.py

import re
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from agents.base_agent import BaseAgent


REPAIR_PROMPT = """
Tu es un expert HTML/CSS. Le code HTML suivant contient des erreurs ou est incomplet.
Répare-le pour qu'il soit valide, complet et fonctionnel.

Règles :
- Le fichier doit commencer par <!DOCTYPE html> et se terminer par </html>
- Tout le CSS et JS doit être inline (pas de fichiers externes sauf CDN)
- La navigation entre les vues doit fonctionner
- Ne supprime aucune vue, répare seulement les erreurs

CODE HTML À RÉPARER :
{html_code}

Génère UNIQUEMENT le HTML corrigé, sans explication ni balises markdown.
"""


class ExecutorAgent(BaseAgent):
    """
    Agent de validation et préparation du rendu.

    Responsabilités :
    1. Valider que le HTML est complet et structurellement correct
    2. Réparer automatiquement si nécessaire (via LLM)
    3. Préparer le HTML final pour l'affichage dans Streamlit
       via st.components.v1.html()
    """

    # Hauteur par défaut de l'iframe Streamlit (px)
    DEFAULT_HEIGHT = 800

    def __init__(self):
        super().__init__(name="ExecutorAgent", temperature=0.0)

    def _build_chain(self):
        """Chain de réparation HTML (utilisée seulement si validation échoue)."""
        prompt = ChatPromptTemplate.from_template(REPAIR_PROMPT)
        return prompt | self.llm | StrOutputParser()

    def run(self, state: dict) -> dict:
        """
        Valide et finalise le HTML pour Streamlit.

        Args:
            state: AgentState — doit contenir 'html_code'

        Returns:
            dict: AgentState avec 'final_result' et 'render_height' mis à jour
        """
        self._log("Validation du HTML en cours...")

        html_code = state.get("html_code", "")

        if not html_code:
            error_msg = "ExecutorAgent : 'html_code' vide, rien à valider."
            self._log(f"❌ {error_msg}")
            return {
                **state,
                "final_result": "ERROR",
                "errors": state.get("errors", []) + [error_msg]
            }

        # --- Étape 1 : Validation structurelle ---
        issues = self._validate(html_code)

        if issues:
            self._log(f"⚠️  {len(issues)} problème(s) détecté(s) : {issues}")
            self._log("Tentative de réparation automatique...")

            html_code = self._repair(html_code)

            # Revalide après réparation
            remaining = self._validate(html_code)
            if remaining:
                self._log(f"⚠️  Problèmes résiduels après réparation : {remaining}")
            else:
                self._log("✅ Réparation réussie")
        else:
            self._log("✅ HTML valide, aucune réparation nécessaire")

        # --- Étape 2 : Injection du script de resize automatique ---
        # Permet à Streamlit d'ajuster la hauteur de l'iframe dynamiquement
        html_code = self._inject_resize_script(html_code)

        # --- Étape 3 : Calcul de la hauteur de rendu ---
        render_height = self._estimate_height(html_code)

        self._log(f"✅ Prêt pour Streamlit (hauteur estimée : {render_height}px)")

        return {
            **state,
            "html_code": html_code,
            "final_result": "OK",
            "render_height": render_height,
            "errors": state.get("errors", [])
        }

    # ------------------------------------------------------------------ #
    #  Méthodes privées                                                    #
    # ------------------------------------------------------------------ #

    def _validate(self, html: str) -> list[str]:
        """
        Vérifie les problèmes structurels basiques.
        Retourne une liste de problèmes (vide = HTML valide).
        """
        issues = []

        if not html.strip().lower().startswith("<!doctype html"):
            issues.append("Manque <!DOCTYPE html>")

        if "<html" not in html.lower():
            issues.append("Manque balise <html>")

        if "</html>" not in html.lower():
            issues.append("Manque balise </html>")

        if "<body" not in html.lower():
            issues.append("Manque balise <body>")

        if "<head" not in html.lower():
            issues.append("Manque balise <head>")

        # Vérifie que la navigation JS est présente
        if "onclick" not in html and "addEventListener" not in html:
            issues.append("Aucune interaction JS détectée (navigation peut être absente)")

        return issues

    def _repair(self, html: str) -> str:
        """Utilise le LLM pour réparer le HTML invalide."""
        try:
            repaired = self.chain.invoke({"html_code": html})
            repaired = repaired.strip()

            # Nettoyage des balises markdown éventuelles
            if repaired.startswith("```"):
                lines = repaired.split("\n")
                repaired = "\n".join(lines[1:])
            if repaired.endswith("```"):
                repaired = repaired[:-3].strip()

            return repaired
        except Exception as e:
            self._log(f"❌ Réparation échouée : {e}")
            return html  # Retourne l'original si la réparation plante

    def _inject_resize_script(self, html: str) -> str:
        """
        Injecte un script qui envoie la hauteur réelle du document
        à Streamlit pour ajuster l'iframe automatiquement.
        """
        resize_script = """
<script>
  // Communique la hauteur réelle au parent Streamlit
  function notifyHeight() {
    const height = document.body.scrollHeight;
    window.parent.postMessage({ type: 'streamlit:setFrameHeight', height: height }, '*');
  }
  window.addEventListener('load', notifyHeight);
  window.addEventListener('resize', notifyHeight);
  // Re-notifie après les transitions de navigation
  document.addEventListener('click', () => setTimeout(notifyHeight, 300));
</script>
"""
        # Insère juste avant </body>
        if "</body>" in html:
            return html.replace("</body>", resize_script + "\n</body>")
        else:
            return html + resize_script

    def _estimate_height(self, html: str) -> int:
        """
        Estime une hauteur de rendu raisonnable basée sur la taille du HTML.
        Streamlit utilisera cette valeur comme hauteur initiale de l'iframe.
        """
        # Heuristique simple : plus le HTML est riche, plus la hauteur est grande
        line_count = html.count("\n")

        if line_count < 100:
            return 600
        elif line_count < 300:
            return 800
        elif line_count < 600:
            return 900
        else:
            return 1000