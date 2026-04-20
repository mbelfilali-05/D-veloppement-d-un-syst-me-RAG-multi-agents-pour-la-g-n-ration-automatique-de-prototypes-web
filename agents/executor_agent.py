# agents/executor_agent.py

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from agents.base_agent import BaseAgent


REPAIR_PROMPT = """
Tu es un expert front-end. Le code HTML suivant contient des erreurs structurelles
ou est incomplet. Répare-le en préservant EXACTEMENT son intention et sa stack technique.

STACK TECHNIQUE UTILISÉE — À PRÉSERVER ABSOLUMENT :
• Tailwind CSS via CDN (cdn.tailwindcss.com)
• Flowbite (flowbite.min.css et flowbite.min.js)
• Alpine.js (x-data, x-show, @click, :class) pour toute la navigation et l'interactivité
• Google Fonts Inter
• Une seule vue visible à la fois via x-show="currentView === 'nom_vue'"

RÈGLES DE RÉPARATION :
1. Le fichier DOIT commencer par <!DOCTYPE html> et se terminer par </html>
2. Les balises <html>, <head>, <body> doivent être présentes et bien fermées
3. NE PAS remplacer Alpine.js par du JS vanilla — garder @click, x-show, x-data
4. NE PAS modifier les classes Tailwind existantes
5. NE PAS ajouter class="view" sur les divs de vue (x-show gère seul la visibilité)
6. NE PAS supprimer de vues — toutes doivent rester présentes
7. Si une balise est ouverte mais pas fermée, la fermer au bon endroit
8. Si le <head> manque des CDN, les ajouter (Tailwind, Flowbite, Alpine, Inter)

CODE HTML À RÉPARER :
{html_code}

Génère UNIQUEMENT le HTML corrigé, sans explication ni balises markdown.
Commence par <!DOCTYPE html> et termine par </html>.
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

    DEFAULT_HEIGHT = 900

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

            remaining = self._validate(html_code)
            if remaining:
                self._log(f"⚠️  Problèmes résiduels après réparation : {remaining}")
            else:
                self._log("✅ Réparation réussie")
        else:
            self._log("✅ HTML valide, aucune réparation nécessaire")

        # --- Étape 2 : Injection du script de resize automatique ---
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

        Détecte à la fois JS vanilla (onclick, addEventListener) et Alpine.js
        (@click, x-show, x-data) pour ne pas déclencher de réparation inutile
        sur du HTML parfaitement valide utilisant Alpine.
        """
        issues = []
        html_lower = html.lower()

        if not html.strip().lower().startswith("<!doctype html"):
            issues.append("Manque <!DOCTYPE html>")

        if "<html" not in html_lower:
            issues.append("Manque balise <html>")

        if "</html>" not in html_lower:
            issues.append("Manque balise </html> (HTML potentiellement tronqué)")

        if "<body" not in html_lower:
            issues.append("Manque balise <body>")

        if "<head" not in html_lower:
            issues.append("Manque balise <head>")

        # Détection d'interactivité : JS vanilla OU Alpine.js
        has_vanilla_js = "onclick" in html_lower or "addeventlistener" in html_lower
        has_alpine_js  = "@click" in html or "x-show" in html or "x-data" in html

        if not has_vanilla_js and not has_alpine_js:
            issues.append("Aucune interaction JS détectée (ni vanilla ni Alpine.js)")

        return issues

    def _repair(self, html: str) -> str:
        """Utilise le LLM pour réparer le HTML invalide."""
        try:
            repaired = self.chain.invoke({"html_code": html})
            repaired = repaired.strip()

            if repaired.startswith("```"):
                lines = repaired.split("\n")
                repaired = "\n".join(lines[1:])
            if repaired.endswith("```"):
                repaired = repaired[:-3].strip()

            return repaired
        except Exception as e:
            self._log(f"❌ Réparation échouée : {e}")
            return html

    def _inject_resize_script(self, html: str) -> str:
        """
        Injecte un script qui envoie la hauteur réelle du document
        à Streamlit pour ajuster l'iframe automatiquement.
        """
        resize_script = """
<script>
  function notifyHeight() {
    const height = document.body.scrollHeight;
    window.parent.postMessage({ type: 'streamlit:setFrameHeight', height: height }, '*');
  }
  window.addEventListener('load', notifyHeight);
  window.addEventListener('resize', notifyHeight);
  document.addEventListener('click', () => setTimeout(notifyHeight, 300));
</script>
"""
        if "</body>" in html:
            return html.replace("</body>", resize_script + "\n</body>")
        else:
            return html + resize_script

    def _estimate_height(self, html: str) -> int:
        """
        Estime une hauteur de rendu raisonnable pour Streamlit.
        Avec Tailwind + Flowbite, les vues sont denses — les paliers sont
        ajustés à la hausse par rapport à l'heuristique initiale.
        """
        line_count = html.count("\n")

        if line_count < 100:
            return 700
        elif line_count < 300:
            return 900
        elif line_count < 600:
            return 1100
        elif line_count < 1000:
            return 1300
        else:
            return 1500