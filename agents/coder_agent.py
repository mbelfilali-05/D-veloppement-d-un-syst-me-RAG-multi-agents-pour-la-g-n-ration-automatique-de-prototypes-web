# agents/coder_agent.py

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from agents.base_agent import BaseAgent


PROMPT_TEMPLATE = """
Tu es un expert en développement front-end et en prototypage d'interfaces web.

À partir de la description structurée des pages/vues fournie ci-dessous,
génère un prototype HTML complet, interactif et navigable.

CONTRAINTES TECHNIQUES (obligatoires) :
- Un seul fichier HTML autonome (tout inline : CSS + JS)
- Navigation entre les vues sans rechargement de page (single-page app)
- Utilise Tailwind CSS via CDN pour le style
- Barre de navigation latérale (sidebar) ou barre supérieure avec les liens vers chaque vue
- Une seule vue visible à la fois, les autres masquées (display:none / classList)
- Données d'exemple réalistes dans les tableaux, formulaires, cartes, etc.
- Responsive (mobile-friendly)
- Commentaires HTML clairs pour délimiter chaque vue

QUALITÉ VISUELLE :
- Design propre et professionnel (pas un prototype brouillon)
- Couleurs cohérentes sur toutes les vues
- Composants UI soignés : boutons avec hover, formulaires avec focus, cartes avec ombre
- Icônes via Heroicons (CDN) ou emojis si nécessaire

---
DESCRIPTION DES PAGES/VUES :
{summary}
---

Génère UNIQUEMENT le code HTML complet, sans explication, sans balises markdown.
Commence directement par <!DOCTYPE html> et termine par </html>.
"""


class CoderAgent(BaseAgent):
    """
    Agent de génération de code.
    Transforme la description structurée du CRAgent
    en un prototype HTML multi-vues navigable.
    """

    def __init__(self):
        super().__init__(name="CoderAgent", temperature=0.2, model="gpt-4o")
        # temperature=0.2 : légèrement créatif pour le rendu visuel,
        # mais toujours structuré

    def _build_chain(self):
        """Chain LCEL : prompt → LLM → texte brut (le HTML)"""
        prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        return prompt | self.llm | StrOutputParser()

    def run(self, state: dict) -> dict:
        """
        Génère le prototype HTML depuis le résumé du CRAgent.

        Args:
            state: AgentState — doit contenir 'summary'

        Returns:
            dict: AgentState avec 'html_code' mis à jour
        """
        self._log("Génération du prototype HTML en cours...")

        summary = state.get("summary", "")

        if not summary:
            error_msg = "CoderAgent : 'summary' vide, impossible de générer le HTML."
            self._log(f"❌ {error_msg}")
            return {
                **state,
                "html_code": "",
                "errors": state.get("errors", []) + [error_msg]
            }

        try:
            html_code = self.chain.invoke({"summary": summary})

            # Nettoyage : retire les éventuels backticks markdown si le LLM en ajoute
            html_code = self._clean_html(html_code)

            self._log(f"✅ HTML généré ({len(html_code)} caractères)")

            return {
                **state,
                "html_code": html_code,
                "errors": state.get("errors", [])
            }

        except Exception as e:
            error_msg = f"Erreur CoderAgent : {str(e)}"
            self._log(f"❌ {error_msg}")
            return {
                **state,
                "html_code": "",
                "errors": state.get("errors", []) + [error_msg]
            }

    def _clean_html(self, raw: str) -> str:
        """
        Retire les balises markdown que le LLM peut parfois ajouter
        autour du HTML (```html ... ```).
        """
        raw = raw.strip()
        if raw.startswith("```"):
            # Retire la première ligne (```html ou ```)
            lines = raw.split("\n")
            raw = "\n".join(lines[1:])
        if raw.endswith("```"):
            raw = raw[:-3].strip()
        return raw