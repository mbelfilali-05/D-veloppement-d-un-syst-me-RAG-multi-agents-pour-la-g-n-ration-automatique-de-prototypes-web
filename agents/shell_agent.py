# agents/shell_agent.py
#
# Agent 2 — Shell Generator
#
# Génère le squelette HTML complet : <head>, navbar/sidebar, footer,
# et une <div> vide avec x-show pour chaque vue.
#
# Le contenu des vues sera injecté par le View Generator (Agent 3)
# puis assemblé par l'Assembler (Agent 4).
#
# Input : design_config (JSON de l'Agent 1)
# Output : HTML complet avec des marqueurs <!-- VIEW_CONTENT:view_id -->
#
# Modèle : gpt-4o (le shell est visible sur TOUTES les pages, il doit être parfait)

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from agents.base_agent import BaseAgent


# ═══════════════════════════════════════════════════════════════
#  PROMPT
# ═══════════════════════════════════════════════════════════════

SHELL_PROMPT = """
Tu es un développeur front-end senior. Tu génères UNIQUEMENT le squelette HTML
d'une application — PAS le contenu des pages. Le contenu sera ajouté séparément.

Tu reçois une configuration de design en JSON. Utilise-la comme source de vérité
pour TOUS les noms, couleurs, fonctionnalités.

CONFIGURATION DE DESIGN :
{design_config_text}

═══════════════════════════════════════════════════════════════
CE QUE TU DOIS GÉNÉRER
═══════════════════════════════════════════════════════════════

1. Le <head> complet avec :
   - <title> avec le vrai nom du projet (depuis design_config.project_name)
   - Les 5 CDN obligatoires :
     <script src="https://cdn.tailwindcss.com"></script>
     <link href="https://cdn.jsdelivr.net/npm/flowbite@2.5.2/dist/flowbite.min.css" rel="stylesheet"/>
     <script src="https://cdn.jsdelivr.net/npm/flowbite@2.5.2/dist/flowbite.min.js"></script>
     <script defer src="https://cdn.jsdelivr.net/npm/alpinejs@3.14.1/dist/cdn.min.js"></script>
   - Google Fonts : la police heading ET Inter pour le body
   - Le style [x-cloak] {{ display: none !important; }}
   - font-family sur body

2. Le <body> avec x-data="{{ currentView: '{default_view}' }}" x-cloak

3. La NAVIGATION selon layout :
   Si layout == "navbar_top" :
     - <nav> fixe en haut (h-16, bg-white, border-b, z-50)
     - Logo à gauche avec le NOM EXACT du projet (style="color:{{primary_color}}")
     - Liens de navigation au centre (un @click par vue)
     - À droite : les features globales :
       • Si "language_selector" dans features → boutons de langue (FR, AR, EN, ES selon config)
       • Si "cart" dans features → icône panier SVG (pas de Material Icons texte)
       • Si "account" dans features → icône compte SVG
   
   Si layout == "sidebar_left" :
     - <aside> fixe (w-60, h-screen, bg-white, border-r)
     - Logo en haut + liens de navigation (un @click par vue)

4. Le <main> avec une <div> pour CHAQUE vue listée dans design_config.views :
   <div x-show="currentView === 'view_id'">
     <!-- VIEW_CONTENT:view_id -->
   </div>
   
   IMPORTANT : le contenu de chaque vue est UNIQUEMENT le commentaire marqueur.
   NE PAS générer de contenu dans les vues. Elles seront remplies séparément.

5. Le <footer> avec :
   - Si "social_links" dans features → liens Facebook, Instagram, Twitter
   - Si "partners" dans features → section "Nos partenaires"
   - Copyright avec le nom du projet et l'année

═══════════════════════════════════════════════════════════════
RÈGLES STRICTES
═══════════════════════════════════════════════════════════════

- Utilise TOUJOURS le nom du projet depuis design_config, JAMAIS de placeholder
- Utilise les couleurs exactes de design_config.palette.primary pour l'accent
- Pour les icônes (panier, compte, menu) → utilise des SVG inline simples, 
  PAS de noms texte comme "shopping_cart"
- Le lien de navigation actif utilise :style="currentView==='id' ? 'color:{{primary_color}}' : ''"
- Chaque div de vue DOIT avoir exactement le commentaire <!-- VIEW_CONTENT:view_id -->
- NE PAS ajouter class="hidden" — Alpine x-show gère la visibilité
- Le premier lien actif par défaut est la première vue de design_config.views

Génère UNIQUEMENT le HTML complet, sans explication, sans markdown.
Commence par <!DOCTYPE html> et termine par </html>.
"""


# ═══════════════════════════════════════════════════════════════
#  CLASSE
# ═══════════════════════════════════════════════════════════════

class ShellAgent(BaseAgent):
    """
    Génère le squelette HTML (navbar + footer + vues vides)
    à partir de la configuration de design.
    """

    def __init__(self):
        super().__init__(name="ShellAgent", temperature=0.1, model="gpt-4o")

    def _build_chain(self):
        prompt = ChatPromptTemplate.from_template(SHELL_PROMPT)
        return prompt | self.llm | StrOutputParser()

    def run(self, state: dict) -> dict:
        """
        Génère le shell HTML.

        Args:
            state: AgentState avec 'design_config' (dict)

        Returns:
            state enrichi avec 'shell_html' (str)
        """
        self._log("Génération du shell HTML...")

        design_config = state.get("design_config")
        if not design_config:
            error_msg = "ShellAgent : design_config manquant."
            self._log(f"❌ {error_msg}")
            return {
                **state,
                "shell_html": "",
                "errors": state.get("errors", []) + [error_msg],
            }

        try:
            # Sérialise le design_config en texte lisible pour le LLM
            import json
            config_text = json.dumps(design_config, ensure_ascii=False, indent=2)

            # Vue par défaut = première vue
            views = design_config.get("views", [])
            default_view = views[0]["id"] if views else "accueil"
            primary_color = design_config.get("palette", {}).get("primary", "#4F46E5")

            raw = self.chain.invoke({
                "design_config_text": config_text,
                "default_view": default_view,
                "primary_color": primary_color,
            })

            shell_html = self._clean_html(raw)

            # Vérifications de base
            views_found = []
            for view in views:
                marker = f"VIEW_CONTENT:{view['id']}"
                if marker in shell_html:
                    views_found.append(view["id"])
                else:
                    self._log(f"⚠️  Marqueur manquant : <!-- {marker} -->")

            if "</html>" not in shell_html.lower():
                self._log("⚠️  HTML potentiellement tronqué — </html> absent")

            self._log(f"✅ Shell généré ({len(shell_html):,} chars) — "
                      f"{len(views_found)}/{len(views)} marqueurs de vue trouvés")

            return {
                **state,
                "shell_html": shell_html,
                "errors": state.get("errors", []),
            }

        except Exception as e:
            error_msg = f"Erreur ShellAgent : {str(e)}"
            self._log(f"❌ {error_msg}")
            return {
                **state,
                "shell_html": "",
                "errors": state.get("errors", []) + [error_msg],
            }

    def _clean_html(self, raw: str) -> str:
        raw = raw.strip()
        if raw.startswith("```"):
            lines = raw.split("\n")
            raw = "\n".join(lines[1:])
        if raw.endswith("```"):
            raw = raw[:-3].strip()
        return raw