# agents/assembler.py
#
# Agent 4 — Assembler (Python pur — AUCUN appel LLM)
#
# Prend le shell HTML (Agent 2) et remplace chaque marqueur
# <!-- VIEW_CONTENT:view_id --> par le HTML de la vue correspondante (Agent 3).
#
# Produit le HTML final complet, prêt à être évalué par le ReviewerAgent.

import re


class Assembler:
    """
    Assemble le shell HTML avec les contenus des vues générées.
    Aucun appel LLM — Python pur, instantané, 100% fiable.
    """

    def run(self, state: dict) -> dict:
        """
        Assemble shell_html + views_html → html_code.

        Args:
            state: AgentState avec :
                - shell_html (str) : squelette avec marqueurs <!-- VIEW_CONTENT:id -->
                - views_html (dict) : {view_id: html_content}
                - design_config (dict) : pour la liste des vues attendues

        Returns:
            state enrichi avec 'html_code' (str)
        """
        print(f"[Assembler] Assemblage du HTML final...")

        shell = state.get("shell_html", "")
        views = state.get("views_html", {})
        design_config = state.get("design_config", {})

        if not shell:
            error_msg = "Assembler : shell_html vide."
            print(f"[Assembler] ❌ {error_msg}")
            return {
                **state,
                "html_code": "",
                "errors": state.get("errors", []) + [error_msg],
            }

        html = shell
        expected_views = [v["id"] for v in design_config.get("views", [])]
        
        injected = []
        missing = []

        for view_id in expected_views:
            marker = f"<!-- VIEW_CONTENT:{view_id} -->"
            
            if view_id in views and views[view_id]:
                # Injection du contenu de la vue
                view_content = views[view_id]
                html = html.replace(marker, view_content)
                injected.append(view_id)
            elif marker in html:
                # Le marqueur existe mais pas de contenu → laisser un placeholder visible
                fallback = (
                    f'<div class="p-12 text-center text-gray-400">'
                    f'<p class="text-lg">Vue « {view_id} » — contenu en cours de génération</p>'
                    f'</div>'
                )
                html = html.replace(marker, fallback)
                missing.append(view_id)
            else:
                # Ni marqueur ni contenu
                missing.append(view_id)

        # Vérification : y a-t-il des marqueurs non remplacés ?
        remaining_markers = re.findall(r'<!-- VIEW_CONTENT:(\w+) -->', html)
        if remaining_markers:
            print(f"[Assembler] ⚠️  Marqueurs non remplacés : {remaining_markers}")

        print(f"[Assembler] ✅ Assemblé : {len(injected)} vues injectées"
              f"{f', {len(missing)} manquante(s) : {missing}' if missing else ''}")
        print(f"[Assembler]    HTML final : {len(html):,} caractères / "
              f"{html.count(chr(10))} lignes")

        return {
            **state,
            "html_code": html,
            "errors": state.get("errors", []),
        }