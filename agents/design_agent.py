# agents/design_agent.py
#
# Agent 1 — Design System Agent
#
# Lit le summary du CRAgent et extrait une configuration de design structurée (JSON).
# Cette config est la source unique de vérité pour tous les agents suivants :
#   - Shell Generator (Agent 2) : utilise palette, layout, features, project_name
#   - View Generator (Agent 3) : utilise palette, locale, typography, view types
#
# Modèle : gpt-4o-mini (extraction structurée, pas de créativité nécessaire)

import json

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from agents.base_agent import BaseAgent


# ═══════════════════════════════════════════════════════════════
#  PROMPT
# ═══════════════════════════════════════════════════════════════

DESIGN_PROMPT = """
Tu es un directeur artistique web. Tu lis la description d'une application
et tu produis une configuration de design structurée en JSON.

Tu ne génères PAS de HTML. Tu extrais les informations visuelles et
structurelles pour qu'un développeur puisse ensuite coder le site.

DESCRIPTION DE L'APPLICATION :
{summary}

═══════════════════════════════════════════════════════════════
RÈGLES D'EXTRACTION
═══════════════════════════════════════════════════════════════

1. project_name : le nom EXACT du projet tel qu'il apparaît dans la description.
   Si c'est une association, utilise son nom complet.
   JAMAIS un nom inventé. JAMAIS un nom générique.

2. type_site : détermine le type parmi :
   - "ecommerce_public" → site marchand (acheter, réserver, panier)
   - "dashboard_admin" → back-office (gestion, administration)
   - "vitrine" → site institutionnel (présentation, association)
   - "hybride" → front public + back-office

3. palette : 
   PRIORITÉ 1 — Si la description mentionne des couleurs ou une charte graphique,
   utilise-les exactement. Convertis en hex si nécessaire.
   PRIORITÉ 2 — Sinon, déduis du domaine :
     Escape game/aventure → primary #E85D04 (orange chaud)
     Patrimoine/culture → primary #B45309 (ambre)
     Éducation → primary #0284C7 (bleu confiance)
     Santé → primary #0891B2 (cyan)
     Finance/SaaS → primary #4F46E5 (indigo)
     Restauration → primary #DC2626 (rouge)

4. locale : extrais pays, villes, devise, langues TELS QU'ILS APPARAISSENT
   dans la description. Ne devine pas. Si non mentionné → "non_specifie".

5. typography :
   - Si la description mentionne une typo → utilise-la
   - Sinon : serif élégant pour culture/patrimoine/luxe, sans-serif pour le reste
   - heading : police pour les titres
   - body : police pour le texte courant

6. features : liste les fonctionnalités EXPLICITEMENT mentionnées.
   Ne jamais inventer. Exemples : "language_selector", "cart", "social_links",
   "partners", "newsletter", "faq", "account", "search".

7. views : liste TOUTES les vues/pages mentionnées avec :
   - id : identifiant court en snake_case (ex: "accueil", "nos_parcours")
   - name : nom lisible (ex: "Page d'accueil", "Nos Parcours")
   - type : parmi "landing", "catalog", "detail", "cart", "account", "form", "faq", "dashboard", "list", "other"

═══════════════════════════════════════════════════════════════
FORMAT DE SORTIE (JSON strict)
═══════════════════════════════════════════════════════════════

Réponds UNIQUEMENT avec un objet JSON valide.
Aucun markdown, aucun texte avant ou après, aucune balise ```.

{{
  "project_name": "<nom exact>",
  "organization": "<nom de l'organisation si différent du projet>",
  "type_site": "<ecommerce_public | dashboard_admin | vitrine | hybride>",
  "domain": "<escape-game, santé, éducation, etc.>",
  "locale": {{
    "country": "<pays>",
    "cities": ["<ville1>", "<ville2>"],
    "currency": "<MAD | EUR | USD | etc.>",
    "currency_symbol": "<DH | € | $ | etc.>",
    "languages": ["FR", "AR", "EN", "ES"],
    "names_style": "<moroccan | french | english | arabic | etc.>"
  }},
  "palette": {{
    "primary": "<#hex>",
    "secondary": "<#hex>",
    "accent": "<#hex ou null>",
    "source": "<charte CDC | déduit du domaine | défaut>"
  }},
  "typography": {{
    "heading": "<nom de la police>",
    "heading_url": "<URL Google Fonts>",
    "body": "Inter"
  }},
  "layout": "<navbar_top | sidebar_left>",
  "features": ["<feature1>", "<feature2>"],
  "views": [
    {{
      "id": "<snake_case>",
      "name": "<Nom lisible>",
      "type": "<landing | catalog | detail | cart | account | form | faq | dashboard | list>"
    }}
  ]
}}
"""


# ═══════════════════════════════════════════════════════════════
#  CLASSE
# ═══════════════════════════════════════════════════════════════

class DesignAgent(BaseAgent):
    """
    Extrait la configuration de design depuis le summary du CRAgent.
    Produit un JSON structuré utilisé par tous les agents suivants.
    """

    def __init__(self):
        super().__init__(name="DesignAgent", temperature=0.0, model="gpt-4o-mini")

    def _build_chain(self):
        prompt = ChatPromptTemplate.from_template(DESIGN_PROMPT)
        return prompt | self.llm | StrOutputParser()

    def run(self, state: dict) -> dict:
        """
        Extrait la configuration de design depuis le summary.

        Args:
            state: AgentState avec 'summary' non vide

        Returns:
            state enrichi avec 'design_config' (dict)
        """
        self._log("Extraction de la configuration de design...")

        summary = state.get("summary", "")
        if not summary:
            error_msg = "DesignAgent : summary vide."
            self._log(f"❌ {error_msg}")
            return {
                **state,
                "design_config": None,
                "errors": state.get("errors", []) + [error_msg],
            }

        try:
            raw = self.chain.invoke({"summary": summary})
            config = self._parse_json(raw)

            if config is None:
                error_msg = "DesignAgent : JSON invalide retourné par le LLM."
                self._log(f"❌ {error_msg}")
                return {
                    **state,
                    "design_config": None,
                    "errors": state.get("errors", []) + [error_msg],
                }

            # Validation minimale
            required_fields = ["project_name", "type_site", "palette", "views"]
            missing = [f for f in required_fields if f not in config]
            if missing:
                error_msg = f"DesignAgent : champs manquants dans le JSON : {missing}"
                self._log(f"⚠️ {error_msg}")
                # On continue quand même — le JSON est utilisable partiellement

            views_count = len(config.get("views", []))
            self._log(f"✅ Design config extraite : {config.get('project_name', '?')}")
            self._log(f"   Type : {config.get('type_site', '?')} | "
                      f"Palette : {config.get('palette', {}).get('primary', '?')} | "
                      f"Vues : {views_count} | "
                      f"Features : {len(config.get('features', []))}")

            return {
                **state,
                "design_config": config,
                "errors": state.get("errors", []),
            }

        except Exception as e:
            error_msg = f"Erreur DesignAgent : {str(e)}"
            self._log(f"❌ {error_msg}")
            return {
                **state,
                "design_config": None,
                "errors": state.get("errors", []) + [error_msg],
            }

    def _parse_json(self, raw: str) -> dict | None:
        """Parsing JSON défensif."""
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            parts = cleaned.split("```")
            if len(parts) >= 3:
                cleaned = parts[1]
                if cleaned.startswith("json"):
                    cleaned = cleaned[4:]
            cleaned = cleaned.strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as e:
            self._log(f"❌ JSONDecodeError : {e}")
            return None