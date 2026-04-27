# agents/view_agent.py
#
# Agent 3 — View Generator
#
# Génère le contenu HTML d'1 ou 2 vues par appel.
# Reçoit le design_config + le summary de la/les vue(s) spécifique(s).
# Produit uniquement le HTML intérieur (pas de <html>, <body>, <nav>).
#
# Appelé N/2 fois (arrondi supérieur) par le workflow.
# Ex: 7 vues AMPD → 4 appels (2+2+2+1)
#
# Modèle : gpt-4o (qualité critique pour le contenu visible)

import json
import re
import math

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from agents.base_agent import BaseAgent


# ═══════════════════════════════════════════════════════════════
#  PROMPT
# ═══════════════════════════════════════════════════════════════

VIEW_PROMPT = """
Tu es un développeur front-end senior. Tu génères le contenu HTML
d'une ou deux vues spécifiques pour une application web.

Tu ne génères PAS de <html>, <head>, <body>, <nav>, <footer>.
Tu génères UNIQUEMENT le contenu intérieur des vues demandées,
qui sera inséré dans un squelette HTML existant.

═══════════════════════════════════════════════════════════════
CONFIGURATION DU PROJET (source de vérité pour noms, couleurs, données)
═══════════════════════════════════════════════════════════════

{design_config_text}

═══════════════════════════════════════════════════════════════
VUE(S) À GÉNÉRER
═══════════════════════════════════════════════════════════════

{views_to_generate}

═══════════════════════════════════════════════════════════════
STACK TECHNIQUE
═══════════════════════════════════════════════════════════════

- Tailwind CSS (classes utilitaires)
- Flowbite (composants)
- Alpine.js pour interactivité interne à la vue (accordéons, toggles, modals)
  MAIS PAS pour la navigation entre vues (c'est géré par le shell)
- Police heading : {heading_font} | Police body : Inter
- Couleur accent : {primary_color}

═══════════════════════════════════════════════════════════════
RÈGLES PAR TYPE DE VUE
═══════════════════════════════════════════════════════════════

TYPE "landing" (page d'accueil) :
  - Hero section GRANDE (min h-[500px]), fond teinté avec l'accent, titre émotionnel
    en text-5xl avec la police heading, sous-titre text-lg, 2 boutons CTA
  - Minimum 5 sections distinctes après le hero
  - Section produits/services : grid grid-cols-3 avec les vrais noms du projet
  - Section "Comment ça marche" : 3 étapes numérotées horizontales
  - Section témoignages : 3 cards avec prénoms adaptés au pays (locale.names_style)
  - Sections supplémentaires selon features : partenaires, newsletter, CTA final

TYPE "catalog" (liste/catalogue) :
  - Barre de filtres horizontale (flex gap-3) selon le contexte
  - Grille grid-cols-3 gap-6 avec MINIMUM 6 cards
  - Chaque card : placeholder image teinté accent, titre exact, description, prix avec devise

TYPE "detail" (fiche produit/service) :
  - Layout grid grid-cols-3 gap-8
  - Gauche (col-span-2) : image grande, description longue, galerie thumbnails
  - Droite (col-span-1) : card prix, infos clés, bouton CTA principal
  - Section infos pratiques sous le contenu principal

TYPE "cart" (panier) :
  - Layout grid grid-cols-3 gap-8
  - Gauche (col-span-2) : tableau articles avec MINIMUM 4 lignes, colonnes :
    Produit | Prix | Quantité | Total | Supprimer
  - Droite (col-span-1) : récap (sous-total, taxes, total TTC, bouton paiement)
  - Données d'exemple avec vrais noms et devise du projet

TYPE "account" (compte) :
  - Sections inscription/connexion OU profil selon le contexte
  - Si historique : tableau avec MINIMUM 6 lignes et badges statut
  - Formulaire avec 4-6 champs pré-remplis

TYPE "form" (formulaire de contact) :
  - Card centrée max-w-2xl mx-auto
  - MINIMUM 5 champs : nom, email, téléphone, sujet (select), message (textarea)
  - Boutons flex justify-end gap-3
  - Section coordonnées à côté ou en dessous

TYPE "faq" :
  - Barre de recherche en haut
  - Accordéons Alpine.js (x-data, @click, x-show) avec MINIMUM 6 questions
  - Questions réalistes liées au domaine du projet

TYPE "dashboard" :
  - 4 KPI cards en haut (grid grid-cols-4 gap-4)
  - Tableau principal avec MINIMUM 6 lignes et badges statut colorés
  - Sidebar stats ou activité récente

═══════════════════════════════════════════════════════════════
RÈGLES DE CONTENU (non négociables)
═══════════════════════════════════════════════════════════════

1. DONNÉES FIDÈLES : utilise les noms de produits, villes, devise depuis design_config.
   Prénoms et noms adaptés au pays (locale.names_style).
   Si "moroccan" → Alaoui, Benali, Tazi, Chraibi, El Fassi, Amrani
   JAMAIS de "Produit A", "John Doe", "Lorem ipsum"

2. DENSITÉ OBLIGATOIRE :
   - Chaque vue : minimum 30 lignes de HTML
   - Tableaux : minimum 4 lignes (6 idéalement)
   - Formulaires : minimum 5 champs
   - Landing : minimum 5 sections

3. PLACEHOLDERS IMAGES : fond teinté accent (opacity 10-15%) avec texte descriptif
   <div class="h-48 rounded-lg flex items-center justify-center"
        style="background: {primary_color}15">
     <span class="text-sm" style="color: {primary_color}">
       Photo : [description contextuelle de l'image attendue]
     </span>
   </div>

4. BOUTONS : accent du projet pour les primaires
   <button style="background: {primary_color}"
           class="px-4 py-2 text-white text-sm font-medium rounded-lg hover:opacity-90">
     Label
   </button>

5. FORMAT DE SORTIE : pour chaque vue, encadre le HTML avec un commentaire :
   <!-- START_VIEW:view_id -->
   ... HTML de la vue ...
   <!-- END_VIEW:view_id -->

Génère UNIQUEMENT le HTML des vues demandées, sans explication, sans markdown.
"""


# ═══════════════════════════════════════════════════════════════
#  CLASSE
# ═══════════════════════════════════════════════════════════════

class ViewAgent(BaseAgent):
    """
    Génère le contenu HTML d'1-2 vues par appel.
    Appelé en boucle par le workflow pour chaque batch de vues.
    """

    VIEWS_PER_BATCH = 2  # nombre de vues par appel LLM

    def __init__(self):
        super().__init__(name="ViewAgent", temperature=0.2, model="gpt-4o")

    def _build_chain(self):
        prompt = ChatPromptTemplate.from_template(VIEW_PROMPT)
        return prompt | self.llm | StrOutputParser()

    def run(self, state: dict) -> dict:
        """
        Génère le contenu de TOUTES les vues en batches de VIEWS_PER_BATCH.

        Args:
            state: AgentState avec 'design_config' et 'summary'

        Returns:
            state enrichi avec 'views_html' (dict: view_id → html_content)
        """
        design_config = state.get("design_config")
        summary = state.get("summary", "")

        if not design_config or not summary:
            error_msg = "ViewAgent : design_config ou summary manquant."
            self._log(f"❌ {error_msg}")
            return {
                **state,
                "views_html": {},
                "errors": state.get("errors", []) + [error_msg],
            }

        views = design_config.get("views", [])
        if not views:
            error_msg = "ViewAgent : aucune vue dans design_config."
            self._log(f"❌ {error_msg}")
            return {
                **state,
                "views_html": {},
                "errors": state.get("errors", []) + [error_msg],
            }

        # Préparer les infos communes
        config_text = json.dumps(design_config, ensure_ascii=False, indent=2)
        primary_color = design_config.get("palette", {}).get("primary", "#4F46E5")
        heading_font = design_config.get("typography", {}).get("heading", "Inter")

        # Découper en batches de VIEWS_PER_BATCH
        batches = []
        for i in range(0, len(views), self.VIEWS_PER_BATCH):
            batches.append(views[i:i + self.VIEWS_PER_BATCH])

        total_batches = len(batches)
        self._log(f"Génération de {len(views)} vues en {total_batches} batch(es) "
                  f"(max {self.VIEWS_PER_BATCH} vues/batch)...")

        all_views_html = {}
        errors = list(state.get("errors", []))

        for batch_idx, batch in enumerate(batches, 1):
            batch_names = [v["name"] for v in batch]
            self._log(f"  Batch {batch_idx}/{total_batches} : {', '.join(batch_names)}")

            # Construire la description des vues demandées
            views_desc = self._build_views_description(batch, summary)

            try:
                raw = self.chain.invoke({
                    "design_config_text": config_text,
                    "views_to_generate": views_desc,
                    "primary_color": primary_color,
                    "heading_font": heading_font,
                })

                raw = self._clean_html(raw)

                # Parser le HTML pour extraire chaque vue
                for view in batch:
                    view_html = self._extract_view(raw, view["id"])
                    if view_html:
                        all_views_html[view["id"]] = view_html
                        lines = view_html.count("\n")
                        self._log(f"    ✅ {view['name']} — {lines} lignes")
                    else:
                        # Fallback : si les marqueurs ne sont pas là,
                        # et qu'il n'y a qu'une seule vue dans le batch,
                        # utilise tout le HTML brut
                        if len(batch) == 1:
                            all_views_html[view["id"]] = raw
                            self._log(f"    ⚠️ {view['name']} — marqueurs absents, HTML brut utilisé")
                        else:
                            self._log(f"    ❌ {view['name']} — non trouvée dans la réponse")
                            errors.append(f"Vue '{view['name']}' non trouvée dans le batch {batch_idx}")

            except Exception as e:
                error_msg = f"Erreur batch {batch_idx} ({batch_names}) : {str(e)}"
                self._log(f"  ❌ {error_msg}")
                errors.append(error_msg)

        self._log(f"✅ {len(all_views_html)}/{len(views)} vues générées")

        return {
            **state,
            "views_html": all_views_html,
            "errors": errors,
        }

    def generate_single_view(self, view_info: dict, design_config: dict, summary: str) -> str:
        """
        Génère une seule vue. Utilisé pour le retry ciblé du ReviewerAgent.

        Args:
            view_info: dict avec id, name, type
            design_config: configuration de design complète
            summary: summary complet (pour le contexte de la vue)

        Returns:
            HTML de la vue (str)
        """
        config_text = json.dumps(design_config, ensure_ascii=False, indent=2)
        primary_color = design_config.get("palette", {}).get("primary", "#4F46E5")
        heading_font = design_config.get("typography", {}).get("heading", "Inter")

        views_desc = self._build_views_description([view_info], summary)

        raw = self.chain.invoke({
            "design_config_text": config_text,
            "views_to_generate": views_desc,
            "primary_color": primary_color,
            "heading_font": heading_font,
        })

        raw = self._clean_html(raw)
        view_html = self._extract_view(raw, view_info["id"])
        return view_html or raw

    # ────────────────────────────────────────────────────────────
    #  Helpers
    # ────────────────────────────────────────────────────────────

    def _build_views_description(self, views: list[dict], summary: str) -> str:
        """
        Construit la description des vues à générer pour le prompt.
        Extrait la section pertinente du summary pour chaque vue.
        """
        parts = []
        for view in views:
            # Essayer d'extraire la section du summary correspondant à cette vue
            view_summary = self._extract_view_summary(view, summary)

            parts.append(
                f"VUE : {view['name']}\n"
                f"  ID : {view['id']}\n"
                f"  TYPE : {view['type']}\n"
                f"  DESCRIPTION DEPUIS LE SUMMARY :\n"
                f"  {view_summary}\n"
            )

        return "\n---\n".join(parts)

    def _extract_view_summary(self, view: dict, summary: str) -> str:
        """
        Extrait la section du summary qui correspond à une vue spécifique.
        Recherche par nom de vue avec plusieurs patterns.
        """
        view_name = view["name"]
        
        # Pattern 1 : ## N. Nom de la vue (jusqu'au prochain ## ou fin)
        pattern = re.compile(
            rf'##\s*\d*\.?\s*{re.escape(view_name)}.*?(?=\n##|\Z)',
            re.DOTALL | re.IGNORECASE
        )
        match = pattern.search(summary)
        if match:
            return match.group(0).strip()

        # Pattern 2 : recherche par mots-clés du nom
        keywords = [w for w in view_name.split() if len(w) > 3]
        if keywords:
            for kw in keywords:
                pattern = re.compile(
                    rf'##[^\n]*{re.escape(kw)}[^\n]*\n(.*?)(?=\n##|\Z)',
                    re.DOTALL | re.IGNORECASE
                )
                match = pattern.search(summary)
                if match:
                    return match.group(0).strip()

        # Pattern 3 : recherche par id
        pattern = re.compile(
            rf'##[^\n]*{re.escape(view["id"])}[^\n]*\n(.*?)(?=\n##|\Z)',
            re.DOTALL | re.IGNORECASE
        )
        match = pattern.search(summary)
        if match:
            return match.group(0).strip()

        # Fallback : retourne tout le summary (le LLM devra chercher)
        return f"(Section spécifique non trouvée — se référer au summary complet pour '{view_name}')"

    def _extract_view(self, html: str, view_id: str) -> str | None:
        """
        Extrait le HTML d'une vue depuis les marqueurs START_VIEW / END_VIEW.
        """
        pattern = re.compile(
            rf'<!--\s*START_VIEW:{re.escape(view_id)}\s*-->(.*?)<!--\s*END_VIEW:{re.escape(view_id)}\s*-->',
            re.DOTALL
        )
        match = pattern.search(html)
        if match:
            return match.group(1).strip()
        return None

    def _clean_html(self, raw: str) -> str:
        raw = raw.strip()
        if raw.startswith("```"):
            lines = raw.split("\n")
            raw = "\n".join(lines[1:])
        if raw.endswith("```"):
            raw = raw[:-3].strip()
        return raw