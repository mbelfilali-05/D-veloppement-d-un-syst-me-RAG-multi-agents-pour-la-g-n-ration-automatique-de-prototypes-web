# agents/coder_agent.py
#
# CHANGEMENTS PAR RAPPORT À LA VERSION PRÉCÉDENTE :
#
# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  CHANGEMENT 1 — Plus d'Étape 0 dans le CoderAgent                        ║
# ║  Avant : le CoderAgent faisait sa propre extraction CDC (commentaire      ║
# ║  <!-- CDC_EXTRACTED -->), dupliquant le travail du CRAgent.               ║
# ║  Maintenant : le CoderAgent CONSOMME la palette et les métadonnées        ║
# ║  déjà extraites par le CRAgent dans le summary. Plus de duplication.      ║
# ╠══════════════════════════════════════════════════════════════════════════════╣
# ║  CHANGEMENT 2 — Palette consommée, plus choisie                          ║
# ║  Avant : le CoderAgent avait une Priorité 1/2/3 pour choisir les         ║
# ║  couleurs → la Priorité 2 (domaine) prenait souvent le dessus sur la     ║
# ║  Priorité 1 (charte CDC) quand le CDC était vague.                       ║
# ║  Maintenant : le CoderAgent lit les hex de la section "Palette de         ║
# ║  couleurs" du summary et les applique directement. Plus de choix.         ║
# ╠══════════════════════════════════════════════════════════════════════════════╣
# ║  CHANGEMENT 3 — Exemples de données NEUTRES et multi-géographiques       ║
# ║  Avant : les exemples citaient "Kasbah des Oudayas", "Alaoui", "MAD",    ║
# ║  "Dr. Benali" — 100% marocains → biais pour tout CDC non-marocain.       ║
# ║  Maintenant : les exemples montrent le PATTERN (utiliser les noms,       ║
# ║  villes, devise du CDC) sans donner de contenu fixe.                     ║
# ╠══════════════════════════════════════════════════════════════════════════════╣
# ║  CHANGEMENT 4 — "Minimum 6 cards" supprimé                               ║
# ║  Avant : le prompt imposait "minimum 6 cards produit" → forçait le LLM   ║
# ║  à inventer des produits quand le CDC en listait moins (ex: 3 parcours   ║
# ║  AMPD → "Parcours Médina de Fès" inventé).                               ║
# ║  Maintenant : "autant de cards que le CDC mentionne de produits.          ║
# ║  Si le CDC dit 3, affiche 3. Si 1, affiche 1."                          ║
# ╠══════════════════════════════════════════════════════════════════════════════╣
# ║  CHANGEMENT 5 — Titre émotionnel plus hardcodé sur AMPD                  ║
# ║  Avant : l'exemple disait "Vivez une aventure inoubliable à Rabat".      ║
# ║  Maintenant : le prompt dit "titre émotionnel adapté au domaine ET au    ║
# ║  nom du projet extraits du summary" — sans exemple fixe.                 ║
# ╠══════════════════════════════════════════════════════════════════════════════╣
# ║  CHANGEMENT 6 — Layouts conditionnels au lieu d'absolus                   ║
# ║  Avant : "ACCUEIL = hero 500px + 4 sections minimum" était une règle     ║
# ║  absolue qui ne convient pas à tous les types de site.                   ║
# ║  Maintenant : chaque layout est préfixé par les types de site auxquels   ║
# ║  il s'applique. Les quantités sont "selon le CDC" et non hardcodées.     ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from agents.base_agent import BaseAgent


PROMPT_TEMPLATE = """
Tu es un développeur front-end senior expert en prototypage web rapide.

═══════════════════════════════════════════════════════════════
3 RÈGLES CRITIQUES (non négociables)
═══════════════════════════════════════════════════════════════

1. FIDÉLITÉ ABSOLUE AU SUMMARY :
   Noms, couleurs, langues, produits, villes, devises, fonctionnalités
   → tels qu'ils apparaissent dans le summary du CRAgent. JAMAIS inventés.

2. PALETTE DU SUMMARY EN PRIORITÉ :
   Le summary contient une section "Palette de couleurs" avec des codes hex
   et des rôles (dominante, accent, secondaire...). UTILISE CES HEX
   EXACTEMENT. Ne choisis pas d'autres couleurs.

3. QUANTITÉ = CELLE DU SUMMARY :
   Si le summary décrit 3 produits, affiche 3 cards. Pas 6. Pas 1.
   Si le summary dit "à venir" pour certaines villes, affiche-les
   comme "bientôt disponible" sans inventer de contenu.

═══════════════════════════════════════════════════════════════
ÉTAPE 1 — LECTURE DU SUMMARY
═══════════════════════════════════════════════════════════════

Le summary contient des informations structurées que tu DOIS lire
avant de coder. Cherche dans le summary :

1. **Palette de couleurs** → tes hex pour tout le HTML
   - Dominante → navbar, titres, fond hero, sidebar
   - Accent    → boutons primaires, liens actifs, CTA
   - Secondaire → badges, highlights, icônes
   - Tertiaire → accents légers, bordures décoratives
   - Alerte    → messages d'erreur

2. **Type de site** → ton layout global
   - ecommerce_public → navbar fixe + main (PAS de sidebar)
   - dashboard_admin  → sidebar fixe + main
   - vitrine          → navbar + sections pleine largeur
   - hybride          → front public + back-office dans le même fichier

3. **Langues demandées** → sélecteur de langues dans la navbar
4. **Nom du projet** → texte du logo dans la navbar
5. **Produits/services** → tes cards, avec les noms et prix EXACTS du summary

═══════════════════════════════════════════════════════════════
STACK TECHNIQUE OBLIGATOIRE
═══════════════════════════════════════════════════════════════

<script src="https://cdn.tailwindcss.com"></script>
<link href="https://cdn.jsdelivr.net/npm/flowbite@2.5.2/dist/flowbite.min.css" rel="stylesheet"/>
<script src="https://cdn.jsdelivr.net/npm/flowbite@2.5.2/dist/flowbite.min.js"></script>
<script defer src="https://cdn.jsdelivr.net/npm/alpinejs@3.14.1/dist/cdn.min.js"></script>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet"/>

Navigation SPA : toutes vues dans le même fichier, Alpine.js uniquement
(x-show, x-data, @click — JAMAIS de JS vanilla).

═══════════════════════════════════════════════════════════════
STRUCTURE HTML GLOBALE (selon le type de site du summary)
═══════════════════════════════════════════════════════════════

Utilise les hex du summary. Dans les exemples ci-dessous, [DOMINANT],
[ACCENT], [SECONDARY] sont des placeholders — remplace-les par les
hex réels de la palette du summary.

Pour ecommerce_public / vitrine :

<body class="bg-white text-gray-900" x-data="{{ currentView: 'accueil' }}" x-cloak>
  <nav class="fixed top-0 left-0 right-0 h-16 bg-white border-b z-50 flex items-center px-6">
    <div class="flex items-center justify-between w-full max-w-7xl mx-auto">
      <div class="font-bold text-lg" style="color:[DOMINANT]">[Nom du projet — depuis le summary]</div>
      <div class="flex gap-6 text-sm">
        <a href="#" @click.prevent="currentView='accueil'"
           :style="currentView==='accueil' ? 'color:[ACCENT]' : ''">Accueil</a>
        <!-- un lien par vue listée dans le summary -->
      </div>
      <div class="flex gap-3 items-center">
        <!-- SI le summary mentionne des langues : un bouton par langue (FR, AR, EN...) -->
        <!-- SI ecommerce : icône panier -->
      </div>
    </div>
  </nav>
  <main class="pt-16"><!-- vues avec x-show --></main>
  <footer><!-- réseaux sociaux, partenaires — uniquement si mentionnés dans le summary --></footer>
</body>

Pour dashboard_admin / saas_b2b :

<body class="bg-gray-50" x-data="{{ currentView: 'dashboard' }}" x-cloak>
  <aside class="fixed top-0 left-0 h-screen w-60 bg-white border-r overflow-y-auto">
    <div class="p-6 border-b"><div class="text-lg font-semibold" style="color:[DOMINANT]">[Nom du projet]</div></div>
    <nav class="p-4 space-y-1"><!-- un lien Alpine par vue --></nav>
  </aside>
  <main class="ml-60 min-h-screen p-8"><!-- vues avec x-show --></main>
</body>

═══════════════════════════════════════════════════════════════
COMPOSANTS UI (utiliser les hex du summary)
═══════════════════════════════════════════════════════════════

BOUTON PRIMAIRE :
<button style="background:[ACCENT]" class="inline-flex items-center px-4 py-2 text-white text-sm font-medium rounded-lg hover:opacity-90">Label</button>

BOUTON SECONDAIRE :
<button class="inline-flex items-center px-4 py-2 bg-white hover:bg-gray-50 border border-gray-300 text-gray-700 text-sm font-medium rounded-lg">Label</button>

KPI CARD :
<div class="bg-white border border-gray-200 rounded-lg p-6">
  <div class="text-xs font-medium text-gray-500 uppercase tracking-wide">Label</div>
  <div class="mt-2 text-2xl font-semibold text-gray-900">Valeur</div>
  <div class="mt-1 text-xs" style="color:[SECONDARY]">Tendance</div>
</div>

CARD PRODUIT :
<div class="bg-white border rounded-xl overflow-hidden hover:shadow-md">
  <div class="h-44 flex items-center justify-center" style="background:[DOMINANT]; opacity:0.10">
    <span class="text-sm font-medium" style="color:[DOMINANT]">
      Photo : [description contextuelle du visuel attendu, adaptée au produit du CDC]
    </span>
  </div>
  <div class="p-5">
    <h3 class="font-semibold">[Nom EXACT du produit — depuis le summary]</h3>
    <p class="text-sm text-gray-500 mt-1">[Description basée sur le summary]</p>
    <div class="flex items-center justify-between mt-4">
      <span class="text-lg font-bold">[Prix EXACT avec devise du summary]</span>
      <button style="background:[ACCENT]" class="px-4 py-2 text-white text-sm rounded-lg">[Libellé CTA du summary]</button>
    </div>
  </div>
</div>

PLACEHOLDERS IMAGES : jamais de bg-gray-200 vide. Toujours un fond teinté
[DOMINANT] opacity-10 avec un texte descriptif adapté au contexte du CDC.

═══════════════════════════════════════════════════════════════
LAYOUTS PAR TYPE DE VUE
═══════════════════════════════════════════════════════════════

ACCUEIL / LANDING (ecommerce_public, vitrine) :
  Hero pleine largeur (h-[500px] min), fond teinté [DOMINANT] opacity-10
  → h1 text-5xl sm:text-6xl avec titre accrocheur adapté au nom du projet
    et au domaine trouvés dans le summary (PAS de texte générique)
  → sous-titre text-lg text-gray-600
  → 2 boutons CTA côte à côte ([ACCENT] pour le primaire)
  Sections suivantes : celles décrites dans le summary pour cette page.
  Le nombre de sections = celui du summary, pas un minimum imposé.

LISTE / CATALOGUE :
  Barre filtres horizontale + grille cards (grid-cols-3 gap-6).
  Nombre de cards = nombre de produits dans le summary. Si le summary
  dit "3 parcours", affiche 3 cards. Pour les items "à venir" ou
  "prochainement", affiche une card grisée avec badge "Bientôt disponible".

DÉTAIL ITEM :
  grid grid-cols-3 gap-8 : gauche (col-span-2) image + description ;
  droite (col-span-1) card prix + infos pratiques + CTA.

FORMULAIRE :
  Card max-w-2xl mx-auto, grid grid-cols-2 gap-4 pour champs courts,
  champs longs pleine largeur, boutons flex justify-end.

DASHBOARD :
  grid grid-cols-4 gap-4 (KPI cards) puis table avec 6-8 lignes d'exemple.

PANIER :
  grid grid-cols-3 gap-8 : articles col-span-2 (table) + récap col-span-1.

FAQ :
  Accordéons Alpine.js. Nombre de questions = celui du summary ou
  4 minimum si le summary ne précise pas.

═══════════════════════════════════════════════════════════════
DONNÉES D'EXEMPLE — RÈGLE D'OR
═══════════════════════════════════════════════════════════════

Toutes les données visibles dans le HTML doivent refléter le summary :

• Noms de produits/services : ceux du summary, JAMAIS inventés.
  Si le summary dit "3 parcours" sans donner les noms → utilise
  "Parcours 1", "Parcours 2", "Parcours 3" — PAS des noms inventés
  comme "Parcours Médina de Fès".

• Devise : celle du summary (MAD, €, $, etc.)

• Noms de personnes dans les tableaux/témoignages : adaptés au pays
  mentionné dans le summary. Si le summary mentionne le Maroc,
  utilise des noms marocains. Si la France, des noms français. Etc.

• Villes : celles du summary uniquement.

• Tableaux : 6-8 lignes de données réalistes cohérentes avec le domaine.

• JAMAIS de Lorem ipsum, "Titre", "Description", "Produit A".

═══════════════════════════════════════════════════════════════
DENSITÉ ET QUALITÉ
═══════════════════════════════════════════════════════════════

• Chaque vue : minimum 3 sections distinctes, 30+ lignes de HTML
• Formulaires : 4-6 champs si le summary ne précise pas
• Commentaire HTML <!-- Vue: NomVue --> avant chaque section de vue
• Visibilité uniquement via x-show — JAMAIS class="hidden" ou .view CSS
• Le site doit être visuellement IMPRESSIONNANT :
  - Utiliser les couleurs de la palette de façon harmonieuse
  - Varier les teintes : fond teinté dominante pour le hero,
    accent pour les boutons, secondaire pour les badges
  - Ajouter des dégradés légers, des ombres subtiles, des bordures fines
  - Les sections doivent respirer : py-16 ou py-20 entre elles

{feedback_section}

═══════════════════════════════════════════════════════════════
RAPPEL FINAL — LES 3 RÈGLES CRITIQUES
═══════════════════════════════════════════════════════════════

Avant de finaliser le HTML, vérifie ces 3 points :

  ☐ Tous les noms, produits, villes et prix sont ceux du summary.
    Rien d'inventé.

  ☐ Les couleurs utilisées sont les hex de la palette du summary.
    Pas d'autres couleurs.

  ☐ Toutes les fonctionnalités listées dans le summary (langues,
    partenaires, réseaux sociaux, etc.) sont présentes dans le HTML.

---
DESCRIPTION DES PAGES/VUES :
{summary}
---

Génère UNIQUEMENT le code HTML complet, sans explication, sans balises markdown.
Commence directement par <!DOCTYPE html> et termine par </html>.
"""


# ═══════════════════════════════════════════════════════════════
#  Template de la section feedback (injectée au retry uniquement)
# ═══════════════════════════════════════════════════════════════

FEEDBACK_SECTION_TEMPLATE = """
═══════════════════════════════════════════════════════════════
🚨 CORRECTIONS PRIORITAIRES — version {iteration}
═══════════════════════════════════════════════════════════════

Score précédent : {score}/5 — {verdict_label}

Scores par critère :
{criteria_summary}

ISSUES À CORRIGER EN PRIORITÉ (par ordre de sévérité) :
{issues_list}

VUES MANQUANTES dans la version précédente :
{missing_views}

TU DOIS :
1. Corriger CHAQUE issue listée ci-dessus dans ta nouvelle version
2. Ajouter les vues manquantes si elles sont dans le summary
3. Préserver les points forts identifiés : {strengths}
4. Ne PAS régresser sur les critères déjà notés ≥ 4

Traite ces corrections comme des exigences strictes, pas des suggestions.
"""


class CoderAgent(BaseAgent):
    """
    Agent de génération de code.
    Transforme la description structurée du CRAgent
    en un prototype HTML multi-vues navigable.
    """

    def __init__(self):
        super().__init__(name="CoderAgent", temperature=0.2, model="gpt-4o")

    def _build_chain(self):
        """Chain LCEL : prompt → LLM → HTML brut"""
        prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        return prompt | self.llm | StrOutputParser()

    def run(self, state: dict) -> dict:
        """
        Génère le prototype HTML depuis le résumé du CRAgent.
        """
        summary = state.get("summary", "")
        review_feedback = state.get("review_feedback")
        iteration = state.get("iteration", 0) + 1

        if review_feedback:
            self._log(f"Génération corrective (itération {iteration})...")
        else:
            self._log(f"Génération initiale du prototype HTML...")

        if not summary:
            error_msg = "CoderAgent : 'summary' vide, impossible de générer le HTML."
            self._log(f"❌ {error_msg}")
            return {
                **state,
                "html_code": "",
                "iteration": iteration,
                "errors": state.get("errors", []) + [error_msg]
            }

        feedback_section = self._build_feedback_section(review_feedback, iteration)

        try:
            html_code = self.chain.invoke({
                "summary":          summary,
                "feedback_section": feedback_section,
            })
            html_code = self._clean_html(html_code)

            if "</html>" not in html_code.lower():
                self._log("⚠️  HTML potentiellement tronqué — </html> absent")

            self._log(f"✅ HTML généré ({len(html_code)} caractères / {html_code.count(chr(10))} lignes)")

            return {
                **state,
                "html_code": html_code,
                "iteration": iteration,
                "errors": state.get("errors", [])
            }

        except Exception as e:
            error_msg = f"Erreur CoderAgent : {str(e)}"
            self._log(f"❌ {error_msg}")
            return {
                **state,
                "html_code": "",
                "iteration": iteration,
                "errors": state.get("errors", []) + [error_msg]
            }

    # ------------------------------------------------------------------ #
    #  Helpers
    # ------------------------------------------------------------------ #

    def _build_feedback_section(self, feedback: dict | None, iteration: int) -> str:
        """
        Version CLEAN :
        - Ne garde que les informations actionnables
        - Supprime le bruit (scores, justifications, commentaire)
        - Produit un format simple et déterministe pour le LLM
        """
        if not feedback or feedback.get("error"):
            return ""

        issues = feedback.get("issues", [])
        missing = feedback.get("missing_views", [])

        severity_order = {"high": 0, "medium": 1, "low": 2}
        issues_sorted = sorted(
            issues,
            key=lambda i: severity_order.get(i.get("severity", "low"), 3),
        )[:3]

        if not issues_sorted and not missing:
            return """
      MODE CORRECTION :
      Aucune correction nécessaire.
      Reproduis exactement le HTML précédent sans modification.
      """

        issues_lines = []
        for issue in issues_sorted:
            issues_lines.append(
                f"- Vue: {issue.get('vue', '?')}\n"
                f"  Problème: {issue.get('description', '')}\n"
                f"  Correction: {issue.get('suggestion', '')}"
            )

        issues_text = "\n".join(issues_lines) if issues_lines else "Aucune"
        missing_text = ", ".join(missing) if missing else "Aucune"

        return f"""
        MODE CORRECTION (itération {iteration}) :

        PROBLÈMES À CORRIGER :
        {issues_text}

        VUES MANQUANTES :
        {missing_text}

        RÈGLES STRICTES :
        - Corrige UNIQUEMENT les problèmes ci-dessus
        - Ne modifie PAS le reste du HTML
        - Ne supprime PAS de contenu correct
        - Ne réorganise PAS les layouts corrects
        """

    def _clean_html(self, raw: str) -> str:
        """Retire les balises markdown autour du HTML."""
        raw = raw.strip()
        if raw.startswith("```"):
            lines = raw.split("\n")
            raw = "\n".join(lines[1:])
        if raw.endswith("```"):
            raw = raw[:-3].strip()
        return raw