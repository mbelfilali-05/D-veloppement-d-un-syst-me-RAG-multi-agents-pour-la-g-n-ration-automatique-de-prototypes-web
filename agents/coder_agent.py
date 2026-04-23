# agents/coder_agent.py

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from agents.base_agent import BaseAgent


# agents/coder_agent.py — NOUVEAU PROMPT_TEMPLATE
# À remplacer dans ton fichier existant.
# agents/coder_agent.py — PROMPT_TEMPLATE RESTRUCTURÉ V3
# Principes :
#   1. Règles CRITIQUES au début ET à la fin (contournement du "Lost in the Middle")
#   2. Extraction CDC obligatoire en Étape 0 (force le LLM à "montrer son travail")
#   3. Priorité des couleurs explicite (charte CDC > domaine > indigo)
#   4. Compression des sections détaillées sans perte d'info utile
#   5. Rappel compact des 3 règles vitales juste avant {summary}

PROMPT_TEMPLATE = """
Tu es un développeur front-end senior expert en prototypage web rapide.

═══════════════════════════════════════════════════════════════
3 RÈGLES CRITIQUES (non négociables)
═══════════════════════════════════════════════════════════════

1. FIDÉLITÉ ABSOLUE AU CDC :
   Noms, couleurs, langues, produits, villes, devises, fonctionnalités
   → tels qu'ils apparaissent dans le CDC, JAMAIS inventés ni remplacés.

2. CHARTE GRAPHIQUE DU CDC EN PRIORITÉ :
   Si le CDC mentionne des couleurs, motifs, typographies → UTILISE-LES.
   Même si elles paraissent criardes ou incohérentes — c'est la volonté
   du client, pas un choix à débattre.

3. FONCTIONNALITÉS EXPLICITES JAMAIS OMISES :
   Sélecteur de langues, partenaires, réseaux sociaux, newsletter,
   témoignages, boutons CTA spécifiques — si mentionnés dans le CDC,
   ils DOIVENT apparaître dans le HTML final.

═══════════════════════════════════════════════════════════════
ÉTAPE 0 — EXTRACTION DU CDC (obligatoire avant de coder)
═══════════════════════════════════════════════════════════════

Avant toute génération, extrais les éléments du CDC et liste-les dans
un commentaire HTML placé juste après <body>. Format strict :

<!--
CDC_EXTRACTED:
  organisation : [nom exact du CDC]
  projet       : [nom du projet/produit]
  domaine      : [escape-game, santé, ecommerce, etc.]
  localisation : [villes/pays mentionnés]
  devise       : [MAD, €, $, DH, etc.]
  langues      : [FR, AR, EN, ES — celles du CDC uniquement]
  produits     : [noms exacts des produits/services/parcours]
  charte       : [couleurs, motifs, typographies demandés explicitement]
  features     : [fonctionnalités listées dans le CDC]
  type_site    : [ecommerce_public | dashboard_admin | vitrine | saas_b2b | hybride]
-->

Cette extraction N'EST PAS optionnelle. Tu l'utilises comme source
unique de vérité pour tous les contenus du HTML.

═══════════════════════════════════════════════════════════════
ÉTAPE 1 — DÉTECTION DU TYPE DE SITE
═══════════════════════════════════════════════════════════════

À partir du champ type_site extrait, applique le layout correspondant :

  ecommerce_public → navbar fixe h-16 + main pt-16 (PAS de sidebar)
                     "acheter", "réserver", "panier" → ce type
  dashboard_admin  → sidebar fixe w-60 + main ml-60 p-8
                     "gestion", "back-office", "admin" → ce type
  vitrine          → navbar + sections pleine largeur (comme ecommerce)
                     "association", "présentation" → ce type
  saas_b2b         → sidebar comme dashboard_admin
  hybride          → les deux dans le même fichier (front + back)

═══════════════════════════════════════════════════════════════
ÉTAPE 2 — PALETTE (ordre de priorité strict)
═══════════════════════════════════════════════════════════════

PRIORITÉ 1 — Charte graphique du CDC :
  Le CDC mentionne des couleurs ? → UTILISE-LES. Point.
  Exemple AMPD : "nuances de bleu, orange, jaune, vert et rouge inspirées
  de l'architecture marocaine" → crée une palette avec ces 5 couleurs,
  choisis une couleur accent principale parmi elles.

PRIORITÉ 2 — Déduction par domaine (si CDC muet sur les couleurs) :

  Escape game / aventure    → #E85D04 (orange)   sombre, mystère
  Patrimoine / culture      → #B45309 (ambre)    chaleureux
  Association / éducation   → #0284C7 (bleu)     clair, confiance
  Santé / médical           → #0891B2 (cyan)     propre
  Environnement / nature    → #16A34A (vert)     naturel
  Luxe / premium            → #1C1917 (noir)     sobre
  Enfants / ludique         → #7C3AED (violet)   joyeux
  Finance / SaaS B2B        → #4F46E5 (indigo)   sérieux
  Restauration / food       → #DC2626 (rouge)    appétissant
  Sport / fitness           → #EA580C (orange)   énergie

PRIORITÉ 3 — Indigo par défaut (uniquement si 1 et 2 ne s'appliquent pas).

Couleurs neutres partagées (quelque soit l'accent) :
  Texte       : #111827 principal, #6B7280 secondaire, #9CA3AF tertiaire
  Bordures    : #E5E7EB
  Surfaces    : #FFFFFF
  Succès      : fond #ECFDF5 / texte #065F46
  Danger      : fond #FEF2F2 / texte #991B1B
  Avertiss.   : fond #FFFBEB / texte #92400E

Remplace [ACCENT] partout dans le code par la couleur hex retenue.

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
STRUCTURE HTML GLOBALE (selon type_site)
═══════════════════════════════════════════════════════════════

Pour ecommerce_public / vitrine :

<body class="bg-white text-gray-900" x-data="{{ currentView: 'accueil' }}" x-cloak>
  <!-- CDC_EXTRACTED ci-dessus -->
  <nav class="fixed top-0 left-0 right-0 h-16 bg-white border-b z-50 flex items-center px-6">
    <div class="flex items-center justify-between w-full max-w-7xl mx-auto">
      <div class="font-bold text-lg" style="color:[ACCENT]">[Nom exact du CDC]</div>
      <div class="flex gap-6 text-sm">
        <a href="#" @click.prevent="currentView='accueil'"
           :style="currentView==='accueil' ? 'color:[ACCENT]' : ''">Accueil</a>
        <!-- un lien par vue -->
      </div>
      <div class="flex gap-3 items-center">
        <!-- SI langues dans le CDC : boutons FR / AR / EN / ES -->
        <!-- SI ecommerce : icône panier + compte -->
      </div>
    </div>
  </nav>
  <main class="pt-16"><!-- vues avec x-show --></main>
  <footer><!-- SI réseaux sociaux/partenaires dans le CDC --></footer>
</body>

Pour dashboard_admin / saas_b2b :

<body class="bg-gray-50" x-data="{{ currentView: 'dashboard' }}" x-cloak>
  <aside class="fixed top-0 left-0 h-screen w-60 bg-white border-r overflow-y-auto">
    <div class="p-6 border-b"><div class="text-lg font-semibold" style="color:[ACCENT]">[Nom CDC]</div></div>
    <nav class="p-4 space-y-1"><!-- un lien Alpine par vue --></nav>
  </aside>
  <main class="ml-60 min-h-screen p-8"><!-- vues avec x-show --></main>
</body>

═══════════════════════════════════════════════════════════════
COMPOSANTS UI (adapter [ACCENT])
═══════════════════════════════════════════════════════════════

BOUTON PRIMAIRE :
<button style="background:[ACCENT]" class="inline-flex items-center px-4 py-2 text-white text-sm font-medium rounded-lg hover:opacity-90">Label</button>

BOUTON SECONDAIRE :
<button class="inline-flex items-center px-4 py-2 bg-white hover:bg-gray-50 border border-gray-300 text-gray-700 text-sm font-medium rounded-lg">Label</button>

BADGES STATUT (fond léger + texte foncé + bordure) :
• Succès   : bg-green-50 text-green-700 border-green-200
• Attente  : bg-amber-50 text-amber-700 border-amber-200
• Annulé   : bg-red-50 text-red-700 border-red-200

KPI CARD :
<div class="bg-white border border-gray-200 rounded-lg p-6">
  <div class="text-xs font-medium text-gray-500 uppercase tracking-wide">Label</div>
  <div class="mt-2 text-2xl font-semibold text-gray-900">Valeur</div>
  <div class="mt-1 text-xs text-green-600">Tendance</div>
</div>

INPUT : label text-sm font-medium + input rounded-lg border px-3 py-2 focus:ring-2
       avec style="--tw-ring-color:[ACCENT]"

TABLE : thead bg-gray-50 / tbody divide-y / tr hover:bg-gray-50

CARD PRODUIT (ecommerce) :
<div class="bg-white border rounded-xl overflow-hidden hover:shadow-md">
  <div class="h-44 flex items-center justify-center" style="background:[ACCENT]; opacity:0.12">
    <span class="text-sm font-medium" style="color:[ACCENT]">
      Photo : [description contextuelle du visuel attendu]
    </span>
  </div>
  <div class="p-5">
    <h3 class="font-semibold">[Nom exact du produit du CDC]</h3>
    <p class="text-sm text-gray-500 mt-1">[Description basée sur le CDC]</p>
    <div class="flex items-center justify-between mt-4">
      <span class="text-lg font-bold">[Prix avec devise du CDC]</span>
      <button style="background:[ACCENT]" class="px-4 py-2 text-white text-sm rounded-lg">Réserver</button>
    </div>
  </div>
</div>

PLACEHOLDERS IMAGES : jamais bg-gray-200 vide. Toujours un fond teinté
[ACCENT] opacity-12 avec un texte décrivant l'image attendue
("Photo : Kasbah des Oudayas au coucher du soleil").

═══════════════════════════════════════════════════════════════
LAYOUTS PAR TYPE DE VUE (patterns obligatoires)
═══════════════════════════════════════════════════════════════

ACCUEIL / LANDING :
  Hero pleine largeur (h-[500px] min), fond teinté [ACCENT] opacity-10
  → h1 text-5xl sm:text-6xl avec titre ÉMOTIONNEL adapté au domaine
  → sous-titre text-lg text-gray-600
  → 2 boutons CTA côte à côte
  Puis minimum 4 sections : produits (grid-cols-3), "Comment ça marche"
  (3 étapes horizontales numérotées), témoignages (3 cards), et selon CDC :
  partenaires, newsletter, CTA final.

LISTE / CATALOGUE :
  Barre filtres horizontale (flex gap-3) + grille cards (grid-cols-3 gap-6),
  minimum 6 cards produit avec placeholder contextuel.

DÉTAIL ITEM :
  grid grid-cols-3 gap-8 : gauche (col-span-2) image + description +
  galerie 3 thumbnails ; droite (col-span-1) card prix + infos + CTA.

FORMULAIRE :
  Card max-w-2xl mx-auto, grid grid-cols-2 gap-4 pour champs courts,
  champs longs pleine largeur, boutons flex justify-end.

DASHBOARD :
  grid grid-cols-4 gap-4 (4 KPI cards obligatoires) puis grid grid-cols-3
  gap-6 (table col-span-2 avec 6-8 lignes + sidebar col-span-1).

PANIER :
  grid grid-cols-3 gap-8 : articles col-span-2 (table) + récap col-span-1.

COMPTE :
  grid grid-cols-4 gap-6 : menu col-span-1 + contenu col-span-3.

FAQ :
  Barre de recherche + grid grid-cols-2 accordéons Alpine.js,
  4 questions minimum par catégorie.

═══════════════════════════════════════════════════════════════
TON VISUEL SELON LE TYPE DE SITE
═══════════════════════════════════════════════════════════════

ecommerce_public / vitrine :
  ÉMOTIONNEL et AÉRÉ. Hero grand format, titres chaleureux et évocateurs
  ("Vivez une aventure inoubliable à Rabat" PAS "Page d'accueil"),
  sections py-20, au moins 6 sections distinctes sur l'accueil.

dashboard_admin / saas_b2b :
  DENSE et FONCTIONNEL. Pas de hero, titres sobres ("Tableau de bord"),
  h1 text-2xl suffit, priorité à l'information.

═══════════════════════════════════════════════════════════════
DONNÉES D'EXEMPLE — FIDÉLITÉ AU CDC
═══════════════════════════════════════════════════════════════

Extrait les données du CDC (Étape 0) et utilise-les telles quelles :
• Produits/services : les noms exacts (PAS "Produit A / Parcours Mystère")
• Devise : celle du CDC (MAD, DH, €, $... jamais € par défaut)
• Personnes : prénoms/noms adaptés au pays du CDC
  (Maroc → Alaoui, Benali, Tazi, Chraibi — PAS Dubois, Rousseau)
• Villes : celles du CDC
• Tableaux : 6-8 lignes minimum (jamais 2-3)
• Formulaires : 4-6 champs pré-remplis avec valeurs réalistes du domaine

Exemples de BONNE adaptation par domaine :
  Escape game Rabat  → "Parcours Kasbah des Oudayas — 150 MAD —
                        Samedi 20 avril, 14h — Alaoui Karim, 4 joueurs"
  Clinique Maroc     → "Dr. Benali — Consultation cardio — 350 MAD —
                        14/04/2026 — Tazi Sara"
  Association sport  → "Tournoi Casablanca — Équipe Atlas —
                        12 joueurs — 05/05/2026"

═══════════════════════════════════════════════════════════════
DENSITÉ ET QUALITÉ (non négociables)
═══════════════════════════════════════════════════════════════

• Chaque vue : minimum 3 sections distinctes, 30+ lignes de HTML
• Tableaux : 6-8 lignes
• Formulaires : 4-6 champs
• Accueil (ecommerce/vitrine) : 6+ sections
• JAMAIS de Lorem ipsum, "Titre", "Description", "Produit 1"
• Commentaire HTML <!-- Vue: NomVue --> avant chaque section de vue
• Visibilité uniquement via x-show — JAMAIS class="hidden" ou .view CSS


{feedback_section}

═══════════════════════════════════════════════════════════════
RAPPEL FINAL — LES 3 RÈGLES CRITIQUES
═══════════════════════════════════════════════════════════════

Avant de finaliser le HTML, vérifie ces 3 points :

  ☐ Tous les noms (app, produits, personnes, villes) sont ceux
    extraits du CDC dans l'Étape 0 — rien d'inventé.

  ☐ La palette suit la priorité : charte du CDC > domaine > indigo.
    Si le CDC mentionne des couleurs, elles DOIVENT apparaître.

  ☐ Toutes les fonctionnalités listées dans le CDC (langues, partenaires,
    réseaux sociaux, etc.) sont présentes dans le HTML final.

Si l'une de ces 3 règles n'est pas respectée, le prototype est un échec.

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
🚨 CORRECTIONS PRIORITAIRES — version {iteration} (précédente version jugée insuffisante)
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

    Depuis l'Axe 2 : peut recevoir un feedback du ReviewerAgent
    pour produire une version corrective (retry du cycle).
    """

    def __init__(self):
        super().__init__(name="CoderAgent", temperature=0.2,model="gpt-4o")

    def _build_chain(self):
        """Chain LCEL : prompt → LLM → HTML brut"""
        prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        return prompt | self.llm | StrOutputParser()

    def run(self, state: dict) -> dict:
        """
        Génère le prototype HTML depuis le résumé du CRAgent.

        Si state contient 'review_feedback' (retry du cycle), la section
        feedback est injectée dans le prompt pour guider les corrections.

        Args:
            state: AgentState — doit contenir 'summary'
                   Optionnel : 'review_feedback' (pour un retry corrigé)

        Returns:
            dict: AgentState avec 'html_code' mis à jour,
                  et 'iteration' incrémenté.
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

        # Construction de la section feedback (vide si premier passage)
        feedback_section = self._build_feedback_section(review_feedback, iteration)

        try:
            html_code = self.chain.invoke({
                "summary":          summary,
                "feedback_section": feedback_section,
            })
            html_code = self._clean_html(html_code)

            # Détection de troncature
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

        # Limiter à 3 issues max (priorité high > medium > low)
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
        """
        Retire les balises markdown que le LLM peut parfois ajouter
        autour du HTML (```html ... ```).
        """
        raw = raw.strip()
        if raw.startswith("```"):
            lines = raw.split("\n")
            raw = "\n".join(lines[1:])
        if raw.endswith("```"):
            raw = raw[:-3].strip()
        return raw