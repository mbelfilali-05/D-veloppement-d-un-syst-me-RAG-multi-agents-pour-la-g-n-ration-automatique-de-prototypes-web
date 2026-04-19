# agents/coder_agent.py

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from agents.base_agent import BaseAgent


# agents/coder_agent.py — NOUVEAU PROMPT_TEMPLATE
# À remplacer dans ton fichier existant.

PROMPT_TEMPLATE = """
Tu es un développeur front-end senior spécialisé dans le prototypage d'applications
web métier (CRM, ERP, dashboards admin, outils SaaS B2B).
 
Ta mission : transformer la description des pages/vues ci-dessous en un prototype HTML
complet, navigable, et visuellement professionnel. Le prototype doit ressembler à une
application réelle de production, pas à un wireframe.
 
═══════════════════════════════════════════════════════════════
STACK TECHNIQUE OBLIGATOIRE
═══════════════════════════════════════════════════════════════
 
Un seul fichier HTML autonome, avec uniquement ces CDN dans le <head> :
 
<script src="https://cdn.tailwindcss.com"></script>
<link href="https://cdn.jsdelivr.net/npm/flowbite@2.5.2/dist/flowbite.min.css" rel="stylesheet"/>
<script src="https://cdn.jsdelivr.net/npm/flowbite@2.5.2/dist/flowbite.min.js"></script>
<script defer src="https://cdn.jsdelivr.net/npm/alpinejs@3.14.1/dist/cdn.min.js"></script>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet"/>
 
Navigation SPA : toutes les vues dans le même fichier, une seule visible à la fois.
Utilise Alpine.js (x-show, x-data, @click) pour la gestion d'état et la navigation.
NE PAS utiliser JS vanilla pour la navigation — Alpine.js uniquement.
 
═══════════════════════════════════════════════════════════════
DESIGN SYSTEM — CLEAN CORPORATE (style Linear / Stripe / Notion)
═══════════════════════════════════════════════════════════════
 
PALETTE (hex fixes — ne pas inventer d'autres couleurs) :
• Fond principal      : #F9FAFB (gris très clair)
• Fond surfaces       : #FFFFFF (blanc)
• Bordures            : #E5E7EB (gris clair)
• Bordures hover      : #D1D5DB (gris moyen clair)
• Texte principal     : #111827 (quasi-noir)
• Texte secondaire    : #6B7280 (gris moyen)
• Texte tertiaire     : #9CA3AF (gris clair)
• Accent primaire     : #4F46E5 (indigo-600)
• Accent hover        : #4338CA (indigo-700)
• Succès              : fond #ECFDF5 / texte #065F46
• Avertissement       : fond #FFFBEB / texte #92400E
• Danger              : fond #FEF2F2 / texte #991B1B
• Info                : fond #EFF6FF / texte #1E40AF
 
TYPOGRAPHIE :
• Font famille        : Inter, via Google Fonts
• H1 page             : text-2xl font-semibold text-gray-900
• H2 section          : text-lg font-semibold text-gray-900
• H3 sous-section     : text-base font-medium text-gray-900
• Corps de texte      : text-sm text-gray-700
• Labels / meta       : text-xs text-gray-500
 
SPACING & FORMES :
• Coins arrondis      : rounded-lg (cards, modals, boutons)
• Bordures            : border border-gray-200 (jamais plus épais que 1px)
• Ombres              : shadow-sm sur cards, shadow-md sur modals, jamais ailleurs
• Padding interne card: p-6
• Gap entre éléments  : space-y-4 (vertical), gap-4 (horizontal)
 
LAYOUT GLOBAL :
• Sidebar gauche fixe : largeur 240px (w-60), fond #FFFFFF, bordure droite #E5E7EB
• Main content        : ml-60, padding p-8, fond #F9FAFB
• Header sidebar      : logo + nom app, padding p-6, bordure basse
• Items navigation    : padding px-4 py-2, rounded-md, hover:bg-gray-100
                        Item actif : bg-indigo-50 text-indigo-700 font-medium
 
═══════════════════════════════════════════════════════════════
COMPOSANTS UI — utilise ces patterns précis
═══════════════════════════════════════════════════════════════
 
BOUTON PRIMAIRE :
<button class="inline-flex items-center px-4 py-2 bg-indigo-600 hover:bg-indigo-700
               text-white text-sm font-medium rounded-md transition-colors">
  Label
</button>
 
BOUTON SECONDAIRE :
<button class="inline-flex items-center px-4 py-2 bg-white hover:bg-gray-50
               border border-gray-300 text-gray-700 text-sm font-medium rounded-md">
  Label
</button>
 
BADGE STATUT :
<span class="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium
             bg-green-50 text-green-700 border border-green-200">Actif</span>
<span class="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium
             bg-amber-50 text-amber-700 border border-amber-200">En attente</span>
<span class="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium
             bg-red-50 text-red-700 border border-red-200">Retard</span>
<span class="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium
             bg-gray-100 text-gray-600 border border-gray-200">Archivé</span>
 
KPI CARD (dashboard) :
<div class="bg-white border border-gray-200 rounded-lg p-6">
  <div class="text-xs font-medium text-gray-500 uppercase tracking-wide">Label</div>
  <div class="mt-2 text-2xl font-semibold text-gray-900">€ 128 450</div>
  <div class="mt-1 text-xs text-green-600">+ 12.4% vs mois dernier</div>
</div>
 
INPUT FORMULAIRE :
<div>
  <label class="block text-sm font-medium text-gray-700 mb-1">Label</label>
  <input type="text"
         class="block w-full rounded-md border border-gray-300 px-3 py-2 text-sm
                focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500"
         placeholder="Valeur d'exemple"/>
</div>
 
TABLE :
<div class="bg-white border border-gray-200 rounded-lg overflow-hidden">
  <table class="min-w-full divide-y divide-gray-200">
    <thead class="bg-gray-50">
      <tr>
        <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Colonne</th>
      </tr>
    </thead>
    <tbody class="divide-y divide-gray-200">
      <tr class="hover:bg-gray-50">
        <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-900">Cellule</td>
      </tr>
    </tbody>
  </table>
</div>
 
═══════════════════════════════════════════════════════════════
PATTERNS DE LAYOUT PAR TYPE DE VUE — OBLIGATOIRES
═══════════════════════════════════════════════════════════════
 
PAGE D'ACCUEIL / LANDING :
→ Hero section : titre h1 large + sous-titre + 2 boutons CTA côte à côte
→ Grille 3 colonnes (grid grid-cols-3 gap-6) : cards produits avec titre, description, prix, bouton
→ Section "Comment ça marche" : 3 étapes HORIZONTALES (grid grid-cols-3) avec numéro + titre + texte
→ Section avis clients : 3 cards côte à côte (grid grid-cols-3 gap-4)
→ Au moins 6 éléments de contenu distincts au total
 
PAGE LISTE / CATALOGUE :
→ Barre de filtres horizontale en haut (flex gap-3) : boutons ou selects de filtre
→ Grille 3 colonnes (grid grid-cols-3 gap-6) : cards avec image placeholder (div bg-gray-200 h-40),
   titre, sous-titre, badge ville/statut, prix, bouton "Voir détail"
→ Au moins 6 cards dans la grille
 
PAGE DÉTAIL ITEM :
→ Layout 2 colonnes : div class="grid grid-cols-3 gap-8"
   - Colonne gauche (col-span-2) : image placeholder grande, description longue, galerie 3 thumbnails
   - Colonne droite (col-span-1) : card sticky avec prix, infos clés, bouton CTA principal
→ Section infos pratiques en grille 2x2 sous le contenu principal
 
PAGE FORMULAIRE (inscription, contact, création) :
→ Card centrée (max-w-2xl mx-auto) avec header titre + sous-titre
→ Grille 2 colonnes pour champs courts (grid grid-cols-2 gap-4) : prénom/nom, ville/code postal
→ Champs longs sur toute la largeur : email, message, description, textarea
→ Boutons d'action en bas à droite (flex justify-end gap-3)
 
DASHBOARD / RAPPORTS :
→ Grille 4 KPI cards OBLIGATOIRE en haut (grid grid-cols-4 gap-4)
→ Layout 2 colonnes sous les KPI (grid grid-cols-3 gap-6) :
   - Tableau principal (col-span-2) avec 6-8 lignes et badges colorés
   - Sidebar stats (col-span-1) : liste de métriques ou activité récente
→ Si rapport : section graphique placeholder (div bg-gray-100 rounded-lg h-64 flex items-center
   justify-center text-gray-400 text-sm) avec texte "Graphique des ventes"
 
PAGE PANIER / CHECKOUT :
→ Layout 2 colonnes (grid grid-cols-3 gap-8) :
   - Liste articles (col-span-2) : table avec colonnes article/quantité/prix/total + bouton supprimer
   - Récapitulatif (col-span-1) : card avec sous-total, taxes, total TTC en gras, bouton paiement
 
PAGE COMPTE / PROFIL :
→ Layout 2 colonnes (grid grid-cols-4 gap-6) :
   - Menu latéral interne (col-span-1) : liens Profil / Commandes / Sécurité
   - Contenu (col-span-3) : section active avec formulaire ou tableau historique
→ Tableau historique commandes : 6 lignes avec date, référence, montant, statut badge
 
PAGE FAQ :
→ Header avec barre de recherche (input pleine largeur)
→ 2 colonnes de catégories (grid grid-cols-2 gap-6)
→ Accordéon dans chaque colonne : au moins 4 questions par catégorie
   Utiliser Alpine.js : x-data="{{open: false}}" @click="open=!open" x-show="open"
 
═══════════════════════════════════════════════════════════════
DONNÉES D'EXEMPLE — CRUCIAL pour la crédibilité
═══════════════════════════════════════════════════════════════
 
• Noms français réalistes : Dubois, Rousseau, Alaoui, Benali, Leroy, Martin, Tazi, Chraibi
• Emails cohérents : m.dubois@entreprise.fr, s.benali@acme.ma
• Montants en euros avec espaces : € 4 280, € 18 900, € 1 240
• Dates françaises : 16 avril 2026, 03/04/2026
• Au moins 6-8 lignes dans chaque tableau — jamais 2-3 lignes
• Données adaptées au DOMAINE du CDC (escape-game → parcours, scores, villes marocaines)
 
═══════════════════════════════════════════════════════════════
EXEMPLE COMPLET — VUE DASHBOARD DE RÉFÉRENCE
═══════════════════════════════════════════════════════════════
 
Voici la densité et le niveau de détail attendus pour chaque vue.
 
<div id="view-dashboard" x-show="currentView === 'dashboard'">
 
  <div class="flex items-center justify-between mb-6">
    <div>
      <h1 class="text-2xl font-semibold text-gray-900">Tableau de bord</h1>
      <p class="mt-1 text-sm text-gray-500">Vue d'ensemble de l'activité</p>
    </div>
    <div class="flex gap-2">
      <button class="inline-flex items-center px-4 py-2 bg-white hover:bg-gray-50
                     border border-gray-300 text-gray-700 text-sm font-medium rounded-md">
        Exporter
      </button>
      <button class="inline-flex items-center px-4 py-2 bg-indigo-600 hover:bg-indigo-700
                     text-white text-sm font-medium rounded-md">
        + Nouvelle commande
      </button>
    </div>
  </div>
 
  <div class="grid grid-cols-4 gap-4 mb-6">
    <div class="bg-white border border-gray-200 rounded-lg p-6">
      <div class="text-xs font-medium text-gray-500 uppercase tracking-wide">CA du mois</div>
      <div class="mt-2 text-2xl font-semibold text-gray-900">€ 128 450</div>
      <div class="mt-1 text-xs text-green-600">+ 12.4% vs mois dernier</div>
    </div>
    <div class="bg-white border border-gray-200 rounded-lg p-6">
      <div class="text-xs font-medium text-gray-500 uppercase tracking-wide">Réservations</div>
      <div class="mt-2 text-2xl font-semibold text-gray-900">342</div>
      <div class="mt-1 text-xs text-green-600">+ 8 cette semaine</div>
    </div>
    <div class="bg-white border border-gray-200 rounded-lg p-6">
      <div class="text-xs font-medium text-gray-500 uppercase tracking-wide">Taux conversion</div>
      <div class="mt-2 text-2xl font-semibold text-gray-900">68.2 %</div>
      <div class="mt-1 text-xs text-red-600">− 2.1 points</div>
    </div>
    <div class="bg-white border border-gray-200 rounded-lg p-6">
      <div class="text-xs font-medium text-gray-500 uppercase tracking-wide">Nouveaux clients</div>
      <div class="mt-2 text-2xl font-semibold text-gray-900">47</div>
      <div class="mt-1 text-xs text-gray-500">sur 30 jours</div>
    </div>
  </div>
 
  <div class="grid grid-cols-3 gap-6">
    <div class="col-span-2 bg-white border border-gray-200 rounded-lg overflow-hidden">
      <div class="px-6 py-4 border-b border-gray-200 flex items-center justify-between">
        <h2 class="text-lg font-semibold text-gray-900">Commandes récentes</h2>
        <a href="#" class="text-sm text-indigo-600 hover:text-indigo-700">Voir tout →</a>
      </div>
      <table class="min-w-full divide-y divide-gray-200">
        <thead class="bg-gray-50">
          <tr>
            <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Référence</th>
            <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Client</th>
            <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Montant</th>
            <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Statut</th>
          </tr>
        </thead>
        <tbody class="divide-y divide-gray-200">
          <tr class="hover:bg-gray-50">
            <td class="px-6 py-4 text-sm font-medium text-gray-900">#CMD-2026-0142</td>
            <td class="px-6 py-4 text-sm text-gray-700">Alaoui Karim</td>
            <td class="px-6 py-4 text-sm text-gray-900">€ 4 280</td>
            <td class="px-6 py-4"><span class="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-green-50 text-green-700 border border-green-200">Payé</span></td>
          </tr>
          <tr class="hover:bg-gray-50">
            <td class="px-6 py-4 text-sm font-medium text-gray-900">#CMD-2026-0141</td>
            <td class="px-6 py-4 text-sm text-gray-700">Benali Sofia</td>
            <td class="px-6 py-4 text-sm text-gray-900">€ 18 900</td>
            <td class="px-6 py-4"><span class="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-amber-50 text-amber-700 border border-amber-200">En attente</span></td>
          </tr>
          <tr class="hover:bg-gray-50">
            <td class="px-6 py-4 text-sm font-medium text-gray-900">#CMD-2026-0140</td>
            <td class="px-6 py-4 text-sm text-gray-700">Rousseau Marie</td>
            <td class="px-6 py-4 text-sm text-gray-900">€ 2 180</td>
            <td class="px-6 py-4"><span class="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-red-50 text-red-700 border border-red-200">Retard</span></td>
          </tr>
        </tbody>
      </table>
    </div>
    <div class="col-span-1 bg-white border border-gray-200 rounded-lg p-6">
      <h2 class="text-lg font-semibold text-gray-900 mb-4">Activité récente</h2>
      <div class="space-y-4">
        <div class="flex items-start gap-3">
          <div class="w-2 h-2 rounded-full bg-green-500 mt-1.5 flex-shrink-0"></div>
          <div><p class="text-sm text-gray-900">Nouvelle réservation Rabat</p><p class="text-xs text-gray-500">Il y a 5 min</p></div>
        </div>
        <div class="flex items-start gap-3">
          <div class="w-2 h-2 rounded-full bg-indigo-500 mt-1.5 flex-shrink-0"></div>
          <div><p class="text-sm text-gray-900">Paiement confirmé #0141</p><p class="text-xs text-gray-500">Il y a 23 min</p></div>
        </div>
        <div class="flex items-start gap-3">
          <div class="w-2 h-2 rounded-full bg-amber-500 mt-1.5 flex-shrink-0"></div>
          <div><p class="text-sm text-gray-900">Nouveau client inscrit</p><p class="text-xs text-gray-500">Il y a 1h</p></div>
        </div>
      </div>
    </div>
  </div>
 
</div>
 
═══════════════════════════════════════════════════════════════
STRUCTURE HTML GLOBALE ATTENDUE
═══════════════════════════════════════════════════════════════
 
<!DOCTYPE html>
<html lang="fr">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>[Nom de l'application]</title>
  <!-- Les 5 CDN listés plus haut -->
  <style>
    body {{ font-family: 'Inter', system-ui, sans-serif; }}
    [x-cloak] {{ display: none !important; }}
  </style>
</head>
<body class="bg-gray-50 text-gray-900" x-data="{{ currentView: 'dashboard' }}" x-cloak>
 
  <!-- SIDEBAR fixe gauche -->
  <aside class="fixed top-0 left-0 h-screen w-60 bg-white border-r border-gray-200 overflow-y-auto">
    <div class="p-6 border-b border-gray-200">
      <div class="text-lg font-semibold text-gray-900">[Nom app]</div>
    </div>
    <nav class="p-4 space-y-1">
      <a href="#" @click.prevent="currentView='dashboard'"
         :class="currentView==='dashboard' ? 'bg-indigo-50 text-indigo-700 font-medium' : 'text-gray-700 hover:bg-gray-100'"
         class="block px-4 py-2 rounded-md text-sm">Tableau de bord</a>
      <!-- Un lien Alpine par vue — MÊME PATTERN -->
    </nav>
  </aside>
 
  <!-- MAIN content — décalé de la largeur sidebar -->
  <main class="ml-60 min-h-screen p-8">
    <!-- TOUTES les vues ici, chacune avec x-show -->
    <!-- NE PAS ajouter de classe CSS pour cacher les vues — x-show s'en charge -->
  </main>
 
</body>
</html>
 
═══════════════════════════════════════════════════════════════
RÈGLES FINALES
═══════════════════════════════════════════════════════════════
 
1. Génère TOUTES les pages identifiées dans la description, aucune omission.
2. Chaque vue respecte le pattern de layout défini ci-dessus selon son type.
3. CRITIQUE — visibilité des vues : utilise UNIQUEMENT x-show="currentView === 'nom'"
   NE PAS ajouter class="view" ou tout autre classe CSS qui cache les éléments.
   x-show est le seul mécanisme de visibilité — ne jamais le doubler avec du CSS.
4. Dans les formulaires, pré-remplis les champs avec des valeurs d'exemple réalistes.
5. Ajoute des commentaires HTML <!-- Vue: NomVue --> avant chaque section de vue.
6. Si une fonctionnalité est floue dans le CDC, propose une implémentation standard cohérente.
7. Ne jamais utiliser : emojis décoratifs, gradients criards, ombres lourdes,
   couleurs hors palette, polices autres que Inter.
 
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