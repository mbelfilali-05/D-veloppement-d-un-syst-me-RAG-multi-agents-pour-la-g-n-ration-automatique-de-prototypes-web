# experiments/configs.py
#
# Ajoute ici de nouvelles RAGConfig sans jamais toucher à cr_agent.py.
# Chaque config = 1 ligne dans le rapport final.

from experiments.models import RAGConfig


# ─────────────────────────────────────────────────────────────────────────────
#  PROMPT TEMPLATES
# ─────────────────────────────────────────────────────────────────────────────

# V1 — baseline : liste à puces, format libre
PROMPT_V1 = """
Tu es un expert en analyse de cahiers des charges et en conception d'interfaces web.

À partir des extraits du cahier des charges fournis ci-dessous, ta mission est d'identifier
et de décrire toutes les pages/vues de l'application web à prototyper.

Pour CHAQUE page/vue identifiée, fournis une description structurée avec :
- Nom de la page (ex: "Page d'accueil", "Dashboard", "Formulaire de contact")
- Objectif principal de la page
- Composants UI présents (navbar, formulaires, tableaux, boutons, cartes, etc.)
- Données affichées ou saisies
- Actions utilisateur possibles (clic, soumission, navigation, etc.)
- Liens/navigation vers d'autres pages

Sois précis et exhaustif. Si une information n'est pas mentionnée dans le cahier des charges,
indique-le clairement plutôt que d'inventer.

---
EXTRAITS DU CAHIER DES CHARGES :
{context}
---

DESCRIPTION STRUCTURÉE DES PAGES/VUES :
"""

# V2 — format ## strict, sans exemple, sortie parseable par CoderAgent
PROMPT_V2 = """
Tu es un architecte front-end senior. Analyse le cahier des charges ci-dessous.

Produis une liste structurée de TOUTES les vues/pages à prototyper.
Respecte ce format strict pour chaque vue sans exception :

## [Numéro]. [Nom de la vue]
**Rôle** : (1 phrase — à quoi sert cette page)
**Composants** : liste précise des éléments UI (navbar, tableau, formulaire, modal, bouton, etc.)
**Données affichées** : ce qui est visible à l'écran sans interaction
**Données saisies** : champs, filtres, recherche
**Actions** : ce que l'utilisateur peut faire
**Navigation** : vers quelles autres pages et dans quelles conditions

Règles :
- N'invente AUCUN élément absent du CDC.
- Si une information manque, écris [Non spécifié].
- Sois précis : "tableau colonnes Nom | Date | Statut" vaut mieux que "un tableau".

---
EXTRAITS DU CAHIER DES CHARGES :
{context}
---

LISTE DES VUES :
"""

# V4 — format ## avec exemple login, règles strictes
PROMPT_V4 = """
Tu es un architecte front-end senior spécialisé dans l'analyse de cahiers des charges
pour applications web métier. Ta réponse sera utilisée directement par un agent de
génération de code pour produire un prototype HTML interactif — chaque composant que
tu identifies sera traduit en code concret.

CONTEXTE :
Les extraits ci-dessous proviennent d'un cahier des charges (CDC). Ils peuvent être
incomplets ou mal ordonnés. Ton rôle est d'en extraire une description structurée
et exploitable de toutes les pages/vues de l'application.

FORMAT DE SORTIE OBLIGATOIRE — respecte-le pour chaque page sans exception :

## [N°]. [Nom de la page]
**URL suggérée** : /chemin
**Objectif** : (1 phrase — à quoi sert cette page)
**Composants UI** : liste précise (ex: navbar, tableau triable, formulaire 3 champs,
                    bouton primaire "Valider", modal de confirmation...)
**Données affichées** : ce qui est visible à l'écran
**Données saisies** : champs de formulaire, filtres, recherche...
**Actions utilisateur** : ce que l'utilisateur peut faire (clic, soumission, export...)
**Navigation** : vers quelles autres pages + conditions (ex: → Dashboard si succès)

EXEMPLE DE SORTIE ATTENDUE :

## 1. Page de connexion
**URL suggérée** : /login
**Objectif** : Authentifier l'utilisateur avant accès à l'application
**Composants UI** : logo centré, formulaire (2 champs), bouton "Se connecter",
                    lien "Mot de passe oublié"
**Données affichées** : message d'erreur si échec
**Données saisies** : email (texte), mot de passe (masqué)
**Actions utilisateur** : soumettre le formulaire, cliquer sur lien reset
**Navigation** : → Dashboard (authentification réussie),
                 → Réinitialisation mot de passe

RÈGLES STRICTES :
1. N'invente AUCUNE page absente du CDC — si tu n'es pas sûr, ne l'inclus pas
2. Si une information est absente du CDC, écris : [Non spécifié]
3. Sois précis sur les composants : "tableau avec colonnes Nom/Date/Statut" vaut
   mieux que "un tableau de données"
4. Si une fonctionnalité est floue, ajoute [À PRÉCISER] et propose 2 interprétations

---
EXTRAITS DU CAHIER DES CHARGES :
{context}
---

LISTE COMPLÈTE DES PAGES/VUES :
"""

# V5 — meilleur prompt : éléments globaux + 2 exemples + colonnes + boutons + rôles
PROMPT_V5 = """
Tu es un architecte front-end senior. Ta sortie sera transmise directement à un générateur
de code HTML. Chaque composant que tu décris sera traduit en code concret — sois précis
sur les libellés, les colonnes, les champs, les boutons.

MISSION : Analyse les extraits du cahier des charges ci-dessous et produis une description
complète de toutes les pages/vues de l'application.

═══════════════════════════════════════════
ÉTAPE 1 — ÉLÉMENTS GLOBAUX (sidebar/navbar partagés)
═══════════════════════════════════════════
Identifie d'abord les éléments présents sur toutes les pages :
**Navigation globale** : [liste des entrées de menu et leurs destinations]
**Rôles utilisateur** : [admin / manager / agent / etc. — selon le CDC]

═══════════════════════════════════════════
ÉTAPE 2 — LISTE DES PAGES (une section par page)
═══════════════════════════════════════════
Format obligatoire pour chaque page :

## [N°]. [Nom de la page]
**Objectif** : (1 phrase)
**Composants UI** :
  - Formulaires : [champ1 (type, ex: email texte obligatoire), champ2, ...]
  - Tableaux : [colonnes : Nom | Statut | Date | Actions]
  - Boutons : ["Valider" (primaire), "Annuler" (secondaire), ...]
  - Autres : [cartes, modals, filtres, graphiques, badges, etc.]
**Données affichées** : [ce qui est visible sans interaction]
**Données saisies** : [champs + types + contraintes si mentionnées]
**Actions utilisateur** : [clic sur X → Y, soumission → Z, etc.]
**Navigation** : [→ /page-cible SI condition, → /autre SINON]
**Visible pour** : [tous / admin seulement / etc. — si mentionné]

EXEMPLES DE RÉFÉRENCE (ne pas reproduire, format uniquement) :

## 1. Tableau de bord
**Objectif** : Vue d'ensemble des indicateurs clés pour le manager
**Composants UI** :
  - Tableaux : Dernières commandes — colonnes : N° Commande | Client | Montant | Statut | Date
  - Boutons : "Voir détail" (lien par ligne), "Exporter CSV" (secondaire)
  - Autres : 3 cartes KPI (Commandes du jour / CA mensuel / Taux d'annulation)
**Données affichées** : KPIs temps réel, 10 dernières commandes
**Données saisies** : Filtre date (date picker : début / fin)
**Actions utilisateur** : Cliquer sur une ligne → Page Détail Commande, Exporter
**Navigation** : → /commande/:id (clic ligne), → /commandes (lien "Voir tout")
**Visible pour** : manager, admin

## 2. Formulaire de création de commande
**Objectif** : Saisir une nouvelle commande
**Composants UI** :
  - Formulaires : Client (select, obligatoire), Produit (select), Quantité (number, min=1),
                  Date livraison souhaitée (datepicker), Notes (textarea, optionnel)
  - Boutons : "Créer la commande" (primaire), "Annuler" (secondaire)
**Données affichées** : Prix unitaire (calculé selon produit sélectionné)
**Données saisies** : voir Formulaires ci-dessus
**Actions utilisateur** : soumettre → création + redirection, annuler → retour liste
**Navigation** : → /commandes (succès ou annulation)
**Visible pour** : agent, manager

═══════════════════════════════════════════
RÈGLES STRICTES
═══════════════════════════════════════════
1. N'invente AUCUNE page, fonctionnalité ou champ absent du CDC.
2. Information absente du CDC → écris [Non spécifié] (ne l'omets pas, ne l'invente pas).
3. Fonctionnalité ambiguë → ajoute [Ambigu : interprétation A / interprétation B].
4. Sois précis : "tableau 5 colonnes : Nom | Email | Rôle | Statut | Actions" vaut infiniment
   mieux que "un tableau de gestion des utilisateurs".
5. Inclus TOUTES les pages, même celles très brièvement mentionnées dans le CDC.

---
EXTRAITS DU CAHIER DES CHARGES :
{context}
---

DESCRIPTION COMPLÈTE DE L'APPLICATION :
"""


# ─────────────────────────────────────────────────────────────────────────────
#  REQUÊTES
# ─────────────────────────────────────────────────────────────────────────────

# Requête unique large — baseline single-query
QUERY_ORIGINAL = (
    "Quelles sont toutes les pages, vues, écrans et fonctionnalités "
    "de l'application web décrite dans ce cahier des charges ? "
    "Quels sont les composants d'interface, les formulaires, "
    "les tableaux de bord et les interactions utilisateur ?"
)

# 4 requêtes thématiques — couverture par domaine UI
QUERIES_MULTI_THEMATIQUE = [
    "Pages principales et écrans de l'application web",
    "Formulaires de saisie et interactions utilisateur",
    "Tableaux de bord, listes et affichage de données",
    "Navigation, menus et structure de l'application",
]

# 5 requêtes — comme thématique + rôles/règles métier (dimension souvent manquée)
QUERIES_MULTI_COMPLET = [
    "Pages principales et écrans de l'application web",
    "Formulaires de saisie et interactions utilisateur",
    "Tableaux de bord, listes et affichage de données",
    "Navigation, menus et structure de l'application",
    "Rôles utilisateur, droits d'accès et règles fonctionnelles",
]

# 4 requêtes formulées en langage naturel (questions directes)
QUERIES_MULTI_NATUREL = [
    "Quelles sont les pages et écrans présents dans cette application ?",
    "Comment l'utilisateur interagit-il avec l'interface ? Quels formulaires et actions sont disponibles ?",
    "Quelles informations et données sont affichées à l'écran ?",
    "Comment navigue-t-on entre les différentes sections et pages de l'application ?",
]

# 4 requêtes courtes — mots-clés concis pour un rappel maximal
QUERIES_COURTES = [
    "pages écrans vues interface application",
    "formulaires champs saisie boutons actions",
    "tableaux listes données affichage résultats",
    "navigation menu liens structure sections",
]



# ─────────────────────────────────────────────────────────────────────────────
#  CONFIGURATIONS
# ─────────────────────────────────────────────────────────────────────────────

EXPERIMENTS: list[RAGConfig] = [

    # =========================================================================
    # TIER 1 — Baseline V1 (référence)
    # =========================================================================

    RAGConfig(
        name="baseline_single_k8",
        strategy="single",
        queries=[QUERY_ORIGINAL],
        k=8,
        prompt_template=PROMPT_V1,
        description="Baseline : requête unique large, k=8, prompt V1",
    ),
    RAGConfig(
        name="single_k4",
        strategy="single",
        queries=[QUERY_ORIGINAL],
        k=4,
        prompt_template=PROMPT_V1,
        description="Requête unique, contexte court (k=4)",
    ),
    RAGConfig(
        name="single_k12",
        strategy="single",
        queries=[QUERY_ORIGINAL],
        k=12,
        prompt_template=PROMPT_V1,
        description="Requête unique, contexte large (k=12)",
    ),
    RAGConfig(
        name="multi_thematique_k3",
        strategy="multi",
        queries=QUERIES_MULTI_THEMATIQUE,
        k=3,
        prompt_template=PROMPT_V1,
        description="4 requêtes thématiques × k=3",
    ),
    RAGConfig(
        name="multi_thematique_k5",
        strategy="multi",
        queries=QUERIES_MULTI_THEMATIQUE,
        k=5,
        prompt_template=PROMPT_V1,
        description="4 requêtes thématiques × k=5",
    ),
    RAGConfig(
        name="multi_naturel_k3",
        strategy="multi",
        queries=QUERIES_MULTI_NATUREL,
        k=3,
        prompt_template=PROMPT_V1,
        description="4 requêtes naturelles × k=3",
    ),
    RAGConfig(
        name="multi_naturel_k5",
        strategy="multi",
        queries=QUERIES_MULTI_NATUREL,
        k=5,
        prompt_template=PROMPT_V1,
        description="4 requêtes naturelles × k=5",
    ),
    RAGConfig(
        name="multi_court_k3",
        strategy="multi",
        queries=QUERIES_COURTES,
        k=3,
        prompt_template=PROMPT_V1,
        description="4 requêtes courtes × k=3",
    ),
    RAGConfig(
        name="multi_court_k5",
        strategy="multi",
        queries=QUERIES_COURTES,
        k=5,
        prompt_template=PROMPT_V1,
        description="4 requêtes courtes × k=5",
    ),

    # =========================================================================
    # TIER 2 — Prompt V4 (format ## + exemple login)
    # =========================================================================

    RAGConfig(
        name="baseline_single_k8_v4",
        strategy="single",
        queries=[QUERY_ORIGINAL],
        k=8,
        prompt_template=PROMPT_V4,
        description="Baseline V4 : requête unique, k=8",
    ),
    RAGConfig(
        name="single_k4_v4",
        strategy="single",
        queries=[QUERY_ORIGINAL],
        k=4,
        prompt_template=PROMPT_V4,
        description="V4, requête unique, k=4",
    ),
    RAGConfig(
        name="single_k12_v4",
        strategy="single",
        queries=[QUERY_ORIGINAL],
        k=12,
        prompt_template=PROMPT_V4,
        description="V4, requête unique, k=12",
    ),
    RAGConfig(
        name="multi_thematique_k3_v4",
        strategy="multi",
        queries=QUERIES_MULTI_THEMATIQUE,
        k=3,
        prompt_template=PROMPT_V4,
        description="V4 + 4 requêtes thématiques × k=3",
    ),
    RAGConfig(
        name="multi_thematique_k5_v4",
        strategy="multi",
        queries=QUERIES_MULTI_THEMATIQUE,
        k=5,
        prompt_template=PROMPT_V4,
        description="V4 + 4 requêtes thématiques × k=5",
    ),
    RAGConfig(
        name="multi_thematique_k8_v4",
        strategy="multi",
        queries=QUERIES_MULTI_THEMATIQUE,
        k=8,
        prompt_template=PROMPT_V4,
        description="V4 + 4 requêtes thématiques × k=8 — candidat fort",
    ),
    RAGConfig(
        name="multi_naturel_k3_v4",
        strategy="multi",
        queries=QUERIES_MULTI_NATUREL,
        k=3,
        prompt_template=PROMPT_V4,
        description="V4 + 4 requêtes naturelles × k=3",
    ),
    RAGConfig(
        name="multi_naturel_k5_v4",
        strategy="multi",
        queries=QUERIES_MULTI_NATUREL,
        k=5,
        prompt_template=PROMPT_V4,
        description="V4 + 4 requêtes naturelles × k=5",
    ),
    RAGConfig(
        name="multi_court_k3_v4",
        strategy="multi",
        queries=QUERIES_COURTES,
        k=3,
        prompt_template=PROMPT_V4,
        description="V4 + 4 requêtes courtes × k=3",
    ),
    RAGConfig(
        name="multi_court_k5_v4",
        strategy="multi",
        queries=QUERIES_COURTES,
        k=5,
        prompt_template=PROMPT_V4,
        description="V4 + 4 requêtes courtes × k=5",
    ),

    # =========================================================================
    # TIER 3 — Prompt V2 (format ## pur, sans exemple — teste si la structure
    #          seule suffit sans le bruit d'un exemple)
    # =========================================================================

    RAGConfig(
        name="multi_thematique_k5_v2",
        strategy="multi",
        queries=QUERIES_MULTI_THEMATIQUE,
        k=5,
        prompt_template=PROMPT_V2,
        description="V2 (## strict, sans exemple) + thématique × k=5",
    ),
    RAGConfig(
        name="multi_thematique_k8_v2",
        strategy="multi",
        queries=QUERIES_MULTI_THEMATIQUE,
        k=8,
        prompt_template=PROMPT_V2,
        description="V2 + thématique × k=8",
    ),

    # =========================================================================
    # TIER 4 — Prompt V5 + QUERIES_MULTI_COMPLET (meilleurs candidats)
    # =========================================================================

    RAGConfig(
        name="multi_thematique_k5_v5",
        strategy="multi",
        queries=QUERIES_MULTI_THEMATIQUE,
        k=5,
        prompt_template=PROMPT_V5,
        description="V5 + thématique × k=5",
    ),
    RAGConfig(
        name="multi_thematique_k8_v5",
        strategy="multi",
        queries=QUERIES_MULTI_THEMATIQUE,
        k=8,
        prompt_template=PROMPT_V5,
        description="V5 + thématique × k=8 — candidat principal",
    ),
    RAGConfig(
        name="multi_complet_k5_v5",
        strategy="multi",
        queries=QUERIES_MULTI_COMPLET,
        k=5,
        prompt_template=PROMPT_V5,
        description="V5 + 5 requêtes (incl. rôles/règles) × k=5",
    ),
    RAGConfig(
        name="multi_complet_k8_v5",
        strategy="multi",
        queries=QUERIES_MULTI_COMPLET,
        k=8,
        prompt_template=PROMPT_V5,
        description="V5 + 5 requêtes (incl. rôles/règles) × k=8 — meilleur candidat",
    ),
]
