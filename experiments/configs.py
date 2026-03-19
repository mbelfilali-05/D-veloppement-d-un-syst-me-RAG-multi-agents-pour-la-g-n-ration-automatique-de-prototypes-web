# experiments/configs.py
#
# Ajoute ici de nouvelles RAGConfig sans jamais toucher à cr_agent.py.
# Chaque config = 1 ligne dans le rapport final.
 
from experiments.models import RAGConfig
 
 
# ─────────────────────────────────────────────
#  PROMPT TEMPLATES
# ─────────────────────────────────────────────
 
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
 
PROMPT_V2 = """
Tu es un architecte front-end senior. Analyse le cahier des charges ci-dessous.
 
Produis une liste structurée de TOUTES les vues/pages à prototyper, en suivant ce format
strict pour chaque vue :
 
## [Numéro]. [Nom de la vue]
**Rôle** : (1 phrase)
**URL suggérée** : /chemin
**Composants** : liste des éléments UI (navbar, tableau, formulaire, modal, etc.)
**Données** : ce qui est affiché ou saisi
**Actions** : ce que l'utilisateur peut faire
**Navigation** : vers quelles autres vues
 
---
EXTRAITS DU CAHIER DES CHARGES :
{context}
---
 
LISTE DES VUES :
"""
 
PROMPT_V3_CONCIS = """
Analyse ce cahier des charges. Liste toutes les pages/vues de l'application web.
Pour chaque page, donne : nom, composants UI principaux, données, actions utilisateur.
Sois concis et factuel.
 
CAHIER DES CHARGES :
{context}
 
PAGES IDENTIFIÉES :
"""
 
 
# ─────────────────────────────────────────────
#  REQUÊTES
# ─────────────────────────────────────────────
 
QUERY_ORIGINAL = (
    "Quelles sont toutes les pages, vues, écrans et fonctionnalités "
    "de l'application web décrite dans ce cahier des charges ? "
    "Quels sont les composants d'interface, les formulaires, "
    "les tableaux de bord et les interactions utilisateur ?"
)
 
QUERIES_MULTI_THEMATIQUE = [
    "Pages principales et écrans de l'application web",
    "Formulaires de saisie et interactions utilisateur",
    "Tableaux de bord, listes et affichage de données",
    "Navigation, menus et structure de l'application",
]
 
QUERIES_MULTI_NATUREL = [
    "Quels écrans l'utilisateur voit-il dans cette application ?",
    "Quelles actions l'utilisateur peut-il effectuer dans l'interface ?",
    "Quelles données sont affichées ou saisies dans l'application ?",
    "Comment la navigation est-elle organisée entre les pages ?",
]
 
QUERIES_COURTES = [
    "pages et vues interface",
    "formulaires saisie données",
    "navigation menus application",
]
 
 
# ─────────────────────────────────────────────
#  CONFIGURATIONS À TESTER
# ─────────────────────────────────────────────
 
EXPERIMENTS: list[RAGConfig] = [
 
    # --- Baseline : config actuelle de cr_agent.py ---
    RAGConfig(
        name="baseline_single_k8",
        strategy="single",
        queries=[QUERY_ORIGINAL],
        k=8,
        prompt_template=PROMPT_V1,
        description="Config actuelle de cr_agent.py — point de référence",
    ),
 
    # --- Variation k ---
    RAGConfig(
        name="single_k4",
        strategy="single",
        queries=[QUERY_ORIGINAL],
        k=4,
        prompt_template=PROMPT_V1,
        description="Même requête, moins de chunks — contexte plus court",
    ),
    RAGConfig(
        name="single_k12",
        strategy="single",
        queries=[QUERY_ORIGINAL],
        k=12,
        prompt_template=PROMPT_V1,
        description="Même requête, plus de chunks — contexte plus large",
    ),
 
    # --- Multi-query thématique ---
    RAGConfig(
        name="multi_thematique_k3",
        strategy="multi",
        queries=QUERIES_MULTI_THEMATIQUE,
        k=3,
        prompt_template=PROMPT_V1,
        description="4 requêtes thématiques × k=3 — couverture par sujet",
    ),
    RAGConfig(
        name="multi_thematique_k5",
        strategy="multi",
        queries=QUERIES_MULTI_THEMATIQUE,
        k=5,
        prompt_template=PROMPT_V1,
        description="4 requêtes thématiques × k=5 — plus de contexte par sujet",
    ),
 
    # --- Multi-query phrases naturelles ---
    RAGConfig(
        name="multi_naturel_k3",
        strategy="multi",
        queries=QUERIES_MULTI_NATUREL,
        k=3,
        prompt_template=PROMPT_V1,
        description="Requêtes formulées en phrases naturelles",
    ),
 
    # --- Requêtes courtes ---
    RAGConfig(
        name="multi_court_k5",
        strategy="multi",
        queries=QUERIES_COURTES,
        k=5,
        prompt_template=PROMPT_V1,
        description="Requêtes très courtes (style mot-clé)",
    ),
 
    # --- Variation prompt ---
    RAGConfig(
        name="baseline_prompt_v2",
        strategy="single",
        queries=[QUERY_ORIGINAL],
        k=8,
        prompt_template=PROMPT_V2,
        description="Prompt structuré avec format Markdown imposé",
    ),
    RAGConfig(
        name="baseline_prompt_v3_concis",
        strategy="single",
        queries=[QUERY_ORIGINAL],
        k=8,
        prompt_template=PROMPT_V3_CONCIS,
        description="Prompt minimaliste — teste si la concision aide",
    ),
]
 