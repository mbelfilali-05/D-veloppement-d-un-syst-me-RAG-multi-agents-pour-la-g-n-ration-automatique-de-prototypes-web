# agents/cr_agent.py
#
# CHANGEMENTS PAR RAPPORT À LA VERSION PRÉCÉDENTE :
#
# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  CHANGEMENT 1 — Étape 0 : PALETTE HEX CONCRÈTE                           ║
# ║  Avant : le CRAgent extrayait "bleu, orange, jaune, vert, rouge" en       ║
# ║  texte libre, sans codes hex ni rôles assignés. Le CoderAgent devait      ║
# ║  deviner seul quelle couleur utiliser où → tombait sur la Priorité 2      ║
# ║  de son prompt (orange pour escape game) au lieu de respecter la charte.  ║
# ║  Maintenant : le CRAgent DOIT produire une palette de 5 hex avec rôles   ║
# ║  assignés (dominant, accent, secondaire, tertiaire, alerte). Si le CDC    ║
# ║  ne donne pas de hex, le CRAgent choisit des hex cohérents avec les       ║
# ║  couleurs nommées. Le CoderAgent n'a plus qu'à copier-coller.             ║
# ╠══════════════════════════════════════════════════════════════════════════════╣
# ║  CHANGEMENT 2 — Étape 0 : TYPE DE SITE EXPLICITE                         ║
# ║  Avant : le type de site (ecommerce, dashboard, vitrine, hybride) n'était ║
# ║  pas transmis dans le summary → le CoderAgent devinait.                   ║
# ║  Maintenant : le CRAgent doit choisir parmi les types connus et           ║
# ║  l'indiquer explicitement dans l'Étape 0.                                ║
# ╠══════════════════════════════════════════════════════════════════════════════╣
# ║  CHANGEMENT 3 — Exemples de référence NEUTRES et MULTI-DOMAINES          ║
# ║  Avant : les 2 exemples étaient un Tableau de bord B2B et un Formulaire  ║
# ║  de commande — biais systématique vers le back-office.                    ║
# ║  Maintenant : 3 exemples couvrant e-commerce public (catalogue),         ║
# ║  site vitrine (page d'accueil) et dashboard admin — pour montrer le      ║
# ║  format attendu sur tous les types de site.                              ║
# ╠══════════════════════════════════════════════════════════════════════════════╣
# ║  CHANGEMENT 4 — Règle "quantité = celle du CDC"                          ║
# ║  Avant : pas de guidance sur le nombre de produits à lister.              ║
# ║  Maintenant : règle explicite "N'invente pas de produits/services         ║
# ║  supplémentaires. Indique le nombre exact mentionné dans le CDC."         ║
# ╠══════════════════════════════════════════════════════════════════════════════╣
# ║  CHANGEMENT 5 — Requêtes par défaut = QUERIES_MULTI_THEMATIQUE (4)       ║
# ║  Avant : défaut = QUERIES_MULTI_COMPLET (6 requêtes incluant rôles       ║
# ║  et style visuel). L'analyse a montré que les 2 requêtes                  ║
# ║  supplémentaires n'apportent rien sur des CDC courts et diluent le        ║
# ║  contexte.                                                               ║
# ║  Maintenant : défaut = 4 requêtes thématiques. k=5 au lieu de 8.         ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from agents.base_agent import BaseAgent
from core.vector_store import VectorStore
from utils.token_tracker import TokenTracker


# ─────────────────────────────────────────────────────────────────────────────
#  PROMPT V5 CORRIGÉ
# ─────────────────────────────────────────────────────────────────────────────

PROMPT_V5 = """
Tu es un architecte front-end senior. Ta sortie sera transmise directement à un générateur
de code HTML. Chaque composant que tu décris sera traduit en code concret — sois précis
sur les libellés, les colonnes, les champs, les boutons.

MISSION : Analyse les extraits du cahier des charges ci-dessous et produis une description
complète de toutes les pages/vues de l'application.

═══════════════════════════════════════════
ÉTAPE 0 — IDENTITÉ VISUELLE, CONTEXTE ET PALETTE
═══════════════════════════════════════════

Extrais ces informations puis DÉCIDE la palette concrète.

**Nom du Projet** : [nom officiel trouvé dans le document, NE PAS inventer]
**Organisation** : [nom de l'organisation/client]
**Domaine** : [escape-game, santé, ecommerce, éducation, etc.]
**Localisation** : [villes/pays mentionnés]
**Devise** : [MAD, €, $, etc. — celle du CDC]
**Langues demandées** : [FR, AR, EN, ES — uniquement celles citées dans le CDC]
**Ton & Style** : [éducatif, ludique, professionnel, etc.]

**Type de site** : [choisis parmi :]
  - ecommerce_public : si le site vend des produits/services au public
  - dashboard_admin : si c'est un back-office de gestion
  - vitrine : si c'est une présentation sans achat en ligne
  - hybride : si le CDC décrit à la fois un front public ET un back-office

**Palette de couleurs** (OBLIGATOIRE — tu DOIS produire 5 codes hex) :
  Lis ce que le CDC dit sur les couleurs. Puis traduis en hex concrets :
  - Dominante (navbar, titres, fond hero) : #______
  - Accent (boutons CTA, liens actifs)    : #______
  - Secondaire (badges, highlights)       : #______
  - Tertiaire (icônes, accents légers)    : #______
  - Alerte (erreurs, urgence)             : #______
  - Fond général : #FFFFFF ou autre si le CDC précise
  - Texte principal : #111827

  Règle de décision pour la palette :
  → Si le CDC cite des couleurs nommées (ex: "bleu, orange, vert") :
    choisis un hex pour chacune et assigne-leur un rôle ci-dessus.
    La couleur la plus représentative du domaine/ambiance = dominante.
  → Si le CDC cite des hex précis : utilise-les tels quels.
  → Si le CDC ne mentionne aucune couleur : choisis une palette
    adaptée au domaine (ludique = vif, médical = sobre, etc.)

**Éléments graphiques** : [motifs, textures, style d'icônes — si mentionnés dans le CDC]

═══════════════════════════════════════════
ÉTAPE 1 — ÉLÉMENTS GLOBAUX (navbar/sidebar partagés)
═══════════════════════════════════════════

Identifie les éléments présents sur toutes les pages :
**Navigation globale** : [liste des entrées de menu et leurs destinations]
**Rôles utilisateur** : [admin / client / visiteur / etc. — selon le CDC]
**Éléments transverses** : [sélecteur de langues, icône panier, réseaux sociaux — si cités]

═══════════════════════════════════════════
ÉTAPE 2 — LISTE DES PAGES (une section par page)
═══════════════════════════════════════════

Format obligatoire pour chaque page :

## [N°]. [Nom de la page]
**Objectif** : (1 phrase)
**Composants UI** :
  - Formulaires : [champ1 (type, contraintes), champ2, ...]
  - Tableaux : [colonnes : Nom | Statut | Date | Actions]
  - Boutons : ["Libellé" (primaire/secondaire), ...]
  - Autres : [cartes, modals, filtres, graphiques, badges, etc.]
**Données affichées** : [ce qui est visible sans interaction]
**Données saisies** : [champs + types + contraintes si mentionnées]
**Actions utilisateur** : [clic sur X → Y, soumission → Z, etc.]
**Navigation** : [→ /page-cible SI condition, → /autre SINON]
**Visible pour** : [tous / admin seulement / etc. — si mentionné]

EXEMPLES DE RÉFÉRENCE (ne pas reproduire le contenu, suivre le format uniquement) :

## 1. Page d'accueil (site vitrine/ecommerce)
**Objectif** : Présenter l'offre et inciter à l'action
**Composants UI** :
  - Autres : hero pleine largeur avec titre + sous-titre + 2 boutons CTA,
    section "Nos offres" (grille de 3 cards produit), section "Comment ça marche"
    (3 étapes numérotées), section témoignages (3 cards avis clients),
    section partenaires (logos en bande horizontale), footer avec liens réseaux sociaux
  - Boutons : "Réserver" (primaire), "En savoir plus" (secondaire)
**Données affichées** : produits/services du CDC avec noms et prix exacts
**Données saisies** : [Non spécifié]
**Actions utilisateur** : clic sur un produit → fiche détail, clic CTA → page réservation
**Navigation** : → /catalogue (lien "Voir tout"), → /produit/:id (clic card)
**Visible pour** : tous

## 2. Catalogue / liste de produits
**Objectif** : Parcourir et filtrer les produits/services disponibles
**Composants UI** :
  - Autres : barre de filtres horizontale (par catégorie, par ville, par prix),
    grille de cards produit (autant que le CDC en mentionne)
  - Boutons : "Filtrer" (secondaire), "Réserver" (primaire par card)
**Données affichées** : nom, description courte, prix, catégorie — pour chaque produit du CDC
**Données saisies** : filtres (select catégorie, select ville)
**Actions utilisateur** : filtrer la liste, cliquer sur une card → fiche détail
**Navigation** : → /produit/:id (clic card)
**Visible pour** : tous

## 3. Tableau de bord (back-office)
**Objectif** : Vue d'ensemble des indicateurs clés pour l'administrateur
**Composants UI** :
  - Tableaux : dernières commandes — colonnes : N° | Client | Montant | Statut | Date
  - Boutons : "Voir détail" (lien par ligne), "Exporter" (secondaire)
  - Autres : 4 cartes KPI (indicateurs adaptés au domaine du CDC)
**Données affichées** : KPIs, 10 dernières commandes
**Données saisies** : filtre date (date picker début / fin)
**Actions utilisateur** : cliquer sur une ligne → détail, exporter les données
**Navigation** : → /commande/:id (clic ligne)
**Visible pour** : admin

═══════════════════════════════════════════
RÈGLES STRICTES
═══════════════════════════════════════════

1. N'invente AUCUNE page, fonctionnalité ou champ absent du CDC.
2. Information absente du CDC → écris [Non spécifié] (ne l'omets pas, ne l'invente pas).
3. Fonctionnalité ambiguë → ajoute [Ambigu : interprétation A / interprétation B].
4. Sois précis : "tableau 5 colonnes : Nom | Email | Rôle | Statut | Actions" vaut
   infiniment mieux que "un tableau de gestion des utilisateurs".
5. Inclus TOUTES les pages, même celles très brièvement mentionnées dans le CDC.
6. N'invente PAS de produits/services supplémentaires. Indique le nombre exact
   mentionné dans le CDC. Si le CDC dit "3 parcours à Rabat", décris 3 parcours,
   pas 6. Pour les villes "à venir", mentionne-les comme prévues mais non disponibles.
   Si les noms exacts des produits ne sont pas dans le CDC, donne des noms DESCRIPTIFS
   basés sur le contexte — JAMAIS de noms génériques comme "Parcours 1", "Produit A".
7. Le nom du projet = celui trouvé dans le document. Si aucun nom de projet n'est
   explicite, utilise le nom de l'organisation. Ne mets JAMAIS [Non spécifié] pour
   le nom du projet ou l'organisation — il y a TOUJOURS un nom quelque part dans
   le document (en-tête, titre, signature, logo). Cherche-le.
8. La palette hex de l'Étape 0 est OBLIGATOIRE — ne la saute jamais.

---
EXTRAITS DU CAHIER DES CHARGES :
{context}
---

DESCRIPTION COMPLÈTE DE L'APPLICATION :
"""


# ─────────────────────────────────────────────────────────────────────────────
#  REQUÊTES DE RETRIEVAL
# ─────────────────────────────────────────────────────────────────────────────

# Défaut : 4 requêtes thématiques (suffisantes pour les CDC courts à moyens)
QUERIES_MULTI_THEMATIQUE = [
    "Pages principales et écrans de l'application web",
    "Formulaires de saisie et interactions utilisateur",
    "Tableaux de bord, listes et affichage de données",
    "Navigation, menus et structure de l'application",
    "Nom du projet, organisation, charte graphique, couleurs, typographie, identité visuelle",
    "Produits, services, parcours, tarifs, prix, offres proposées",
]


# ─────────────────────────────────────────────────────────────────────────────
#  CLASSE CRAgent
# ─────────────────────────────────────────────────────────────────────────────

class CRAgent(BaseAgent):
    """
    Agent d'analyse du cahier des charges.
    Utilise le RAG pour interroger le PDF et extraire
    une description structurée de toutes les vues à prototyper.
    """

    def __init__(self, vector_store: VectorStore,
                 retrieval_k: int = 6,                                  # ← CHANGEMENT 5 : k=5 au lieu de 8
                 retrieval_query: str | list[str] = None,
                 prompt_template: str = PROMPT_V5
                ):
        """
        Args:
            vector_store: Instance VectorStore déjà chargée avec le PDF
            retrieval_k: nombre de chunks à récupérer par requête
            retrieval_query: requête unique (str) ou liste de requêtes (list[str])
            prompt_template: template du prompt (peut être modifié)
        """
        self.vector_store = vector_store
        self.retrieval_k = retrieval_k
        self.retrieval_query = retrieval_query or QUERIES_MULTI_THEMATIQUE  # ← CHANGEMENT 5 : défaut = thématique (4)
        self.prompt_template = prompt_template
        self.last_token_usage = 0
        super().__init__(name="CRAgent", temperature=0.0)

    def _build_chain(self):
        """Chain LCEL : prompt → LLM → texte brut."""
        prompt = ChatPromptTemplate.from_template(self.prompt_template)
        return prompt | self.llm | StrOutputParser()

    def run(self, state: dict) -> dict:
        """
        1. Interroge ChromaDB pour récupérer les chunks pertinents
        2. Injecte ces chunks dans le prompt
        3. Retourne l'AgentState avec le champ 'summary' rempli
        """
        self._log("Analyse du cahier des charges en cours...")

        try:
            retriever = self.vector_store.get_retriever(k=self.retrieval_k)

            queries = self.retrieval_query if isinstance(self.retrieval_query, list) else [self.retrieval_query]

            if len(queries) == 1:
                docs = retriever.invoke(queries[0])
            else:
                seen: dict = {}
                for query in queries:
                    for doc in retriever.invoke(query):
                        key = (
                            doc.metadata.get("page", 0),
                            doc.metadata.get("chunk_index", 0),
                        )
                        if key not in seen:
                            seen[key] = doc
                docs = list(seen.values())

            if not docs:
                raise ValueError("Aucun document récupéré depuis ChromaDB.")

            docs_sorted = sorted(
                docs,
                key=lambda d: (
                    d.metadata.get("page", 0),
                    d.metadata.get("chunk_index", 0),
                )
            )

            context = "\n\n---\n\n".join(
                f"[Page {doc.metadata.get('page', '?')}]\n{doc.page_content}"
                for doc in docs_sorted
            )

            self._log(f"{len(docs)} chunks récupérés pour le contexte RAG")

            with TokenTracker("CRAgent") as tracker:
                summary = self.chain.invoke({"context": context})
            self.last_token_usage = tracker.total_tokens
            tracker.report()

            self._log("✅ Analyse terminée")

            return {
                **state,
                "summary": summary,
                "errors": state.get("errors", [])
            }

        except Exception as e:
            error_msg = f"Erreur CRAgent : {str(e)}"
            self._log(f"❌ {error_msg}")
            return {
                **state,
                "summary": "",
                "status": "error",
                "errors": state.get("errors", []) + [error_msg]
            }