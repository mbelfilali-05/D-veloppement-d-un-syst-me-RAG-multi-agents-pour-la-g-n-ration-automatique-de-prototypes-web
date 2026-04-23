# agents/cr_agent.py

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from agents.base_agent import BaseAgent
from core.vector_store import VectorStore
from utils.token_tracker import TokenTracker


#prompt par defaut pour CRAgent — peut être modifié à l'instanciation
PROMPT_TEMPLATE = """
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

# V5 — meilleur prompt : éléments globaux + 2 exemples + colonnes + boutons + rôles
PROMPT_V5 = """
Tu es un architecte front-end senior. Ta sortie sera transmise directement à un générateur
de code HTML. Chaque composant que tu décris sera traduit en code concret — sois précis
sur les libellés, les colonnes, les champs, les boutons.

MISSION : Analyse les extraits du cahier des charges ci-dessous et produis une description
complète de toutes les pages/vues de l'application.

═══════════════════════════════════════════
ÉTAPE 0 — IDENTITÉ VISUELLE ET STYLE (Directives transversales)
═══════════════════════════════════════════
Extrais ici tout ce qui définit l'apparence et le contexte global :
- **Nom du Projet & Organisation** : [Nom officiel, abréviations]
- **Charte Graphique** : [Couleurs citées, codes hex, ambiance, style (ex: moderne, traditionnel)]
- **Localisation & Devises** : [Villes spécifiques, monnaies (MAD, €, etc.)]
- **Ton & Style** : [ex: éducatif, ludique, professionnel, sombre, clair]

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
6. Si le nom du projet n'est pas explicitement nommé 'Projet X', utilise le nom de l'organisation ou le titre principal trouvé en haut du document

---
EXTRAITS DU CAHIER DES CHARGES :
{context}
---

DESCRIPTION COMPLÈTE DE L'APPLICATION :
"""


#apres experimentation, j'ai trouvé la strategie la plus adapté est:
#muli-thematique k=5 avec prompt v_1
QUERIES_MULTI_THEMATIQUE = [
    "Pages principales et écrans de l'application web",
    "Formulaires de saisie et interactions utilisateur",
    "Tableaux de bord, listes et affichage de données",
    "Navigation, menus et structure de l'application",
    "Style visuel, typographie et ambiance du site",]

QUERIES_MULTI_COMPLET = [
    "Pages principales et écrans de l'application web",
    "Formulaires de saisie et interactions utilisateur",
    "Tableaux de bord, listes et affichage de données",
    "Navigation, menus et structure de l'application",
    "Rôles utilisateur, droits d'accès et règles fonctionnelles",
    "Style visuel, typographie et ambiance du site",
]

class CRAgent(BaseAgent):
    """
    Agent d'analyse du cahier des charges.
    Utilise le RAG pour interroger le PDF et extraire
    une description structurée de toutes les vues à prototyper.
    """

    def __init__(self, vector_store: VectorStore,
                 retrieval_k: int = 5,
                 retrieval_query: str | list[str] = QUERIES_MULTI_THEMATIQUE,
                 prompt_template: str = PROMPT_V5
                ):
        """
        Args:
            vector_store: Instance VectorStore déjà chargée avec le PDF
            retrieval_k: nombre de chunks à récupérer
            retrieval_query: requête unique (str) ou liste de requêtes thématiques (list[str])
            prompt_template: template du prompt (peut être modifié)
        """
        self.vector_store = vector_store
        self.retrieval_k = retrieval_k
        self.retrieval_query = retrieval_query or (
            "Quelles sont toutes les pages, vues, écrans et fonctionnalités "
            "de l'application web décrite dans ce cahier des charges ? "
            "Quels sont les composants d'interface, les formulaires, "
            "les tableaux de bord et les interactions utilisateur ?"
        )
        self.prompt_template = prompt_template
        self.last_token_usage = 0   # pour stocker le dernier compteur
        # BaseAgent.__init__ appelle _build_chain() — vector_store doit
        # exister AVANT super().__init__()
        super().__init__(name="CRAgent", temperature=0.0)

    def _build_chain(self):
        """
        Chain LCEL : prompt → LLM → texte brut
        Le contexte RAG est injecté dans run() via le retriever.
        """
        prompt = ChatPromptTemplate.from_template(self.prompt_template)
        return prompt | self.llm | StrOutputParser()

    def run(self, state: dict) -> dict:
        """
        1. Interroge ChromaDB pour récupérer les chunks pertinents
        2. Injecte ces chunks dans le prompt
        3. Retourne l'AgentState avec le champ 'summary' rempli

        Args:
            state: AgentState — doit contenir 'pdf_path'

        Returns:
            dict: AgentState avec 'summary' mis à jour
        """
        self._log("Analyse du cahier des charges en cours...")

        try:
            retriever = self.vector_store.get_retriever(k=self.retrieval_k)

            queries = self.retrieval_query if isinstance(self.retrieval_query, list) else [self.retrieval_query]

            if len(queries) == 1:
                # Stratégie single : une seule requête
                docs = retriever.invoke(queries[0])
            else:
                # Stratégie multi : N requêtes avec déduplication par (page, chunk_index)
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

            # Tri chronologique : par page d'abord, puis par chunk_index
            docs_sorted = sorted(
                docs,
                key=lambda d: (
                    d.metadata.get("page", 0),
                    d.metadata.get("chunk_index", 0),
                )
            )


             # Concatène les chunks récupérés en un seul contexte
            context = "\n\n---\n\n".join(
                f"[Page {doc.metadata.get('page', '?')}]\n{doc.page_content}"
                for doc in docs_sorted
            )

            self._log(f"{len(docs)} chunks récupérés pour le contexte RAG")

            # Invoque la chain LCEL
            with TokenTracker("CRAgent") as tracker:
                summary = self.chain.invoke({"context": context})
            self.last_token_usage = tracker.total_tokens   # capture du nombre de tokens
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
                "status": "error",          # <-- signal explicite pour le graph, Tu catches toutes les exceptions et retournes un summary vide — 
                #c'est bien, le workflow ne crashe pas. Mais le CoderAgent va recevoir summary="" et probablement générer du HTML vide ou planter silencieusement.
                #Et dans graph/workflow.py, tu pourras router différemment si state["status"] == "error" au lieu de continuer vers le CoderAgent avec un contexte vide.
                "errors": state.get("errors", []) + [error_msg]
            }