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


class CRAgent(BaseAgent):
    """
    Agent d'analyse du cahier des charges.
    Utilise le RAG pour interroger le PDF et extraire
    une description structurée de toutes les vues à prototyper.
    """

    def __init__(self, vector_store: VectorStore,
                 retrieval_k: int = 4,
                 retrieval_query: str = None,
                 prompt_template: str = PROMPT_TEMPLATE
                ):
        """
        Args:
            vector_store: Instance VectorStore déjà chargée avec le PDF
            retrieval_k: nombre de chunks à récupérer
            retrieval_query: requête à utiliser pour le retrieval (si None, utilise une requête par défaut)
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

            # On interroge sur les pages et fonctionnalités du projet
            # query = (
            #     "Quelles sont toutes les pages, vues, écrans et fonctionnalités "
            #     "de l'application web décrite dans ce cahier des charges ? "
            #     "Quels sont les composants d'interface, les formulaires, "
            #     "les tableaux de bord et les interactions utilisateur ?"
            # )

            docs = retriever.invoke(self.retrieval_query)

            if not docs:
                raise ValueError("Aucun document récupéré depuis ChromaDB.")

            # 🔽 TRI CHRONOLOGIQUE : par page d'abord, puis par index global
            docs_sorted = sorted(
                docs, 
                key=lambda d: (
                    d.metadata.get("page", 0),           # d'abord par page
                    d.metadata.get("chunk_index", 0)      # ensuite par index (si disponible)
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
                "errors": state.get("errors", []) + [error_msg]
            }