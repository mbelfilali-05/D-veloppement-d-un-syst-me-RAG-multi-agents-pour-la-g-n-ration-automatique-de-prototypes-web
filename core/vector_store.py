#core/vector_store.py

import os 
import shutil
from typing import List
#from langchain_community.vectorstores import Chroma 
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever
from core.llm_config import get_embeddings

class VectorStore:
    """
    Gere la base de données vectorielle ChromDB

    Responsabilités:
    -Vectoriser et stocher les chunks du PDF
    -Rechercher les chunks pertinents pour une question donnée par l'utilisateur
    -Persister la base de données sur le disque pour eviter de recreer les vecteurs a chaque lancement
    """
    #constructeur
    def __init__(
            self,
            persist_directory: str = ".chroma",
            collection_name: str = "pdf_collection"
    ):
        """
        Args:
            persist_directory: Dossier où ChromaDB stocke et sauvegarde les vecteurs
            collection_name: Nom de la collection dans ChromaDB
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.embeddings = get_embeddings()
        self.db = None # sera initialisé dans create() ou load()


    def create(self, chunks: List[Document]):
        """"
        Vectorise les chunks et crée la base de données ChromaDB
        Si une base existe deja, elle est remplacée.

        Args:
            chunks: Liste de Documents à vectoriser et stocker venant de PDFLoader 
        
        """
         # Supprime l'ancienne base si elle existe
        if os.path.exists(self.persist_directory):
            shutil.rmtree(self.persist_directory)
            print(f"🗑️  Ancienne base supprimée")

        # 🔽 Ajout d'un index chronologique global à chaque chunk
        for idx, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = idx
            # Optionnel : ajouter aussi un index par page
            # chunk.metadata["page_index"] = idx  # si vous voulez un tri plus fin

        print(f"⚙️  Vectorisation de {len(chunks)} chunks...")
        print(f"   (Cette opération appelle l'API OpenAI)")

        self.db = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=self.persist_directory,
            collection_name=self.collection_name
        )

        print(f"✅ Base créée : {len(chunks)} vecteurs stockés dans '{self.persist_directory}'")


    def load(self):
        """"
        Charge unne ase ChromaDB existante depuis le disque.
        Evite de re-vectoriser si le PDF n'a pas changé.

        Returns: 
            bool : True si la base existe et est chargée, False sinon
        """
        if not os.path.exists(self.persist_directory):
            print(f"⚠️  Aucune base trouvée dans '{self.persist_directory}'")
            return False
    
        self.db = Chroma(
            persist_directory = self.persist_directory,
            embedding_function = self.embeddings,
            collection_name = self.collection_name
        )
        count = self.db._collection.count()
        print(f"✅ Base chargée : {count} vecteurs")
        return True
    
    def get_retriever(self, k: int = 4) -> VectorStoreRetriever:
        """
        Retourne un retriever — l'objet qui fait la recherche sémantique.
        C'est lui que les agents utilisent pour trouver les chunks pertinents.

        Args:
            k: Nombre de chunks à retourner pour chaque question

        Returns:
            VectorStoreRetriever: Prêt à être utilisé dans une chain
        """
        if self.db is None:
            raise RuntimeError(
                "Base non initialisée. "
                "Appelle create() ou load() d'abord."
            )

        return self.db.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k}
        )

    def search(self, query: str, k: int = 4) -> List[Document]:
        """
        Recherche directe — retourne les chunks les plus pertinents.
        Utile pour tester et déboguer.

        Args:
            query: La question ou le texte à rechercher
            k: Nombre de résultats à retourner

        Returns:
            List[Document]: Les chunks les plus proches sémantiquement
        """
        if self.db is None:
            raise RuntimeError(
                "Base non initialisée. "
                "Appelle create() ou load() d'abord."
            )

        return self.db.similarity_search(query, k=k)
    
    def reset(self) -> None:
        """
        Supprime complètement la base vectorielle.
        Utile quand on change de PDF.
        """
        if os.path.exists(self.persist_directory):
            shutil.rmtree(self.persist_directory)
            self.db = None
            print(f"🗑️  Base réinitialisée")
        else:
            print(f"ℹ️  Aucune base à supprimer")