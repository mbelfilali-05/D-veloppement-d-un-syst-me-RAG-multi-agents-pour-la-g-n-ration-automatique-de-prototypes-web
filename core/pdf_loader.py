# core/pdf_loader.py

import os
from typing import List
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


class PDFLoader:
    """
    Responsable du chargement et du découpage des fichiers PDF.
    Retourne des Documents LangChain prêts à être vectorisés.
    """

    def __init__(
        self,
        chunk_size: int = 600,
        chunk_overlap: int = 150
    ):
        """
        Args:
            chunk_size: Taille maximale de chaque chunk en caractères
            chunk_overlap: Chevauchement entre chunks consécutifs
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            # découpe dans cet ordre de priorité :
            # paragraphe → phrase → mot → caractère
            separators=["\n\n", "\n", " ", ""]
        )

    def load(self, pdf_path: str) -> List[Document]:
        """
        Charge un PDF et retourne une liste de Documents découpés.

        Args:
            pdf_path: Chemin vers le fichier PDF

        Returns:
            List[Document]: Liste de chunks prêts pour ChromaDB

        Raises:
            FileNotFoundError: Si le PDF n'existe pas
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF introuvable : {pdf_path}")

        # Charge le PDF page par page
        loader = PyMuPDFLoader(pdf_path)
        pages = loader.load()

        print(f"📄 PDF chargé : {len(pages)} pages")

        # Découpe en chunks
        raw_chunks = self.text_splitter.split_documents(pages)

        print(f"✂️  Découpage : {len(raw_chunks)} chunks "
              f"(taille={self.chunk_size}, overlap={self.chunk_overlap})")

        # Ajoute chunk_index par page pour permettre la déduplication multi-query
        page_counters: dict = {}
        for chunk in raw_chunks:
            p = chunk.metadata.get("page", 0)
            page_counters[p] = page_counters.get(p, 0)
            chunk.metadata["chunk_index"] = page_counters[p]
            page_counters[p] += 1

        # Filtre les chunks trop courts (parasites sémantiques)
        MIN_CHUNK_SIZE = 30  # caractères
        chunks = [c for c in raw_chunks if len(c.page_content.strip()) >= MIN_CHUNK_SIZE]
        print(f"✂️  Après filtrage : {len(chunks)} chunks valides")
   
        return chunks 

    def get_stats(self, chunks: List[Document]) -> dict:
        """
        Retourne des statistiques sur les chunks générés.
        Utile pour comprendre et ajuster les paramètres.
        """
        sizes = [len(chunk.page_content) for chunk in chunks]

        return {
            "total_chunks": len(chunks),
            "avg_size": round(sum(sizes) / len(sizes)),
            "min_size": min(sizes),
            "max_size": max(sizes),
        }