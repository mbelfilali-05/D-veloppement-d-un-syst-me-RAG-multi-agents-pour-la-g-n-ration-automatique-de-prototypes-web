# core/llm_config.py

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings

# Charge les variables du fichier .env
load_dotenv()


def get_llm(temperature: float = None, model: str = None) -> ChatOpenAI:
    """
    Retourne une instance configurée du LLM.
    
    Args:
        temperature: Si None, utilise la valeur du .env
                     Sinon, utilise la valeur fournie.
        model: Si None, utilise la valeur du .env
               Sinon, utilise la valeur fournie.
    Returns:
        ChatOpenAI: Instance prête à utiliser
    """
    temp = temperature if temperature is not None else float(
        os.getenv("OPENAI_TEMPERATURE", 0.0)
    )
    
    #gpt-4o-mini est plus rapide et moins cher que gpt-4, idéal pour les tests et l'évaluation, et parfais pour generer des resumes de cahier des charges. Pour une analyse plus approfondie ou des résumés plus longs, gpt-4 peut être préféré.
    selected_model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    return ChatOpenAI(
        model=selected_model,
        temperature=temp,
        api_key=os.getenv("OPENAI_API_KEY")
    )


def get_embeddings() -> OpenAIEmbeddings:
    """
    Retourne une instance configurée du modèle d'embeddings.
    Utilisé pour vectoriser le contenu des PDFs dans ChromaDB.
    
    Returns:
        OpenAIEmbeddings: Instance prête à utiliser
    """
    return OpenAIEmbeddings(
        model="text-embedding-3-small",  # le plus récent et moins cher
        api_key=os.getenv("OPENAI_API_KEY")
    )