# agents/base_agent.py

from abc import ABC, abstractmethod
from langchain_openai import ChatOpenAI
from core.llm_config import get_llm
from utils.token_tracker import TokenTracker


class BaseAgent(ABC):
    """
    Classe abstraite dont tous les agents héritent.
    Garantit une structure cohérente : chaque agent a un LLM,
    un nom, et implémente obligatoirement run() et _build_chain().
    """

    def __init__(self, name: str, temperature: float = 0.0, model: str = None):
        """
        Args:
            name: Identifiant de l'agent (ex: "CRAgent")
            temperature: Créativité du LLM (0.0 = déterministe)
            model: Modèle LLM à utiliser
        """
        self.name = name
        self.llm: ChatOpenAI = get_llm(temperature=temperature, model=model)
        self.chain = self._build_chain()

        print(f"✅ {self.name} initialisé")

    @abstractmethod
    def _build_chain(self):
        """
        Construit et retourne la chain LCEL spécifique à l'agent.
        Appelé automatiquement à l'initialisation.
        Format : prompt | llm | output_parser
        """
        pass

    @abstractmethod
    def run(self, state: dict) -> dict:
        """
        Point d'entrée principal de l'agent.
        Reçoit l'AgentState, le modifie, et le retourne.

        Args:
            state: Dictionnaire partagé entre tous les agents
                   (pdf_path, summary, html_code, final_result, errors)

        Returns:
            dict: L'AgentState mis à jour
        """
        pass
    def _tracked_invoke(self, inputs: dict) -> str:
    #Invoke la chain avec tracking des tokens.
        with TokenTracker(self.name) as tracker:
            result = self.chain.invoke(inputs)
        self.last_token_usage = tracker.total_tokens
        tracker.report()
        return result
    
    def _log(self, message: str):
        """Helper pour logger avec le nom de l'agent."""
        print(f"[{self.name}] {message}")