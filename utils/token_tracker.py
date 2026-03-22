# utils/token_tracker.py

# from langchain_community.callbacks import get_openai_callback

# class TokenTracker:
#     """
#     Context manager qui wrappe n'importe quel appel LangChain
#     et loggue les tokens consommés + le coût estimé.

#     Usage :
#         with TokenTracker("CRAgent") as tracker:
#             result = chain.invoke(...)
#         tracker.report()
#     """

    # def __init__(self, label: str = ""):
    #     self.label = label
    #     self._cb = None

    # def __enter__(self):
    #     self._ctx = get_openai_callback()
    #     self._cb = self._ctx.__enter__()
    #     return self

    # def __exit__(self, *args):
    #     self._ctx.__exit__(*args)

    # def report(self):
    #     cb = self._cb
    #     print(f"\n [{self.label}] Consommation tokens")
    #     print(f"   Prompt tokens   : {cb.prompt_tokens:,}")
    #     print(f"   Completion tokens: {cb.completion_tokens:,}")
    #     print(f"   Total tokens    : {cb.total_tokens:,}")
    #     print(f"   Coût estimé     : ${cb.total_cost:.6f}")
    #     return {
    #         "prompt_tokens":     cb.prompt_tokens,
    #         "completion_tokens": cb.completion_tokens,
    #         "total_tokens":      cb.total_tokens,
    #         "cost_usd":          cb.total_cost
    #     }

# utils/token_tracker.py

from langchain_community.callbacks import get_openai_callback


class TokenTracker:
    """
    Context manager qui wrappe n'importe quel appel LangChain
    et loggue les tokens consommés + le coût estimé.

    Usage :
        with TokenTracker("CRAgent") as tracker:
            result = chain.invoke(...)
        tracker.report()
        print(tracker.total_tokens)   # accès direct possible
    """

    def __init__(self, label: str = ""):
        self.label = label
        self._ctx = None
        self._cb  = None

    def __enter__(self):
        self._ctx = get_openai_callback()
        self._cb  = self._ctx.__enter__()
        return self

    def __exit__(self, *args):
        self._ctx.__exit__(*args)

    # ── Propriétés directes ──────────────────────────────────────
    # Permettent d'écrire tracker.total_tokens au lieu de tracker._cb.total_tokens

    @property
    def prompt_tokens(self) -> int:
        return self._cb.prompt_tokens if self._cb else 0

    @property
    def completion_tokens(self) -> int:
        return self._cb.completion_tokens if self._cb else 0

    @property
    def total_tokens(self) -> int:
        return self._cb.total_tokens if self._cb else 0

    @property
    def cost_usd(self) -> float:
        return self._cb.total_cost if self._cb else 0.0

    # ── Rapport terminal ─────────────────────────────────────────

    def report(self):
        cb = self._cb
        print(f"\n💰 [{self.label}] Consommation tokens")
        print(f"   Prompt tokens    : {cb.prompt_tokens:,}")
        print(f"   Completion tokens: {cb.completion_tokens:,}")
        print(f"   Total tokens     : {cb.total_tokens:,}")
        print(f"   Coût estimé      : ${cb.total_cost:.6f}")
        return {
            "prompt_tokens":     cb.prompt_tokens,
            "completion_tokens": cb.completion_tokens,
            "total_tokens":      cb.total_tokens,
            "cost_usd":          cb.total_cost,
        }