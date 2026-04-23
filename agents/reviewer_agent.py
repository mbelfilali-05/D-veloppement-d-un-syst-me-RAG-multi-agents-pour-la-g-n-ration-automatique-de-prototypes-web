# agents/reviewer_agent.py
#
# Agent de review qui évalue le HTML produit par CoderAgent
# selon 5 critères et produit un feedback structuré (JSON) qui sera
# réinjecté dans CoderAgent pour une itération corrective.
#
# Pattern : LLM-as-a-Judge, inspiré directement d'experiments/evaluate.py
# qui applique le même principe pour évaluer les summaries du CRAgent.

import json

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from agents.base_agent import BaseAgent


# ═══════════════════════════════════════════════════════════════
#  CRITÈRES ET PONDÉRATIONS
# ═══════════════════════════════════════════════════════════════

CRITERIA = ["completude", "densite", "donnees", "interactivite", "coherence"]

CRITERIA_LABELS = {
    "completude":     "Complétude",
    "densite":        "Densité",
    "donnees":        "Données",
    "interactivite":  "Interactivité",
    "coherence":      "Cohérence",
}

# Pondérations pour le score global (somme = 1.0)
WEIGHTS = {
    "completude":    0.30,   # toutes les vues présentes ? priorité max
    "densite":       0.25,   # chaque vue riche ?
    "donnees":       0.20,   # données cohérentes avec le CDC ?
    "interactivite": 0.15,   # navigation fonctionnelle ?
    "coherence":     0.10,   # cohérence visuelle ?
}


# ═══════════════════════════════════════════════════════════════
#  PROMPT D'ÉVALUATION
# ═══════════════════════════════════════════════════════════════

REVIEW_PROMPT = """
Tu es un expert senior en conception d'interfaces web et en évaluation de prototypes HTML.
Tu évalues la qualité d'un prototype HTML produit automatiquement à partir d'un
cahier des charges (CDC) et d'une description structurée (summary).

Tu disposes de deux éléments :
1. Le summary (description des pages attendues, source de vérité)
2. Le HTML généré (ce que tu évalues)

---
SUMMARY (ATTENDU) :
{summary}
---

---
HTML GÉNÉRÉ (À ÉVALUER) :
{html_code}
---

ÉVALUE selon les 5 critères ci-dessous.
Pour chaque critère : un score entier de 1 à 5, et une justification
factuelle en 1 phrase (cite un exemple concret du HTML : nom de vue,
classe CSS, nombre de lignes, etc.)

═══════════════════════════════════════════════════════════════
RUBRIQUES DE SCORING
═══════════════════════════════════════════════════════════════

1. completude — Toutes les pages/vues du summary sont-elles présentes
   dans le HTML (détectables via x-show="currentView === '...'") ?
   1 = moins de 40% des vues présentes
   2 = 40-60% des vues
   3 = 60-80% des vues, vues secondaires manquantes
   4 = 80-95% des vues, une seule vue mineure manquante
   5 = 100% des vues, toutes implémentées

2. densite — Chaque vue a-t-elle le niveau de contenu attendu selon
   son type (dashboard : 4 KPI + table 6+ lignes ; liste : 6+ cards ;
   formulaire : 4-6 champs ; accueil : 5+ sections) ?
   1 = vues quasi vides (juste un titre et 1-2 éléments)
   2 = vues avec 1-2 sections, tableaux de 2-3 lignes
   3 = densité moyenne, certaines vues correctes, d'autres pauvres
   4 = bonne densité partout, 1-2 vues un peu légères
   5 = densité maximale, toutes vues riches et crédibles

3. donnees — Les données d'exemple sont-elles cohérentes avec le CDC
   (noms de produits exacts, devise correcte, prénoms adaptés au pays,
   villes mentionnées dans le CDC) ?
   1 = données génériques (Lorem ipsum, Produit A, Nom Prénom)
   2 = données plausibles mais déconnectées du CDC
   3 = données adaptées en partie (bonnes villes mais mauvaise devise)
   4 = données quasiment toutes cohérentes avec le CDC
   5 = données 100% fidèles au CDC (noms exacts, devise exacte, contexte réaliste)

4. interactivite — Le HTML utilise-t-il Alpine.js correctement
   (x-data sur body, x-show sur chaque vue, @click sur liens navigation,
   pas de JS vanilla qui casserait la navigation SPA) ?
   1 = aucune interactivité ou JS vanilla cassant la navigation
   2 = Alpine partiel, x-data présent mais navigation incomplète
   3 = navigation Alpine fonctionnelle mais composants interactifs
     (accordéon, modal) manquants
   4 = Alpine bien utilisé, navigation + 1-2 composants interactifs
   5 = Alpine maîtrisé partout : navigation, accordéons FAQ, modals,
     toggles, filtres

5. coherence — Le HTML est-il visuellement cohérent sur toutes les vues
   (même sidebar/navbar partout, même palette de couleurs accent, même
   typographie, bordures/ombres homogènes) ?
   1 = chaque vue a son propre style, incohérences majeures
   2 = cohérence partielle, certaines vues dérivent
   3 = cohérence structurelle OK, mais accent couleur variable
   4 = très cohérent, 1-2 détails divergents mineurs
   5 = cohérence parfaite sur toutes les vues

═══════════════════════════════════════════════════════════════
ISSUES SPÉCIFIQUES À REMONTER
═══════════════════════════════════════════════════════════════

En plus des scores, identifie 2 à 5 issues précises à corriger.
Pour chaque issue :
- vue : nom exact de la vue concernée (depuis x-show)
- severity : "high" (bloquant) | "medium" (dégrade la qualité) | "low" (détail)
- description : le problème observé (factuel)
- suggestion : action précise à appliquer pour corriger

Une issue est "high" si elle viole directement le summary (vue manquante,
fonctionnalité explicite du CDC omise, données inventées au lieu de celles du CDC).

═══════════════════════════════════════════════════════════════
VERDICT GLOBAL
═══════════════════════════════════════════════════════════════

Calcule le score_global pondéré :
- completude    : 30%
- densite       : 25%
- donnees       : 20%
- interactivite : 15%
- coherence     : 10%

verdict = "good" SI score_global >= 4.0 ET aucune issue severity="high"
verdict = "insufficient" SINON

═══════════════════════════════════════════════════════════════
FORMAT DE RÉPONSE (JSON strict)
═══════════════════════════════════════════════════════════════

Réponds UNIQUEMENT avec un objet JSON valide.
Aucun markdown, aucun texte avant ou après, aucune balise ```.

{{
  "criteria": {{
    "completude":    {{"score": <1-5>, "justification": "<exemple factuel>"}},
    "densite":       {{"score": <1-5>, "justification": "<exemple factuel>"}},
    "donnees":       {{"score": <1-5>, "justification": "<exemple factuel>"}},
    "interactivite": {{"score": <1-5>, "justification": "<exemple factuel>"}},
    "coherence":     {{"score": <1-5>, "justification": "<exemple factuel>"}}
  }},
  "score_global": <score pondéré arrondi à 1 décimale>,
  "verdict": "good" | "insufficient",
  "issues": [
    {{
      "vue": "<nom de la vue>",
      "severity": "high" | "medium" | "low",
      "description": "<problème observé>",
      "suggestion": "<action corrective précise>"
    }}
  ],
  "missing_views": ["<vue du summary absente du HTML>"],
  "strengths": ["<point fort factuel #1>", "<point fort factuel #2>"],
  "commentaire": "<2 phrases : point fort principal + axe d'amélioration prioritaire>"
}}
"""


# ═══════════════════════════════════════════════════════════════
#  CLASSE ReviewerAgent
# ═══════════════════════════════════════════════════════════════

class ReviewerAgent(BaseAgent):
    """
    Agent de review qui évalue le HTML généré par CoderAgent.

    Responsabilités :
    1. Lire state["html_code"] et state["summary"]
    2. Appeler le LLM juge avec les 5 critères
    3. Parser le JSON retourné (parsing défensif)
    4. Calculer le score pondéré et le verdict hybride (score ≥ 4 ET aucun high)
    5. Enrichir state avec review_feedback, quality_score, verdict
    """

    # Seuils de sortie de la boucle (config du verdict hybride)
    SCORE_THRESHOLD = 4.0
    BLOCKING_SEVERITY = "high"

    def __init__(self):
        # Temperature 0 pour un juge déterministe et reproductible
        super().__init__(name="ReviewerAgent", temperature=0.0, model= "gpt-4o")

    def _build_chain(self):
        """Chain LCEL : prompt → LLM juge → texte brut JSON."""
        prompt = ChatPromptTemplate.from_template(REVIEW_PROMPT)
        return prompt | self.llm | StrOutputParser()

    def run(self, state: dict) -> dict:
        """
        Évalue le HTML et enrichit l'état avec le feedback structuré.

        Args:
            state: AgentState — doit contenir 'html_code' et 'summary'

        Returns:
            dict: AgentState avec 'review_feedback', 'quality_score', 'verdict'
        """
        self._log("Évaluation du HTML en cours...")

        html_code = state.get("html_code", "")
        summary   = state.get("summary", "")

        if not html_code or not summary:
            error_msg = "ReviewerAgent : html_code ou summary vide."
            self._log(f"❌ {error_msg}")
            return {
                **state,
                "review_feedback": None,
                "quality_score":   0.0,
                "verdict":         "insufficient",
                "errors":          state.get("errors", []) + [error_msg],
            }

        try:
            # Invocation du LLM juge
            raw = self.chain.invoke({
                "summary":   summary,
                "html_code": html_code,
            })

            # Parsing défensif du JSON (inspiré d'evaluate.py)
            feedback = self._parse_json(raw)
            if feedback is None:
                # Fallback : score par défaut déclenchant un retry
                self._log("⚠️  JSON invalide, fallback sur score par défaut 2.0")
                return {
                    **state,
                    "review_feedback": {"error": "JSON parsing failed", "raw": raw[:500]},
                    "quality_score":   2.0,
                    "verdict":         "insufficient",
                }

            # Recalcul du score pondéré (ne pas faire confiance au LLM)
            score = self._compute_weighted_score(feedback)

            # Verdict hybride : score ≥ seuil ET aucune issue high
            verdict = self._compute_verdict(score, feedback)

            feedback["score_global"] = score
            feedback["verdict"]      = verdict

            self._log(f"✅ Review terminée — score {score}/5 — verdict {verdict}")
            self._log_summary(feedback)

            return {
                **state,
                "review_feedback": feedback,
                "quality_score":   score,
                "verdict":         verdict,
                "errors":          state.get("errors", []),
            }

        except Exception as e:
            error_msg = f"Erreur ReviewerAgent : {str(e)}"
            self._log(f"❌ {error_msg}")
            # Fallback : on retourne un score bas pour déclencher un retry
            return {
                **state,
                "review_feedback": {"error": str(e)},
                "quality_score":   2.0,
                "verdict":         "insufficient",
                "errors":          state.get("errors", []) + [error_msg],
            }

    # ------------------------------------------------------------------ #
    #  Méthodes privées
    # ------------------------------------------------------------------ #

    def _parse_json(self, raw: str) -> dict | None:
        """
        Parsing JSON défensif — retire les backticks markdown éventuels.
        Retourne None si le parsing échoue.
        """
        cleaned = raw.strip()

        # Retrait d'éventuels ```json ... ```
        if cleaned.startswith("```"):
            # Coupe le premier ``` (et éventuellement "json")
            parts = cleaned.split("```")
            if len(parts) >= 3:
                cleaned = parts[1]
                if cleaned.startswith("json"):
                    cleaned = cleaned[4:]
            cleaned = cleaned.strip()

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as e:
            self._log(f"❌ JSONDecodeError : {e}")
            return None

    def _compute_weighted_score(self, feedback: dict) -> float:
        """
        Recalcule le score global pondéré depuis les scores par critère.
        Ne fait pas confiance au score_global du LLM.
        """
        criteria = feedback.get("criteria", {})
        total = 0.0
        for key, weight in WEIGHTS.items():
            crit = criteria.get(key, {})
            score = crit.get("score", 0)
            try:
                total += float(score) * weight
            except (TypeError, ValueError):
                pass
        return round(total, 1)

    def _compute_verdict(self, score: float, feedback: dict) -> str:
        """
        Verdict hybride : bon SI score ≥ seuil ET aucune issue 'high'.
        """
        if score < self.SCORE_THRESHOLD:
            return "insufficient"

        issues = feedback.get("issues", [])
        has_blocking = any(
            issue.get("severity") == self.BLOCKING_SEVERITY
            for issue in issues
        )
        if has_blocking:
            return "insufficient"

        return "good"

    def _log_summary(self, feedback: dict):
        """Logge un résumé compact du feedback pour debug console."""
        criteria = feedback.get("criteria", {})
        scores_str = " | ".join(
            f"{CRITERIA_LABELS[k]}: {criteria.get(k, {}).get('score', '?')}"
            for k in CRITERIA
        )
        self._log(f"   {scores_str}")

        issues = feedback.get("issues", [])
        high_issues = [i for i in issues if i.get("severity") == "high"]
        if high_issues:
            self._log(f"   🚨 {len(high_issues)} issue(s) bloquante(s) :")
            for i in high_issues[:3]:
                self._log(f"      - [{i.get('vue', '?')}] {i.get('description', '')[:80]}")