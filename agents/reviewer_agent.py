# agents/reviewer_agent.py
#
# ReviewerAgent V2 — architecture 3 couches
#
# Couche 1 : Vérification mécanique (Python pur, 0 tokens)
#   → Détecte les défauts évidents : placeholders, tableaux courts, icônes cassées
#
# Couche 2 : Vérification de conformité (Python, summary ↔ HTML)
#   → Compare les vues attendues dans le summary aux vues présentes dans le HTML
#   → Détecte les fonctionnalités CDC manquantes
#
# Couche 3 : Évaluation sémantique (LLM, GPT-4o)
#   → Évalue la qualité de ce qui EST présent
#   → Reçoit les résultats des couches 1 et 2 comme contexte
#   → Produit les scores finaux et les suggestions d'amélioration

import re
import json
from collections import Counter

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

WEIGHTS = {
    "completude":    0.30,
    "densite":       0.25,
    "donnees":       0.20,
    "interactivite": 0.15,
    "coherence":     0.10,
}


# ═══════════════════════════════════════════════════════════════
#  COUCHE 1 — VÉRIFICATION MÉCANIQUE (Python pur)
# ═══════════════════════════════════════════════════════════════

def mechanical_check(html: str) -> list[dict]:
    """
    Détecte les défauts évidents par analyse statique du HTML.
    Retourne une liste d'issues au format standard.
    Zéro appel LLM — 100% fiable, instantané.
    """
    issues = []

    # 1. Placeholders non remplacés : [Nom exact du CDC], [ACCENT], [À compléter], etc.
    placeholders = re.findall(r'\[(?:Nom|ACCENT|À |Non |Prix|Image|Description|Titre)[^\]]{0,60}\]', html)
    if placeholders:
        unique = list(set(placeholders))[:5]  # max 5 exemples
        issues.append({
            "vue": "global",
            "severity": "high",
            "description": f"Placeholders non remplacés trouvés ({len(placeholders)} occurrences) : {', '.join(unique[:3])}",
            "suggestion": "Remplacer chaque placeholder par la vraie valeur extraite du CDC. '[Nom exact du CDC]' doit devenir le vrai nom du projet.",
            "source": "mechanical",
        })

    # 2. Texte Material Icons non rendu (ex: "shopping_cart" en texte brut)
    icon_patterns = re.findall(r'(?<!["\'-])(?:shopping_cart|account_circle|menu|search|close|arrow_back|favorite|delete|edit|visibility)(?!["\'-])', html)
    if icon_patterns:
        # Vérifie que Material Icons n'est pas chargé
        has_material = 'material' in html.lower() and ('icons' in html.lower() or 'symbols' in html.lower())
        if not has_material and icon_patterns:
            issues.append({
                "vue": "global",
                "severity": "medium",
                "description": f"Noms d'icônes Material Icons en texte brut (pas de CDN Material Icons chargé) : {', '.join(list(set(icon_patterns)[:3]))}",
                "suggestion": "Ajouter le CDN Material Icons dans le <head>, ou remplacer par des icônes Flowbite/SVG inline.",
                "source": "mechanical",
            })

    # 3. Tableaux trop courts (< 4 lignes dans tbody)
    tables = re.findall(r'<tbody[^>]*>(.*?)</tbody>', html, re.DOTALL)
    for i, tbody in enumerate(tables):
        row_count = len(re.findall(r'<tr', tbody))
        if row_count < 4:
            issues.append({
                "vue": f"tableau_{i+1}",
                "severity": "high" if row_count <= 2 else "medium",
                "description": f"Tableau {i+1} n'a que {row_count} ligne(s) de données (minimum attendu : 6)",
                "suggestion": f"Ajouter {6 - row_count} lignes supplémentaires avec des données réalistes du domaine.",
                "source": "mechanical",
            })

    # 4. Formulaires trop courts (< 4 inputs)
    # Cherche les sections de formulaire
    forms = re.findall(r'<form[^>]*>(.*?)</form>', html, re.DOTALL)
    if not forms:
        # Pas de <form> explicite, cherche les groupements d'inputs
        input_count = len(re.findall(r'<(?:input|textarea|select)\b', html))
        # On ne flag que s'il y a très peu d'inputs au total
        if 0 < input_count < 3:
            issues.append({
                "vue": "formulaire",
                "severity": "medium",
                "description": f"Seulement {input_count} champ(s) de formulaire détecté(s) dans tout le HTML (minimum attendu : 4-6 par formulaire)",
                "suggestion": "Enrichir les formulaires avec les champs mentionnés dans le summary (nom, email, téléphone, message, sujet, etc.)",
                "source": "mechanical",
            })

    # 5. DOCTYPE et structure de base
    if '<!doctype html>' not in html.lower()[:100]:
        issues.append({
            "vue": "global",
            "severity": "high",
            "description": "Le DOCTYPE HTML est manquant ou mal placé",
            "suggestion": "Le fichier doit commencer par <!DOCTYPE html>",
            "source": "mechanical",
        })

    if '</html>' not in html.lower():
        issues.append({
            "vue": "global",
            "severity": "high",
            "description": "Balise </html> fermante absente — HTML probablement tronqué",
            "suggestion": "Regénérer le HTML complet ou augmenter max_tokens.",
            "source": "mechanical",
        })

    # 6. Données génériques évidentes
    generic_patterns = [
        (r'\bLorem ipsum\b', "Lorem ipsum détecté"),
        (r'\bProduit [A-D]\b', "Nom de produit générique 'Produit A/B/C'"),
        (r'\bJohn Doe\b', "Nom générique 'John Doe'"),
        (r'\bexample\.com\b', "URL placeholder example.com"),
        (r'\b\d+\.\d{2}\s*€(?!\s*[\w])', None),  # skip, pas toujours un problème
    ]
    for pattern, msg in generic_patterns:
        if msg and re.search(pattern, html, re.IGNORECASE):
            issues.append({
                "vue": "global",
                "severity": "medium",
                "description": msg,
                "suggestion": "Remplacer par des données réalistes adaptées au domaine et au pays du CDC.",
                "source": "mechanical",
            })

    # 7. Alpine.js : x-data doit exister sur body
    if 'x-data' not in html:
        issues.append({
            "vue": "global",
            "severity": "high",
            "description": "Aucun attribut x-data trouvé — la navigation Alpine.js ne fonctionnera pas",
            "suggestion": "Ajouter x-data=\"{ currentView: 'accueil' }\" sur la balise <body>.",
            "source": "mechanical",
        })

    # 8. Vérifier qu'il y a au moins 2 vues avec x-show
    xshow_views = re.findall(r'x-show\s*=\s*["\'].*?currentView\s*===?\s*["\'](\w+)["\']', html)
    if len(set(xshow_views)) < 2:
        issues.append({
            "vue": "global",
            "severity": "high",
            "description": f"Seulement {len(set(xshow_views))} vue(s) détectée(s) via x-show (minimum : les vues du summary)",
            "suggestion": "Chaque page du summary doit avoir une section avec x-show=\"currentView === 'nom_vue'\".",
            "source": "mechanical",
        })

    return issues


# ═══════════════════════════════════════════════════════════════
#  COUCHE 2 — VÉRIFICATION DE CONFORMITÉ (summary ↔ HTML)
# ═══════════════════════════════════════════════════════════════

def conformity_check(html: str, summary: str) -> list[dict]:
    """
    Compare le summary au HTML pour détecter les manques de conformité.
    Extraction robuste : fonctionne que le summary utilise ##, -, ●, ou du texte libre.
    """
    issues = []
    html_lower = html.lower()

    # 1. Extraire les vues attendues depuis le summary
    #    On cherche plusieurs patterns possibles : "## 1. Nom", "## Nom", "**Nom de la page**", etc.
    view_patterns = [
        re.findall(r'##\s*\d*\.?\s*(.+)', summary),                          # ## 1. Page d'accueil
        re.findall(r'\*\*(?:Page|Vue)\s*:?\s*(.+?)\*\*', summary),           # **Page : Accueil**
        re.findall(r'(?:^|\n)\d+\.\s+(?:Page\s+)?(.+?)(?:\n|$)', summary),   # 1. Page d'accueil
    ]
    
    # Prend le pattern qui a trouvé le plus de résultats
    expected_views = max(view_patterns, key=len) if any(view_patterns) else []
    expected_views = [v.strip().rstrip(':').strip() for v in expected_views]

    # Extraire les vues présentes dans le HTML via x-show
    html_views = set(re.findall(r'x-show\s*=\s*["\'].*?currentView\s*===?\s*["\'](\w+)["\']', html))

    if expected_views and html_views:
        # Pour chaque vue attendue, vérifie si un x-show correspondant existe
        # Comparaison fuzzy : normalise les noms
        def normalize(name):
            return re.sub(r'[^a-z0-9]', '', name.lower())

        html_views_normalized = {normalize(v): v for v in html_views}

        missing_views = []
        for view_name in expected_views:
            norm = normalize(view_name)
            # Cherche une correspondance partielle
            found = any(
                norm in hn or hn in norm or 
                # correspondance par mots clés
                any(word in hn for word in norm.split() if len(word) > 3)
                for hn in html_views_normalized.keys()
            )
            if not found:
                missing_views.append(view_name)

        if missing_views:
            issues.append({
                "vue": "global",
                "severity": "high",
                "description": f"Vue(s) du summary absente(s) du HTML : {', '.join(missing_views[:5])}",
                "suggestion": f"Ajouter les vues manquantes avec x-show correspondant. Vues HTML détectées : {', '.join(html_views)}",
                "source": "conformity",
                "missing_views": missing_views,
            })

    # 2. Fonctionnalités CDC mentionnées dans le summary mais absentes du HTML
    feature_checks = [
        {
            "keywords_summary": ["langue", "langu", "fr/ar", "fr, ar", "français.*arabe", "multilingue"],
            "keywords_html": ["lang", "fr", "ar", "en", "es", "language", "langue"],
            "min_html_matches": 3,  # Au moins 3 codes de langue doivent être présents
            "name": "Sélecteur de langues",
        },
        {
            "keywords_summary": ["partenaire"],
            "keywords_html": ["partenaire", "partner"],
            "min_html_matches": 1,
            "name": "Section partenaires",
        },
        {
            "keywords_summary": ["réseaux sociaux", "facebook", "instagram", "twitter", "réseau social"],
            "keywords_html": ["facebook", "instagram", "twitter", "linkedin", "réseaux", "social"],
            "min_html_matches": 1,
            "name": "Réseaux sociaux",
        },
        {
            "keywords_summary": ["newsletter", "inscription.*email", "s'abonner"],
            "keywords_html": ["newsletter", "abonner", "subscribe"],
            "min_html_matches": 1,
            "name": "Newsletter",
        },
        {
            "keywords_summary": ["faq", "foire aux questions", "questions fréquentes"],
            "keywords_html": ["faq", "question", "accordéon", "accordion"],
            "min_html_matches": 1,
            "name": "FAQ",
        },
        {
            "keywords_summary": ["panier", "cart", "achat"],
            "keywords_html": ["panier", "cart", "basket"],
            "min_html_matches": 1,
            "name": "Panier",
        },
    ]

    summary_lower = summary.lower()
    for check in feature_checks:
        # La fonctionnalité est-elle mentionnée dans le summary ?
        mentioned_in_summary = any(
            re.search(kw, summary_lower) for kw in check["keywords_summary"]
        )
        if not mentioned_in_summary:
            continue

        # Est-elle présente dans le HTML ?
        html_matches = sum(
            1 for kw in check["keywords_html"]
            if kw.lower() in html_lower
        )
        if html_matches < check["min_html_matches"]:
            issues.append({
                "vue": "global",
                "severity": "high",
                "description": f"Fonctionnalité '{check['name']}' mentionnée dans le summary mais absente ou incomplète dans le HTML",
                "suggestion": f"Ajouter la fonctionnalité '{check['name']}' dans le HTML. Chercher dans le summary les détails attendus.",
                "source": "conformity",
            })

    # 3. Devise — vérifier que la bonne devise est utilisée
    devise_summary = None
    if re.search(r'\b(?:MAD|DH|dirham)\b', summary, re.IGNORECASE):
        devise_summary = "MAD"
    elif re.search(r'\b(?:EUR|€|euro)\b', summary, re.IGNORECASE):
        devise_summary = "EUR"
    elif re.search(r'\b(?:USD|\$|dollar)\b', summary, re.IGNORECASE):
        devise_summary = "USD"

    if devise_summary:
        if devise_summary == "MAD" and '€' in html and 'MAD' not in html.upper():
            issues.append({
                "vue": "global",
                "severity": "high",
                "description": "Le CDC utilise MAD/DH mais le HTML utilise € comme devise",
                "suggestion": "Remplacer toutes les occurrences de € par MAD ou DH.",
                "source": "conformity",
            })

    return issues


# ═══════════════════════════════════════════════════════════════
#  COUCHE 3 — ÉVALUATION SÉMANTIQUE (LLM)
# ═══════════════════════════════════════════════════════════════

REVIEW_PROMPT = """
Tu es un expert senior en évaluation de prototypes HTML.
Tu es EXIGEANT et PRÉCIS. Tu ne donnes JAMAIS un score de 5/5 sauf si
le HTML est parfait sur ce critère — sans aucune exception.

Tu évalues un prototype HTML produit à partir d'un summary (source de vérité).

AVANT de noter, une vérification automatique a déjà détecté ces défauts :

DÉFAUTS DÉTECTÉS AUTOMATIQUEMENT (couches 1 et 2) :
{pre_check_issues}

Ces défauts sont CONFIRMÉS — tu ne peux PAS les ignorer dans ta notation.
Un HTML contenant des placeholders non remplacés comme [Nom exact du CDC]
ne peut PAS recevoir plus de 2/5 en Données.
Un HTML avec des vues manquantes ne peut PAS recevoir plus de 3/5 en Complétude.

---
SUMMARY (ATTENDU — source de vérité) :
{summary}
---

---
HTML GÉNÉRÉ (À ÉVALUER) :
{html_code}
---

ÉVALUE selon les 5 critères. Score de 1 à 5, justification factuelle.

1. completude — Vues du summary présentes dans le HTML (via x-show) ?
   1 = moins de 40%  |  2 = 40-60%  |  3 = 60-80%  |  4 = 80-95%  |  5 = 100%
   ⚠️ Si des vues manquantes ont été détectées ci-dessus, score ≤ 3.

2. densite — Chaque vue est-elle riche ?
   Dashboard : 4+ KPI + table 6+ lignes. Liste : 6+ cards. Formulaire : 4-6 champs.
   Accueil : 5+ sections distinctes. Panier : table 4+ lignes + récap.
   1 = quasi vide  |  2 = squelette  |  3 = moyen  |  4 = riche  |  5 = très riche
   ⚠️ Si des tableaux courts ont été détectés ci-dessus, score ≤ 3.

3. donnees — Données fidèles au CDC ?
   Noms de produits exacts, devise correcte, prénoms du pays, villes du CDC.
   1 = Lorem ipsum / génériques  |  2 = plausibles mais inventées  |  3 = partiellement fidèles
   4 = quasi toutes fidèles  |  5 = 100% fidèles
   ⚠️ Si des placeholders [...] ont été détectés, score ≤ 2.
   ⚠️ Si la devise est fausse, score ≤ 3.

4. interactivite — Alpine.js correct ?
   x-data sur body, x-show par vue, @click sur navigation, composants interactifs.
   1 = aucune  |  2 = partiel  |  3 = navigation OK mais pas de composants
   4 = navigation + 1-2 composants  |  5 = complet (accordéons, modals, toggles)

5. coherence — Cohérence visuelle sur toutes les vues ?
   Même navbar/sidebar, même palette, même typographie, même style de cartes.
   1 = chaos  |  2 = partiel  |  3 = structure OK, couleurs variables
   4 = très cohérent  |  5 = parfait

ISSUES : identifie 2-5 issues SUPPLÉMENTAIRES (en plus de celles déjà détectées).
Ne répète PAS les issues déjà listées ci-dessus.
Concentre-toi sur ce que l'analyse automatique ne peut PAS détecter :
- Ton visuel inadapté au domaine
- Données sémantiquement fausses (pas juste des placeholders)
- Structure de page inadaptée (hero sur un dashboard, sidebar sur un ecommerce)
- Éléments du summary présents mais mal implémentés

Pour chaque issue :
- vue : nom de la vue
- severity : "high" | "medium" | "low"
- description : problème factuel
- suggestion : action corrective précise

Réponds UNIQUEMENT en JSON valide, sans markdown ni texte autour.

{{
  "criteria": {{
    "completude":    {{"score": <1-5>, "justification": "<factuel>"}},
    "densite":       {{"score": <1-5>, "justification": "<factuel>"}},
    "donnees":       {{"score": <1-5>, "justification": "<factuel>"}},
    "interactivite": {{"score": <1-5>, "justification": "<factuel>"}},
    "coherence":     {{"score": <1-5>, "justification": "<factuel>"}}
  }},
  "issues": [
    {{"vue": "<>", "severity": "<>", "description": "<>", "suggestion": "<>"}}
  ],
  "strengths": ["<point fort #1>", "<point fort #2>"],
  "commentaire": "<2 phrases : point fort + amélioration prioritaire>"
}}
"""


# ═══════════════════════════════════════════════════════════════
#  CLASSE ReviewerAgent V2
# ═══════════════════════════════════════════════════════════════

class ReviewerAgent(BaseAgent):
    """
    ReviewerAgent V2 — architecture 3 couches.

    Couche 1 (mécanique)   : détecte placeholders, tableaux courts, icônes cassées
    Couche 2 (conformité)  : compare summary ↔ HTML (vues, fonctionnalités, devise)
    Couche 3 (sémantique)  : LLM évalue qualité avec les résultats des couches 1-2

    Le LLM ne peut plus ignorer les défauts évidents parce qu'ils sont
    injectés dans son prompt comme des faits confirmés.
    """

    SCORE_THRESHOLD = 4.0
    BLOCKING_SEVERITY = "high"

    def __init__(self):
        super().__init__(name="ReviewerAgent", temperature=0.0, model="gpt-4o")

    def _build_chain(self):
        prompt = ChatPromptTemplate.from_template(REVIEW_PROMPT)
        return prompt | self.llm | StrOutputParser()

    def run(self, state: dict) -> dict:
        self._log("Évaluation du HTML en cours (3 couches)...")

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

        # ── Couche 1 : Vérification mécanique ─────────────────────
        self._log("  Couche 1 — vérification mécanique...")
        mechanical_issues = mechanical_check(html_code)
        self._log(f"  → {len(mechanical_issues)} défaut(s) mécanique(s) détecté(s)")

        # ── Couche 2 : Vérification de conformité ─────────────────
        self._log("  Couche 2 — vérification de conformité...")
        conformity_issues = conformity_check(html_code, summary)
        self._log(f"  → {len(conformity_issues)} défaut(s) de conformité détecté(s)")

        # ── Combiner les issues des couches 1-2 ───────────────────
        pre_check_issues = mechanical_issues + conformity_issues

        # Formatage texte pour injection dans le prompt LLM
        if pre_check_issues:
            pre_check_text = "\n".join(
                f"- [{issue['severity'].upper()}] ({issue.get('source', '?')}) "
                f"{issue['description']}"
                for issue in pre_check_issues
            )
        else:
            pre_check_text = "(Aucun défaut détecté par l'analyse automatique)"

        # ── Couche 3 : Évaluation sémantique LLM ─────────────────
        self._log("  Couche 3 — évaluation sémantique LLM...")
        try:
            raw = self._tracked_invoke({
                "summary":          summary,
                "html_code":        html_code,
                "pre_check_issues": pre_check_text,
            })

            llm_feedback = self._parse_json(raw)
            if llm_feedback is None:
                self._log("⚠️  JSON invalide du LLM, fallback")
                return self._fallback_result(state, pre_check_issues)

        except Exception as e:
            self._log(f"❌ Erreur LLM : {e}")
            return self._fallback_result(state, pre_check_issues)

        # ── Fusion des résultats ──────────────────────────────────
        feedback = self._merge_results(llm_feedback, pre_check_issues)

        # Score pondéré (recalculé, jamais confiance au LLM)
        score = self._compute_weighted_score(feedback)

        # Verdict hybride
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

    # ────────────────────────────────────────────────────────────
    #  Méthodes privées
    # ────────────────────────────────────────────────────────────

    def _merge_results(self, llm_feedback: dict, pre_check_issues: list) -> dict:
        """
        Fusionne les issues LLM avec les issues mécaniques/conformité.
        Les issues mécaniques ont priorité (elles sont confirmées).
        """
        # Issues LLM (marquées comme source "llm")
        llm_issues = llm_feedback.get("issues", [])
        for issue in llm_issues:
            issue["source"] = "llm"

        # Combine : pré-checks d'abord (prioritaires), puis LLM
        all_issues = pre_check_issues + llm_issues

        # Extraire les vues manquantes depuis les issues de conformité
        missing_views = []
        for issue in pre_check_issues:
            if "missing_views" in issue:
                missing_views.extend(issue["missing_views"])

        return {
            "criteria":      llm_feedback.get("criteria", {}),
            "issues":        all_issues,
            "missing_views": missing_views,
            "strengths":     llm_feedback.get("strengths", []),
            "commentaire":   llm_feedback.get("commentaire", ""),
            "pre_check_count": len(pre_check_issues),
            "llm_issue_count": len(llm_issues),
        }

    def _fallback_result(self, state: dict, pre_check_issues: list) -> dict:
        """
        Résultat de secours si le LLM échoue.
        Utilise les issues mécaniques/conformité pour calculer un score.
        """
        high_count = sum(1 for i in pre_check_issues if i.get("severity") == "high")
        medium_count = sum(1 for i in pre_check_issues if i.get("severity") == "medium")

        # Score heuristique basé sur les défauts détectés
        score = max(1.0, 5.0 - (high_count * 1.0) - (medium_count * 0.3))
        score = round(score, 1)

        feedback = {
            "criteria": {},
            "issues": pre_check_issues,
            "missing_views": [],
            "strengths": [],
            "commentaire": f"Évaluation de secours (LLM indisponible). {high_count} défaut(s) critique(s), {medium_count} défaut(s) moyens détectés mécaniquement.",
            "score_global": score,
            "verdict": "insufficient" if high_count > 0 else "good",
            "pre_check_count": len(pre_check_issues),
            "llm_issue_count": 0,
            "fallback": True,
        }

        return {
            **state,
            "review_feedback": feedback,
            "quality_score":   score,
            "verdict":         feedback["verdict"],
            "errors":          state.get("errors", []),
        }

    def _parse_json(self, raw: str) -> dict | None:
        cleaned = raw.strip()
        if cleaned.startswith("```"):
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
        criteria = feedback.get("criteria", {})
        if not criteria:
            return feedback.get("score_global", 2.0)
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
        criteria = feedback.get("criteria", {})
        if criteria:
            scores_str = " | ".join(
                f"{CRITERIA_LABELS.get(k, k)}: {criteria.get(k, {}).get('score', '?')}"
                for k in CRITERIA
            )
            self._log(f"   {scores_str}")

        pre = feedback.get("pre_check_count", 0)
        llm = feedback.get("llm_issue_count", 0)
        self._log(f"   Issues : {pre} mécanique(s) + {llm} sémantique(s)")

        issues = feedback.get("issues", [])
        high_issues = [i for i in issues if i.get("severity") == "high"]
        if high_issues:
            self._log(f"   🚨 {len(high_issues)} issue(s) bloquante(s) :")
            for i in high_issues[:5]:
                self._log(f"      - [{i.get('source', '?')}] {i.get('description', '')[:100]}")