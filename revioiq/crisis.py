"""Rule- and heuristic-based crisis detection for customer reviews.

Detects high-risk business crisis events:
- Food poisoning & health violations
- Legal threats & litigation
- Discrimination & civil rights violations
- Fraud & unauthorized charges
- Media exposure & investigative reporting threats

Outputs crisis flag, crisis type, severity level, and matched evidence.
Prioritizes high recall to ensure human review of all critical incidents.
"""

from dataclasses import dataclass, field
import re
from typing import Dict, List, Optional, Tuple


@dataclass
class CrisisResult:
    """Detection result for a single review."""

    crisis_flag: bool
    crisis_type: str
    severity: str  # "none", "low", "medium", "high", "critical"
    matched_evidence: List[str] = field(default_factory=list)
    routing_action: str = "auto_reply"  # "auto_reply" or "human_review"

    def to_dict(self) -> Dict:
        return {
            "crisis_flag": self.crisis_flag,
            "crisis_type": self.crisis_type,
            "severity": self.severity,
            "matched_evidence": self.matched_evidence,
            "routing_action": self.routing_action,
        }


# Pattern definitions by crisis category
CRISIS_PATTERNS: Dict[str, List[str]] = {
    "food_poisoning": [
        r"\b(?:food[\s-]poison(?:ing|ed)?)\b",
        r"\b(?:salmonella|e\.?\s*coli|norovirus|listeria)\b",
        r"\b(?:vomit(?:ing|ed)?|diarrhea|stomach\s+pumped)\b",
        r"\b(?:emergency\s+room|er|hospital(?:ized)?)\b",
        r"\b(?:bacterial\s+gastroenteritis|food[\s-]borne\s+illness)\b",
        r"\b(?:health\s+department|health\s+inspector)\b",
    ],
    "legal_threat": [
        r"\b(?:attorney|lawyer|legal\s+counsel)\b",
        r"\b(?:lawsuit|litigation|sue|suing)\b",
        r"\b(?:civil\s+court|small\s+claims|before\s+a\s+judge)\b",
        r"\b(?:summons|service\s+of\s+process|notice\s+of\s+legal\s+action)\b",
        r"\b(?:retained\s+legal|retaining\s+counsel)\b",
        r"\b(?:breach\s+of\s+contract|gross\s+negligence)\b",
    ],
    "discrimination": [
        r"\b(?:racist|racism|racial\s+slur[s]?)\b",
        r"\b(?:discriminat(?:ion|ed|ing|ory))\b",
        r"\b(?:service\s+dog|ada\s+violation|violation\s+of\s+the\s+ada)\b",
        r"\b(?:disabled|handicap(?:ped)?)\b",
        r"\b(?:gender\s+identity|sexual\s+orientation|eeoc)\b",
        r"\b(?:hate\s+speech|civil\s+rights)\b",
    ],
    "fraud_accusation": [
        r"\b(?:credit\s+card\s+fraud|financial\s+fraud)\b",
        r"\b(?:unauthorized\s+(?:tip|charge|transaction|billing))\b",
        r"\b(?:double[\s-]charg(?:ing|ed)|duplicate\s+billing)\b",
        r"\b(?:fraud\s+investigation|fraudulent\s+scheme)\b",
        r"\b(?:stole\s+my\s+money|theft|scam(?:mer|med)?)\b",
        r"\b(?:fake\s+charge|unauthorized\s+\$\d+)\b",
    ],
    "media_threat": [
        r"\b(?:investigative\s+reporter|investigative\s+journalist[s]?)\b",
        r"\b(?:channel\s+\d+|tv\s+news|local\s+news|news\s+desk[s]?)\b",
        r"\b(?:consumer\s+watchdog|local\s+press|press\s+release)\b",
        r"\b(?:expose\s+video|broadcast|tagging\s+media)\b",
        r"\b(?:tiktok|instagram|social\s+media\s+following|followers)\b.*?\b(?:expose|media|outlets|press)\b",
        r"\b(?:rodent\s+infestation|cockroach|maggots?)\b.*?\b(?:press|news|public|photos?)\b",
    ],
}


class CrisisDetector:
    """Evaluates reviews against crisis rules and severity guidelines."""

    def __init__(self):
        self.compiled_rules = {
            category: [re.compile(p, re.IGNORECASE) for p in patterns]
            for category, patterns in CRISIS_PATTERNS.items()
        }

    def detect(self, text: str, sentiment: Optional[str] = None) -> CrisisResult:
        """Analyze text for crisis patterns and determine risk severity."""
        if not text or not isinstance(text, str):
            return CrisisResult(
                crisis_flag=False,
                crisis_type="none",
                severity="none",
                matched_evidence=[],
                routing_action="auto_reply",
            )

        matches_by_category: Dict[str, List[str]] = {}

        for category, regex_list in self.compiled_rules.items():
            for regex in regex_list:
                found = regex.findall(text)
                if found:
                    if category not in matches_by_category:
                        matches_by_category[category] = []
                    for item in found:
                        if isinstance(item, tuple):
                            item = " ".join([x for x in item if x])
                        item_str = str(item).strip()
                        if item_str and item_str not in matches_by_category[category]:
                            matches_by_category[category].append(item_str)

        if not matches_by_category:
            return CrisisResult(
                crisis_flag=False,
                crisis_type="none",
                severity="none",
                matched_evidence=[],
                routing_action="auto_reply",
            )

        # Pick primary crisis type (the category with most or highest severity matches)
        sorted_categories = sorted(
            matches_by_category.items(),
            key=lambda kv: len(kv[1]),
            reverse=True,
        )
        primary_category = sorted_categories[0][0]
        all_evidence = []
        for cat, ev_list in sorted_categories:
            all_evidence.extend([f"[{cat}] {ev}" for ev in ev_list])

        total_match_count = sum(len(v) for v in matches_by_category.values())

        # Determine severity
        critical_types = {"food_poisoning", "legal_threat", "discrimination"}
        if total_match_count >= 2 or primary_category in critical_types:
            severity = "critical"
        elif total_match_count >= 1:
            severity = "high"
        else:
            severity = "medium"

        # Crisis reviews are NEVER auto-sent; always routed to a human
        return CrisisResult(
            crisis_flag=True,
            crisis_type=primary_category,
            severity=severity,
            matched_evidence=all_evidence,
            routing_action="human_review",
        )


_GLOBAL_DETECTOR: Optional[CrisisDetector] = None


def get_crisis_detector() -> CrisisDetector:
    """Singleton getter for CrisisDetector."""
    global _GLOBAL_DETECTOR
    if _GLOBAL_DETECTOR is None:
        _GLOBAL_DETECTOR = CrisisDetector()
    return _GLOBAL_DETECTOR
