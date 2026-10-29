"""
Evaluation of a learned ontology H against the target O, as the original
ExactLearner-LLM scores its results (``org.analysis.ResultAnalyzer`` with
``OntologyManipulator.getAllPossibleAxiomsCombinationsOWL``).

Every statement of four shapes over the signature is checked against O and
against H, and the two are compared as a binary classification with "O entails
it" as the positive class:

  1. A ⊑ B            A ≠ B                          cn·(cn−1)
  2. A ⊓ B ⊑ C        {A, B} unordered, A ≠ B, C ∉ {A, B}   C(cn,2)·(cn−2)
  3. B ⊑ ∃r.A                                       cn²·rn
  4. ∃r.A ⊑ B                                       cn²·rn

For animals.owl (17 concept names, 4 roles) that is 4624 statements, 272 of
them entailed by O; the paper's Mistral recall of 271/272 = 0.99632 fits.

Upstream takes the names from the OWL API's class signature, which holds
owl:Thing whenever the file mentions it (football.owl: ``Game ⊑ owl:Thing``).
``top`` adds ⊤ as one more name to match: football then has 1050 statements,
163 entailed, as upstream counts.  Pass the names of ``OntologyInfo`` and an
O-entailment over the raw file (``load_ontology``) to get upstream's counts
for generations.owl too (7000 statements, 1159 entailed).
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Callable, Iterable, Iterator

from learner.el_algorithm import ELConcept, GCI


def all_statements(classes: Iterable[str], roles: Iterable[str], top: bool = False) -> Iterator[GCI]:
    """The four statement shapes over the given names (and ⊤ with ``top``), in a fixed order."""
    # ⊤ is the empty conjunction: ⊤ ⊓ B is B, and ∃r.⊤ is an ordinary existential.
    cs = [ELConcept({c}) for c in sorted(classes)] + ([ELConcept()] if top else [])
    rs = sorted(roles)
    for a in cs:
        for b in cs:
            if a != b:
                yield GCI(a, b)
    for a, b in combinations(cs, 2):
        for c in cs:
            if c not in (a, b):
                yield GCI(ELConcept(a.atoms | b.atoms), c)
    for b in cs:
        for r in rs:
            for a in cs:
                yield GCI(b, ELConcept(existentials={(r, a)}))
    for a in cs:
        for r in rs:
            for b in cs:
                yield GCI(ELConcept(existentials={(r, a)}), b)


@dataclass
class OntologyScore:
    tp: int = 0   # O entails, H entails
    fn: int = 0   # O entails, H does not
    fp: int = 0   # O does not, H entails
    tn: int = 0   # neither

    @property
    def total(self) -> int:
        return self.tp + self.fn + self.fp + self.tn

    @property
    def accuracy(self) -> float:
        return (self.tp + self.tn) / self.total if self.total else float("nan")

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else float("nan")

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else float("nan")

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if self.tp else 0.0


def score_ontology(
    O_entails: Callable[[GCI], bool],
    H_entails: Callable[[GCI], bool],
    classes: Iterable[str],
    roles: Iterable[str],
    top: bool = False,
) -> OntologyScore:
    """Compare H with O on every statement of ``all_statements``."""
    score = OntologyScore()
    for gci in all_statements(classes, roles, top):
        o, h = O_entails(gci), H_entails(gci)
        if o and h:
            score.tp += 1
        elif o:
            score.fn += 1
        elif h:
            score.fp += 1
        else:
            score.tn += 1
    return score
