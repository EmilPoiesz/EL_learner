"""
PAC-sampling equivalence queries, following the original ExactLearner-LLM
(``org.pac.Pac`` and ``LaunchLearner.getCounterExample``).

Instead of asking the oracle whether H ≡ O, an equivalence query draws random
statements over the signature and returns the first one that H does not entail
but the oracle's MQ does.  If none is found within the sample budget, H is
accepted as probably approximately correct:

    budget = round((h · ln|X| − ln δ) / ε)

where h is the size of the target ontology and |X| the number of statements
that can be drawn.  The budget is global: samples drawn by one EQ are not
drawn again by the next, and learning stops once the budget is spent.
"""

from __future__ import annotations

import logging
import math
import random
from typing import Callable, Iterable, Optional

from learner.el_algorithm import ELConcept, GCI, Oracle

logger = logging.getLogger(__name__)


def role_names(gcis: Iterable[GCI]) -> set[str]:
    """Every role name that occurs in an existential of the given GCIs."""
    roles: set[str] = set()

    def _walk(concept: ELConcept) -> None:
        for role, filler in concept.existentials:
            roles.add(role)
            _walk(filler)

    for gci in gcis:
        _walk(gci.lhs)
        _walk(gci.rhs)
    return roles


class PACSampler:
    """
    Draws statements uniformly from three shapes, as upstream's ``Pac``:

      1. A ⊓ B ⊑ C     (A may equal B; C differs from both)
      2. B ⊑ ∃r.A
      3. ∃r.A ⊑ B

    Three indices are drawn over concept names + roles; an index triple that
    fits none of the shapes is redrawn.  Sample i uses ``random.Random(seed + i)``,
    so the sequence is reproducible and does not depend on how many EQs drew it.
    Python's generator differs from Java's, so the statements are not the same
    ones upstream draws, only drawn the same way.
    """

    def __init__(
        self,
        classes: Iterable[str],
        roles: Iterable[str],
        epsilon: float = 0.2,
        delta: float = 0.1,
        hypothesis_size: int = 1,
        seed: int = 0,
    ):
        self.classes = [ELConcept({c}) for c in sorted(classes)]
        self.roles = sorted(roles)
        self.epsilon = epsilon
        self.delta = delta
        self.seed = seed
        cn, rn = len(self.classes), len(self.roles)
        self.instance_space_size = cn * cn * (cn - 1) + 2 * cn * cn * rn
        self.budget = round(
            (hypothesis_size * math.log(self.instance_space_size) - math.log(delta)) / epsilon
        )
        self.provided = 0

    def next_statement(self) -> Optional[GCI]:
        """The next random statement, or None once the budget is spent."""
        if self.provided >= self.budget:
            return None
        rng = random.Random(self.seed + self.provided)
        cn = len(self.classes)
        span = cn + len(self.roles)
        statement = None
        while statement is None:
            i, j, k = rng.randrange(span), rng.randrange(span), rng.randrange(span)
            if i < cn and j < cn and k < cn:
                if k in (i, j):
                    continue  # C must differ from A and B
                statement = GCI(
                    ELConcept(self.classes[i].atoms | self.classes[j].atoms),
                    self.classes[k],
                )
            elif i < cn and j >= cn and k < cn:
                statement = GCI(
                    self.classes[i],
                    ELConcept(existentials={(self.roles[j - cn], self.classes[k])}),
                )
            elif i >= cn and j < cn and k < cn:
                statement = GCI(
                    ELConcept(existentials={(self.roles[i - cn], self.classes[j])}),
                    self.classes[k],
                )
        self.provided += 1
        return statement


class PACOracle(Oracle):
    """
    Wraps any oracle and replaces its equivalence query by PAC sampling.

    MQs, the signature and H-entailment are the inner oracle's.  ``mq_count``
    counts the learner's MQs only; the MQs asked while sampling are counted in
    ``sample_mq_count`` (the inner oracle's ``mq_count`` holds both).

    Parameters
    ----------
    inner : Oracle
        The oracle answering MQs (an LLMOracle, or a ReasonerOracle for testing).
    roles : set[str]
        Role names to sample with; ``None`` takes those in ``inner.axioms``.
    epsilon, delta : float
        PAC accuracy and confidence; upstream defaults 0.2 and 0.1.
    hypothesis_size : int | None
        h in the budget; ``None`` takes the number of GCIs in ``inner.axioms``.
    seed : int
        Seed of the statement sequence (upstream: 0).
    """

    def __init__(
        self,
        inner: Oracle,
        roles: Optional[set[str]] = None,
        epsilon: float = 0.2,
        delta: float = 0.1,
        hypothesis_size: Optional[int] = None,
        seed: int = 0,
    ):
        super().__init__()
        self.inner = inner
        if roles is None:
            roles = role_names(inner.axioms)
        if hypothesis_size is None:
            hypothesis_size = len(inner.axioms)
        self.sampler = PACSampler(inner.signature, roles, epsilon, delta, hypothesis_size, seed)
        self.sample_mq_count = 0
        self._H_MQ: Optional[Callable[[GCI], bool]] = None
        logger.info(
            "PAC sampling: ε=%s δ=%s h=%d |X|=%d → budget %d statements",
            epsilon, delta, hypothesis_size, self.sampler.instance_space_size, self.sampler.budget,
        )

    @property
    def signature(self) -> set[str]:
        return self.inner.signature

    @property
    def axioms(self) -> set[GCI]:
        return self.inner.axioms

    def _MQ(self, gci: GCI) -> bool:
        return self.inner.MQ(gci)

    def _EQ(self, hypothesis: set[GCI]) -> Optional[GCI]:
        H_MQ = self._H_MQ or self.inner.make_H_MQ(hypothesis)
        while (statement := self.sampler.next_statement()) is not None:
            # H first: it is a local reasoner call, the MQ may be an LLM call.
            if H_MQ(statement):
                continue
            self.sample_mq_count += 1
            if self.inner.MQ(statement):
                logger.info(
                    "PAC sample %d/%d is a counterexample: %s",
                    self.sampler.provided, self.sampler.budget, statement,
                )
                return statement
        logger.info("PAC budget of %d samples spent without a counterexample.", self.sampler.budget)
        return None

    def make_H_MQ(self, H: set[GCI]) -> Callable[[GCI], bool]:
        self._H_MQ = self.inner.make_H_MQ(H)
        return self._H_MQ

    def on_H_add(self, gci: GCI) -> None:
        self.inner.on_H_add(gci)

    def close(self) -> None:
        close = getattr(self.inner, "close", None)
        if close is not None:
            close()

    def __enter__(self) -> "PACOracle":
        return self

    def __exit__(self, *args) -> None:
        self.close()
