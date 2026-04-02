# EL Ontology Learner

An implementation of an exact learning algorithm for EL terminologies, based on Angluin-style queries to a DL reasoner (HermiT).

Given a target OWL ontology expressed in the EL description logic fragment, the learner reconstructs an equivalent terminology by issuing:
- **Membership queries (MQ)** — "Does O entail C ⊑ D?"
- **Equivalence queries (EQ)** — "Is my current hypothesis equivalent to O?"

The algorithm is described in:
> Duarte et al., *ExactLearner: a Tool for Exact Learning of EL Ontologies*.

---

## Prerequisites

- **Python 3.10+**
- **Java 11+** (Java 21 recommended)

Install Python dependencies:

```bash
pip install -r requirements.txt
```

---

## Files

| File | Description |
|---|---|
| `EL_algorithm.py` | Core learning algorithm and EL concept data structures |
| `HermiT_reasoner.py` | `ReasonerOracle` — bridges Python to OWL API + HermiT via py4j |
| `OWLGateway.java` | Java gateway process exposing `add_gci`, `entails`, `clear` over py4j |
| `demo.py` | End-to-end demo on `ontologies/medical.ttl` |
| `unit_tests.py` | Unit and integration tests |
| `java_env.bash` | Helper script to compile `OWLGateway.java` |
| `ontologies/` | Example OWL/Turtle ontology files |

---

## Setup: Compile the Java gateway

The Java gateway wraps OWL API + HermiT and must be compiled before use. The helper script auto-detects the required jars from your Python environment:

```bash
bash java_env.bash
```

This produces `OWLGateway.class` in the project directory.

> **Note:** The script expects `owlready2` and `py4j` to be installed (they bundle the required jars). If you use a virtual environment, activate it first.

---

## Usage

```python
from HermiT_reasoner import ReasonerOracle
from EL_algorithm import learn_el_terminology

oracle = ReasonerOracle(
    path="ontologies/medical.ttl",
    gateway_jar_dir="."        # directory containing OWLGateway.class
)
H = learn_el_terminology(oracle)

for gci in sorted(H, key=str):
    print(gci)

oracle.close()
```

`learn_el_terminology` returns a `set[GCI]` equivalent to the target ontology O.

---

## Demo

```bash
python demo.py          # run on ontologies/medical.ttl
python demo.py -v       # verbose (shows all logging)
```

---

## Tests

```bash
python unit_tests.py        # compact output
python unit_tests.py -v     # verbose
```

Tests cover all algorithm sub-routines (saturate, merge, decompose, normalise) as well as a full end-to-end integration run on `ontologies/test_minimal.ttl`. Tests that require HermiT are skipped automatically if the gateway cannot start.

---

## Concept encoding (Python ↔ Java)

Concepts are encoded as strings when passed to the Java gateway:

| EL concept   | Encoding                      |
|---|---|
| ⊤            | `TOP`                         |
| A            | `A`                           |
| A ⊓ B        | `AND:(A),(B)`                 |
| ∃r.A         | `SOME:r:(A)`                  |
| A ⊓ ∃r.B    | `AND:(A),(SOME:r:(B))`        |

---

## Oracle skills (optional)

`ReasonerOracle` accepts an `oracle_skills` dict that probabilistically transforms counterexamples before returning them, simulating a more varied teacher:

```python
oracle = ReasonerOracle(
    path="ontologies/medical.ttl",
    oracle_skills={
        "saturate_left":    0.8,   # add concept names to the LHS
        "unsaturate_right": 0.5,   # remove atoms from the RHS
        "compose_left":     0.6,   # replace LHS atom with a subclass
        "compose_right":    0.6,   # replace RHS atom with a superclass
    }
)
```

Omit `oracle_skills` (or pass `{}`) for a deterministic oracle that always returns the first unentailed axiom from O.
