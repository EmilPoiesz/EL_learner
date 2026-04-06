# EL Ontology Learner

An implementation of an exact learning algorithm for EL terminologies, based on Angluin-style queries to an oracle.

Given a target OWL ontology expressed in the EL description logic fragment, the learner reconstructs an equivalent terminology by issuing:
- **Membership queries (MQ)** — "Does O entail C ⊑ D?"
- **Equivalence queries (EQ)** — "Is my current hypothesis equivalent to O?"

The oracle answering these queries is pluggable. The current implementation provides a reasoner-backed oracle (ELK or HermiT), and the architecture is designed to allow the O-oracle to be replaced by a language model (`LLMOracle`).

The algorithm is described in:
> Duarte et al., *ExactLearner: a Tool for Exact Learning of EL Ontologies*.

---

## Prerequisites

- **Python 3.10+**
- **Java 11+** (Java 21 recommended)
- **ELK jar** — `elk-owlapi-standalone-0.4.2-bin.jar` from [Maven Central](https://repo1.maven.org/maven2/org/semanticweb/elk/elk-owlapi-standalone/0.4.2/), renamed to `elk-owlapi-standalone-0.4.2.jar` and placed in the project directory (or pointed to via the `ELK_JAR` environment variable).

> **ELK version note:** ELK 0.4.2 is required because it targets OWL API 3.x, which is compatible with the HermiT jar bundled by `owlready2`. Newer ELK versions target OWL API 4+ and are incompatible.

Install Python dependencies:

```bash
pip install -r requirements.txt
```

---

## Files

| File | Description |
|---|---|
| `EL_algorithm.py` | Core learning algorithm, EL concept data structures, and `Oracle` ABC |
| `Reasoner.py` | `HypothesisReasoner` and Java gateway helpers — shared by all oracle implementations |
| `ReasonerOracle.py` | `ReasonerOracle` — O-oracle backed by OWL API + ELK/HermiT via py4j |
| `LLMOracle.py` | `LLMOracle` — O-oracle skeleton to be backed by a language model |
| `OWLGateway.java` | Java gateway process exposing `add_gci`, `entails`, `clear` over py4j |
| `demo.py` | End-to-end demo on `ontologies/medical.ttl` |
| `unit_tests.py` | Unit and integration tests |
| `java_env.bash` | Helper script to compile `OWLGateway.java` |
| `ontologies/` | Example OWL/Turtle ontology files |

---

## Setup: Compile the Java gateway

The Java gateway wraps OWL API + ELK/HermiT and must be compiled before use. The helper script auto-detects the required jars from your Python environment:

```bash
bash java_env.bash
```

This produces `OWLGateway.class` in the project directory.

> **Note:** The script expects `owlready2`, `py4j`, and the ELK jar to be available. Activate your virtual environment first, and ensure the ELK jar is in the project directory or on `$VIRTUAL_ENV`.

---

## Architecture

### Oracle abstraction

The learning algorithm interacts with the outside world exclusively through the `Oracle` ABC defined in `EL_algorithm.py`. Any oracle must implement:

| Method | Description |
|---|---|
| `MQ(gci)` | Membership query — does O entail this GCI? |
| `EQ(H)` | Equivalence query — return a counterexample GCI, or `None` if H ≡ O |
| `signature` | Σ_O — the set of atomic concept names in O |
| `make_H_MQ(H)` | Return a callable for H-entailment (default: structural EL check) |
| `on_H_add(gci)` | Called when a GCI is added to H (override to keep an external reasoner in sync) |

Two oracle implementations are provided:

- **`ReasonerOracle`** (`ReasonerOracle.py`) — answers MQ and EQ using a DL reasoner (ELK or HermiT). The O-axioms are loaded once at construction; H-entailment is delegated to a `HypothesisReasoner`.
- **`LLMOracle`** (`LLMOracle.py`) — skeleton oracle that accepts a `HypothesisReasoner` for H-entailment and delegates MQ/EQ to a language model (not yet implemented).

### HypothesisReasoner

`HypothesisReasoner` (`Reasoner.py`) is a standalone component shared by both oracle types. It maintains a running Java OWLGateway process whose ontology is kept in sync with H via `add()` calls, and answers H-entailment queries via `entails()`. It is passed into `LLMOracle` as a constructor parameter, so the H-entailment side does not need to be reimplemented when swapping the O-oracle.

### Java gateway (ReasonerOracle only)

`ReasonerOracle` relies on Java reasoners (ELK and HermiT) which have no Python bindings. Communication happens via **py4j**, which runs a small Java process (`OWLGateway`) that listens on a local TCP port and exposes three methods to Python:

- `add_gci(lhs, rhs)` — load an axiom into the reasoner
- `entails(lhs, rhs)` — subsumption query
- `clear()` — reset the ontology

`ReasonerOracle` starts two separate gateway processes:

- **O-reasoner** — holds the target ontology O, used to answer MQ
- **H-reasoner** (`HypothesisReasoner`) — holds the current hypothesis H, used during EQ

Concepts are serialised as strings when passed over the py4j bridge (see [Concept encoding](#concept-encoding-python--java) below).

---

## Reasoners (ReasonerOracle)

Two reasoners are supported, selected via the `reasoner` parameter of `ReasonerOracle`:

| Reasoner | Value | Notes |
|---|---|---|
| **ELK** | `"elk"` | Default. OWL 2 EL reasoner; fast and sound/complete for the EL profile. Requires the ELK jar (see Prerequisites). |
| **HermiT** | `"hermit"` | Full OWL 2 DL reasoner; supports expressive axioms beyond EL but is slower. Bundled with `owlready2`, no extra download needed. |

ELK is the default because the learning algorithm targets the EL profile, making HermiT's extra expressivity unnecessary.

### ELK jar lookup order

The ELK jar is located by searching in this order:

1. `ELK_JAR` environment variable (explicit path)
2. Project directory (`gateway_jar_dir`)
3. Active virtual environment (`$VIRTUAL_ENV`)
4. Common system locations (`/usr/local/share`, `/usr/share`, `/opt/homebrew`, `~/.local/share`)

Set `ELK_JAR=/path/to/elk-owlapi-standalone-0.4.2.jar` to bypass the search entirely.

---

## Usage

```python
from ReasonerOracle import ReasonerOracle
from EL_algorithm import learn_el_terminology

# ELK is the default reasoner
oracle = ReasonerOracle(
    path="ontologies/medical.ttl",
    gateway_jar_dir="."        # directory containing OWLGateway.class
)
H = learn_el_terminology(oracle)

for gci in sorted(H, key=str):
    print(gci)

oracle.close()
```

To use HermiT instead:

```python
oracle = ReasonerOracle(
    path="ontologies/medical.ttl",
    reasoner="hermit"
)
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

Tests cover all algorithm sub-routines (saturate, merge, decompose, normalise) as well as a full end-to-end integration run on `ontologies/test_minimal.ttl`. Tests that require a running gateway are skipped automatically if the Java process cannot start.

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
        "compose_left":     0.6,   # replace LHS atom with a subclass from O
        "compose_right":    0.6,   # replace RHS atom with a superclass from O
    }
)
```

Each skill is applied independently with the given probability whenever a counterexample is found. All produced counterexamples are guaranteed to be O-entailed and H-not-entailed.

Omit `oracle_skills` (or pass `{}`) for a deterministic oracle that always returns the first unentailed axiom from O.
