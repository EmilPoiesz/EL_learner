# EL Ontology Learner

An implementation of an exact learning algorithm for EL terminologies, based on Angluin-style queries to an oracle.

Given a target OWL ontology expressed in the EL description logic fragment, the learner reconstructs an equivalent terminology by issuing:
- **Membership queries (MQ)** — "Does O entail C ⊑ D?"
- **Equivalence queries (EQ)** — "Is my current hypothesis equivalent to O?"

The oracle answering these queries is pluggable. Two implementations are provided:
- **`ReasonerOracle`** — backed by a DL reasoner (ELK or HermiT)
- **`LLMOracle`** — backed by a local HuggingFace language model

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

The `LLMOracle` additionally requires `transformers`, `accelerate`, and `torch`, which are included in `requirements.txt`.

---

## Files

| File | Description |
|---|---|
| `el_algorithm.py` | Core learning algorithm, EL concept data structures, and `Oracle` ABC |
| `hypothesis_reasoner.py` | `HypothesisReasoner` and Java gateway helpers — shared by all oracle implementations |
| `reasoner_oracle.py` | `ReasonerOracle` — O-oracle backed by OWL API + ELK/HermiT via py4j |
| `llm_oracle.py` | `LLMOracle` — O-oracle backed by a local HuggingFace language model; Manchester syntax serializer and parser |
| `demo.py` | End-to-end demo on `ontologies/medical.ttl` using `ReasonerOracle` |
| `demo_llm.py` | End-to-end demo using `LLMOracle` with a local HuggingFace model |
| `utils/java_utils.py` | Jar discovery and gateway lifecycle utilities |
| `utils/owl_parser.py` | OWL/Turtle ontology parser (rdflib-based) |
| `java/OWLGateway.java` | Java gateway process exposing `add_gci`, `entails`, `clear` over py4j |
| `java/java_env.bash` | Helper script to compile `OWLGateway.java` |
| `tests/test_el_algorithm.py` | Pytest test suite — unit and integration tests |
| `conftest.py` | Pytest fixtures and `--reasoner` CLI option |
| `ontologies/` | Example OWL/Turtle ontology files |
| `papers/` | Reference papers |

---

## Setup: Compile the Java gateway

The Java gateway wraps OWL API + ELK/HermiT and must be compiled before use. The helper script auto-detects the required jars from your Python environment:

```bash
bash java/java_env.bash
```

This produces `OWLGateway.class` in the `java/` directory.

> **Note:** The script expects `owlready2`, `py4j`, and the ELK jar to be available. Activate your virtual environment first, and ensure the ELK jar is in the project directory or on `$VIRTUAL_ENV`.

---

## Architecture

### Oracle abstraction

The learning algorithm interacts with the outside world exclusively through the `Oracle` ABC defined in `el_algorithm.py`. Any oracle must implement:

| Method | Description |
|---|---|
| `MQ(gci)` | Membership query — does O entail this GCI? |
| `EQ(H)` | Equivalence query — return a counterexample GCI, or `None` if H ≡ O |
| `signature` | Σ_O — the set of atomic concept names in O |
| `make_H_MQ(H)` | Return a callable for H-entailment (default: structural EL check) |
| `on_H_add(gci)` | Called when a GCI is added to H (override to keep an external reasoner in sync) |

Two oracle implementations are provided:

- **`ReasonerOracle`** (`reasoner_oracle.py`) — answers MQ and EQ using a DL reasoner (ELK or HermiT). The O-axioms are loaded once at construction; H-entailment is delegated to a `HypothesisReasoner`.
- **`LLMOracle`** (`llm_oracle.py`) — answers MQ and EQ by prompting a local HuggingFace language model using Manchester syntax. H-entailment is delegated to a `HypothesisReasoner`.

### HypothesisReasoner

`HypothesisReasoner` (`hypothesis_reasoner.py`) is a standalone component shared by both oracle types. It maintains a running Java OWLGateway process whose ontology is kept in sync with H via `add()` calls, and answers H-entailment queries via `entails()`. It is passed into `LLMOracle` as a constructor parameter, so the H-entailment side does not need to be reimplemented when swapping the O-oracle.

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

### ReasonerOracle

```python
from reasoner_oracle import ReasonerOracle
from el_algorithm import learn_el_terminology

with ReasonerOracle(
    path="ontologies/medical.ttl",
    gateway_jar_dir="."        # directory containing OWLGateway.class
) as oracle:
    H = learn_el_terminology(oracle)

for gci in sorted(H, key=str):
    print(gci)
```

To use HermiT instead of ELK:

```python
with ReasonerOracle(
    path="ontologies/medical.ttl",
    reasoner="hermit"
) as oracle:
    H = learn_el_terminology(oracle)
```

### LLMOracle

The `LLMOracle` uses a local HuggingFace model as the O-oracle. The model is loaded via `transformers.pipeline` and communicates through Manchester syntax prompts. H-entailment is still handled by `HypothesisReasoner` (Java/ELK).

```python
import os
from utils.java_utils import build_classpath
from hypothesis_reasoner import HypothesisReasoner
from llm_oracle import LLMOracle
from el_algorithm import learn_el_terminology

gateway_jar_dir = os.path.dirname(os.path.abspath(__file__))
h_reasoner = HypothesisReasoner(build_classpath(gateway_jar_dir), "elk")

with LLMOracle(
    model_name_or_path="HuggingFaceTB/SmolLM2-1.7B-Instruct",
    signature={"Person", "Male", "Female", "Parent", "Father", "Mother"},
    h_reasoner=h_reasoner,
    max_new_tokens=128,
    device="cpu",       # or "cuda" / "mps"
    verbose=False,
) as oracle:
    H = learn_el_terminology(oracle)
```

`learn_el_terminology` returns a `set[GCI]` representing what the language model knows about the domain.

#### Manchester syntax

`llm_oracle.py` exports module-level helpers for converting between `ELConcept`/`GCI` objects and Manchester syntax strings:

```python
from llm_oracle import concept_to_manchester, gci_to_manchester, parse_manchester_gci, parse_manchester_concept
```

| Function | Direction |
|---|---|
| `concept_to_manchester(concept)` | `ELConcept` → string |
| `gci_to_manchester(gci)` | `GCI` → string |
| `parse_manchester_concept(s)` | string → `ELConcept` |
| `parse_manchester_gci(s)` | string → `GCI` (raises `ValueError` on invalid input) |

#### How MQ and EQ work

**MQ** sends a single prompt asking the model to answer yes or no:

```
Concept names in scope: Father, Female, Male, Mother, Parent, Person.
Does the target ontology O entail the following GCI?
  Father SubClassOf Parent
Reply with exactly one word: yes or no.
```

**EQ** uses two sequential prompts:

1. **Judge** — asks whether H is complete:
```
Does H capture all subsumptions entailed by the target ontology O?
Reply with exactly one word: yes or no.
```

2. **Counterexample** — sent only when the judge answers "no", asks for a missing GCI in Manchester syntax:
```
Name one GCI that is entailed by target ontology O but is NOT entailed by H.
Output exactly one line in Manchester syntax, e.g.: Cat SubClassOf Animal
Output nothing else.
```

Parsing is strict: any response that does not parse as a valid GCI raises `ValueError`.

#### Verbose mode

Pass `verbose=True` to `LLMOracle` (or `-v` in `demo_llm.py`) to print every prompt and response, labelled by query type:

```
--- MQ ---
[PROMPT] ...
[RESPONSE] yes

--- EQ (judge) ---
[PROMPT] ...
[RESPONSE] no

--- EQ (counterexample) ---
[PROMPT] ...
[RESPONSE] Male and Parent SubClassOf Father
```

---

## Demo

### ReasonerOracle demo

```bash
python demo.py                      # run on ontologies/medical.ttl with ELK
python demo.py --reasoner=hermit    # use HermiT instead
python demo.py -v                   # verbose logging
```

### LLMOracle demo

```bash
python demo_llm.py                  # run learning loop (CPU)
python demo_llm.py --device=mps     # Apple Silicon GPU
python demo_llm.py --device=cuda    # NVIDIA GPU
python demo_llm.py -v               # print all prompts and responses
python demo_llm.py --dry-run        # preview prompts without loading the model
```

The demo uses `HuggingFaceTB/SmolLM2-1.7B-Instruct` on a kinship domain (`{Person, Male, Female, Parent, Father, Mother}`). The model is downloaded automatically on first run and cached in `~/.cache/huggingface/`. To pre-download:

```bash
huggingface-cli download HuggingFaceTB/SmolLM2-1.7B-Instruct
```

---

## Tests

```bash
pytest                          # ELK reasoner, compact output
pytest -v                       # verbose output
pytest --reasoner=hermit        # HermiT reasoner
pytest --reasoner=hermit -v
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

## Oracle skills (optional, ReasonerOracle only)

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
