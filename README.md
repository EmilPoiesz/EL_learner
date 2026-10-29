# EL Ontology Learner

An implementation of an exact learning algorithm for EL terminologies, based on Angluin-style queries to an oracle.

[![Test](https://github.com/EmilPoiesz/EL_Learner/actions/workflows/check.yml/badge.svg?event=push)]()
![Codecov](https://img.shields.io/codecov/c/github/EmilPoiesz/EL_Learner)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

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
- **Java 11**
- **ELK jar** — `elk-owlapi-standalone-0.4.2-bin.jar` from [Maven Central](https://repo1.maven.org/maven2/org/semanticweb/elk/elk-owlapi-standalone/0.4.2/), renamed to `elk-owlapi-standalone-0.4.2.jar` and placed in the `java/` directory (or pointed to via the `ELK_JAR` environment variable).

> **ELK version note:** ELK 0.4.2 is required because it targets OWL API 3.x. The ELK standalone jar does not include the OWL API itself; those classes come from the `HermiT.jar` bundled with `owlready2` (located by `find_owlapi_jar()`), so that jar is always on the classpath, whichever reasoner is selected. Newer ELK versions target OWL API 4+ and are incompatible.

Install Python dependencies:

```bash
pip install -r requirements.txt
```

Alternatively, if you use [Poetry](https://python-poetry.org/):

```bash
poetry install
```

The `LLMOracle` additionally requires `transformers`, `accelerate`, and `torch`, which are included in `requirements.txt`.

---

## Project structure

```
.
├── learner/
│   ├── el_algorithm.py          # Core learning algorithm, EL concept data structures, Oracle ABC
│   ├── hypothesis_reasoner.py   # HypothesisReasoner — shared H-entailment component
│   ├── reasoner_oracle.py       # ReasonerOracle — O-oracle backed by OWL API + ELK/HermiT
│   ├── llm_oracle.py            # LLMOracle — O-oracle backed by a local HuggingFace model
│   └── cache/
│       ├── __init__.py
│       ├── hashing.py           # stable_hash() — deterministic SHA-256 key generation
│       ├── backend.py           # SQLiteCacheBackend — persistent SQLite storage
│       └── cache.py             # LLMCache — two-level cache (in-memory + SQLite)
├── utils/
│   ├── java_utils.py            # Jar discovery and gateway lifecycle utilities
│   └── owl_parser.py            # OWL/Turtle ontology parser (rdflib-based)
├── java/
│   ├── OWLGateway.java          # Java gateway process exposing add_gci, entails, clear over py4j
│   ├── OWLGateway.class         # Compiled gateway (Java 11)
│   └── java_env.bash            # Helper script to recompile OWLGateway.java
├── tests/
│   ├── conftest.py              # Pytest fixtures and --reasoner CLI option
│   ├── test_el_algorithm.py     # Unit and integration tests
│   └── test_cache.py            # Unit tests for the LLMCache layer
├── ontologies/
│   ├── medical.ttl              # Example medical ontology
│   ├── test_minimal.ttl         # Minimal ontology used by the test suite
│   └── animals.owl              # Small Animal ontology 
├── .github/workflows/
│   └── check.yml                # CI workflow — runs tests with coverage on every push
├── demo.py                      # Unified demo entry point
├── requirements.txt             # Pinned dependencies
└── pyproject.toml               # Poetry project config
```

---

## Setup: Compile the Java gateway

`OWLGateway.class` is not committed. Compile it once after cloning, and again after any change to `OWLGateway.java`, using the helper script, which auto-detects the required jars from your Python environment:

```bash
bash java/java_env.bash
```

This produces `OWLGateway.class` (Java 11 bytecode, also when compiled with a newer JDK) in the `java/` directory.

> **Note:** The script expects `owlready2` and `py4j` to be installed in the active virtual environment. The ELK jar is not needed to compile, since `OWLGateway` loads ELK by reflection at runtime.

---

## Architecture

### Oracle abstraction

The learning algorithm interacts with the outside world exclusively through the `Oracle` ABC defined in `el_algorithm.py`. The learner uses these members:

| Method | Description |
|---|---|
| `MQ(gci)` | Membership query — does O entail this GCI? |
| `EQ(H)` | Equivalence query — return a counterexample GCI, or `None` if H ≡ O |
| `signature` | Σ_O — the set of atomic concept names in O |
| `make_H_MQ(H)` | Return a callable answering H ⊨ α. Both shipped oracles return their `HypothesisReasoner` |
| `on_H_add(gci)` | Called after each GCI is added to H; keeps the H-reasoner in sync |

`MQ` and `EQ` are concrete wrappers that count calls (`mq_count`, `eq_count`) and delegate to `_MQ` / `_EQ`. A new oracle implements `signature`, `_MQ`, `_EQ`, `make_H_MQ` and `on_H_add`; overriding `MQ`/`EQ` directly would bypass the counters.

Two oracle implementations are provided:

- **`ReasonerOracle`** (`reasoner_oracle.py`) — answers MQ and EQ using a DL reasoner (ELK or HermiT). The O-axioms are loaded once at construction; H-entailment is delegated to a `HypothesisReasoner`.
- **`LLMOracle`** (`llm_oracle.py`) — answers MQ and EQ by prompting a local HuggingFace language model using Manchester syntax. H-entailment is delegated to a `HypothesisReasoner`.

### Algorithm internals

The core of the learner is `learn_el_terminology` in `el_algorithm.py`.  When a counterexample `C ⊑ D` is received from the oracle it is processed by one of two pipelines depending on which side is atomic.

**Right O-essential** (C is atomic):

1. **Concept saturation** (`saturate_concept_rhs`) — close the RHS under O-subsumptions to add all implied atomic names.
2. **Sibling merge** (`sibling_merge`) — merge same-role existentials validated against O.
3. **RHS decomposition** (`decompose_rhs`) — split off sub-GCIs already known to H.

**Left O-essential** (D is atomic):

1. **Left saturation** (`saturate_left`) — add to each LHS node every atomic name implied by H.
2. **LHS decomposition** (`decompose_left`) — prune existentials and descend into sub-concepts whose conjunction alone forces D.
3. **Left desaturation** (`unsaturate_left`) — iterate over every node in the LHS tree and remove each atomic concept name whose removal still leaves the GCI entailed by O.  This minimises the left-hand side so the learner does not commit to an unnecessarily strong precondition.

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

## Caching

### LLMOracle — two-level response cache

`LLMOracle` automatically caches model responses to avoid redundant inference across runs. The cache is keyed on a deterministic SHA-256 hash of the model name and the full message list, so identical prompts always resolve to the same key regardless of run order.

Two storage layers are consulted in order:

| Layer | Implementation | Scope |
|---|---|---|
| **Memory** | `dict[str, str]` | Current process only |
| **Disk** | SQLite (`llm_cache.db`) | Persistent across runs |

A cache hit on the memory layer requires no disk I/O. A miss falls through to SQLite and promotes the value to memory. The disk cache is never invalidated automatically — delete the `.db` file to start fresh.

By default the database is `llm_cache.db` in the working directory, i.e. the project's top directory when `demo.py` is run from there (the file is git-ignored).

The cache key covers the model name, the messages and the generation settings (`max_new_tokens`, `do_sample`), so a response generated under one token budget is never returned for another.

The cache is controlled by three environment variables:

| Variable | Default | Effect |
|---|---|---|
| `LLM_CACHE_ENABLED` | `1` | Set to `0` to disable all caching |
| `LLM_CACHE_DB` | `llm_cache.db` | Path to the SQLite database file (relative paths resolve against the working directory) |
| `LLM_CACHE_STORE_PROMPTS` | `0` | Set to `1` to also store the raw prompt string alongside the response (useful for debugging) |

Example — run with a custom DB path and prompt logging:

```bash
LLM_CACHE_DB=/tmp/my_cache.db LLM_CACHE_STORE_PROMPTS=1 python demo.py --oracle llm
```

Example — disable caching entirely:

```bash
LLM_CACHE_ENABLED=0 python demo.py --oracle llm
```

### ReasonerOracle — in-memory MQ cache

`ReasonerOracle` maintains a lightweight in-memory `dict[GCI, bool]` that memoises the result of every membership query against the target ontology O. Repeated MQ calls for the same GCI within a single learning run skip the Java/py4j round-trip entirely. This cache lives only for the lifetime of the oracle instance and is not persisted to disk.

---

## Reasoners (ReasonerOracle)

Two reasoners are supported, selected via the `reasoner` parameter of `ReasonerOracle`:

| Reasoner | Value | Notes |
|---|---|---|
| **ELK** | `"elk"` | Default. OWL 2 EL reasoner; fast and sound/complete for the EL profile. Requires the ELK jar (see Prerequisites). |
| **HermiT** | `"hermit"` | Full OWL 2 DL reasoner; supports expressive axioms beyond EL but is slower. Bundled with `owlready2`, no extra download needed. |

ELK is the default everywhere (`ReasonerOracle`, `HypothesisReasoner`, `demo.py`, `pytest`, and the `OWLGateway` entry point) because the learning algorithm targets the EL profile, so HermiT's extra expressivity is unnecessary. HermiT remains available as an alternative, e.g. to cross-check ELK's answers: pass `reasoner="hermit"`, `--reasoner hermit` to `demo.py`, or `--reasoner=hermit` to `pytest`. With `ReasonerOracle`, the choice applies to both the O-reasoner and the H-reasoner. The LLM demos always use ELK for the H-reasoner.

### ELK jar lookup order

The ELK jar is located by searching in this order:

1. `ELK_JAR` environment variable (explicit path)
2. `java/` directory (`gateway_jar_dir`)
3. Active virtual environment (`$VIRTUAL_ENV`)
4. Common system locations (`/usr/local/share`, `/usr/share`, `/opt/homebrew`, `~/.local/share`)

Set `ELK_JAR=/path/to/elk-owlapi-standalone-0.4.2.jar` to bypass the search entirely.

---

## Usage

### ReasonerOracle

```python
from learner.reasoner_oracle import ReasonerOracle
from learner.el_algorithm import learn_el_terminology

with ReasonerOracle(
        path="ontologies/medical.ttl",
        gateway_jar_dir="java"  # directory containing OWLGateway.class
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
from learner.hypothesis_reasoner import HypothesisReasoner
from learner.llm_oracle import LLMOracle
from learner.el_algorithm import learn_el_terminology

gateway_jar_dir = "java"
h_reasoner = HypothesisReasoner(build_classpath(gateway_jar_dir), "elk")

with LLMOracle(
        model_name_or_path="HuggingFaceTB/SmolLM2-1.7B-Instruct",
        signature={"Person", "Male", "Female", "Parent", "Father", "Mother"},
        h_reasoner=h_reasoner,
        max_new_tokens=128,
        device="cpu",  # or "cuda" / "mps"
        verbose=False,
) as oracle:
    H = learn_el_terminology(oracle)
```

`learn_el_terminology` returns a `set[GCI]` representing what the language model knows about the domain.

In this parametric mode (no `ontology_path`) the prompts list no object properties, so the model is effectively limited to atomic subsumptions. Pass `ontology_path` for grounded mode, where the signature, roles and axioms are taken from the ontology file and placed in the system prompt.

#### Manchester syntax

`llm_oracle.py` exports module-level helpers for converting between `ELConcept`/`GCI` objects and Manchester syntax strings:

```python
from learner.llm_oracle import concept_to_manchester, gci_to_manchester, parse_manchester_gci, parse_manchester_concept
```

| Function | Direction |
|---|---|
| `concept_to_manchester(concept)` | `ELConcept` → string |
| `gci_to_manchester(gci)` | `GCI` → string |
| `parse_manchester_concept(s)` | string → `ELConcept` |
| `parse_manchester_gci(s)` | string → `GCI` (raises `ValueError` on invalid input) |

The parsers accept the keywords `SubClassOf` (also `SubClassOf:`), `and` and `some` in any case, and names with `:`, `.` or `-` inside them (`ex:hasPart`, `has-part`).

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

The answer to an MQ or judge prompt is read from the text after any `<think>…</think>` block: the first yes/no on the last non-empty line, or, if that line has none, the last yes/no anywhere. A response with no yes/no at all (often reasoning cut off by `max_new_tokens`) raises `OracleError`.

The counterexample is read from the last non-empty line after any `<think>` block, skipping code fences and removing backticks, quotes, a list dash, an `Answer:` label and a final full stop. It is usable when it parses, uses only the concept names and object properties in scope, and the model's own MQ accepts it. An unusable answer is asked for again, up to 3 attempts in all, with one line per earlier failure added to the prompt (e.g. "Your previous answer could not be used: Animal, Cat are not concept names in scope."). Generation is deterministic, so the changed prompt is what gets a different answer. If no attempt gives a usable counterexample, the oracle raises `OracleError`.

`learn_el_terminology` stops on an `OracleError` with a warning that gives the reason, and returns the H learned so far. A response that does not parse is never cached, so a rerun (for example with a larger `max_new_tokens`) asks the model again instead of replaying it; a cached response that no longer parses is regenerated.

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

Both oracle backends are accessed through a single entry point. Use `--oracle` to select the backend.

```bash
# ReasonerOracle (default)
python demo.py                                     # ELK on ontologies/medical.ttl
python demo.py --reasoner hermit                   # HermiT instead of ELK
python demo.py --ontology ontologies/medical.ttl   # explicit ontology path
python demo.py -v                                  # verbose logging

# LLMOracle
python demo.py --oracle llm                        # run learning loop (CPU)
python demo.py --oracle llm --device mps           # Apple Silicon GPU
python demo.py --oracle llm --device cuda          # NVIDIA GPU
python demo.py --oracle llm -v                     # print all prompts and responses
python demo.py --oracle llm --dry-run              # preview prompts without loading the model
python demo.py --oracle llm --model <hf-path>      # use a different HuggingFace model
```

The LLM demo uses `HuggingFaceTB/SmolLM2-1.7B-Instruct` on a kinship domain (`{Person, Male, Female, Parent, Father, Mother}`). The model is downloaded automatically on first run and cached in `~/.cache/huggingface/`. To pre-download:

```bash
huggingface-cli download HuggingFaceTB/SmolLM2-1.7B-Instruct
```

---

## CI

Every push to `master`, `main`, `develop`, or `feature/**` / `updates/**` branches runs the test suite automatically via GitHub Actions (`.github/workflows/check.yml`). The workflow:

- Runs on Ubuntu, Windows, and macOS
- Downloads the ELK jar from Maven Central
- Recompiles `OWLGateway.java` with Java 11
- Runs all tests with coverage and uploads the report to Codecov

A `CODECOV_TOKEN` repository secret is required for the Codecov upload step.

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

Each skill is applied independently with the given probability whenever a counterexample is found. All produced counterexamples are guaranteed to be O-entailed and H-not-entailed: the base counterexample is an axiom of O not entailed by H; `saturate_left` only adds conjuncts to the LHS, which keeps the GCI O-entailed, and checks that H still does not entail it; every other skill re-checks both conditions before accepting a change.

The skills draw from a `random.Random(seed)` owned by the oracle (`seed=0` by default; `None` for a fresh seed each run). Omit `oracle_skills` (or pass `{}`) for an oracle without randomness that always returns the first unentailed axiom of O, in the order of their string forms.

**Reproducibility:** the learner and `ReasonerOracle` iterate signatures, existentials, O and H in sorted order, never in Python's set order (which changes per process with hash randomisation). With the same ontology and seed, a run asks the same queries in the same order, and gives the same H and the same MQ/EQ counts, in every process.

**Progress guard:** none of the cases below happens with a correct oracle; all can with an LLM, whose answers can contradict each other.
- If the oracle returns a counterexample already entailed by H, `learn_el_terminology` stops with a warning instead of looping until `max_iterations`.
- If the O-essential computed from a counterexample is already entailed by H, it is skipped with a warning and learning continues. Learning stops only if the same counterexample comes back with H unchanged, as a deterministic LLM judge would return it. With PAC sampling, every equivalence query draws fresh statements, so the run continues, as upstream's does.
- If a counterexample cannot be normalised to a GCI with a concept name on one side (Lemma 4), the oracle accepted it but rejected every smaller GCI it implies. A consistent oracle always admits one, so this means the oracle contradicted itself. The counterexample is skipped with a warning saying so, under the same rule: learning stops only if it comes back with H unchanged.
