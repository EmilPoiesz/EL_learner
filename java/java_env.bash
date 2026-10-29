#!/usr/bin/env bash

set -e

# Resolve script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Ensure venv is active
if [ -z "$VIRTUAL_ENV" ]; then
    echo "Error: virtual environment not activated"
    exit 1
fi

# --- OWL API 3 (bundled inside owlready2's HermiT.jar) ---
OWLAPI_JAR=$(python3 -c "import owlready2, os; print(os.path.join(os.path.dirname(owlready2.__file__), 'hermit', 'HermiT.jar'))")

# --- py4j (from venv) ---
PY4J_JAR=$(find "$VIRTUAL_ENV" -name "py4j*.jar" | head -1)

# ELK and log4j are not needed at compile time: OWLGateway loads ELK by
# reflection, and they are only put on the classpath at runtime.

# --- Debug prints ---
echo "OWLAPI: $OWLAPI_JAR"
echo "py4j:   $PY4J_JAR"

# --- Classpath ---
CLASSPATH="$OWLAPI_JAR:$PY4J_JAR"

# --- Compile (Java 11 bytecode, even on a newer JDK) ---
javac --release 11 -cp "$CLASSPATH" "$SCRIPT_DIR/OWLGateway.java"