#!/usr/bin/env bash

set -e

# Resolve script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Ensure venv is active
if [ -z "$VIRTUAL_ENV" ]; then
    echo "Error: virtual environment not activated"
    exit 1
fi

# --- HermiT (from owlready2) ---
HERMIT_JAR=$(python3 -c "import owlready2, os; print(os.path.join(os.path.dirname(owlready2.__file__), 'hermit', 'HermiT.jar'))")

# --- py4j (from venv) ---
PY4J_JAR=$(find "$VIRTUAL_ENV" -name "py4j*.jar" | head -1)

# --- ELK (from venv) ---
ELK_JAR=$(find "$VIRTUAL_ENV" -name "elk-owlapi-standalone*.jar" | head -1)

# --- log4j (from owlready2 pellet dir) ---
PELLET_DIR=$(python3 -c "import owlready2, os; print(os.path.join(os.path.dirname(owlready2.__file__), 'pellet'))")
LOG4J_JARS=$(find "$PELLET_DIR" -maxdepth 1 -name "log4j*.jar" | tr '\n' ':' | sed 's/:$//')

# --- Debug prints ---
echo "HermiT: $HERMIT_JAR"
echo "py4j:   $PY4J_JAR"
echo "ELK:    ${ELK_JAR:-not found}"
echo "log4j:  ${LOG4J_JARS:-not found}"

# --- Classpath ---
CLASSPATH="$HERMIT_JAR:$PY4J_JAR:$ELK_JAR:$LOG4J_JARS"

# --- Compile ---
javac -cp "$CLASSPATH" "$SCRIPT_DIR/OWLGateway.java"