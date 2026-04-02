HERMIT_JAR=$(python3 -c "import owlready2, os; print(os.path.join(os.path.dirname(owlready2.__file__), 'hermit', 'HermiT.jar'))")

# Try multiple locations for the py4j jar
PY4J_JAR=$(find /usr/local/share/py4j /opt/homebrew /usr/share/py4j ~/.local/share/py4j 2>/dev/null -name "py4j*.jar" | head -1)

# Fallback: search the active venv
if [ -z "$PY4J_JAR" ]; then
    PY4J_JAR=$(find "$VIRTUAL_ENV" -name "py4j*.jar" 2>/dev/null | head -1)
fi

echo "HermiT: $HERMIT_JAR"
echo "py4j:   $PY4J_JAR"

javac -cp "$HERMIT_JAR:$PY4J_JAR" OWLGateway.java