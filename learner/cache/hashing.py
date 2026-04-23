import hashlib
import json


def stable_hash(obj: dict) -> str:
    """
    Deterministic hash of a JSON-serializable object.
    """
    serialized = json.dumps(obj, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()