import hashlib
import json


def normalized_signature(value: object) -> str:
    normalized = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(normalized.encode()).hexdigest()
