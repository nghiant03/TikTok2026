"""Static, controller-owned checks for candidate implementation entrypoints."""

from __future__ import annotations

import ast

_REQUIRED_ARGUMENTS = {
    "--output-dir",
    "--seed",
    "--fidelity",
    "--data-manifest",
    "--source-commit",
    "--execution-id",
    "--dataset-manifest-sha256",
    "--data-root",
}
_OPTIONAL_ARGUMENTS = {"--dataset-view-sha256"}
_PREDICTION_ENVELOPE_KEYS = {
    "schema_version",
    "manifest_id",
    "manifest_sha256",
    "dataset_view_sha256",
    "source_commit",
    "execution_id",
    "split",
    "rows",
}
_CHECKPOINT_ENVELOPE_KEYS = {
    "schema_version",
    "checkpoint_id",
    "data_manifest_id",
    "seed",
    "source_commit",
    "execution_id",
    "fidelity",
    "prediction_artifact_id",
    "prediction_artifact",
    "prediction_sha256",
    "dataset_view_sha256",
}


def _literal_string(node: ast.AST | None) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _dictionary_keys(node: ast.Dict) -> set[str]:
    return {
        key
        for key_node in node.keys
        if (key := _literal_string(key_node)) is not None
    }


def _named_dictionary_keys(tree: ast.AST) -> dict[str, set[str]]:
    dictionaries: dict[str, set[str]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        value = node.value
        if not isinstance(value, ast.Dict):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else (node.target,)
        for target in targets:
            if isinstance(target, ast.Name):
                dictionaries[target.id] = _dictionary_keys(value)
    return dictionaries


def _argument_contract(tree: ast.AST) -> tuple[set[str], set[str]]:
    options: set[str] = set()
    required: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Attribute) or node.func.attr != "add_argument":
            continue
        option = _literal_string(node.args[0] if node.args else None)
        if option is None:
            continue
        options.add(option)
        for keyword in node.keywords:
            if (
                keyword.arg == "required"
                and isinstance(keyword.value, ast.Constant)
                and keyword.value.value is True
            ):
                required.add(option)
    return options, required


def check_static_training_contract(source: str) -> str | None:
    """Check the training contract without importing or executing ``source``.

    This function deliberately accepts source text and only uses :mod:`ast`.
    It must remain a structural check: candidate top-level code is never
    compiled with execution enabled, imported, or evaluated.
    """
    try:
        tree = ast.parse(source, filename="train.py", mode="exec")
    except SyntaxError as error:
        return f"static contract check failed: syntax error at line {error.lineno}"

    options, required = _argument_contract(tree)
    missing_options = (_REQUIRED_ARGUMENTS | _OPTIONAL_ARGUMENTS) - options
    if missing_options:
        return "static contract check failed: missing argparse options " + ", ".join(
            sorted(missing_options)
        )
    missing_required = _REQUIRED_ARGUMENTS - required
    if missing_required:
        return "static contract check failed: options are not required " + ", ".join(
            sorted(missing_required)
        )

    dictionaries = _named_dictionary_keys(tree)
    if not dictionaries.get("prediction_payload", set()) >= _PREDICTION_ENVELOPE_KEYS:
        return "static contract check failed: prediction artifact envelope keys are incomplete"
    if not dictionaries.get("bundle_payload", set()) >= _CHECKPOINT_ENVELOPE_KEYS:
        return "static contract check failed: checkpoint artifact envelope keys are incomplete"

    strings = {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    if {"predictions.json", "checkpoint_bundle.json"} - strings:
        return "static contract check failed: required artifact filenames are not constructed"
    return None
