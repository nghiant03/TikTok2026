import ast
from pathlib import Path

ROOT = Path(__file__).parents[2]
GRAPH = ROOT / "src/tiktok2026/graph"

FORBIDDEN_MODULES = {
    "mlflow",
    "sqlite3",
    "subprocess",
    "tiktok2026.evaluation.registry",
    "tiktok2026.execution.docker",
    "tiktok2026.persistence.repositories",
    "tiktok2026.repository.worktrees",
}


def imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text())
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            modules.add(node.module)
    return modules


def test_graph_modules_do_not_import_privileged_implementations() -> None:
    violations: dict[str, set[str]] = {}
    for path in GRAPH.glob("*.py"):
        forbidden = imported_modules(path) & FORBIDDEN_MODULES
        if forbidden:
            violations[str(path.relative_to(ROOT))] = forbidden

    assert violations == {}


def test_graph_nodes_depend_on_controller_protocol_only() -> None:
    imports = imported_modules(GRAPH / "nodes.py")

    assert imports <= {
        "__future__",
        "typing",
        "tiktok2026.graph.state",
    }
