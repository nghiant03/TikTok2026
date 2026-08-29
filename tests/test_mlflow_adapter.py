from pathlib import Path

from tiktok2026.observability.mlflow import MlflowTelemetry


class RecordingMlflow:
    def __init__(self) -> None:
        self.tracking_uri: str | None = None
        self.metrics: dict[str, float] = {}
        self.params: dict[str, str] = {}

    def set_tracking_uri(self, uri: str) -> None:
        self.tracking_uri = uri

    def log_metrics(self, metrics: dict[str, float]) -> None:
        self.metrics = metrics

    def log_params(self, params: dict[str, str]) -> None:
        self.params = params


def test_mlflow_adapter_records_only_telemetry_and_references(tmp_path: Path) -> None:
    client = RecordingMlflow()
    telemetry = MlflowTelemetry((tmp_path / "mlruns").resolve().as_uri(), client)

    telemetry.record_evaluation(
        run_id="run-1",
        experiment_id="experiment-1",
        metrics={"NDCG@10": 0.75, "Recall@50": 0.9},
        artifact_ids=("artifact-1",),
    )

    assert client.tracking_uri == (tmp_path / "mlruns").resolve().as_uri()
    assert client.metrics == {"NDCG@10": 0.75, "Recall@50": 0.9}
    assert client.params == {
        "run_id": "run-1",
        "experiment_id": "experiment-1",
        "artifact_ids": "artifact-1",
    }
