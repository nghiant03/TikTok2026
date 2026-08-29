from typing import Protocol


class MlflowClient(Protocol):
    def set_tracking_uri(self, uri: str) -> None: ...

    def log_metrics(self, metrics: dict[str, float]) -> None: ...

    def log_params(self, params: dict[str, str]) -> None: ...


class MlflowTelemetry:
    def __init__(self, tracking_uri: str, client: MlflowClient) -> None:
        self.client = client
        self.client.set_tracking_uri(tracking_uri)

    def record_evaluation(
        self,
        run_id: str,
        experiment_id: str,
        metrics: dict[str, float],
        artifact_ids: tuple[str, ...],
    ) -> None:
        self.client.log_params(
            {
                "run_id": run_id,
                "experiment_id": experiment_id,
                "artifact_ids": ",".join(artifact_ids),
            }
        )
        self.client.log_metrics(metrics)
