from typing import Protocol

from tiktok2026.agents.common.client import OpenAICompatibleClient
from tiktok2026.agents.common.structured import invoke_structured
from tiktok2026.contracts import AgentFailure, AgentRole, ImplementationResult
from tiktok2026.policies.paths import check_changed_paths


class WriteCapability(Protocol):
    def write(self, path: str, content: str) -> None: ...


class ImplementorAgent:
    def __init__(self, client: OpenAICompatibleClient, repository: WriteCapability) -> None:
        self.client = client
        self.repository = repository

    async def invoke(
        self,
        request_id: str,
        experiment_id: str,
        allowed_scopes: tuple[str, ...],
    ) -> ImplementationResult | AgentFailure:
        result = await invoke_structured(
            self.client,
            AgentRole.IMPLEMENTOR,
            request_id,
            ImplementationResult,
            "Describe a faithful implementation within the approved paths.",
            {"experiment_id": experiment_id, "allowed_scopes": allowed_scopes},
        )
        if isinstance(result, ImplementationResult):
            decision = check_changed_paths(result.changed_files, allowed_scopes)
            if not decision.allowed:
                return AgentFailure(
                    request_id=request_id,
                    role=AgentRole.IMPLEMENTOR,
                    kind="policy",
                    message=decision.reason,
                    repair_attempts=0,
                )
            if result.experiment_id != experiment_id:
                return AgentFailure(
                    request_id=request_id,
                    role=AgentRole.IMPLEMENTOR,
                    kind="policy",
                    message="experiment identity changed",
                    repair_attempts=0,
                )
        return result
