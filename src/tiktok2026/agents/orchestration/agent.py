from tiktok2026.agents.common.client import OpenAICompatibleClient
from tiktok2026.agents.common.structured import invoke_structured
from tiktok2026.contracts import AgentFailure, AgentRole, OrchestrationDecision


class OrchestrationAgent:
    def __init__(self, client: OpenAICompatibleClient) -> None:
        self.client = client

    async def invoke(
        self, request_id: str, context: dict[str, object]
    ) -> OrchestrationDecision | AgentFailure:
        result = await invoke_structured(
            self.client,
            AgentRole.ORCHESTRATION,
            request_id,
            OrchestrationDecision,
            "Select exactly one action allowed by the context. Do not execute it.",
            context,
        )
        if isinstance(result, OrchestrationDecision):
            allowed = context.get("allowed_actions")
            if isinstance(allowed, list) and result.action.value not in allowed:
                return AgentFailure(
                    request_id=request_id,
                    role=AgentRole.ORCHESTRATION,
                    kind="policy",
                    message="orchestration selected a disallowed action",
                    repair_attempts=0,
                )
        return result
