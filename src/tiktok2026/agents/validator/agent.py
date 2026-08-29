from tiktok2026.agents.common.client import OpenAICompatibleClient
from tiktok2026.agents.common.structured import invoke_structured
from tiktok2026.contracts import AgentFailure, AgentRole, ValidationReport


class ValidatorAgent:
    def __init__(self, client: OpenAICompatibleClient) -> None:
        self.client = client

    async def invoke(
        self, request_id: str, subject: dict[str, object]
    ) -> ValidationReport | AgentFailure:
        return await invoke_structured(
            self.client,
            AgentRole.VALIDATOR,
            request_id,
            ValidationReport,
            "Adversarially validate the subject without changing or executing it.",
            subject,
        )
