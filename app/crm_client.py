import json
from pathlib import Path
from typing import Any

import httpx

from app.config import Settings

SAMPLE_CRM_PATH = Path(__file__).resolve().parent.parent / "data" / "sample_crm.json"


class CRMClient:
    def __init__(self, settings: Settings):
        self.settings = settings

    async def get_treatments(self, branch_id: str) -> list[dict[str, Any]]:
        if not self.settings.crm_base_url:
            return self._sample_branch(branch_id)["treatments"]
        return await self._get(f"/ai/branches/{branch_id}/treatments")

    async def get_clients(self, branch_id: str) -> list[dict[str, Any]]:
        if not self.settings.crm_base_url:
            return self._sample_branch(branch_id)["clients"]
        return await self._get(f"/ai/branches/{branch_id}/clients")

    async def post_analysis(self, consultation_id: str, payload: dict[str, Any]) -> None:
        if not self.settings.crm_base_url:
            return
        await self._post(f"/ai/consultations/{consultation_id}/analysis", payload)

    async def post_recommendations(self, payload: dict[str, Any]) -> None:
        if not self.settings.crm_base_url:
            return
        await self._post("/ai/recommendations", payload)

    async def _get(self, path: str) -> Any:
        async with httpx.AsyncClient(base_url=self.settings.crm_base_url, timeout=self.settings.crm_timeout_seconds) as client:
            response = await client.get(path, headers=self._headers())
            response.raise_for_status()
            return response.json()

    async def _post(self, path: str, payload: dict[str, Any]) -> None:
        async with httpx.AsyncClient(base_url=self.settings.crm_base_url, timeout=self.settings.crm_timeout_seconds) as client:
            response = await client.post(path, json=payload, headers=self._headers())
            response.raise_for_status()

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.settings.crm_api_key:
            headers["Authorization"] = f"Bearer {self.settings.crm_api_key}"
        return headers

    def _sample_branch(self, branch_id: str) -> dict[str, Any]:
        data = json.loads(SAMPLE_CRM_PATH.read_text(encoding="utf-8"))
        branches = data["branches"]
        if branch_id not in branches:
            raise ValueError(f"Unknown branch_id {branch_id!r}. Configure CRM_BASE_URL or add sample data.")
        return branches[branch_id]

