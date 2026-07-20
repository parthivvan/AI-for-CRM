import base64
import logging
import mimetypes
from typing import TYPE_CHECKING, Any

import httpx

from app.config import Settings
from app.schemas import ImageType, VisionResult

if TYPE_CHECKING:
    from app.local_model_provider import LocalModelProvider

logger = logging.getLogger(__name__)


class AIProvider:
    def __init__(self, settings: Settings, local_model: "LocalModelProvider | None" = None):
        self.settings = settings
        self._local_model = local_model

    async def analyze_image(self, image_url: str, image_type: ImageType) -> VisionResult:
        """Route analysis through local model → Gemini → deterministic fallback."""
        # --- 1. Local ONNX model (primary) ---
        if self._local_model and self._local_model.is_ready():
            try:
                image_bytes, _ = await self._download_image(image_url)
                result = self._local_model.analyze(image_bytes, image_type)
                if result.confidence >= self.settings.model_threshold:
                    return result
                # Low confidence — fall through to Gemini if enabled
                logger.warning(
                    "LocalModel confidence %.3f below threshold %.3f — trying Gemini fallback.",
                    result.confidence,
                    self.settings.model_threshold,
                )
            except Exception:
                logger.exception("LocalModel inference error — trying Gemini fallback.")

        # --- 2. Gemini fallback ---
        if self.settings.enable_gemini_fallback and self.settings.gemini_api_key:
            logger.warning("Using Gemini fallback for image analysis.")
            try:
                return await self._analyze_with_gemini(image_url, image_type)
            except Exception:
                logger.exception("Gemini fallback failed — using deterministic fallback.")

        # --- 3. Deterministic fallback ---
        logger.warning("Using deterministic fallback for image analysis.")
        return self._fallback_vision(image_url, image_type)

    async def generate_consultant_brief(
        self,
        flags: list[str],
        treatments: list[dict[str, Any]],
        image_type: ImageType,
    ) -> str:
        if self.settings.gemini_api_key:
            prompt = (
                "Write a concise consultant-facing brief for a clinic CRM. "
                "Do not diagnose. Mention that human review is required. "
                f"Image type: {image_type.value}. Flags: {flags}. Treatments: {treatments}."
            )
            try:
                return await self._generate_text(prompt)
            except Exception:
                pass
        treatment_names = ", ".join(item["name"] for item in treatments) or "a follow-up consultation"
        flag_text = ", ".join(flags) or "no strong visual indicators"
        return (
            f"Review suggested for {image_type.value} image. Detected indicators: {flag_text}. "
            f"Consider discussing {treatment_names}. Human review is required before sharing recommendations."
        )

    async def generate_upsell_text(self, client: dict[str, Any], score_result: dict[str, Any]) -> dict[str, str]:
        signals = score_result.get("signals", [])
        offer = score_result["suggested_offer"]
        if self.settings.gemini_api_key:
            prompt = (
                "Create a short CRM upsell explanation and a polite outreach message. "
                "Return plain text with labels Reason and Message. "
                f"Client data: {client}. Score signals: {signals}. Suggested offer: {offer}."
            )
            try:
                text = await self._generate_text(prompt)
                return self._split_reason_message(text, signals, offer)
            except Exception:
                pass
        reason = f"Client is a good follow-up candidate because {', '.join(signals) or 'their profile matches pilot rules'}."
        message = f"Hi, we noticed it may be a good time to discuss {offer}. Would you like us to help schedule a follow-up?"
        return {"reason": reason, "draft_message": message}

    async def _analyze_with_gemini(self, image_url: str, image_type: ImageType) -> VisionResult:
        image_bytes, mime_type = await self._download_image(image_url)
        prompt = (
            "Analyze this clinic consultation image for visible non-diagnostic indicators only. "
            "Return JSON with keys detected_flags and confidence. Allowed flags: "
            "pigmentation, uneven_texture, redness, hair_density, dryness. "
            f"Image type: {image_type.value}."
        )
        endpoint = self._gemini_endpoint()
        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": prompt},
                        {
                            "inline_data": {
                                "mime_type": mime_type,
                                "data": base64.b64encode(image_bytes).decode("ascii"),
                            }
                        },
                    ]
                }
            ],
            "generationConfig": {"response_mime_type": "application/json"},
        }
        async with httpx.AsyncClient(timeout=self.settings.crm_timeout_seconds) as client:
            response = await client.post(endpoint, json=payload)
            response.raise_for_status()
        data = response.json()
        text = data["candidates"][0]["content"]["parts"][0]["text"]
        parsed = httpx.Response(200, content=text).json()
        flags = [flag for flag in parsed.get("detected_flags", []) if isinstance(flag, str)]
        confidence = float(parsed.get("confidence", 0.7))
        return VisionResult(
            detected_flags=flags,
            confidence=max(0.0, min(confidence, 1.0)),
            provider_used="gemini_fallback",
            raw=data,
        )

    async def _generate_text(self, prompt: str) -> str:
        endpoint = self._gemini_endpoint()
        payload = {"contents": [{"parts": [{"text": prompt}]}]}
        async with httpx.AsyncClient(timeout=self.settings.crm_timeout_seconds) as client:
            response = await client.post(endpoint, json=payload)
            response.raise_for_status()
        data = response.json()
        return data["candidates"][0]["content"]["parts"][0]["text"].strip()

    async def _download_image(self, image_url: str) -> tuple[bytes, str]:
        async with httpx.AsyncClient(timeout=self.settings.crm_timeout_seconds, follow_redirects=True) as client:
            response = await client.get(image_url)
            response.raise_for_status()
        mime_type = response.headers.get("content-type") or mimetypes.guess_type(image_url)[0] or "image/jpeg"
        return response.content, mime_type.split(";")[0]

    def _gemini_endpoint(self) -> str:
        return (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.settings.gemini_model}:generateContent?key={self.settings.gemini_api_key}"
        )

    def _fallback_vision(self, image_url: str, image_type: ImageType) -> VisionResult:
        source = image_url.lower()
        flags: list[str] = []
        keyword_flags = {
            "pigment": "pigmentation",
            "spot": "pigmentation",
            "texture": "uneven_texture",
            "red": "redness",
            "hair": "hair_density",
            "scalp": "hair_density",
            "dry": "dryness",
        }
        for keyword, flag in keyword_flags.items():
            if keyword in source and flag not in flags:
                flags.append(flag)
        if not flags:
            flags = ["hair_density"] if image_type in {ImageType.hair, ImageType.scalp} else ["pigmentation", "uneven_texture"]
        return VisionResult(
            detected_flags=flags,
            confidence=0.72,
            provider_used="deterministic_fallback",
            raw={"provider": "deterministic_fallback"},
        )

    def _split_reason_message(self, text: str, signals: list[str], offer: str) -> dict[str, str]:
        reason = ""
        message = ""
        for line in text.splitlines():
            lower = line.lower()
            if lower.startswith("reason"):
                reason = line.split(":", 1)[-1].strip()
            if lower.startswith("message"):
                message = line.split(":", 1)[-1].strip()
        if not reason:
            reason = f"Client is a good follow-up candidate because {', '.join(signals) or 'their profile matches pilot rules'}."
        if not message:
            message = f"Hi, we noticed it may be a good time to discuss {offer}. Would you like us to help schedule a follow-up?"
        return {"reason": reason, "draft_message": message}

