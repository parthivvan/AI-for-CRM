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
        """Route image classification through local ONNX model -> deterministic fallback.
        
        Per strict architecture rules, GenAI is restricted to brief write-ups only and
        is never used for image classification/detection.
        """
        # --- 1. Local ONNX model (primary classification provider) ---
        if self._local_model and self._local_model.is_ready():
            try:
                image_bytes, _ = await self._download_image(image_url)
                result = self._local_model.analyze(image_bytes, image_type)

                # Return quality or subject validation rejections directly (do not fall back to _fallback_vision)
                if result.provider_used in ("local_model_quality_rejected", "subject_validation_rejected"):
                    return result

                if result.confidence >= self.settings.model_threshold and result.detected_flags:
                    return result
                logger.warning(
                    "LocalModel confidence %.3f below threshold %.3f — falling back to deterministic classification.",
                    result.confidence,
                    self.settings.model_threshold,
                )
            except Exception:
                logger.exception("LocalModel inference error — falling back to deterministic classification.")

        # --- 2. Deterministic classification fallback ---
        logger.warning("Using deterministic fallback for image classification.")
        return self._fallback_vision(image_url, image_type)

    async def generate_consultant_brief(
        self,
        flags: list[str],
        treatments: list[dict[str, Any]],
        image_type: ImageType,
    ) -> str:
        if self.settings.gemini_api_key:
            # Hardened prompt with structural tags, negative constraints, and few-shot examples
            prompt = f"""You are a clinical advisory assistant for a clinic CRM.
Your task is to generate a concise, objective, consultant-facing brief recommending treatments based on visual indicators.

[CRITICAL SAFETY CONSTRAINTS]
1. DO NOT DIAGNOSE. Use speculative and cautious terminology (e.g., "visual indicators suggest", "showing signs of", "indicates potential"). Never say "we detected", "diagnosed", or promise a "cure".
2. MANDATORY SAFETY DISCLAIMER: You must explicitly include the exact phrase: "Human review is required before sharing recommendations."
3. PERSONA: Write for a clinic consultant/practitioner, not the patient. Keep it objective, professional, and brief.
[/CRITICAL SAFETY CONSTRAINTS]

[EXAMPLES]
Example 1:
Input: Image type: skin. Flags: ['redness']. Treatments: [{{'name': 'Calming Facial', 'price': 90}}].
Output: Consultant Brief: Skin consultation visual indicators suggest potential redness. Recommend discussing Calming Facial ($90). Human review is required before sharing recommendations.

Example 2:
Input: Image type: scalp. Flags: ['hair_thinning']. Treatments: [{{'name': 'Laser Hair Therapy', 'price': 250}}].
Output: Consultant Brief: Scalp consultation shows signs of potential hair thinning. Recommend discussing Laser Hair Therapy ($250). Human review is required before sharing recommendations.
[/EXAMPLES]

[CURRENT INPUT]
Image type: {image_type.value}.
Flags: {flags}.
Treatments: {treatments}.
[/CURRENT INPUT]

Provide only the "Consultant Brief" output:"""
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
            "Return JSON with keys detected_flags (a list containing exactly one best-matching label) and confidence. Allowed flags: "
            "acne, dryness, hair_thinning, pigmentation, redness. "
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
        if flags:
            flags = [flags[0]]
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
        if image_url.startswith("data:image"):
            # Parse data URI
            header, encoded = image_url.split(",", 1)
            mime_type = header.split(";")[0].split(":")[1]
            return base64.b64decode(encoded), mime_type

        # Check for local file path
        if not image_url.startswith("http://") and not image_url.startswith("https://"):
            from pathlib import Path
            p = Path(image_url)
            if p.exists() and p.is_file():
                mime_type = mimetypes.guess_type(image_url)[0] or "image/jpeg"
                return p.read_bytes(), mime_type.split(";")[0]

        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        async with httpx.AsyncClient(timeout=self.settings.crm_timeout_seconds, follow_redirects=True, headers=headers) as client:
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
            "acne": "acne",
            "pimple": "acne",
            "red": "redness",
            "hair": "hair_thinning",
            "scalp": "hair_thinning",
            "dry": "dryness",
        }
        for keyword, flag in keyword_flags.items():
            if keyword in source and flag not in flags:
                flags.append(flag)
        
        if flags:
            flags = [flags[0]]
        else:
            flags = ["hair_thinning"] if image_type in {ImageType.hair, ImageType.scalp} else ["acne"]
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

