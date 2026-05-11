"""Curriculum agent: single LLM call to produce a Curriculum from ExplorerFindings."""

import json
import logging
import re

from llm.client import LLMClient
from prompts.curriculum import CURRICULUM_SYSTEM, CURRICULUM_USER

logger = logging.getLogger(__name__)


class CurriculumAgent:
    """Generate a Curriculum from ExplorerFindings with a single LLM call."""

    def __init__(self, llm_client: LLMClient) -> None:
        self.llm = llm_client

    def generate(self, findings: dict) -> dict:
        """Generate curriculum from findings.

        Args:
            findings: ExplorerFindings dict.

        Returns:
            Curriculum dict.

        Raises:
            ValueError: If the LLM response is not parseable JSON.
        """
        max_notebooks = self.llm.gen.get("max_notebooks", 8)
        prompt = CURRICULUM_USER.format(
            findings_json=json.dumps(findings, indent=2),
            max_notebooks=max_notebooks,
        )
        response = self.llm.chat(
            messages=[{"role": "user", "content": prompt}],
            system=CURRICULUM_SYSTEM,
            model=self.llm.model_for_step("curriculum"),
            max_tokens=16000,
        )

        parsed = self._parse_json(response)
        if parsed is None:
            raise ValueError(
                f"CurriculumAgent: LLM response is not valid JSON.\n{response[:500]}"
            )

        return self._validate(parsed, findings)

    def _validate(self, curriculum: dict, findings: dict) -> dict:
        """Fill missing fields with defaults. Lenient — never raises."""
        # Top-level defaults
        if not curriculum.get("title"):
            curriculum["title"] = f"Understanding {findings.get('title', 'the Project')}"
        if not curriculum.get("mental_model"):
            curriculum["mental_model"] = findings.get("mental_model", "")
        if "concepts" not in curriculum:
            curriculum["concepts"] = []

        # Renumber notebook IDs sequentially
        notebooks = curriculum.get("notebooks", [])
        for i, nb in enumerate(notebooks):
            nb["id"] = f"{i + 1:02d}"
            # Fill optional fields with defaults
            nb.setdefault("prerequisites", [])
            nb.setdefault("key_source_files", [])
            nb.setdefault("key_symbols", [])
            nb.setdefault("learning_objectives", [])
            nb.setdefault("exercise_description", "")
            nb.setdefault("visualization_idea", "")
        curriculum["notebooks"] = notebooks

        # Capstone defaults
        capstone = curriculum.get("capstone", {})
        if not capstone.get("title"):
            project = findings.get("title", "project")
            capstone["title"] = f"mini-{project.lower().replace(' ', '-')}"
        capstone.setdefault("description", "")
        capstone.setdefault("estimated_hours", 4)
        capstone.setdefault("modules", [])
        capstone.setdefault(
            "integration_test",
            {
                "description": "",
                "setup_code": "",
                "success_metric": "",
                "expected_output_check": "",
            },
        )
        curriculum["capstone"] = capstone

        return curriculum

    def _parse_json(self, text: str) -> dict | None:
        """Strip ```json fences, then parse. Returns None on failure."""
        text = text.strip()

        # Strip ```json ... ``` fences
        fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
        if fence_match:
            text = fence_match.group(1).strip()

        try:
            result = json.loads(text)
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            pass

        # Try to find a JSON object if extra text surrounds it
        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            try:
                result = json.loads(match.group())
                if isinstance(result, dict):
                    return result
            except json.JSONDecodeError:
                pass

        return None
