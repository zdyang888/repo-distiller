"""Explorer agent: navigates a repo with an LLM-driven tool-call loop."""

import json
import logging
import re

from llm.client import LLMClient
from prompts.explorer import EXPLORER_SYSTEM
from tools.repo import RepoTool

logger = logging.getLogger(__name__)

_FINDINGS_DEFAULTS = {
    "title": "Unknown Project",
    "one_liner": "",
    "mental_model": "",
    "domain": "other",
    "concepts": [],
    "dependency_order": [],
    "key_files": [],
    "skip_files": [],
}


class ExplorerAgent:
    """Navigate a repo with an LLM tool-call loop to produce ExplorerFindings."""

    def __init__(self, llm_client: LLMClient, repo_tool: RepoTool) -> None:
        self.llm = llm_client
        self.repo = repo_tool
        self.max_steps: int = llm_client.gen.get("max_explore_steps", 20)
        self.usage_log: list[dict] = []

    def explore(self) -> dict:
        """Run the loop and return ExplorerFindings dict."""
        # Seed the conversation with a directory listing + hint
        seed_listing = self.repo.list_dir("")
        priority = self.repo.get_priority_files()
        priority_hint = ""
        if priority:
            priority_hint = "\n\nHigh-value files to prioritize:\n" + "\n".join(
                f"  - {p}" for p in priority
            )

        messages: list[dict] = [
            {
                "role": "user",
                "content": f"Repository root:\n{seed_listing}{priority_hint}\n\nBegin exploring.",
            }
        ]

        for step in range(self.max_steps):
            response = self.llm.chat(
                messages,
                system=EXPLORER_SYSTEM,
                model=self.llm.model_for_step("explore"),
            )
            self.usage_log.append(
                {"step": step, **self.llm.last_usage}
            )
            logger.debug("Explorer step %d response: %s", step, response[:200])

            parsed = self._parse_json(response)

            if parsed is None:
                logger.warning("Explorer step %d: invalid JSON — sending correction", step)
                messages.append({"role": "assistant", "content": response})
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Your response was not valid JSON. "
                            "Respond with EXACTLY one JSON object."
                        ),
                    }
                )
                continue

            messages.append({"role": "assistant", "content": response})

            action = parsed.get("action")

            if action == "done":
                findings = parsed.get("findings", {})
                return self._fill_findings_defaults(findings)

            if action == "tool":
                result = self._execute_tool(parsed)
                logger.debug("Tool %s result: %s", parsed.get("tool"), result[:200])
                messages.append({"role": "user", "content": result})
            else:
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"Unknown action {action!r}. Use 'tool' or 'done'."
                        ),
                    }
                )

        # Max steps reached — force finalize
        logger.warning("Explorer hit max_steps=%d, forcing finalize", self.max_steps)
        messages.append(
            {
                "role": "user",
                "content": (
                    "You have reached the maximum number of steps. "
                    "You MUST respond NOW with the 'done' action and your best findings."
                ),
            }
        )
        final_response = self.llm.chat(
            messages,
            system=EXPLORER_SYSTEM,
            model=self.llm.model_for_step("explore"),
        )
        self.usage_log.append({"step": self.max_steps, **self.llm.last_usage})

        parsed = self._parse_json(final_response)
        if parsed and parsed.get("action") == "done":
            return self._fill_findings_defaults(parsed.get("findings", {}))

        return self._extract_findings_fallback(final_response)

    def _execute_tool(self, call: dict) -> str:
        """Dispatch a tool call to the RepoTool."""
        tool = call.get("tool", "")
        if tool == "list_dir":
            return self.repo.list_dir(call.get("path", ""))
        if tool == "read_file":
            return self.repo.read_file(call.get("path", ""))
        if tool == "search_code":
            return self.repo.search_code(
                call.get("keyword", ""),
                file_pattern=call.get("file_pattern", "*.py"),
                max_results=call.get("max_results", 20),
            )
        return f"Unknown tool: {tool!r}. Valid tools: list_dir, read_file, search_code."

    def _parse_json(self, text: str) -> dict | None:
        """Extract and parse a JSON object from text. Returns None on failure."""
        # Try direct parse
        text = text.strip()
        try:
            result = json.loads(text)
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            pass

        # Try to find a JSON object within the text
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                result = json.loads(match.group())
                if isinstance(result, dict):
                    return result
            except json.JSONDecodeError:
                pass

        return None

    def _extract_findings_fallback(self, text: str) -> dict:
        """Best-effort findings when even forced finalize fails."""
        logger.error("Explorer fallback: could not parse findings from: %s", text[:300])
        return dict(_FINDINGS_DEFAULTS)

    def _fill_findings_defaults(self, findings: dict) -> dict:
        """Merge findings with defaults for any missing fields."""
        result = dict(_FINDINGS_DEFAULTS)
        result.update(findings)
        return result
