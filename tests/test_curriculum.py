"""Tests for CurriculumAgent."""

import json

import pytest

from agents.curriculum import CurriculumAgent
from tests.conftest import MockLLM

_FINDINGS = {
    "title": "TestProject",
    "one_liner": "A test project",
    "mental_model": "It does X via Y.",
    "domain": "other",
    "concepts": [{"name": "Core", "description": "The core concept", "complexity": "basic"}],
    "dependency_order": ["Core"],
    "key_files": ["src/core.py"],
    "skip_files": [],
}

_VALID_CURRICULUM = {
    "title": "Understanding TestProject",
    "mental_model": "It does X via Y.",
    "concepts": [{"name": "Core", "description": "The core concept", "complexity": "basic"}],
    "notebooks": [
        {
            "id": "01",
            "title": "Core Concepts",
            "concept": "Core",
            "description": "Build the core",
            "prerequisites": [],
            "key_source_files": ["src/core.py"],
            "key_symbols": ["CoreClass"],
            "learning_objectives": ["Understand Core"],
            "exercise_description": "Implement Core",
            "visualization_idea": "Draw Core",
        }
    ],
    "capstone": {
        "title": "mini-testproject",
        "description": "Build a mini version",
        "estimated_hours": 4,
        "modules": [
            {
                "name": "MiniCore",
                "description": "Core module",
                "depends_on": [],
                "interface_sketch": "class MiniCore:\n    pass",
                "test_behaviors": ["Given X, returns Y"],
            }
        ],
        "integration_test": {
            "description": "End-to-end",
            "setup_code": "x = 1",
            "success_metric": "output == 1",
            "expected_output_check": "assert output == 1",
        },
    },
}


def test_curriculum_generates_valid_structure():
    """LLM returns valid curriculum, agent passes through."""
    llm = MockLLM(responses=[json.dumps(_VALID_CURRICULUM)])
    agent = CurriculumAgent(llm)
    result = agent.generate(_FINDINGS)
    assert result["title"] == "Understanding TestProject"
    assert len(result["notebooks"]) == 1
    assert result["notebooks"][0]["id"] == "01"


def test_curriculum_strips_json_fences():
    """LLM wraps response in ```json ... ```, agent extracts."""
    wrapped = f"```json\n{json.dumps(_VALID_CURRICULUM)}\n```"
    llm = MockLLM(responses=[wrapped])
    result = CurriculumAgent(llm).generate(_FINDINGS)
    assert result["title"] == "Understanding TestProject"


def test_curriculum_fills_missing_notebook_fields():
    """LLM omits optional fields, agent fills with defaults."""
    minimal_curriculum = {
        "title": "Understanding TestProject",
        "mental_model": "X via Y",
        "concepts": [],
        "notebooks": [
            {
                "id": "01",
                "title": "Core",
                "concept": "Core",
                "description": "Do core",
                # Missing: prerequisites, key_source_files, key_symbols,
                #          learning_objectives, exercise_description, visualization_idea
            }
        ],
        "capstone": {
            "title": "mini-testproject",
            "description": "Mini",
            "estimated_hours": 4,
            "modules": [],
            "integration_test": {
                "description": "",
                "setup_code": "",
                "success_metric": "",
                "expected_output_check": "",
            },
        },
    }
    llm = MockLLM(responses=[json.dumps(minimal_curriculum)])
    result = CurriculumAgent(llm).generate(_FINDINGS)
    nb = result["notebooks"][0]
    assert nb["prerequisites"] == []
    assert nb["key_source_files"] == []
    assert nb["key_symbols"] == []
    assert nb["learning_objectives"] == []
    assert nb["exercise_description"] == ""
    assert nb["visualization_idea"] == ""


def test_curriculum_renumbers_notebook_ids():
    """LLM uses ids ['1', '3', '5'], agent renumbers to ['01', '02', '03']."""
    curriculum_with_bad_ids = dict(_VALID_CURRICULUM)
    curriculum_with_bad_ids["notebooks"] = [
        {**_VALID_CURRICULUM["notebooks"][0], "id": "1"},
        {**_VALID_CURRICULUM["notebooks"][0], "id": "3", "title": "Second"},
        {**_VALID_CURRICULUM["notebooks"][0], "id": "5", "title": "Third"},
    ]
    llm = MockLLM(responses=[json.dumps(curriculum_with_bad_ids)])
    result = CurriculumAgent(llm).generate(_FINDINGS)
    ids = [nb["id"] for nb in result["notebooks"]]
    assert ids == ["01", "02", "03"]


def test_curriculum_raises_on_invalid_json():
    """LLM returns 'not json at all', agent raises ValueError."""
    llm = MockLLM(responses=["not json at all"])
    with pytest.raises(ValueError, match="not valid JSON"):
        CurriculumAgent(llm).generate(_FINDINGS)


def test_curriculum_fills_missing_title_from_findings():
    """If LLM omits title, agent fills from findings."""
    no_title = dict(_VALID_CURRICULUM)
    del no_title["title"]
    llm = MockLLM(responses=[json.dumps(no_title)])
    result = CurriculumAgent(llm).generate(_FINDINGS)
    assert "TestProject" in result["title"]


def test_curriculum_fills_missing_capstone_title():
    """If capstone.title missing, agent fills from project name."""
    curriculum = dict(_VALID_CURRICULUM)
    capstone = dict(curriculum["capstone"])
    del capstone["title"]
    curriculum["capstone"] = capstone
    llm = MockLLM(responses=[json.dumps(curriculum)])
    result = CurriculumAgent(llm).generate(_FINDINGS)
    assert result["capstone"]["title"] != ""
