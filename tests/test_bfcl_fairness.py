from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    from env_service.environments.bfcl.bfcl_env import (
        parse_assistant_content_to_tool_calls,
    )
    from env_service.environments.bfcl.env_handler import EnvHandler
except ModuleNotFoundError as exc:
    if exc.name and exc.name.startswith("env_service"):
        raise
    parse_assistant_content_to_tool_calls = None
    EnvHandler = None


def _require_bfcl_eval() -> bool:
    return EnvHandler is not None and parse_assistant_content_to_tool_calls is not None


def _tool_call(name: str, arguments: dict) -> dict:
    return {
        "id": f"{name}_1",
        "type": "function",
        "function": {
            "name": name,
            "arguments": arguments,
        },
    }


def test_strict_parser_rejects_malformed_tool_tags():
    if not _require_bfcl_eval():
        return

    parsed = parse_assistant_content_to_tool_calls(
        {
            "role": "assistant",
            "content": "<tool_call>not json</tool_call>",
        },
        strict=True,
    )

    assert parsed["tool_calls"] == []
    assert "_bfcl_parse_error" in parsed


def test_unavailable_tool_is_rejected_before_execution():
    if not _require_bfcl_eval():
        return

    handler = object.__new__(EnvHandler)

    try:
        handler._convert_tool_calls_to_execution_format(
            [_tool_call("sort", {"file_name": "final_report.pdf"})],
            available_function_names={"cd", "grep"},
        )
    except ValueError as exc:
        assert "not available" in str(exc)
    else:
        raise AssertionError("Unavailable tools should be rejected before execution.")


def test_rejected_tool_calls_are_not_extracted_for_eval():
    if not _require_bfcl_eval():
        return

    handler = object.__new__(EnvHandler)
    rejected = _tool_call("sort", {"file_name": "final_report.pdf"})
    accepted = _tool_call("cd", {"folder": "document"})

    messages = [
        {"role": "user", "content": "Do the task."},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [],
            "_bfcl_rejected_tool_calls": [rejected],
        },
        {
            "role": "env",
            "content": "[ERROR] Invalid tool call format: Tool 'sort' is not available.",
        },
        {"role": "assistant", "content": "", "tool_calls": [accepted]},
        {"role": "tool", "content": "ok", "tool_call_id": "cd_1"},
        {"role": "assistant", "content": "done", "tool_calls": []},
        {"role": "user", "content": "Next turn."},
    ]

    assert handler._extract_multi_turn_responses(messages)[0] == [
        "cd(folder='document')"
    ]


if __name__ == "__main__":
    test_strict_parser_rejects_malformed_tool_tags()
    test_unavailable_tool_is_rejected_before_execution()
    test_rejected_tool_calls_are_not_extracted_for_eval()
