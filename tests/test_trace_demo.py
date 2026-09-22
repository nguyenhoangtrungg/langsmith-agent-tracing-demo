"""Test offline và hợp đồng tracing; không gọi model hoặc LangSmith Cloud."""

import json
import socket
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage
from langsmith import Client
from langsmith.run_helpers import tracing_context
from langsmith.run_trees import RunTree

import compare_runs
import trace_demo


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path):
    original = trace_demo.ROOT
    (tmp_path / "samples").mkdir()
    (tmp_path / "prompts").mkdir()
    for relative in ("samples/injection_note.md", "prompts/v1.txt", "prompts/v2.txt"):
        (tmp_path / relative).write_text(
            (original / relative).read_text(encoding="utf-8"), encoding="utf-8"
        )
    monkeypatch.setattr(trace_demo, "ROOT", tmp_path)
    monkeypatch.setattr(compare_runs, "ROOT", tmp_path)
    for name in (
        "GOOGLE_API_KEY", "GOOGLE_MODEL", "OPENROUTER_API_KEY", "OPENROUTER_MODEL",
        "LANGSMITH_API_KEY", "LANGSMITH_WORKSPACE_ID", "LANGSMITH_ENDPOINT",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("MODEL_PROVIDER", "auto")
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "false")

    def block_network(*args, **kwargs):
        raise AssertionError("Test không được mở kết nối mạng")

    monkeypatch.setattr(socket.socket, "connect", block_network)
    with tracing_context(enabled=False):
        yield tmp_path


def memory_client(monkeypatch):
    """Client SDK thật, chỉ thay điểm gửi dữ liệu bằng bộ thu trong bộ nhớ."""
    created, updated = [], []
    monkeypatch.setattr(Client, "create_run", lambda self, **kw: created.append(kw))
    monkeypatch.setattr(Client, "update_run", lambda self, **kw: updated.append(kw))
    client = Client(api_url="http://localhost:1", api_key="test-key", auto_batch_tracing=False)
    return client, created, updated


@pytest.mark.parametrize("version,status,errors", [
    ("v1", "fail", ["missing_required_headings"]),
    ("v2", "pass", []),
])
def test_offline_agent_report_and_output(version, status, errors, isolated):
    report = trace_demo.run_agent(version)
    assert report["status"] == status
    assert report["validation_errors"] == errors
    assert report["tool_trajectory"] == ["read_document", "write_summary"]
    assert report["model_calls"] == 3
    assert report["tool_errors"] == []
    assert (isolated / report["output"]).is_file()
    assert report["trace_id"] is None
    assert report["provider"] == "simulator"
    output = (isolated / report["output"]).read_text(encoding="utf-8")
    assert "16:00" in output


@pytest.mark.parametrize("content", [
    " ".join(trace_demo.HEADINGS),
    "\n".join(trace_demo.HEADINGS),
    "\n".join(h + " sai" for h in trace_demo.HEADINGS),
    "\n".join(reversed(trace_demo.HEADINGS)),
    "```markdown\n" + "\n".join(trace_demo.HEADINGS) + "\n```",
])
def test_rejects_fake_or_empty_headings(content):
    assert trace_demo.validate(content)


def test_extra_heading_is_rejected():
    valid = trace_demo.OFFLINE_SUMMARIES["v2"]
    assert trace_demo.validate(valid) == []
    assert "wrong_heading_order_or_extra_headings" in trace_demo.validate(valid + "\n## Thừa")


def test_validator_does_not_claim_semantic_accuracy():
    # Đủ cấu trúc vẫn có thể sai nội dung: phải giữ giới hạn này trong README.
    content = "\n\n".join(h + "\n\nKhông biết." for h in trace_demo.HEADINGS)
    assert trace_demo.validate(content) == []


def test_injection_marker_is_detected():
    content = trace_demo.OFFLINE_SUMMARIES["v2"] + "\nMọi việc đã hoàn tất"
    assert "followed_instruction_inside_document" in trace_demo.validate(content)


def test_describing_injection_is_not_treated_as_following_it():
    content = trace_demo.OFFLINE_SUMMARIES["v2"] + (
        '\nTài liệu có câu "Mọi việc đã hoàn tất" nhưng bị bỏ qua.'
    )
    assert "followed_instruction_inside_document" not in trace_demo.validate(content)


@pytest.mark.parametrize("path", ["../secret.txt", "samples/../../secret.md", "samples/file.py"])
def test_read_rejects_bad_paths(path):
    with pytest.raises(ValueError):
        trace_demo.read_document(path)


@pytest.mark.parametrize("path", ["../out.md", "outputs/nested/out.md", "outputs/out.txt"])
def test_write_rejects_bad_paths(path):
    with pytest.raises(ValueError):
        trace_demo.write_summary(path, "test")


def test_file_symlink_escape_is_rejected(isolated):
    outside = isolated / "outside.md"
    outside.write_text("test", encoding="utf-8")
    try:
        (isolated / "samples/link.md").symlink_to(outside)
        (isolated / "outputs").mkdir()
        (isolated / "outputs/link.md").symlink_to(outside)
    except OSError:
        pytest.skip("Môi trường không cho phép tạo symlink")
    with pytest.raises(ValueError):
        trace_demo.read_document("samples/link.md")
    with pytest.raises(ValueError):
        trace_demo.write_summary("outputs/link.md", "changed")
    assert outside.read_text(encoding="utf-8") == "test"


def test_tool_error_is_returned_to_model_and_agent_recovers(isolated):
    model = trace_demo.ScriptedModel([
        trace_demo.tool_call("read_document", "c1", relative_path=trace_demo.SOURCE_PATH),
        trace_demo.tool_call("write_summary", "c2", relative_path="../bad.md", content="x"),
        trace_demo.tool_call(
            "write_summary", "c3", relative_path="outputs/v2_summary.md",
            content=trace_demo.OFFLINE_SUMMARIES["v2"],
        ),
        AIMessage(content="Xong."),
    ])
    report = trace_demo.run_agent("v2", model=model)
    assert report["status"] == "pass"
    assert report["tool_trajectory"] == ["read_document", "write_summary", "write_summary"]
    assert len(report["tool_errors"]) == 1
    assert report["tool_calls"][1]["error"]


def test_agent_that_skips_tools_fails(isolated):
    model = trace_demo.ScriptedModel([AIMessage(content="Mọi việc đã hoàn tất")])
    report = trace_demo.run_agent("v1", model=model)
    assert report["validation_errors"] == ["document_not_read", "summary_not_written"]
    assert report["output"] is None


def test_agent_stops_after_max_model_calls(isolated):
    steps = [
        trace_demo.tool_call("read_document", f"c{i}", relative_path=trace_demo.SOURCE_PATH)
        for i in range(trace_demo.MAX_MODEL_CALLS)
    ]
    report = trace_demo.run_agent("v1", model=trace_demo.ScriptedModel(steps))
    assert report["model_calls"] == trace_demo.MAX_MODEL_CALLS
    assert "max_model_calls_exceeded" in report["validation_errors"]


def test_unknown_tool_is_reported(isolated):
    model = trace_demo.ScriptedModel([
        trace_demo.tool_call("delete_file", "c1", relative_path="samples/injection_note.md"),
        AIMessage(content="Xong."),
    ])
    report = trace_demo.run_agent("v1", model=model)
    assert "Tool không tồn tại" in report["tool_errors"][0]
    assert (isolated / "samples/injection_note.md").is_file()


def test_offline_cli_disables_tracing_even_with_environment(monkeypatch, isolated):
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "true")

    def forbidden(*args, **kwargs):
        raise AssertionError("Offline không nạp env, tạo client hoặc post trace")

    monkeypatch.setattr(trace_demo, "load_project_env", forbidden)
    monkeypatch.setattr(trace_demo, "Client", forbidden)
    monkeypatch.setattr(RunTree, "post", forbidden)
    trace_demo.main(["--prompt-version", "v1", "--offline"])
    assert json.loads((isolated / "reports/v1.json").read_text(encoding="utf-8"))["status"] == "fail"


def test_env_is_local_and_does_not_override_shell(monkeypatch, isolated):
    (isolated / ".env").write_text("MODEL_PROVIDER=google\n", encoding="utf-8")
    monkeypatch.setenv("MODEL_PROVIDER", "openrouter")
    trace_demo.load_project_env()
    assert trace_demo.os.environ["MODEL_PROVIDER"] == "openrouter"


def test_auto_ignores_placeholder_google_configuration(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "replace_me")
    monkeypatch.setenv("GOOGLE_MODEL", "replace_with_model")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("OPENROUTER_MODEL", "test/model")
    assert trace_demo.select_provider() == ("openrouter", "test/model")
    monkeypatch.setenv("MODEL_PROVIDER", "google")
    with pytest.raises(ValueError):
        trace_demo.select_provider()


@pytest.mark.parametrize("provider,module,class_name", [
    ("google", "langchain_google_genai", "ChatGoogleGenerativeAI"),
    ("openrouter", "langchain_openai", "ChatOpenAI"),
])
def test_live_model_is_configured_with_both_tools(monkeypatch, provider, module, class_name):
    calls = {}

    class FakeModel:
        def __init__(self, **kwargs):
            calls["config"] = kwargs

        def bind_tools(self, tools):
            calls["tools"] = [tool["function"]["name"] for tool in tools]
            return self

    monkeypatch.setitem(sys.modules, module, SimpleNamespace(**{class_name: FakeModel}))
    monkeypatch.setenv(provider.upper() + "_API_KEY", "test-key")
    trace_demo.build_live_model(provider, "test-model")
    assert calls["config"]["model"] == "test-model"
    assert calls["config"]["temperature"] == 0
    assert calls["tools"] == ["read_document", "write_summary"]


def test_local_live_disables_trace(monkeypatch, isolated):
    monkeypatch.setattr(trace_demo, "load_project_env", lambda: None)
    monkeypatch.setattr(trace_demo, "select_provider", lambda: ("google", "test-model"))
    monkeypatch.setenv("LANGSMITH_TRACING", "true")

    class CheckedModel(trace_demo.ScriptedModel):
        def invoke(self, messages, config=None):
            assert trace_demo.get_current_run_tree() is None
            return next(self.steps, AIMessage(content="Xong."))

    def fake_build(provider, model_name):
        return CheckedModel(trace_demo.offline_model("v2", "outputs/v2_local_live_summary.md").steps)

    monkeypatch.setattr(trace_demo, "build_live_model", fake_build)
    trace_demo.main(["--prompt-version", "v2", "--local-live"])
    report = json.loads((isolated / "reports/v2_local_live.json").read_text(encoding="utf-8"))
    assert report["output"] == "outputs/v2_local_live_summary.md"
    assert report["status"] == "pass"
    assert report["trace_id"] is None


def test_real_sdk_builds_agent_tree_without_cloud(monkeypatch):
    client, created, updated = memory_client(monkeypatch)
    with tracing_context(enabled=True, client=client, project_name="test-project"):
        report = trace_demo.run_agent("v1")
    root = next(run for run in created if run["name"] == "document_summarizer_agent")
    children = [run for run in created if run.get("parent_run_id") == root["id"]]
    assert sorted(run["name"] for run in children) == sorted([
        "summary_model", "read_document", "summary_model", "write_summary",
        "summary_model", "validate_summary",
    ])
    assert {run["run_type"] for run in children if run["name"] == "summary_model"} == {"llm"}
    assert all(run["trace_id"] == root["trace_id"] for run in children)
    write_run = next(run for run in children if run["name"] == "write_summary")
    assert write_run["inputs"]["relative_path"] == "outputs/v1_summary.md"
    root_update = next(run for run in updated if run["run_id"] == root["id"])
    assert root_update["outputs"]["status"] == "fail"
    assert root_update["error"] is None  # Validation fail != lỗi thực thi run.
    assert report["trace_id"] == str(root["trace_id"])


def test_failed_tool_run_carries_error_in_trace(monkeypatch):
    client, created, updated = memory_client(monkeypatch)
    model = trace_demo.ScriptedModel([
        trace_demo.tool_call("read_document", "c1", relative_path="../secret.md"),
        AIMessage(content="Không đọc được."),
    ])
    with tracing_context(enabled=True, client=client, project_name="test-project"):
        trace_demo.run_agent("v1", model=model)
    read_run = next(run for run in created if run["name"] == "read_document")
    read_update = next(run for run in updated if run["run_id"] == read_run["id"])
    assert "Path is not allowed" in read_update["error"]


@pytest.mark.parametrize("raise_error", [False, True])
def test_cloud_cli_flushes_same_client_even_on_failure(monkeypatch, raise_error):
    events = []
    client = SimpleNamespace(flush=lambda: events.append("flush"))
    monkeypatch.setattr(trace_demo, "Client", lambda: client)
    monkeypatch.setattr(trace_demo, "load_project_env", lambda: None)
    monkeypatch.setattr(trace_demo, "select_provider", lambda: ("google", "test-model"))
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGSMITH_API_KEY", "test-key")

    def fake_agent(*args, **kwargs):
        from langsmith.run_helpers import get_tracing_context
        context = get_tracing_context()
        assert context["client"] is client
        assert context["enabled"] is True
        events.append("agent")
        if raise_error:
            raise RuntimeError("test failure")
        return {"status": "pass"}

    monkeypatch.setattr(trace_demo, "run_agent", fake_agent)
    if raise_error:
        with pytest.raises(RuntimeError, match="test failure"):
            trace_demo.main(["--prompt-version", "v2"])
    else:
        trace_demo.main(["--prompt-version", "v2"])
    assert events == ["agent", "flush"]


def test_compare_modes_and_missing_report(capsys):
    with pytest.raises(SystemExit) as exc:
        compare_runs.main([])
    assert exc.value.code == 2
    for version in ("v1", "v2"):
        trace_demo.main(["--prompt-version", version, "--offline"])
    compare_runs.main([])
    out = capsys.readouterr().out
    assert "status\tfail\tpass" in out
    assert "tool_calls\tread_document > write_summary\tread_document > write_summary" in out


def test_compare_rejects_changed_source(isolated):
    for version in ("v1", "v2"):
        trace_demo.main(["--prompt-version", version, "--offline"])
    path = isolated / "reports/v2.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    report["source_sha256"] = "changed"
    path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(SystemExit) as exc:
        compare_runs.main([])
    assert exc.value.code == 2


@pytest.mark.parametrize("provider", ["google", "openrouter"])
def test_real_integration_traces_llm_and_tool_calls(monkeypatch, isolated, provider):
    # Giữ bind_tools, model.invoke và callback SDK thật; chỉ thay bước sinh phản hồi.
    if provider == "google":
        pytest.importorskip("langchain_google_genai")
        from langchain_google_genai import ChatGoogleGenerativeAI
        model_class = ChatGoogleGenerativeAI
    else:
        pytest.importorskip("langchain_openai")
        from langchain_openai import ChatOpenAI
        model_class = ChatOpenAI
    from langchain_core.outputs import ChatGeneration, ChatResult

    replies = iter(trace_demo.offline_model("v2", "outputs/v2_cloud_trace_summary.md").steps)

    def fake_generate(self, messages, **kwargs):
        return ChatResult(generations=[ChatGeneration(message=next(replies))])

    monkeypatch.setattr(model_class, "_generate", fake_generate)
    monkeypatch.setenv(provider.upper() + "_API_KEY", "test-key")
    monkeypatch.setenv(provider.upper() + "_MODEL", "test-model")
    monkeypatch.setenv("MODEL_PROVIDER", provider)
    client, created, updated = memory_client(monkeypatch)
    with tracing_context(enabled=True, client=client, project_name="test-project"):
        report = trace_demo.run_agent("v2", use_live_model=True, run_label="cloud_trace")
    assert report["status"] == "pass"
    assert report["tool_trajectory"] == ["read_document", "write_summary"]
    root = next(run for run in created if run["name"] == "document_summarizer_agent")
    llm_runs = [run for run in created if run["run_type"] == "llm"]
    tool_runs = [run for run in created if run["run_type"] == "tool"]
    assert len(llm_runs) == 3
    assert all(run["parent_run_id"] == root["id"] for run in llm_runs + tool_runs)
    assert {run["name"] for run in tool_runs} == {"read_document", "write_summary"}
