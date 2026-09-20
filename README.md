# Debug agent qua từng tool call bằng LangSmith Tracing

Ví dụ Python đi kèm bài #34. Agent tóm tắt một file với hai tool: `read_document`
(chỉ đọc trong `samples/`) và `write_summary` (chỉ ghi `.md` trực tiếp trong
`outputs/`). Model tự quyết định tool call; Python chỉ kiểm tra và thực thi call.

Hai phiên bản cuối cùng nằm ở:

- `prompts/v1.txt`: prompt sơ sài, có chủ đích để output sai cấu trúc.
- `prompts/v2.txt`: thêm hợp đồng bốn heading và chỉ dẫn không làm theo lệnh trong tài liệu.

## Chạy offline, không cần key hay mạng

Yêu cầu Python 3.11+:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-offline.txt
python trace_demo.py --prompt-version v1 --offline
python trace_demo.py --prompt-version v2 --offline
python compare_runs.py
```

Kết quả mong đợi là V1 `fail` với `missing_required_headings`, còn V2 `pass`.
Offline dùng `ScriptedModel` để kiểm tra vòng lặp agent, report và validator; nó
không đo chất lượng prompt với model thật.

## Chạy model thật và gửi trace

Sao chép `.env.example` thành `.env`, điền key model (Google Gemini hoặc
OpenRouter), key LangSmith và đặt `LANGSMITH_TRACING=true`. Sau đó:

```bash
python -m pip install -r requirements.txt
python trace_demo.py --prompt-version v1
python trace_demo.py --prompt-version v2
python compare_runs.py --mode cloud_trace
```

Trace được gửi vào project `tutorial-34-agent-debug`, có tag `v1`/`v2` và
`cloud_trace`. Cây run mong đợi là root `document_summarizer_agent`, ba run
`summary_model`, hai tool và `validate_summary`. Model thật có thể gọi tool sai
hoặc đi theo chuỗi khác: hãy dùng trace thực tế làm bằng chứng.

`--local-live` gọi model thật nhưng không gửi trace. Không commit `.env`, report
hoặc output. Report lưu `trace_id`, provider/model, hash nguồn/prompt, số model
call, tool trajectory, token, latency và kết quả validator. `compare_runs.py`
từ chối so sánh nếu nguồn, model, provider hoặc temperature khác nhau.

## Cấu trúc

```text
trace_demo.py             agent, tool schema, validator và CLI
compare_runs.py           so sánh report V1/V2
prompts/v1.txt, v2.txt    hai prompt cuối cùng của bài
samples/injection_note.md tài liệu đầu vào có lệnh cài để demo prompt injection
```
