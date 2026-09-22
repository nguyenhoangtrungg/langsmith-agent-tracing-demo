# Phạm vi kiểm chứng

Rà soát ngày 17/09/2026 trên Ubuntu, Python 3.12, môi trường ảo mới, cài các phiên bản ghim trong `requirements.txt` (langsmith 0.12.1, langchain-core 1.6.3, langchain-google-genai 4.4.0, langchain-openai 1.6.0).

## Đã kiểm tra

- `python trace_demo.py --prompt-version v1 --offline`: `status=fail`, `validation_errors=["missing_required_headings"]`, `tool_trajectory=["read_document", "write_summary"]`, 3 lần gọi model.
- `python trace_demo.py --prompt-version v2 --offline`: `status=pass`, cùng chuỗi tool call.
- `python compare_runs.py`: in fail/pass, mã lỗi và chuỗi tool call của hai report.
- `python -m pytest -q`: 36 test qua.

Bộ test kiểm tra:

- Vòng lặp agent offline cho V1/V2.
- Bộ kiểm tra cấu trúc: heading giả/thiếu/sai thứ tự/thừa, code fence, chuỗi injection.
- Giới hạn đường dẫn của hai tool.
- Tool lỗi được trả lại model và agent tự sửa; model bỏ qua tool; vượt 5 lần gọi; tool không tồn tại.
- Offline/local-live không gửi trace; `.env` không ghi đè biến shell; bỏ giá trị `replace_`.
- `build_live_model` gắn đúng hai tool; `flush` chạy kể cả khi agent ném lỗi; `compare_runs` từ chối report khác nguồn.
- **SDK LangSmith thật** với bộ thu trong bộ nhớ: root `document_summarizer_agent` có ba run `llm`, hai run `tool` và `validate_summary`, cùng `trace_id`; run tool lỗi mang thông tin lỗi; `status=fail` không đặt `error` của root.
- **ChatGoogleGenerativeAI và ChatOpenAI thật** (`bind_tools`, `invoke`, callback) với phản hồi giả thay cho lời gọi mạng: sinh đúng run `llm` và run `tool` dưới root.

Test chặn mọi kết nối mạng.

## Ngoài phạm vi kiểm chứng cục bộ

- Hành vi của một model Google/OpenRouter cụ thể: có gọi tool đúng không, output ra sao.
- Key, quota, model ID của người chạy.
- Việc ingest và hiển thị trace trên LangSmith Cloud; token và chi phí của lần chạy thật.
- Khả năng chống các biến thể prompt injection.

Kiểm chứng cloud bằng cách chạy theo README mục 4 và đối chiếu `trace_id` trong report với trace trên LangSmith.
