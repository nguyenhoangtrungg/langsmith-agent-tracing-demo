from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="So sánh hai report cùng chế độ.")
    parser.add_argument("--mode", choices=("offline", "local_live", "cloud_trace"), default="offline")
    args = parser.parse_args(argv)
    suffix = "" if args.mode == "offline" else f"_{args.mode}"
    try:
        before = json.loads((ROOT / f"reports/v1{suffix}.json").read_text(encoding="utf-8"))
        after = json.loads((ROOT / f"reports/v2{suffix}.json").read_text(encoding="utf-8"))
    except FileNotFoundError:
        parser.error(f"Thiếu report: hãy chạy cả V1 và V2 ở chế độ {args.mode}.")
    for field in ("generation_mode", "provider", "model", "temperature", "source_sha256", "tool_trajectory"):
        if field not in before or field not in after:
            parser.error("Report cũ thiếu metadata; chạy lại bằng code đi kèm thư mục này.")
    for field in ("generation_mode", "provider", "model", "temperature", "source_sha256"):
        if before[field] != after[field]:
            parser.error(f"Hai report khác {field}; chưa thể so sánh chỉ ảnh hưởng của prompt.")
    print(f"mode={args.mode}; chỉ đối chiếu hai lần chạy, không phải benchmark")
    print("metric\tv1\tv2")
    print(f"status\t{before['status']}\t{after['status']}")
    print(f"validation_errors\t{','.join(before['validation_errors']) or '-'}\t{','.join(after['validation_errors']) or '-'}")
    print(f"tool_calls\t{' > '.join(before['tool_trajectory']) or '-'}\t{' > '.join(after['tool_trajectory']) or '-'}")
    print(f"model_calls\t{before['model_calls']}\t{after['model_calls']}")
    print(f"total_tokens\t{before['total_tokens']}\t{after['total_tokens']}")
    print(f"latency_ms\t{before['latency_ms']}\t{after['latency_ms']}")


if __name__ == "__main__":
    main()
