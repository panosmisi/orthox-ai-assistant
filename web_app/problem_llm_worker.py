from __future__ import annotations

import json
import sys
from pathlib import Path

from .main import _run_problem_llm_writer_inline


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: python -m web_app.problem_llm_worker <input_json> <output_json>", file=sys.stderr)
        return 2
    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    result = _run_problem_llm_writer_inline(
        payload["extracted_cards"],
        payload["timeline"],
        payload["red_flags"],
        payload["follow_up_questions"],
        payload["clinical_considerations"],
        payload["evidence_pack"],
        payload["clinical_reasoning"],
        payload["clinical_direction"],
    )
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
