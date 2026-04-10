import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import copy

PROJECT_ROOT = Path(__file__).resolve().parents[1]   # SCIR/
UTILS_DIR    = PROJECT_ROOT / "utils"
sys.path.insert(0, str(UTILS_DIR))

from run_api import LLMAPIClient, resolve_default_paths, run_task  # noqa: E402
from util import load_yaml, write_jsonl, make_EE_results, make_NER_results, make_RE_results                           # noqa: E402

table = {
    "EE": make_EE_results,
    "NER": make_NER_results,
    "RE": make_RE_results,
}

TASK     = "EE"
LANGUAGE = "zh" 

def main() -> None:
    # 1. 解析默认路径（自动兼容 .yaml/.yml/.ymal 后缀）
    paths = resolve_default_paths(PROJECT_ROOT)

    api_config_path    = paths["api_config"]
    prompt_config_path = paths["prompt_config"]
    data_dir           = paths["data_dir"]
    output_dir         = paths["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    # 2. 初始化 API 客户端
    api_config = load_yaml(api_config_path)
    client     = LLMAPIClient(api_config)

    # 3. 运行 EE 任务
    results = run_task(
        task=TASK,
        language=LANGUAGE,
        data_dir=data_dir,
        client=client,
        prompt_config_path=prompt_config_path,
    )

    missing, redundancy, correct = table[TASK](results)

    out_path = output_dir / f"{TASK}_{LANGUAGE}/missing.jsonl"
    write_jsonl(missing, out_path)
    out_path = output_dir / f"{TASK}_{LANGUAGE}/redundancy.jsonl"
    write_jsonl(redundancy, out_path)
    out_path = output_dir / f"{TASK}_{LANGUAGE}/correct.jsonl"
    write_jsonl(correct, out_path)


if __name__ == "__main__":
    main()
