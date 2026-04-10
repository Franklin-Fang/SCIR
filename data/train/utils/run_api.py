import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

from util import (
    ensure_list_data,
    load_json,
    load_yaml,
    parse_json_like,
    replace_prompt_by_task_language,
    resolve_existing_path,
    write_json,
    write_jsonl,
)


class LLMAPIClient:
    def __init__(self, api_config: Dict[str, Any]):
        api = api_config.get("api", {})
        self.base_url = str(api.get("base_url", "")).rstrip("/")
        self.api_key = api.get("api_key", "")
        self.model = api.get("model", "")
        self.temperature = float(api.get("temperature", 0.0))
        self.max_tokens = int(api.get("max_tokens", 2048))
        self.request_delay = float(api.get("request_delay", 0.0))
        self.max_retries = int(api.get("max_retries", 3))
        self.retry_delay = float(api.get("retry_delay", 1.0))

        if not self.base_url:
            raise ValueError("api.base_url 不能为空")
        if not self.model:
            raise ValueError("api.model 不能为空")

        self.session = requests.Session()

    def _candidate_urls(self) -> List[str]:
        return [
            f"{self.base_url}/v1/chat/completions",
            f"{self.base_url}/chat/completions",
            self.base_url,
        ]

    @staticmethod
    def _extract_content(resp_json: Dict[str, Any]) -> str:
        # OpenAI 风格
        choices = resp_json.get("choices")
        if isinstance(choices, list) and choices:
            message = choices[0].get("message", {})
            if isinstance(message, dict) and "content" in message:
                return str(message["content"])

        # 其他常见风格兼容
        data = resp_json.get("data")
        if isinstance(data, dict):
            for k in ("content", "text", "output"):
                if k in data:
                    return str(data[k])

        for k in ("content", "text", "output", "result", "response"):
            if k in resp_json:
                return str(resp_json[k])

        return json.dumps(resp_json, ensure_ascii=False)

    def chat(self, prompt: str) -> str:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        last_error: Optional[Exception] = None
        urls = self._candidate_urls()

        for attempt in range(1, self.max_retries + 1):
            for url in urls:
                try:
                    response = self.session.post(url, headers=headers, json=payload, timeout=120)
                    response.raise_for_status()
                    resp_json = response.json()
                    content = self._extract_content(resp_json)
                    if self.request_delay > 0:
                        time.sleep(self.request_delay)
                    return content
                except Exception as e:
                    last_error = e

            if attempt < self.max_retries:
                time.sleep(self.retry_delay)

        raise RuntimeError(f"API 请求失败，重试后仍未成功: {last_error}")


def build_user_prompt(sample: Dict[str, Any]) -> str:
    instruction = sample.get("instruction", "")
    schema = sample.get("schema", "")
    text = sample.get("input", "")

    return (
        f"{instruction}\n\n"
        f"schema:\n{schema}\n\n"
        f"input:\n{text}\n"
    )


def run_task(
    task: str,
    language: str,
    data_dir: Path,
    client: LLMAPIClient,
    prompt_config_path: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    task = task.upper()
    data_file = data_dir / f"{task}.json"
    if not data_file.exists():
        raise FileNotFoundError(f"未找到任务数据文件: {data_file}")

    raw_data = load_json(data_file)
    samples = ensure_list_data(raw_data)

    results: List[Dict[str, Any]] = []
    for sample in samples:
        patched_sample = replace_prompt_by_task_language(
            sample=sample,
            task=task,
            language=language,
            prompt_config_path=prompt_config_path,
        )
        prompt = build_user_prompt(patched_sample)
        raw_output = client.chat(prompt)
        output = parse_json_like(raw_output)

        results.append(
            {
                "input": sample.get("input", ""),
                "output": output,
                "label": sample.get("label", ""),
            }
        )

    return results


def resolve_default_paths(project_root: Path) -> Dict[str, Path]:
    config_dir = project_root / "config"
    api_config = resolve_existing_path(
        [
            config_dir / "api_config.yaml",
            config_dir / "api_config.yml",
            config_dir / "api_config.ymal",
        ]
    )
    prompt_config = resolve_existing_path(
        [
            config_dir / "prompt_config.yaml",
            config_dir / "prompt_config.yml",
            config_dir / "prompt_config.ymal",
        ]
    )

    if api_config is None:
        raise FileNotFoundError("未找到 api_config.yaml/yml/ymal")
    if prompt_config is None:
        raise FileNotFoundError("未找到 prompt_config.yaml/yml/ymal")

    return {
        "api_config": api_config,
        "prompt_config": prompt_config,
        "data_dir": project_root / "data",
        "output_dir": project_root / "output",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行 EE/NER/RE 信息抽取任务")
    parser.add_argument("--task", type=str, default="all", help="任务类型: EE / NER / RE / all")
    parser.add_argument("--language", type=str, default="zh", help="提示词语言: zh / en")
    parser.add_argument("--project-root", type=str, default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--api-config", type=str, default="")
    parser.add_argument("--prompt-config", type=str, default="")
    parser.add_argument("--data-dir", type=str, default="")
    parser.add_argument("--output-dir", type=str, default="")
    parser.add_argument("--save-json", action="store_true", help="是否额外保存合并后的JSON")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    defaults = resolve_default_paths(project_root)

    api_config_path = Path(args.api_config).resolve() if args.api_config else defaults["api_config"]
    prompt_config_path = Path(args.prompt_config).resolve() if args.prompt_config else defaults["prompt_config"]
    data_dir = Path(args.data_dir).resolve() if args.data_dir else defaults["data_dir"]
    output_dir = Path(args.output_dir).resolve() if args.output_dir else defaults["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    api_config = load_yaml(api_config_path)
    client = LLMAPIClient(api_config)

    task_arg = args.task.upper()
    tasks = ["EE", "NER", "RE"] if task_arg == "ALL" else [task_arg]

    all_results: Dict[str, List[Dict[str, Any]]] = {}
    for task in tasks:
        task_results = run_task(
            task=task,
            language=args.language,
            data_dir=data_dir,
            client=client,
            prompt_config_path=prompt_config_path,
        )
        all_results[task] = task_results

        jsonl_path = output_dir / f"{task}_results.jsonl"
        write_jsonl(task_results, jsonl_path)
        print(f"[{task}] 已输出: {jsonl_path}")

    if args.save_json:
        merged_path = output_dir / "all_results.json"
        write_json(all_results, merged_path, indent=2)
        print(f"合并结果已输出: {merged_path}")


if __name__ == "__main__":
    main()
