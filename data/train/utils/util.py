import copy
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union, Tuple

import yaml


def resolve_existing_path(path_candidates: Iterable[Union[str, Path]]) -> Optional[Path]:
    """返回第一个存在的路径。"""
    for candidate in path_candidates:
        p = Path(candidate)
        if p.exists():
            return p
    return None


def load_yaml(file_path: Union[str, Path]) -> Dict[str, Any]:
    path = Path(file_path)
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML 内容必须是字典: {path}")
    return data


def load_json(file_path: Union[str, Path]) -> Any:
    path = Path(file_path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(data: Any, file_path: Union[str, Path], indent: int = 2) -> None:
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)


def write_jsonl(records: Iterable[Dict[str, Any]], file_path: Union[str, Path]) -> None:
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def ensure_list_data(data: Any) -> List[Dict[str, Any]]:
    """将输入样本统一转为字典列表。"""
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if isinstance(data, dict):
        return [data]
    raise ValueError("数据格式不合法，必须是 dict 或 list[dict]")


def parse_json_like(text: str) -> Any:
    """尽量将模型输出解析为 JSON；失败时返回原文本。"""
    if not isinstance(text, str):
        return text

    content = text.strip()
    if not content:
        return content

    # 直接 JSON 解析
    try:
        return json.loads(content)
    except Exception:
        pass

    # 处理 ```json ... ```
    if content.startswith("```"):
        content = content.strip("`")
        if content.lower().startswith("json"):
            content = content[4:].strip()
        try:
            return json.loads(content)
        except Exception:
            return text

    return text


def _safe_template_render(template: str, sample: Dict[str, Any]) -> str:
    """支持在提示词里用 {schema}/{input}/{label} 等变量。"""

    class _SafeDict(dict):
        def __missing__(self, key: str) -> str:
            return "{" + key + "}"

    return template.format_map(_SafeDict(sample))


def get_prompt_template(
    task: str,
    language: str,
    prompt_config_path: Optional[Union[str, Path]] = None,
) -> str:
    """
    根据任务与语言获取提示词模板。
    兼容 prompt_config.yaml / .yml / .ymal。
    """
    task = task.upper()
    language = language.lower()

    if prompt_config_path is None:
        base = Path(__file__).resolve().parents[1] / "config"
        prompt_config_path = resolve_existing_path(
            [
                base / "prompt_config.yaml",
                base / "prompt_config.yml",
                base / "prompt_config.ymal",
            ]
        )
        if prompt_config_path is None:
            raise FileNotFoundError("未找到 prompt_config.yaml/yml/ymal")

    prompt_config = load_yaml(prompt_config_path)

    if task not in prompt_config:
        raise KeyError(f"提示词配置中不存在任务: {task}")

    lang_block = prompt_config[task]
    if not isinstance(lang_block, dict):
        raise ValueError(f"提示词配置格式错误，{task} 对应值应为字典")

    if language not in lang_block:
        raise KeyError(f"提示词配置中 {task} 不存在语言: {language}")

    template = lang_block[language]
    if not isinstance(template, str):
        raise ValueError(f"提示词模板必须是字符串: task={task}, language={language}")

    return template


def replace_prompt_by_task_language(
    sample: Dict[str, Any],
    task: str,
    language: str,
    prompt_config_path: Optional[Union[str, Path]] = None,
    instruction_key: str = "instruction",
) -> Dict[str, Any]:
    """
    根据任务(task)与语言(language)从 prompt_config 中提取目标提示词，
    并替换样本中的 instruction 字段。
    """
    if not isinstance(sample, dict):
        raise ValueError("sample 必须是字典")

    template = get_prompt_template(task=task, language=language, prompt_config_path=prompt_config_path)

    updated = copy.deepcopy(sample)
    updated[instruction_key] = _safe_template_render(template, updated)
    return updated

def make_EE_results(results: list) -> Tuple[list, list, list]:
    missing = []
    redundancy = []
    correct = []
    print(results)
    for item in results:
        output = item["output"]
        label  = json.loads(item["label"])
        for key in label['arguments']:
            if key not in output['arguments']:
                continue
            elif output['arguments'][key] == label['arguments'][key]:
                continue
            elif output['arguments'][key] == 'NAN':
                temp = copy.deepcopy(label)
                temp['arguments'][key] = 'NAN'
                temp = json.dumps(temp, ensure_ascii=False)
                missing.append({'input': item['input'], 'output': temp, 'label': json.dumps({key: label['arguments'][key]}, ensure_ascii=False)})
            elif label['arguments'][key] == 'NAN':
                temp = copy.deepcopy(label)
                temp['arguments'][key] = output['arguments'][key]
                temp = json.dumps(temp, ensure_ascii=False)
                redundancy.append({'input': item['input'], 'output': temp, 'label': json.dumps({key: output['arguments'][key]}, ensure_ascii=False)})
        correct.append({'input': item['input'], 'output': item["label"], 'label': '{Correct}'})
    return missing, redundancy, correct

def make_NER_results(results: list) -> Tuple[list, list, list]:
    missing = []
    redundancy = []
    correct = []
    for item in results:
        output = item["output"]
        label  = json.loads(item["label"])
        has_diff = False
        for key in label:
            if key not in output:
                continue
            label_set  = set(label[key])
            output_set = set(output[key])
            # 漏抽：label 有但 output 没有
            for entity in label_set - output_set:
                temp = copy.deepcopy(label)
                temp[key].remove(entity)
                missing.append({'input': item['input'], 'output': json.dumps(temp, ensure_ascii=False), 'label': json.dumps({key:[entity]}, ensure_ascii=False)})
            # 多抽：output 有但 label 没有
            for entity in output_set - label_set:
                temp = copy.deepcopy(output)
                temp[key].append(entity)
                redundancy.append({'input': item['input'], 'output': json.dumps(temp, ensure_ascii=False), 'label': json.dumps({key:[entity]}, ensure_ascii=False)})
        correct.append({'input': item['input'], 'output': item["label"], 'label': '{Correct}'})
    return missing, redundancy, correct

def make_RE_results(results: list) -> Tuple[list, list, list]:
    missing = []
    redundancy = []
    correct = []
    for item in results:
        output = item["output"]
        label  = json.loads(item["label"])
        has_diff = False
        for key in label:
            if key not in output:
                continue
            label_set = [json.dumps(item, ensure_ascii=False) for item in label[key]]
            output_set = [json.dumps(item, ensure_ascii=False) for item in output[key]]
            label_set  = set(label_set)
            output_set = set(output_set)
            # 漏抽：label 有但 output 没有
            for entity in label_set - output_set:
                temp = copy.deepcopy(label)
                temp[key].remove(json.loads(entity))
                missing.append({'input': item['input'], 'output': json.dumps(temp, ensure_ascii=False), 'label': json.dumps({key:[json.loads(entity)]}, ensure_ascii=False)})
            # 多抽：output 有但 label 没有
            for entity in output_set - label_set:
                temp = copy.deepcopy(output)
                temp[key].append(json.loads(entity))
                redundancy.append({'input': item['input'], 'output': json.dumps(temp, ensure_ascii=False), 'label': json.dumps({key:[json.loads(entity)]}, ensure_ascii=False)})
        correct.append({'input': item['input'], 'output': item["label"], 'label': '{Correct}'})
    return missing, redundancy, correct