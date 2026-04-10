# MBSC Dataset Generation Script

The dataset used in this project is built on top of IEPile (Unearthing Large-Scale Schema-Based Information Extraction Corpus). Since IEPile integrates multiple third-party information extraction datasets, its distribution and use are subject to the copyright and license terms of the original data. As a result, I am currently unable to publicly release this derived dataset.

You are encouraged to consult official channels, such as Hugging Face IEPile, to learn how to obtain IEPile. I can provide the data preprocessing scripts for academic research reference. If formal authorization for IEPile and related data is obtained later, I will organize and publish the dataset as soon as possible.

---

## Contents

- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Data Format](#data-format)
- [Output Description](#output-description)
- [Command-Line Interface](#command-line-interface)
- [Dependency Installation](#dependency-installation)

---

## Features

- **Three IE tasks**: NER, RE, and EE, evaluated within one unified framework
- **Bilingual prompts**: Built-in Chinese (`zh`) and English (`en`) prompt sets, switchable as needed
- **Flexible API integration**: Compatible with OpenAI and other APIs with the same format, with automatic endpoint discovery
- **Structured result comparison**: Field-by-field comparison between model output and ground truth, producing missing, redundant, and correct JSONL files
- **Robust JSON parsing**: Built-in fault-tolerant parsing with support for ```json``` code blocks and raw JSON strings
- **Automatic retry mechanism**: Configurable maximum retries and retry intervals to handle API rate limits or network instability

---

## Project Structure

```
SCIR/
├── config/
│   ├── api_config.yaml       # API connection and parameter settings
│   └── prompt_config.yaml    # Prompt templates for each task/language
├── data/
│   ├── EE.json               # Event extraction test data
│   ├── NER.json              # Named entity recognition test data
│   └── RE.json               # Relation extraction test data
├── output/                   # Result output directory (subdirectories are created automatically at runtime)
├── utils/
│   ├── run_api.py            # API client, task scheduling, prompt construction
│   └── util.py               # Data IO, JSON parsing, and result comparison utilities
├── startup.py                # Entry script for event extraction tasks
├── requirements.txt          # Python dependencies
└── README.md
```

---

## Quick Start

### 1. Data Preparation

Refer to [Data Format](#data-format).

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure the API

Edit `config/api_config.yaml` and fill in your API endpoint and key:

```yaml
api:
  base_url: "https://api.openai.com/v1"
  api_key: "your-api-key-here"
  model: "gpt-4o"
```

### 4. Run the Task

```bash
python startup.py
```

Modify the `TASK` and `LANGUAGE` variables at the top of the script to switch task type and language:

```python
TASK     = "EE"   # Optional: "EE" / "NER" / "RE"
LANGUAGE = "zh"   # Optional: "zh" / "en"
```

## Configuration

### `config/api_config.yaml`

| Field | Description | Default |
|------|------|--------|
| `api.base_url` | API base URL | `https://api.openai.com/v1` |
| `api.api_key` | API key | — |
| `api.model` | Model name | `gpt-4o` |
| `api.temperature` | Sampling temperature (0~1) | `0.0` |
| `api.max_tokens` | Maximum number of generated tokens | `10240` |
| `api.request_delay` | Delay between requests in seconds to avoid rate limits | `1.0` |
| `api.max_retries` | Maximum number of retries | `3` |
| `api.retry_delay` | Retry wait time in seconds | `5.0` |

### `config/prompt_config.yaml`

Prompt templates are organized in a two-level structure: `task type → language`. You can freely modify or add tasks/languages:

```yaml
EE:
  zh: "You are an event extraction expert, please perform the following operations:..."
  en: "You are an event extraction expert..."
NER:
  zh: "..."
  en: "..."
RE:
  zh: "..."
  en: "..."
```

---

## Data Format

Each task corresponds to a JSON file with the same name under the `data/` directory (`EE.json` / `NER.json` / `RE.json`). Both a single dictionary and a list of dictionaries are supported.

Each record contains the following fields:

| Field | Description |
|------|------|
| `instruction` | Task instruction (automatically replaced at runtime by the template in prompt_config) |
| `schema` | Task schema, in JSON string format, describing the target entity / relation / event structure |
| `input` | The original text to be extracted from |
| `label` | Ground-truth answer, in JSON string format |


**EE Example:**

```json
{
    "instruction": "You are an event extraction expert...",
    "schema": "{\"event_type\": \"company_listing\", \"arguments\": [\"market_value\", \"listed_company\", ...]}",
    "input": "Antai Technology's indirectly held company Tianyi Shangjia was listed on the STAR Market on the 22nd...",
    "label": "{\"event_type\": \"company_listing\", \"event_trigger\": \"listed\", \"arguments\": {...}}"
}
```

**NER Example:**

```json
{
    "instruction": "You are an entity extraction expert...",
    "schema": "{\"location\":[], \"product\":[], \"time\":[]}",
    "input": "Homemakers may be most concerned about how to keep their homes cozy and comfortable...",
    "label": "{\"location\":[], \"product\":[\"Haier ZW1800-272 vacuum cleaner\"], \"time\":[\"today\"]}"
}
```

**RE Example:**

```json
{
    "instruction": "You are a relation extraction expert. Please perform the following operations:\n1. Extract head and tail entities from the input that satisfy the relations defined in the schema, and add these head-tail pairs in the form {head: head_entity, tail: tail_entity} to the corresponding relation list in the schema. Keep relations that do not exist as empty lists.\n2. Answer in JSON string format and output only the answer, nothing else.",
    "schema": "{\"composer\":[], \"publisher\":[], \"composer\":[]}",
    "input": "\"Leaving\" was composed by Zhang Yu and performed by him",
    "label": "{\"singer\":[{\"head\": \"Leaving\", \"tail\": \"Zhang Yu\"}], \"publisher\":[], \"composer\":[{\"head\": \"Leaving\", \"tail\": \"Zhang Yu\"}]}"
}
```

---

## Output Description

After execution, the results are saved under `output/{TASK}_{LANGUAGE}/`, with three files in total:

| File | Description |
|------|------|
| `missing.jsonl` | **Missing**: items present in the labels but not correctly extracted by the model |
| `redundancy.jsonl` | **Redundant**: items extracted by the model but not present in the labels |
| `correct.jsonl` | **Correct**: samples where the model output exactly matches the labels |

Each line is one JSON record containing three fields: `input` (original text), `output` (model output), and `label` (corresponding difference annotation).

---

## Command-Line Interface

`utils/run_api.py` supports the following command-line arguments:

| Argument | Description | Default |
|------|------|--------|
| `--task` | Task type: `EE` / `NER` / `RE` / `all` | `all` |
| `--language` | Prompt language: `zh` / `en` | `zh` |
| `--project-root` | Project root path | Parent directory of the script |
| `--api-config` | Path to the api_config file (override default) | — |
| `--prompt-config` | Path to the prompt_config file (override default) | — |
| `--data-dir` | Data directory path (override default) | — |
| `--output-dir` | Output directory path (override default) | — |
| `--save-json` | Whether to additionally save the merged `all_results.json` | `False` |

---

## Dependency Installation

```
requests>=2.31.0
PyYAML>=6.0.1
```

```bash
pip install -r requirements.txt
```

> Recommended: Python 3.8+
