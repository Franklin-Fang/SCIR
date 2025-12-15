import json
import json5
from datetime import datetime
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from src.check.redundancy import check_redundancy
from src.check.missing import check_missing

def format_check(data,task):
    inputs = []
    for item in data:
        instruction = json.loads(item['instruction'])
        input = {}
        input['text'] = instruction['input']
        try:
            result = json5.loads(item['output'])
            if result == {}:
                inputs_item = 'FormatError'
                print('FormatError')
                continue
            if 'ee' in task:
                input['result'] = result
                inputs_item = json.dumps(input,ensure_ascii=False)
            else:
                inputs_item = []
                for name in result:
                    if 'ner' in task:
                        input['result'] = {'entity_type':name,'entity':result[name]}
                    elif 're' in task:
                        input['result'] = {name:result[name]}
                    else:
                        print("Task Error!")
                        return
                    inputs_item.append(json.dumps(input,ensure_ascii=False))
        except:
            inputs_item = 'FormatError'
            print('FormatError')
        inputs.append(inputs_item)
    return inputs

def run_generate(llm,tokenizer,sampling_params,messages_batch,lora_path,task,path,data,inputs):
    texts = [
        tokenizer.apply_chat_template(
            msgs,
            tokenize=False,
            redundancy_generation_prompt=True,
            enable_thinking=False
        ) for msgs in messages_batch
    ]
    outputs = llm.generate(
        texts,
        sampling_params,
        lora_request=LoRARequest(path, 1, lora_path)
    )
    results = []
    for output in outputs:
        results.append(output.outputs[0].text)
    start = 0
    for item,input_item in zip(data,inputs):
        if input_item == 'FormatError':
            item['redundancy'] = 'FormatError'
            print('FormatError')
            continue
        if 'ee' in task:
            result = results[start]
            try:
                answer = json.loads(result)
            except:
                answer = {}
            start += 1
        else:
            end = start + len(input_item)
            result = results[start:end]
            start = end
            answer = {}
            for pair in result:
                try:
                    pair = json.loads(pair)
                    name = list(pair.keys())[0]
                    if len(pair[name]) != 0:
                        answer[name] = pair[name]
                except:
                    continue
        item[path] = answer
    return data


def check(data,task):
    inputs = format_check(data,task)

    base_model_name = "./model/Qwen3-4B/model"  # 以通义千问为例，可替换为其他预训练模型
    tokenizer = AutoTokenizer.from_pretrained(base_model_name,local_files_only=True)
    sampling_params = SamplingParams(
        temperature=0.1, 
        max_tokens=2048
    )
    # Initialize the vLLM engine
    llm = LLM(
        model=base_model_name,
        max_model_len=4096, 
        enable_lora=True,
        gpu_memory_utilization=0.75
    )

    print('Start check redundancy!')
    messages_batch = check_redundancy(inputs,task)
    data = run_generate(llm, tokenizer, sampling_params, messages_batch, './model/Qwen3-4B/redundancy', task, 'redundancy', data, inputs)
    print('Start check missing!')
    messages_batch = check_missing(inputs,task)
    data = run_generate(llm, tokenizer, sampling_params, messages_batch, './model/Qwen3-4B/missing', task, 'missing', data, inputs)
    return data