import json
import json5
from datetime import datetime
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
import torch

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

redundancy_prompt = {
    'zh-ee':"请完成下面的事件抽取结果检测任务：result是从text文本中抽取的结果（若某个论元不存在则显示 \"NAN\"），请检查抽取结果是否过度抽取了result中定义的任何论元。请严格按照以下要求回答：\n1. 如果不存在过度抽取，返回字符串\"Correct\"。\n2. 如果有过度抽取，请返回过度抽取的论元及其抽取结果的JSON字符串，格式为：{过度抽取论元: 对应结果}。\n请确保回答中只包含检查结果，不要包含任何额外的文本，以JSON格式输出。",
    'en-ee':"Please complete the following event extraction result over-detection task: The result is the outcome extracted from the text (if a certain argument does not exist, display \"NAN\"). Please check whether any arguments defined in the result have been over-detected. Please answer strictly according to the following requirements:\n1. If there is no over-detection, return the string \"Correct\".\n2. If there is over-detection, return the JSON string of the over-detected arguments and their extraction results in the format: {Over-Detected Argument: Corresponding Result}.\nPlease ensure that the answer only contains the inspection results and does not include any additional text, and output in JSON format.",
    'zh-ner': "请完成下面的实体抽取结果检测任务：result是从text文本中抽取的结果（若某个实体不存在则用空列表表示），请检查抽取结果是否有不在text中的实体名称被过度抽取的现象。请严格按照以下要求回答：\n1. 如果不存在过度抽取，返回字符串{\"Correct\"}。\n2. 如果有过度抽取，则返回不存在的名称和对应的实体类型组成的字典，格式为：{实体类型:[不存在的名称]}。\n请确保回答中只包含检查结果，不要包含任何额外的文本，以JSON格式输出。",
    'en-ner':"Please complete the following entity extraction result verification task: 'result' is the extraction result from the 'text' (use an empty list if an entity does not exist). Check whether the extraction result includes any entity names not present in the text (over-extraction). Strictly follow these requirements:\n1. If no over-extraction exists, return the string {\"Correct\"}.\n2. If over-extraction exists, return a dictionary of the non-existent names and their corresponding entity types, formatted as: {entity_type: [non-existent_names]}.\nEnsure the response contains only the verification result in JSON format, with no additional text.",
    'zh-re':"请完成下面的关系抽取结果检测任务：result是从text文本中抽取的结果（若某个关系不存在则用空列表表示），请检查抽取结果是否过度抽取了result中定义的关系。请严格按照以下要求回答：\n1. 如果不存在过度抽取，返回字符串{\"Correct\"}。\n2. 如果有过度抽取，请返回过度抽取的关系及头尾节点组成的JSON字符串，格式为：{过度抽取关系: [{head:过度抽取头结点,tail:过度抽取尾节点}]}。\n请确保回答中只包含检查结果，不要包含任何额外的文本，以JSON格式输出。",
    'en-re': "Please complete the following relation extraction result verification task: 'result' is the extracted output from the text (use an empty list if a relation does not exist). Check whether the extraction results contain any over-extracted relations beyond those defined in 'result'. Strictly follow these requirements:\n1. If no over-extraction exists, return the string \"Correct\".\n2. If over-extraction exists, return the over-extracted relations and their head/tail nodes in JSON format: {over-extracted_relation: [{head: over-extracted_head_node, tail: over-extracted_tail_node}]}.\nEnsure the response contains only the verification result without any additional text, and output in JSON format."
}

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

def check_redundancy(data,inputs,task):
    redundancy_lora_path = './model/Qwen3-4B/redundancy'
    
    inst = redundancy_prompt[task]
    print(inst)
    messages_batch = []
    for input_item in inputs:
        if input_item != 'FormatError':
            if 'ee' in  task:
                batch_item = [{"role": "user", "content": inst+'\n'+input_item}]
                messages_batch.append(batch_item)
            else:
                batch_item = [
                    [{"role": "user", "content": inst+'\n'+prompt} ] for prompt in input_item
                ]
                messages_batch += batch_item

    texts = [
        tokenizer.apply_chat_template(
            msgs,
            tokenize=False,
            add_generation_prompt=True
        ) for msgs in messages_batch
    ]
    outputs = llm.generate(
        texts,
        sampling_params,
        lora_request=LoRARequest("redundancy", 1, redundancy_lora_path)
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
                redundancy = json.loads(result)
            except:
                redundancy = {}
            start += 1
        else:
            end = start + len(input_item)
            result = results[start:end]
            start = end
            redundancy = {}
            for pair in result:
                try:
                    pair = json.loads(pair)
                    name = list(pair.keys())[0]
                    if len(pair[name]) != 0:
                        redundancy[name] = pair[name]
                except:
                    continue
        item['redundancy'] = redundancy
    return data

def check(data,task):
    inputs = []
    for index,item in enumerate(data):
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
    print(inputs[0])
    print('Start check redundancy!')
    data = check_redundancy(data,inputs,task)
    for item in data:
        item['missing'] = {}
    return data