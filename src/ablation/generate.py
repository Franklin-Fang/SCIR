import json
import json5
from datetime import datetime
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
import torch

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,2"
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

inst_zh_ee = '你是事件抽取专家，请执行以下操作：\n1. 从input中抽取出符合schema的事件以及属性，按照{event_type: 事件名称, arguments:{属性名称:属性值,...}的JSON字符串的格式回答，缺失论元填NAN。\n'
inst_en_ee = 'You are an event extraction expert. Please perform the following operations:\n1. Extract events and properties that match the schema from the input and answer in the format of a JSON string: {event_type: event_name, arguments: {property_name: property_value, ...}}, fill in NAN for missing arguments.\n'
inst_zh_re = '你是关系抽取专家。请执行以下操作：\n1. 从input中抽取出符合schema定义关系的头尾节点，并将这些头尾节点以 {head:头结点, tail:尾节点} 的形式添加进schema中对应关系的列表中，不存在的关系保持为空列表。\n'
inst_zh_ner = '你是实体抽取专家。请执行以下操作：\n1.从input中抽取出符合schema定义的实体，并将其添加到schema中对应的实体类型的列表里，input不存在的实体类型定义的实体则保持空列表。\n'
inst_en_re = 'You are a specialist in relation extraction. Please perform the following operations:\n1. Extract the head and tail nodes from the input that match the relationships defined in the schema, and add these head and tail nodes in the form of {head: head node, tail: tail node} to the corresponding relationship list in the schema. Relationships that do not exist should remain as empty lists.\n'
inst_en_ner = 'You are an expert specializing in entity extraction. Please perform the following operations:\n1. Extract entities from the input that match the definitions in the schema and add them to the list of corresponding entity types in the schema. If the input does not contain entities defined for a certain entity type, keep that list empty. \n'

base_model_name = "./model/Qwen3-32B"  # 以通义千问为例，可替换为其他预训练模型
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
sampling_params = SamplingParams(
    temperature=0.7, 
    top_p=0.8,
    top_k=20,
    max_tokens=2048
)
# Initialize the vLLM engine
llm = LLM(
    model=base_model_name,
    max_model_len=4096,
    dtype=torch.bfloat16,
    trust_remote_code=True,
    quantization="bitsandbytes",
    tensor_parallel_size=2,
    gpu_memory_utilization=0.8
)

def generate(data,prompts):
    print(prompts[0])
    formatted_prompts = []
    for prompt in prompts:
        msg =[
            {"role": "user", "content": prompt} 
        ] 
        formatted_prompt = tokenizer.apply_chat_template(
            msg,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        ) 
        formatted_prompts.append(formatted_prompt)
    outputs = llm.generate(formatted_prompts,sampling_params)
    for item,output in zip(data,outputs):
        result = output.outputs[0].text
        result = result.replace("```json","")
        result = result.replace("```","")
        result = result.replace("\n","")
        item['output'] = result
    return data

def base_generate(data):
    print('Start base')
    prompts = []
    for item in data:
        prompts.append(item['instruction'])
    return generate(data,prompts)

def en_re(data):
    prompts = []
    print("Start generate en_re")
    for index,item in enumerate(data):
        instruction = json.loads(item['instruction'])
        if item['missing'] == 'FormatError' or item['redundancy'] == 'FormatError':
            addition = '2.Verify whether the generated results conform to the following format:{relation_name: [{head: head_entity, tail: tail_entity}, ...], ...}'
        else:
            missing = item['missing']
            redundancy = item['redundancy']
            try:
                if redundancy == "{}" and missing == "{}":
                    continue
                elif redundancy == "{}" :
                    addition = '2.Check if there are any unextracted relationships in the generated results (e.g., "{}"), and make corresponding modifications to the returned results.'
                    addition = addition.format(missing)
                elif missing == "{}":
                    addition = '2.Check whether the generated results contain incorrectly extracted relationships or non-existent relationships (e.g., "{}"), and make corresponding corrections to the returned results.'
                    addition = addition.format(redundancy)
                else:
                    addition = '2.Check if there are any unextracted relationships in the generated results (e.g., "{}"), and make corresponding corrections to the returned results.\n3.Check whether the generated results contain incorrectly extracted relationships or non-existent relationships (e.g., "{}"), and make corresponding corrections to the returned results.'
                    addition = addition.format(missing,redundancy)
            except:
                continue
        instruction['instruction'] = inst_en_re + addition +'\nPlease respond in JSON string format, only provide the answer without any additional output.'
        prompt = json.dumps(instruction,ensure_ascii=False)
        prompts.append(prompt)
    return generate(data,prompts)

def zh_re(data):
    prompts = []
    print("Start generate zh_re")
    for index,item in enumerate(data):   
        instruction = json.loads(item['instruction'])
        if item['missing'] == 'FormatError' or item['redundancy'] == 'FormatError':
            addition = '2.结果应是符合以下格式：{关系名称: [{head:头结点, tail:尾节点},...],...}。'
        else:
            missing = item['missing']
            redundancy = item['redundancy']
            try:
                if redundancy == "{}" and missing == "{}":
                    continue
                elif redundancy == "{}" :
                    addition = '2.检查生成结果中是否存在关系未抽取（例如：\"{}\"等关系），并进行相应的修改。'
                    addition = addition.format(missing)
                elif missing == "{}":
                    addition = '2.检查生成结果中是否存在关系抽取错误或抽取了不存在的关系（例如：\"{}\"等关系），并进行相应的修改。'
                    addition = addition.format(redundancy)
                else:
                    addition = '2.检查生成结果中是否存在关系未抽取（例如：\"{}\"等关系），并进行相应的修改。\n3.检查生成结果中是否存在关系抽取错误或抽取了不存在的关系（例如：\"{}\"等关系），并进行相应的修改。'
                    addition = addition.format(missing,redundancy)
            except:
                continue
        instruction['instruction'] = inst_zh_re + addition +'\n请按照JSON字符串的格式回答,只回答答案不要输出其他内容。'
        prompt = json.dumps(instruction,ensure_ascii=False)
        prompts.append(prompt)
    return generate(data,prompts)

def en_ner(data):
    prompts = []
    print("Start generate en_ner")
    for index,item in enumerate(data):  
        instruction = json.loads(item['instruction'])
        if item['missing'] == 'FormatError' or item['redundancy'] == 'FormatError':
            addition = '2.Check whether the generated result conforms to the following format: {entity_type: [entity_name_1,..., entity_name_n],...}'
        else:
            missing = item['missing']
            redundancy = item['redundancy']
            try:
                if redundancy == "{}" and missing == "{}":
                    continue
                elif redundancy == "{}" :
                    addition = '2.Check whether there is unextracted entity in the generated result (for example: entities such as "{}"), and make corresponding modifications to the returned result.'
                    addition = addition.format(missing)
                elif missing == "{}":
                    addition = '2.Check whether there are errors in extracting entity or whether non-existent entity are extracted in the generated results (for example: entities such as "{}"), and make corresponding modifications to the returned results.'
                    addition = addition.format(redundancy)
                else:
                    addition = '2.Check whether there is unextracted entity in the generated result (for example: entities such as "{}"), and make corresponding modifications to the returned result.\n3.Check whether there are errors in extracting entity or whether non-existent entity are extracted in the generated results (for example: entities such as "{}"), and make corresponding modifications to the returned results.'
                    addition = addition.format(missing,redundancy)
            except:
                continue
        instruction['instruction'] = inst_en_ner + addition +'\nPlease respond in JSON string format, only provide the answer without any additional output.'
        prompt = json.dumps(instruction,ensure_ascii=False)
        prompts.append(prompt)
    return generate(data,prompts)

def zh_ner(data):
    prompts = []
    print("Start generate zh_ner")
    for index,item in enumerate(data):  
        instruction = json.loads(item['instruction'])
        if item['missing'] == 'FormatError' or item['redundancy'] == 'FormatError':
            addition += '2.检查生成结果是否符合以下格式：{实体类型: [实体名称1,...,实体名称n],...}。'
        else:
            missing = item['missing']
            redundancy = item['redundancy']
            try:
                if redundancy == "{}" and missing == "{}":
                    continue
                elif redundancy == "{}" :
                    addition = '2.检查生成结果中是否存在实体未抽取（例如：\"{}\"等实体），并进行相应的修改。'
                    addition = addition.format(missing)
                elif missing == "{}":
                    addition = '2.检查生成结果中是否存在实体抽取错误或抽取了不存在的实体（例如：\"{}\"等实体），并进行相应的修改。'
                    addition = addition.format(redundancy)
                else:
                    addition = '2.检查生成结果中是否存在实体未抽取（例如：\"{}\"等实体），并进行相应的修改。\n3.检查生成结果中是否存在实体抽取错误或抽取了不存在的实体（例如：\"{}\"等实体），并进行相应的修改。'
                    addition = addition.format(missing,redundancy)
            except:
                continue
        instruction['instruction'] = inst_zh_ner + addition +'\n请按照JSON字符串的格式回答,只回答答案不要输出其他内容。'
        prompt = json.dumps(instruction,ensure_ascii=False)
        prompts.append(prompt)
    return generate(data,prompts)

def zh_ee(data):
    prompts = []
    print("Start generate zh_ee")
    for index,item in enumerate(data):  
        instruction = json.loads(item['instruction'])
        if item['missing'] == 'FormatError' or item['redundancy'] == 'FormatError':
            addition += '2.检查生成结果是否符合以下格式：{event_type: 事件名称, arguments:{属性名称:属性值,...}。'
        else:
            missing = item['missing']
            redundancy = item['redundancy']
            try:
                if redundancy == "{}" and missing == "{}":
                    continue
                elif redundancy == "{}" :
                    addition = '2.检查生成结果中arguments是否存在属性值未抽取（例如：\"{}\"等属性），并对返回结果进行相应的修改。'
                    addition = addition.format(missing)
                elif missing == "{}":
                    addition = '2.检查生成结果中arguments是否存属性的属性值抽取错误或抽取了不存在的属性值（例如：\"{}\"等属性），并对返回结果进行相应的修改。'
                    addition = addition.format(redundancy)
                else:
                    addition = '2.检查生成结果中arguments是否存在属性值未抽取（例如：\"{}\"等属性）\n3.检查生成结果中arguments是否存属性的属性值抽取错误或抽取了不存在的属性值（例如：\"{}\"等属性），并对返回结果进行相应的修改。'
                    addition = addition.format(missing,redundancy)
            except:
                continue
        instruction['instruction'] = inst_zh_ee + addition +'\n请按照JSON字符串的格式回答,只回答答案不要输出其他内容。'
        prompt = json.dumps(instruction,ensure_ascii=False)
        prompts.append(prompt)
    return generate(data,prompts)

def en_ee(data):
    prompts = []
    print("Start generate en_ee")
    for index,item in enumerate(data):  
        instruction = json.loads(item['instruction'])
        if item['missing'] == 'FormatError' or item['redundancy'] == 'FormatError':
            addition += '2. Check if the generated result conforms to the following format: {event_type: event_name, arguments: {property_name: property_value, ...}.'
        else:
            missing = item['missing']
            redundancy = item['redundancy']
            try:
                if redundancy == "{}" and missing == "{}":
                    continue
                elif redundancy == "{}" :
                    addition = '2. Check if there are unextracted property values in the generated result (for example: properties such as "{}"), and modify the returned result accordingly.'
                    addition = addition.format(missing)
                elif missing == "{}":
                    addition = '2. Check if there are property value extraction errors or extraction of non-existent property values in the generated result (for example: properties such as "{}"), and modify the returned result accordingly.'
                    addition = addition.format(redundancy)
                else:
                    addition = '2. Check if there are unextracted property values in the generated result (for example: properties such as "{}")\n3. Check if there are property value extraction errors or extraction of non-existent property values (for example: properties such as "{}") in the generated result, and modify the returned result accordingly.'
                    addition = addition.format(missing,redundancy)
            except:
                continue
        instruction['instruction'] = inst_en_ee + addition +'\nPlease answer in the format of a JSON string, and only provide the answer without any additional output.'
        prompt = json.dumps(instruction,ensure_ascii=False)
        prompts.append(prompt)
    return generate(data,prompts)

def run_generate(data,task):
    if 'zh' in task:
        if 'ner' in task:
            return zh_ner(data)
        elif 're' in task:
            return zh_re(data)
        elif 'ee' in task:
            return zh_ee(data)
    elif 'en' in task:
        if 'ner' in task:
            return en_ner(data)
        elif 're' in task:
            return en_re(data)
        elif 'ee' in task:
            return en_ee(data)
    else:
        return base_generate(data)
