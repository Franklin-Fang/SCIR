import json
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

missing_prompt = {
    'zh-ee':"请完成下面的事件抽取结果检测任务：result是从text文本中抽取的结果（若某个论元不存在则显示 \"NAN\"），请检查抽取结果是否遗漏了result中定义的任何论元。请严格按照以下要求回答：\n1. 如果不存在遗漏，返回字符串\"Correct\"。\n2. 如果有遗漏，请返回遗漏的论元及其抽取结果的JSON字符串，格式为：{遗漏论元: 对应结果}。\n请确保回答中只包含检查结果，不要包含任何额外的文本，以JSON格式输出。",
    'en-ee':"Please complete the following event extraction result detection task: The result is the outcome extracted from the text (if a certain argument does not exist, display \"NAN\"). Please check whether any arguments defined in the result are missing. Please answer strictly according to the following requirements:\n1. If there is no omission, return the string \"Correct\".\n2. If there are omissions, return the JSON string of the omitted arguments and their extraction results in the format: {Omitted_Argument: Corresponding_Result}.\nPlease ensure that the answer only contains the inspection results and does not include any additional text, and output in JSON format.",
    'zh-ner': "请完成下面的实体抽取结果检测任务：result是从text文本中抽取的结果（若某个实体不存在则用空列表表示），请检查抽取结果是否遗漏了text中存在的名称。请严格按照以下要求回答：\n1. 如果不存在遗漏，返回字符串{\"Correct\"}。\n2. 如果有遗漏，则返回遗漏的名称和对应的实体类型组成的字典，格式为：{实体类型:[遗漏的名称]}。\n请确保回答中只包含检查结果，不要包含任何额外的文本，以JSON格式输出。",
    'en-ner':"Please complete the following entity extraction result verification task: 'result' is the extraction result from the 'text' (use an empty list if an entity does not exist). Check whether the extraction result omits any names present in the text. Strictly follow these requirements:\n1. If no omissions exist, return the string {\"Correct\"}.\n2. If omissions exist, return a dictionary of the omitted names and their corresponding entity types, formatted as: {entity_type: [omitted_names]}.\nPlease ensure that the answer only contains the inspection results and does not include any additional text, and output in JSON format.",
    'zh-re':"请完成下面的关系抽取结果检测任务：result是从text文本中抽取的结果（若某个关系不存在则用空列表表示），请检查抽取结果是否遗漏了result中定义的关系。请严格按照以下要求回答：\n1. 如果不存在遗漏，返回字符串{\"Correct\"}。2. 如果有遗漏，请返回遗漏的关系及头尾节点组成的JSON字符串，格式为：{遗漏关系:[{head:遗漏的头结点,tail:遗漏的尾节点}]}。\n请确保回答中只包含检查结果，不要包含任何额外的文本，以JSON格式输出。",
    'en-re': "Please complete the following relation extraction result verification task: 'result' is the extracted output from the text (use an empty list if a relation does not exist). Check whether the extraction results have omitted any relations defined in 'result'. Strictly follow these requirements:\n1. If no omissions exist, return the string \"Correct\".\n2. If omissions exist, return the missing relations and their head/tail nodes in JSON format: {missing_relation:[{head:missing_head_node, tail:missing_tail_node}]}.\nEnsure the response contains only the verification result without any additional text, and output in JSON format."
}

def check_missing(inputs,task):
    inst = missing_prompt[task]
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

    return messages_batch