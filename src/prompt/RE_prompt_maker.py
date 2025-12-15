import json

inst_zh_re = '你是关系抽取专家。请执行以下操作：\n1. 从input中抽取出符合schema定义关系的头尾节点，并将这些头尾节点以 {head:头结点, tail:尾节点} 的形式添加进schema中对应关系的列表中，不存在的关系保持为空列表。\n'
inst_en_re = 'You are a specialist in relation extraction. Please perform the following operations:\n1. Extract the head and tail nodes from the input that match the relationships defined in the schema, and add these head and tail nodes in the form of {head: head node, tail: tail node} to the corresponding relationship list in the schema. Relationships that do not exist should remain as empty lists.\n'

def en_re(data):
    prompts = []
    print("Start generate en_re")
    for item in data:
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
    return prompts

def zh_re(data):
    prompts = []
    print("Start generate zh_re")
    for item in data:
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
    return prompts
