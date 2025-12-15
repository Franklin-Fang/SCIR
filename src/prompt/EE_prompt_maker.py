import json

inst_zh_ee = '你是事件抽取专家，请执行以下操作：\n1. 从input中抽取出符合schema的事件以及属性，按照{event_type: 事件名称, arguments:{属性名称:属性值,...}的JSON字符串的格式回答，缺失论元填NAN。\n'
inst_en_ee = 'You are an event extraction expert. Please perform the following operations:\n1. Extract events and properties that match the schema from the input and answer in the format of a JSON string: {event_type: event_name, arguments: {property_name: property_value, ...}}, fill in NAN for missing arguments.\n'

def zh_ee(data):
    prompts = []
    print("Start generate zh_ee")
    for item in data:  
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
    return prompts

def en_ee(data):
    prompts = []
    print("Start generate en_ee")
    for item in data:  
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
    return prompts