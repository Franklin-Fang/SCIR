import json

inst_zh_ner = '你是实体抽取专家。请执行以下操作：\n1.从input中抽取出符合schema定义的实体，并将其添加到schema中对应的实体类型的列表里，input不存在的实体类型定义的实体则保持空列表。\n'
inst_en_ner = 'You are an expert specializing in entity extraction. Please perform the following operations:\n1. Extract entities from the input that match the definitions in the schema and add them to the list of corresponding entity types in the schema. If the input does not contain entities defined for a certain entity type, keep that list empty. \n'

def en_ner(data):
    prompts = []
    print("Start generate en_ner")
    for item in data:  
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
    return prompts

def zh_ner(data):
    prompts = []
    print("Start generate zh_ner")
    for item in data:
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
    return prompts