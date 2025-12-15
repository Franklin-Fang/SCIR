from src.prompt.EE_prompt_maker import zh_ee, en_ee
from src.prompt.RE_prompt_maker import zh_re, en_re
from src.prompt.NRE_prompt_maker import zh_ner, en_ner

def prompt_maker(data,task):
    if 'zh' in task:
        if 'ner' in task:
            prompts = zh_ner(data)
        elif 're' in task:
            prompts = zh_re(data)
        elif 'ee' in task:
            prompts = zh_ee(data)
    elif 'en' in task:
        if 'ner' in task:
            prompts = en_ner(data)
        elif 're' in task:
            prompts = en_re(data)
        elif 'ee' in task:
            prompts = en_ee(data)
    else:
        prompts = []
        for item in data:
            prompts.append(item['instruction'])
    return prompts