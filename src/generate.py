from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from prompt.prompt_maker import prompt_maker
import torch

def generate(data,prompts):
    
    base_model_name = "./model/Qwen3-8B"
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    sampling_params = SamplingParams(
        temperature=0.7, 
        top_p=0.8,
        top_k=20,
        max_tokens=2048
    )

    llm = LLM(
        model=base_model_name,
        max_model_len=4096,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        tensor_parallel_size=2,
        gpu_memory_utilization=0.8
    )

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

def run_generate(data,task):
    prompts = prompt_maker(data,task)
    results = generate(data, prompts)
    return results
        
