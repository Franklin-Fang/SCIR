import json
import json5
import requests
import argparse

from src.generate import run_generate
from src.check import check

def load_data(file_dir):
    data = []
    with open(file_dir, 'r', encoding='utf-8') as file:
        for line in file:
            item = json.loads(line)
            data.append(item)
    return data

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, help="input_path", required=True)
    parser.add_argument("--output", type=str, help="output_path", required=True)
    parser.add_argument("--task", type=str, help="task", required=True)
    args = parser.parse_args()

    input_path = args.input
    output_path = args.output
    now_task = args.task
    template_output = output_path + '_item_{}.json'
    template_check = output_path + '_check_{}.json'

    index = 0
    K = 2

    data = load_data(input_path)
    data = run_generate(data,'base')
    out_dir = template_output.format(index)
    with open(out_dir, 'w', encoding='utf-8') as f_out:
        for item in data:
            json.dump(item, f_out, ensure_ascii=False)
            f_out.write('\n')
    print(f"Finish generate {index} iter!")
    
    for i in range(K):
        data = check(data,now_task)
        check_dir = template_check.format(index)
        with open(check_dir, 'w', encoding='utf-8') as f_out:
            for item in data:
                json.dump(item, f_out, ensure_ascii=False)
                f_out.write('\n')
        print(f"Finish check {index} iter!")

        data = run_generate(data,now_task)
        index += 1
        out_dir = template_output.format(index)
        with open(out_dir, 'w', encoding='utf-8') as f_out:
            for item in data:
                json.dump(item, f_out, ensure_ascii=False)
                f_out.write('\n')
        print(f"Finish generate {index} iter!")
        

