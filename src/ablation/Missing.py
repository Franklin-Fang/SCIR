import json

from generate import run_generate
from check_missing import check

def load_data(file_dir):
    data = []
    with open(file_dir, 'r', encoding='utf-8') as file:
        for line in file:
            item = json.loads(line)
            data.append(item)
    return data

if __name__ == '__main__':
    with open('./script/Missing.json', 'r', encoding='utf-8') as f:
        commands = json.load(f)
    for command in commands:
        input_path = command['input']
        output_path = command['output']
        now_task = command['task']
        print(f"----------------------- {now_task} ------------------------------")
        template_output = output_path + '_item_{}.json'
        template_check = output_path + '_check_{}.json'
        
        index = 0

        out_dir = input_path
        data = load_data(out_dir)

        for i in range(2):
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
        

