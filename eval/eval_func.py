# coding=utf-8
import json
import os
import sys
sys.path.append('./')
import argparse
from collections import defaultdict
from eval.metric import get_metric
from eval.extracter import get_extracter


def convert_kg(outputs, task):
    kgs = []
    if task == 'NER':
        for it in outputs:
            kgs.append((it['entity'], it['entity_type']))
    elif task == 'RE':
        for it in outputs:
            kgs.append((it.get('head', ''), it.get('relation', ''), it.get('tail', '')))
    elif task == 'EE':
        for it in outputs:
            args = []
            for arg in it['arguments']:
                args.append((arg['argument'], arg['role']))
            kgs.append((it['event_type'], it['event_trigger'], tuple(args)))
    return kgs


def evaluate(options):
    extracter_class = get_extracter(options.task)
    metric_class = get_metric(options.task)

    with open(options.path, 'r') as reader:
        data = json.load(reader)

    total_counter = metric_class(options.match_mode, options.metrics_list)

    extracter = extracter_class()
    for item in data:
        if options.kind == 'retrieve':
            preds = item['rag']
        else:
            preds = item['output']
        label = item['label']


        converted_preds = extracter.extract(preds)
        label_kgs = extracter.extract(label)

        total_counter.count_instance(
            gold_list=label_kgs, pred_list=converted_preds
        )

    cate_results = {}
    total_result = total_counter.compute()

    all_result = {}
    all_result['total'] = total_result
    for key, value in cate_results.items():
        all_result[key] = value
    print(all_result)

def main():
    '''
    python ./eval/eval_func.py \
        --path ./data/result/NER-zh/WEIBONER.json \
        --task NER \
        --kind retrieve
    '''
    parse = argparse.ArgumentParser()
    parse.add_argument("--path", type=str, default="")
    parse.add_argument("--task", type=str, default='re', choices=['NER', 'RE', 'EE', 'SPO', 'EET', 'EEA', 'KG', 'MRC'])
    parse.add_argument("--language", type=str, default='zh', choices=['zh', 'en']) 
    parse.add_argument("--NAN", type=str, default="")
    parse.add_argument("--prefix", type=str, default='')
    parse.add_argument("--kind", type=str, default='')
    parse.add_argument("--match_mode", type=str, default="normal")
    parse.add_argument("--metrics_list", type=str, default="f1")
    options = parse.parse_args()
    evaluate(options)



if __name__=="__main__":
    main()
