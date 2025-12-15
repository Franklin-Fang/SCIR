from .ner_metric import NERMetric
from .re_metric import REMetric
from .ee_metric import EEMetric

def get_metric(task):
    if task == 'NER':
        return NERMetric
    elif task == 'RE':
        return REMetric
    elif task == 'EE':
        return EEMetric
    else:
        raise ValueError("Invalid task: %s" % task)
    
