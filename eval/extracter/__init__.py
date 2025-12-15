from .ee_extracter import EEExtracter
from .re_extracter import REExtracter
from .ner_extracter import NERExtracter


def get_extracter(task):
    if task == 'NER':
        return NERExtracter
    elif task == 'RE':
        return REExtracter
    elif task == 'EE':
        return EEExtracter
    