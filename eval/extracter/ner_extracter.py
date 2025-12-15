import re
import json
from eval.extracter.extracter import Extracter


class NERExtracter(Extracter):
    def __init__(self, language="zh", NAN="NAN", prefix = "输入中包含的实体是：\n", Reject="No entity found."):
        super().__init__(language, NAN, prefix, Reject)

    def post_process(self, result):  
        try:      
            rst = json.loads(result)
        except json.decoder.JSONDecodeError:
            print("json decode error", result)
            return []
        if type(rst) != dict:
            print("type(rst) != dict", result)
            return []
        new_record = []
        for name in rst:
            if type(name) != str or type(rst[name]) != list:
                print("type(name) != str or type(rst[name]) != list", result)
                continue
            for iit in rst[name]:
                if type(iit) != str:
                    print("type(iit) != str", result)
                    continue
                new_record.append((iit, name))
        return new_record

