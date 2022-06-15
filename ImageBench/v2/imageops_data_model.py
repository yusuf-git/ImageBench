# updates on 06-Oct-2020 08:05 PM, 11-Oct-2020 02:30 PM to 11:50 PM, 12-Oct-2020 02:00 AM #
import json
class imageops:
    def __init__(self, image, base_img_path, runtime_img_path, algo, expscore, original_score, result, msg):  
        self.image = image
        self.base_img_path = base_img_path
        self.runtime_img_path = runtime_img_path
        self.algo = algo
        self.expscore = expscore  
        self.original_score = original_score
        self.result = result
        self.msg = msg
    #image_match_outcome_dict = {}

    def toJson(self):
        return json.dumps(self, default=lambda o: o.__dict__)

    def __repr__(self):  
        return "image:% s, base_img_path:% s, runtime_img_path:% s, algo:% s, exp-score:% s, orig-score:% s, result:% s, msg:% s" % (self.image, self.base_img_path, self.runtime_img_path, self.algo, self.expscore, self.original_score, self.result, self.msg)

    image_match_outcome_list = []


    def dump(self):
        return  {'image': self.image,
                 'base_img_path':self.base_img_path,
                 'runtime_img_path':self.runtime_img_path,
                 'algo': self.algo,
                 'expscore': self.expscore,
                 'original_score': self.original_score,
                 'result':self.result,
                 'msg':self.msg}


class CustomEncoder(json.JSONEncoder):
    def default(self, o):
        #if isinstance(o, Custom):
        #    return 'YES-RIGHT'
        return CustomEncoder(self, o)

