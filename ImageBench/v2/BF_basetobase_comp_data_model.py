# created on 06-Nov-2020 11:10 PM #
import json
class BF_basetobase_comp_data_model:
    def __init__(self, image, comp_result, basetobase_latest_kp):  
        self.image = image
        self.comp_result = comp_result
        self.basetobase_latest_kp = basetobase_latest_kp
    #image_match_outcome_dict = {}

    def toJson(self):
        return json.dumps(self, default=lambda o: o.__dict__)

    def __repr__(self):                                                                                                                                                                                                                                       
        return "image:% s, comp_result:% s, basetobase_latest_kp:% s" % (self.image, self.comp_result, self.basetobase_latest_kp)

    basetobase_kp_update_list = []


    def dump(self):
        return  {'image': self.image,
                 'comp_result':self.comp_result,
                 'basetobase_latest_kp':self.basetobase_latest_kp
                }


class CustomEncoder(json.JSONEncoder):
    def default(self, o):
        #if isinstance(o, Custom):
        #    return 'YES-RIGHT'
        return CustomEncoder(self, o)

