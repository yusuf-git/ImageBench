# created on 16-Nov-2020 02:15 AM #
# updates on 17-Nov-2020 03:20 AM, 12:40 PM #
import json
class cons_results_data_model:
    #def __init__(self, image, base_img_path, runtime_img_path, captured_baseline_kp, captured_runtime_kp, captured_good_points, captured_good_points_percent, captured_kp_variance,confirmed_baseline_kp, confirmed_runtime_kp, confirmed_good_points, confirmed_good_points_percent, confirmed_kp_variance, msg):  
    def __init__(self, image, p_hash_score, p_hash_msg, BF_exp_score, BF_orig_score, ssi_score, d_hash_score, d_hash_msg):  
        self.image = image
        #self.base_img_path = base_img_path
        #self.runtime_img_path = runtime_img_path
        #self.captured_baseline_kp = captured_baseline_kp
        #self.captured_runtime_kp = captured_runtime_kp
        #self.captured_good_points = captured_good_points  
        #self.captured_good_points_percent = captured_good_points_percent
        #self.captured_kp_variance = captured_kp_variance
        self.p_hash_score = p_hash_score
        self.p_hash_msg = p_hash_msg
        self.BF_exp_score = BF_exp_score
        self.BF_orig_score = BF_orig_score
        self.ssi_score = ssi_score
        self.d_hash_score = d_hash_score
        self.d_hash_msg = d_hash_msg
    #image_match_outcome_dict = {}

    def toJson(self):
        return json.dumps(self, default=lambda o: o.__dict__)

    def __repr__(self):                                                                                                                                                                                                                                       
        #return "image:% s, base_img_path:% s, runtime_img_path:% s, captured_baseline_kp:% s, captured_runtime_kp:% s, captured_good_points:% s, captured_good_points_percent:% s, captured_kp_variance:% s, confirmed_baseline_kp:% s, confirmed_runtime_kp:% s, confirmed_good_points:% s, confirmed_good_points_percent:% s, confirmed_kp_variance:% s, msg:% s" % (self.image, self.base_img_path, self.runtime_img_path, self.captured_baseline_kp, self.captured_runtime_kp, self.captured_good_points, self.captured_good_points_percent, self.captured_kp_variance, self.confirmed_baseline_kp, self.confirmed_runtime_kp, self.confirmed_good_points, self.confirmed_good_points_percent, self.confirmed_kp_variance, self.msg)
        return "image:% s, p_hash_score:% s, p_hash_msg:% s, BF_exp_score:% s, BF_orig_score:% s, ssi_score:% s, d_hash_score:% s, d_hash_msg:% s" % (self.image, self.p_hash_score, self.p_hash_msg, self.BF_exp_score, self.BF_orig_score, self.ssi_score, self.d_hash_score, self.d_hash_msg)

    consolidated_result_list = []
    #BF_algo_baseline_list = []
    #newimgs_baseline_buffer = []
    #basetobase_kp_update_list = []


    def dump(self):
        return  {'image': self.image,
                 #'base_img_path':self.base_img_path,
                 #'runtime_img_path':self.runtime_img_path,
                 #'captured_baseline_kp':self.captured_baseline_kp,
                 ##'captured_runtime_kp':self.captured_runtime_kp,
                 #'captured_good_points': self.captured_good_points,
                 #'captured_good_points_percent': self.captured_good_points_percent,
                 #'captured_kp_variance':self.captured_kp_variance,
                 'p_hash_score': self.p_hash_score,
                 'p_hash_msg': self.p_hash_msg,
                 'BF_exp_score': self.BF_exp_score,
                 'BF_orig_score': self.BF_orig_score,
                 'ssi_score': self.ssi_score,
                 'd_hash_score': self.d_hash_score,
                 'd_hash_msg': self.d_hash_msg
                }


class CustomEncoder(json.JSONEncoder):
    def default(self, o):
        #if isinstance(o, Custom):
        #    return 'YES-RIGHT'
        return CustomEncoder(self, o)

