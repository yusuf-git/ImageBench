#########################
# created on 17-Oct-2020 11:30 AM
# updates on 18-Oct-2020 03:15 AM, 24-Oct-2020 11:30 AM to 11:50 PM, 25-Oct-2020 12:35 AM to 06:40 AM, 27-Oct-2020 03:50 AM, 09-Nov-2020 03:30 AM, 16-Nov-2020 12:15 PM
#########################

from logging import log
import os
from os.path import join, getsize
import pathlib
import time
from shutil import copyfile
import shutil
from PIL import Image, ImageEnhance
from skimage.metrics import structural_similarity as ssim
#import matplotlib.pyplot as plt
import numpy as np
import cv2
import imutils
import imagehash
#from skimage.measure import compare_ssim
import argparse
import sys
import pyautogui as pygui
import img_comp_utils
from algos_namelist import comp_algos
import jsonpickle
#from di_container import di_container
import img_comparator
from imageops_data_model import imageops
from BF_baseline_data_model import BF_base_data_model
from BF_basetobase_comp_data_model import BF_basetobase_comp_data_model
from cons_results_data_model import cons_results_data_model
import json
import logging

# updates on 27-Oct-2020 02:50 AM #
def delPath(folder):
    if not os.path.exists(folder):
        return
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))
            logging.error('Failed to delete %s. Reason: %s' % (file_path, e))
            #writeToFile("dir_deletion_exception.txt",'Failed to delete %s. Reason: %s' % (file_path, e))
    workspace_base = folder
    if(os.path.exists(workspace_base)):
        logging.info(folder+" still exists")
    else:
        logging.info(folder+" has been deleted")


def writeToFile(file, contents):
    f = open(file, "a")
    f.write(contents)
    f.close()
    #open and read the file after the appending:
    #f = open(file, "r")
    #print(f.read())

def read_BF_algo_result(file):
    #f = open(file, "a")
    f = open(file, "r")
    return f.read()
    #print("BF_algo_baseline_read:",f.read())


# updates on 25-Oct-2020 07:00 PM, 26-Oct-2020 09:55 PM, 27-Oct-2020 02:20 AM #
def createFlatFolders(workspace_folder):
    workspace_base = os.path.join(workspace_folder,"image_ops/flat_base")
    workspace_runtime = os.path.join(workspace_folder,"image_ops/flat_runtime")
    delPath(workspace_base)
    delPath(workspace_runtime)
    if not os.path.exists(workspace_base):
        os.makedirs(workspace_base)
    if not os.path.exists(workspace_runtime):
        os.makedirs(workspace_runtime)
    logging.info("created the flat workspace folders....OK")
    return workspace_base, workspace_runtime

# updates 26-Oct-2020 02:30 AM, 09:50 PM, 27-Oct-2020 01:25 AM, 03:55 AM, 30-Oct-2020 11:30 AM to 02:00 PM #
def copyToFlatTargetPath(srcpath, targetpath, dt, imgList=[], copyToList=False):
    i=0
    t=0
    copied_img_cnt = 0
    e1=0
    l=""
    srcFile = ""
    tmp_imgList = imgList
    imglimit = int(dt["compare_args"]["imgcap"])
    print("***********************")
    for path, subdirs, files in os.walk(srcpath):
        if(copied_img_cnt >= imglimit):
            break
        for name in files:
            if(copied_img_cnt >= imglimit):
                break
            if(copyToList == False and len(imgList) > 0):
                if(name not in imgList): # applicable only for runtime images - copy only those files that were copied during baseline copy
                    continue
            if copyToList == True:
                tmp_imgList.append(name)
            
            srcFile = name
            targetFile = os.path.join(targetpath,os.path.basename(name))
            if(os.path.exists(targetFile)):
                if(srcpath in path):
                    l = str(path).replace(srcpath,"").replace("\\","_")
                    #print("this is the subset path:",l)
                    f1 = os.path.basename(name).split(".")
                    if(len(f1) > 2):
                        f1[2] = ""
                    if(len(f1) > 1):
                        targetFile = targetFile.split(".")[0] + l + "." + targetFile.split(".")[1]
                        srcFile = str(name).split(".")[0] + l + "." + str(name).split(".")[1]
                        print(srcFile)
                    else:
                        targetFile = targetFile + l
                        srcFile = str(name) + l
                    fl = os.path.join(path, srcFile)
                    if(os.path.exists(fl)==False):
                        os.rename(os.path.join(path, name),fl)
                        e1 = e1 + 1
                    else:
                        print("This image already exists")
                t = t + 1
            print("target File:",targetFile)
            f_size = os.stat(os.path.join(path, srcFile)).st_size
            if(f_size > 0):
                copyfile(os.path.join(path, srcFile), targetFile)
                copied_img_cnt = copied_img_cnt + 1
            i = i+1
    logging.info("***********************************")
    logging.info("src path for copy            :{0}".format(srcpath))
    logging.info("destination path for copy    :{0}".format(targetpath))
    logging.info("configured images for copy   :{0}".format(imglimit))
    logging.info("images with conflicting name :{0}".format(t))
    logging.info("renamed files                :{0}".format(e1))
    logging.info("copied images                :{0}".format(copied_img_cnt))
    logging.info("***********************************")
    print("***********************************")
    print("configured images for copy   :{0}".format(imglimit))
    print("images with conflicting name :{0}".format(t))
    print("renamed files                :{0}".format(e1))
    print("copied images                :{0}".format(copied_img_cnt))
    print("***********************************")
    return tmp_imgList

# updates on 11-Nov-2020 01:15 AM ##
def get_algo_configs(dt):
    algos = dt["compare_args"]["similarity"]
    exp_score = ""
    idx = 0
    ret_dict = {}
 
    for i in algos:
        exp_score = str(list(i.values())[1])
        if(exp_score!=""):
            ret_dict[str(list(i.values())[0])] = str(list(i.values())[1])
    if(exp_score==""):
        print("default algo selected : SSI")
        for i in algos:
            if(str(list(i.values())[0])==comp_algos.ssi):
                exp_score = str(list(i.values())[1])
    return ret_dict


# created on 11-Nov-2020 01:35 AM #
def get_algo_runnable_details(dt):
    algos = dt["compare_args"]["similarity"]
    runnable_algos_dict = {}
    for i in algos:
        runnable_algos_dict[str(list(i.values())[0])] = str(list(i.values())[3])
    return runnable_algos_dict


def get_algo_mapper():
    return {
    "SSI":img_comparator.SSI_compare,
    "perceptual_hashing":img_comparator.perceptual_hash_match,
    "BRISK-FLANN":img_comparator.BRISK_FLANN_match,
    "diff_hashing":img_comparator.diff_hash_match
    }


def get_algo_name_list(dt):
    algo_configs = get_algo_configs(dt)
    algo_name_list = []
    for key in algo_configs.keys():
        algo_name_list.append(key)
    return algo_name_list

# not needed for the util #
def get_algo_match_operator(dt):
    algos = dt["compare_args"]["similarity"]
    ret_dict = {}
    for i in algos:
        operator = str(list(i.values())[2])
        if operator == "":
            operator = "and"
        ret_dict[str(list(i.values())[0])] = str(list(i.values())[2])
    return ret_dict

# updates 26-Oct-2020 02:30 AM, 09:50 PM, 05-Nov-2020 02:40 AM #
def call_func(dt, algo_idx, tmp_match_data, preprocessing_needed):
    tmp_res = []
    algo_name_list = get_algo_name_list(dt)
    algo_mapper = get_algo_mapper()
    print("algo active index | algo :",algo_idx, "|", algo_name_list[algo_idx])
    func, dt = algo_mapper[algo_name_list[algo_idx]](dt,tmp_match_data, preprocessing_needed)
    if(callable(func)):
        tmp_res = func()
        return algo_name_list, func, dt
    else:
        return algo_name_list, func, dt

# not needed for the util #
def find_passed_imgs_with_or_operator(n, curr_algo, operator, comp_result_data):
    passed_algos_imgs_with_or_operator = {}
    passed_imgs_with_or_operator = []
    if(comp_result_data[n]["result"] == True and operator.lower() == "or"):
        passed_imgs_with_or_operator.append(comp_result_data[n]["image"])
        passed_algos_imgs_with_or_operator[curr_algo] = passed_imgs_with_or_operator
    return passed_algos_imgs_with_or_operator


 # created on 17-Oct-2020 05:30 PM #
 # updates on 18-Oct-2020 12:25 AM, 03:15 AM #
def analyze_net_result(dt, comp_result_data, op_net_result):
    curr_algo=""
    match_operator = get_algo_match_operator(dt)
    operations_res = True
    comp_res_obj  = iter(comp_result_data)
    n = 0
    passed_imgs_with_or_operator_algo = {}
    tmp_failures_with_algos = ""
    failures_with_algos = ""
    while(len(comp_result_data) > 0 and n < len(comp_result_data)):
        operator = match_operator[comp_result_data[n]["algo"]]
        curr_algo = comp_result_data[n]["algo"]
        #curr_algo = next(comp_res_obj)
        while(comp_result_data[n]["algo"] == curr_algo and n < len(comp_result_data)):
            if(comp_result_data[n]["result"] == False and operator.lower() == "and"):
                operations_res = False
                tmp_failures_with_algos = curr_algo
            n = n + 1
            if(n >= len(comp_result_data)):
                break
        # concatenate algos with comma until before the last algo
        if(n < len(comp_result_data)):
            failures_with_algos = failures_with_algos + tmp_failures_with_algos + ", "
        elif(n > len(comp_result_data)):
            failures_with_algos = failures_with_algos + tmp_failures_with_algos  # concatenate algo without comma for the last algo
        tmp_failures_with_algos = ""
    # in case the algos that have failed ends with comma, remove it
    if(failures_with_algos.strip().endswith(",") == True):
        failures_with_algos = failures_with_algos.strip()[:-1]
    
    if(op_net_result == False):
        return operations_res, failures_with_algos #, passed_imgs_with_or_operator_algo
    else:
        return True, failures_with_algos  #, passed_imgs_with_or_operator_algo # in case the net result is already True(i.e.Pass), return the passed images list with "OR" operator algo with net result being True


## updates on 26-Oct-2020 02:30 AM, 03:30 PM, 17-Nov-2020 02:00 AM #
def WriteJsonlog(curr_algo_log, comp_result_data, _is_img_path_update_needed=True):
    n=0
    #comp_result_data.extend(tmp_match_data)
    while(len(comp_result_data) > 0 and n < len(comp_result_data) and _is_img_path_update_needed == True):
        if(n > 0):
            comp_result_data[n]["base_img_path"] = ""
            comp_result_data[n]["runtime_img_path"] = ""
        n = n + 1
    if(os.path.exists(curr_algo_log)):
        os.unlink(curr_algo_log)    
    img_comp_utils.writeJson(curr_algo_log,comp_result_data,True)


# updates on 05-Nov-2020 03:45 PM #
def write_BF_baseline_Json(BF_baseline, baseline_data_obj, filemode="w", del_current_file=True):
    #comp_result_data.extend(tmp_match_data)
    img_comp_utils.writeJson(BF_baseline,baseline_data_obj,True, filemode, del_current_file)


# created on 17-Nov-2020 02:25 AM #
def write_consolidated_result(cons_result_file, nonserializable_result_obj, stringformat=True, filemode = "w"):
    cons_res_string = json.dumps([o.dump() for o in nonserializable_result_obj])
    serializable_result_obj = json.loads(cons_res_string) 
    img_comp_utils.writeJson(cons_result_file, serializable_result_obj, stringformat, filemode)


# created on 26-Oct-2020 08:15 PM #
# updates on 27-Oct-2020 02:50 AM #
def del_prev_artifacts(workspace_path,should_purge_oldlog):
    try:
        prev_logpath = os.path.join(workspace_path,"image_ops/logs")
        prev_diffpath = os.path.join(workspace_path,"image_ops/diffs")
        if(should_purge_oldlog.lower() == "true" and os.path.exists(prev_logpath) is True):
            shutil.rmtree(prev_logpath)
        if(should_purge_oldlog.lower() == "true" and os.path.exists(prev_diffpath) is True):
            shutil.rmtree(prev_diffpath)
    except:
        print("error :: deleting prev. artifacts....Not OK")


# created on 26-Oct-2020 07:00 PM #
def createartifactspaths(dt):
    import datetime
    workspace_path = str(dt["compare_args"]["workspace_path"])
    should_purge_oldlog = str(dt["compare_args"]["purge_old_artifacts"])
    x = datetime.datetime.now()
    dt_part = "{0}{1}{2}_{3}{4}{5}".format(x.day,x.month,x.year,x.hour,x.minute,x.second)
    del_prev_artifacts(workspace_path, should_purge_oldlog)
    curr_logpath = os.path.join(workspace_path,"image_ops/logs",dt_part)
    curr_diffpath = os.path.join(workspace_path,"image_ops/diffs",dt_part)
    os.makedirs(curr_logpath)
    os.makedirs(curr_diffpath)
    return curr_logpath, curr_diffpath

# created on 31-Oct-2020 11:05 PM #
# updates on 04-Nov-2020 02:35 AM #
def create_BF_algo_baseline_folder(baseline_path):
    if not os.path.exists(baseline_path):
        print("path for BF algo baseline created dynamically:"+baseline_path)
        logging.info("path for BF algo baseline created dynamically:"+baseline_path)
        os.makedirs(baseline_path)


# created on 03-Nov-2020 11:27 PM
# updates on 11-Nov-2020 12:30 AM
def check_BF_baseline_exists(dt):
    BF_algo_baseline = str(dt["compare_args"]["BRISK_FLANN_parametric"]["BRISK_FLANN_parametric_baseline"])
    print("check_BF_baseline_exists",BF_algo_baseline)
    if not os.path.exists(BF_algo_baseline):
        print(":"+ str(dt["compare_args"]["BRISK_FLANN_parametric"]["BRISK_FLANN_parametric_baseline"]))
        logging.info("unable to find BF_parametric baseline:"+ str(dt["compare_args"]["BRISK_FLANN_parametric"]["BRISK_FLANN_parametric_baseline"]))
        return False
    else:
        return True


# created on 03-Nov-2020 11:50 PM
def check_BF_algo_ops_prereq(algo_idx, dt):
    if(get_algo_name_list(dt)[algo_idx] != "BRISK-FLANN"):
        return 2    # different active algo
    if(get_algo_name_list(dt)[algo_idx] == "BRISK-FLANN"):
        tmp = check_BF_baseline_exists(dt)
        if not tmp:
            print("BF algo parametric baseline will be generated...")
            logging.info("BF algo parametric baseline will be generated...")
            return 0 # the BF baseline does not exist
        else: # the Bunable to find BF_parametric baselineF baseline exists
            return 1

# created on 09-Nov-2020 12:40 AM #
# updates on 09-Nov-2020 01:05 AM, 11-Nov-2020 12:30 AM #
########create, read and process BF algo baseline json - add new image entry to the baseline json###############
def add_new_BF_baseline_entry():
    if(len(BF_base_data_model.newimgs_baseline_buffer) <= 0 or _is_BF_active_algo == False):
        return
    i=0
    BF_baseline_file = str(dt["compare_args"]["BRISK_FLANN_parametric"]["BRISK_FLANN_parametric_baseline"])
    #if(len(dt["compare_args"]["newimgs_baselinebuffer"]) > 0): /working fine - however, not needed
    BF_res_obj_json = img_comp_utils.readJson_plain(BF_baseline_file)
           
    # the dt data object for the BF is populated while BF algo operation #
    BF_resJson = json.dumps([o.dump() for o in dt["compare_args"]["newimgs_baselinebuffer"]])
    BF_new_entry_obj_json = json.loads(BF_resJson)
    while(i < len(BF_new_entry_obj_json)):
        BF_res_obj_json.append(BF_new_entry_obj_json[i])
        i = i + 1
    write_BF_baseline_Json(BF_baseline_file, BF_res_obj_json, "w", False)
    print("new image entries are auto-added to BRISK-FLANN baseline json")
    logging.info("new image entries are auto-added to BRISK-FLANN baseline json")

# created on 09-Nov-2020 12:40 AM #
# updates on 11-Nov-2020 12:30 AM #
def generate_BF_baseline_from_dataobject():
    if(_should_generate_BF_baseline_file == True):
        print("active algo : BRISK-FLANN")
        print(dt["compare_args"]["BF_algo_baseline"])
        BF_baseline_file = str(dt["compare_args"]["BRISK_FLANN_parametric"]["BRISK_FLANN_parametric_baseline"])
        # the dt data object for the BF is populated while BF algo operation
        BF_resJson = json.dumps([o.dump() for o in dt["compare_args"]["BF_algo_baseline"]])
        BF_res_obj_json = json.loads(BF_resJson) 
        create_BF_algo_baseline_folder(str(os.path.dirname(BF_baseline_file)))     
        write_BF_baseline_Json(BF_baseline_file, BF_res_obj_json)

# created on 16-Nov-2020 03:25 PM #
def check_resultfiles_presence(result_path):
    pixel_comparison_op = bool(os.path.exists(os.path.join(result_path, ssi_result)))
    ssi_algo_op = bool(os.path.exists(os.path.join(result_path, ssi_result)))
    BF_algo_op = bool(os.path.exists(os.path.join(result_path, BF_result)))
    p_hash_algo_op = bool(os.path.exists(os.path.join(result_path, p_hash_result)))
    d_hash_algo_op = bool(os.path.exists(os.path.join(result_path, d_hash_result)))
    haar_cascade_algo_op = bool(os.path.exists(os.path.join(result_path, ssi_result)))
    return pixel_comparison_op, p_hash_algo_op, BF_algo_op, ssi_algo_op, d_hash_algo_op, haar_cascade_algo_op

# created on 16-Nov-2020 04:30 PM #
def read_result_files(result_file_list):
    res_obj_dict = {}
    res_dict_elem_cnt = {}
    for result_file in result_file_list:
        result_obj = img_comp_utils.readJson_plain(result_file)
        result_str = json.dumps(result_obj)
        result_str = str(result_str)
        result_obj_json = json.loads(result_str)

        if(os.path.basename(result_file)==ssi_result):
            res_obj_dict["ssi"] = result_obj_json
            res_dict_elem_cnt["ssi"] = len(result_obj_json)
        elif(os.path.basename(result_file)==BF_result):
            res_obj_dict["BF"] = result_obj_json
            res_dict_elem_cnt["BF"] = len(result_obj_json)
        elif(os.path.basename(result_file)==p_hash_result):
            res_obj_dict["phash"] = result_obj_json
            res_dict_elem_cnt["phash"] = len(result_obj_json)
        elif(os.path.basename(result_file)==d_hash_result):
            res_obj_dict["dhash"] = result_obj_json
            res_dict_elem_cnt["dhash"] = len(result_obj_json)
        elif(os.path.basename(result_file)==pixel_result):
            res_obj_dict["pixelcomp"] = result_obj_json
            res_dict_elem_cnt["pixelcomp"] = len(result_obj_json)
        elif(os.path.basename(result_file)==pixel_result):
            res_obj_dict["haar"] = result_obj_json
            res_dict_elem_cnt["haar"] = len(result_obj_json)

    return res_obj_dict, res_dict_elem_cnt

# created on 16-Nov-2020 08:15 PM
def get_phash_result_metrics(p_hash_algo_op, res_obj_dict, k):
    phash_img = ""
    phash_score = ""
    phash_msg = ""
    if(p_hash_algo_op == False):
        return phash_img, phash_score, phash_msg
    res_obj = res_obj_dict["phash"]
    phash_img = res_obj[k]["image"]
    phash_score = res_obj[k]["original_score"]
    phash_msg = res_obj[k]["msg"]
    return phash_img, phash_score, phash_msg

# created on 16-Nov-2020 08:30 PM
def get_dhash_result_metrics(d_hash_algo_op, res_obj_dict, k):
    dhash_img = ""
    dhash_score = ""
    dhash_msg = ""
    if(d_hash_algo_op == False):
        return dhash_img, dhash_score, dhash_msg
    res_obj = res_obj_dict["dhash"]
    dhash_img = res_obj[k]["image"]
    dhash_score = res_obj[k]["original_score"]
    dhash_msg = res_obj[k]["msg"]
    return dhash_img, dhash_score, dhash_msg

# created on 16-Nov-2020 08:30 PM
def get_ssi_result_metrics(ssi_algo_op, res_obj_dict, k):
    ssi_img = ""
    ssi_score = ""
    if(ssi_algo_op == False):
        return ssi_img, ssi_score
    res_obj = res_obj_dict["ssi"]
    ssi_img = res_obj[k]["image"]
    ssi_score = res_obj[k]["original_score"]
    return ssi_img, ssi_score

# created on 16-Nov-2020 08:35 PM
def get_BF_result_metrics(BF_algo_op, res_obj_dict, k):
    BF_img = ""
    BF_expscore = ""
    BF_originalscore = ""
    if(BF_algo_op == False):
        return BF_img, BF_expscore, BF_originalscore
    res_obj = res_obj_dict["BF"]
    BF_img = res_obj[k]["image"]
    BF_expscore = res_obj[k]["expscore"]
    BF_originalscore = res_obj[k]["original_score"]
    return BF_img, BF_expscore, BF_originalscore

# created on 16-Nov-2020 11:55 PM
def validate_img_name_is_same(phash_img,BF_img,ssi_img,dhash_img):
    tmp = []
    result = True
    tmp_img = ""
    tmp.append(phash_img)
    tmp.append(BF_img)
    tmp.append(ssi_img)
    tmp.append(dhash_img)
    for i in tmp:
        if(i == ""):
            continue
        if(tmp_img == ""):
            tmp_img = i
        if(tmp_img != i):
            result = False
    return tmp_img, result


# created on 16-Nov-2020 12:25 PM 
# updates on 16-Nov-2020 from 01:30 PM till 11:30 PM, 17-Nov-2020 01:40 PM #
def consolidate_result(result_file_list):
    k=0
    img = ""
    consolidation_result = True
    if len(result_file_list) <= 0:
        logging.info("Not a single result file found. Consolidated result won't be generated")
        print("Not a single result file found. Consolidated result won't be generated")
        return cons_results_data_model.consolidated_result_list, False
    
    pixel_comp_op, p_hash_algo_op,  BF_algo_op, ssi_algo_op, d_hash_algo_op, haar_cascade_algo_op = check_resultfiles_presence(os.path.dirname(result_file_list[0]))

    res_obj_dict, res_dict_elem_cnt = read_result_files(result_file_list)
    #res_max_cnt = max(res_dict_elem_cnt, key= lambda x: res_dict_elem_cnt[x])   # this prints the key with max value
    res_dict_img_cnt_values = res_dict_elem_cnt.values()
    res_max_cnt = max(res_dict_img_cnt_values)
    while(k < res_max_cnt):
        phash_img, phash_score, phash_msg = get_phash_result_metrics(p_hash_algo_op, res_obj_dict, k)
        BF_img, BF_exp_score, BF_orig_score = get_BF_result_metrics(BF_algo_op, res_obj_dict, k)
        ssi_img, ssi_score = get_ssi_result_metrics(ssi_algo_op, res_obj_dict, k)
        dhash_img, dhash_score, dhash_msg = get_dhash_result_metrics(d_hash_algo_op, res_obj_dict, k)

        img, imgname_check = validate_img_name_is_same(phash_img, BF_img, ssi_img, dhash_img)
    
        if imgname_check == False:
            consolidation_result = False
            logging.error("img name :"+phash_img+" ==> image names shouldn't differ among the algos. Idx:"+str(k))
            print("img name :"+phash_img+" ==> image names shouldn't differ among the algos. Idx:"+str(k))
            phash_score = ""
            phash_msg = "image names shouldn't differ among the algos. Idx:"+str(k)
            BF_exp_score = ""
            BF_orig_score = ""
            ssi_score = ""
            dhash_score = ""
            dhash_msg = "image names shouldn't differ among the algos. Idx:"+str(k)
        
        cons_results_data_model.consolidated_result_list.append(cons_results_data_model(img, phash_score, phash_msg, BF_exp_score, BF_orig_score, ssi_score, dhash_score, dhash_msg))
        k = k + 1
    if(consolidation_result == False):
        logging.error("result consolidation is unsuccessful")
    return cons_results_data_model.consolidated_result_list, consolidation_result        


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("-j", "--json", required=True, help="json arg file name")
    args = vars(ap.parse_args())
    return args


#********+********+********+********__main__********+********+********+********
start_time = time.time()
args = parse_args()
#pyautogui.screenshot(args["file"])
#im = pyautogui.screenshot(region=(20,50,500,550))
dt = img_comp_utils.readJson(args["json"], "compare_args")
#base = "D:/Automation/DW/TROS_SDV/src/test-inputs/images"
#base = "D:/Automation/DW/SDV/images_cvLibExp/Set1/widgets/widgets"
base = str(dt["compare_args"]["baseline_path"])
runtime = str(dt["compare_args"]["runtime_path"])

#runtime = "D:/Automation/DW/SDV/images_cvLibExp/Set2/widgets/widgets"
##runtime = "D:/Automation/DW\SDV/images_cvLibExp/Set3/images/images/sitecom-viewer/widgets/Dusk"
workspace_path =  str(dt["compare_args"]["workspace_path"])
logpath, diffpath = createartifactspaths(dt)
errlogfile = str(os.path.join(logpath,'benchmark-util.log')).replace('\\','/')
logging.basicConfig(filename=errlogfile, filemode='a', format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger().setLevel(logging.INFO)
workspace_base, workspace_runtime = createFlatFolders(workspace_path)
dt["compare_args"]["diff_img_path"] = diffpath.replace("\\","/")
logging.info("starting the image benchmarking...")
imgList=[]
imgList = copyToFlatTargetPath(base, workspace_base, dt, imgList, True)
baseline_imgList = imgList
print("baseline imgs count :",len(imgList))
print("image list:",imgList)
logging.info("baseline imgs count :",len(imgList))
print("********+********+********+********+********+********+********+********")
imgList = copyToFlatTargetPath(runtime, workspace_runtime, dt, imgList, False)
print("runtime imgs count  :",len(imgList))
print("image list:",imgList)
logging.info("runtime imgs count  :",len(imgList))
print("********+********+********+********+********+********+********+********")
if(len(baseline_imgList) != len(imgList)):
    print(("mismatch between baseline and runtime count : {0} != {1}".format(len(baseline_imgList),len(imgList))))
    logging.error("mismatch between baseline and runtime count : {0} != {1}".format(len(baseline_imgList),len(imgList)))
else:
    print("no issues in the image count : {0} equals to {1}".format(len(baseline_imgList),len(imgList)))
    logging.info("no issues in the baseline and runtime count : {0} == {1}".format(len(baseline_imgList),len(imgList)))



operation_net_result = False
algo_name_list = []
comp_result_data = []
tmp_match_data = []
match_ops_details=[]
res_obj_json = []
result_file_list = []
algo_cnt = len(get_algo_mapper())
print("available algorithms count:",algo_cnt)
algo_idx = 0
b_tmp_path = ""
r_tmp_path = ""
ssi_result = "ssi_benchmark.json"
BF_result = "brisk-flann_benchmark.json"
p_hash_result = "perceptual_hashing_benchmark.json"
d_hash_result = "diff_hashing_benchmark.json"
pixel_result = "pixel_benchmarking.json"
haar_cascade_result = "haar_cascade_benchmarking.json"
preprocessing_needed = True
_is_BF_active_algo = False
_should_generate_BF_baseline_file = False
algo_name_list_runnables = get_algo_name_list(dt)
runnable_algos_dict = get_algo_runnable_details(dt)
runnable_state_labels = {}
runnable_state_labels["0"] = "off"
runnable_state_labels["1"] = "on"
dt["compare_args"]["missing_imgs"] = {}
preprocessing_finished = False
dt["compare_args"]["workspace_base"] = workspace_base
dt["compare_args"]["workspace_runtime"] = workspace_runtime
#intermediate_output = str(dt["compare_args"]["intermediate_output"])
while(algo_idx < algo_cnt):
    
    # check whether the BF parametric baseline file exists. Accordingly mark the flags
    print("******+******+******+******+*****")
    algo_runnable_state = runnable_algos_dict[algo_name_list_runnables[algo_idx]]
    print("algo           : "+algo_name_list_runnables[algo_idx])
    print("runnable state :",runnable_state_labels[algo_runnable_state])
    print("******+******+******+******+*****")
    logging.info("algo           : "+algo_name_list_runnables[algo_idx])
    logging.info("runnable state :"+runnable_state_labels[algo_runnable_state])
    if(algo_runnable_state=="0"):
        algo_idx = algo_idx + 1
        continue

    BF_algo_prereq = check_BF_algo_ops_prereq(algo_idx, dt)
    if BF_algo_prereq == 0 or BF_algo_prereq == 1:
        _is_BF_active_algo = True
    else:
        _is_BF_active_algo = False
        BF_algo_prereq = 999
        BF_base_data_model.newimgs_baseline_buffer = []
        BF_basetobase_comp_data_model.basetobase_kp_update_list = []
    if BF_algo_prereq == 0 and _is_BF_active_algo == True:
        _should_generate_BF_baseline_file = True
    else:
        _should_generate_BF_baseline_file = False
    # end of parametric baseline check

    print("******+*********+*********")
    if(preprocessing_finished == True):
        preprocessing_needed = False
    
    print("preprocessing finished ?",preprocessing_finished)
    print("preprocessing needed   ?",preprocessing_needed)
    
    algo_name_list, tmp_match_data, dt = call_func(dt,algo_idx,tmp_match_data,preprocessing_needed)
    preprocessing_finished = True
    curr_algo_log = os.path.join(logpath, str(algo_name_list[algo_idx]).lower()+"_benchmark.json")
    result_file_list.append(curr_algo_log) # add the current result json to the result file list for building consolidated result file later
    
    match_ops_rec_cnt = len(tmp_match_data)
    print( algo_name_list[algo_idx], " : image match operational records -->",match_ops_rec_cnt)
    print("******+*********+*********")
    
    #json_file_data = img_comp_utils.readJson_plain("ssi_comp_result.json") #working fine, but without IO ops, the object creation is done below - 14-Oct-2020 03:15 AM
    #res_obj = json_file_data
    m = 0
    k = len(tmp_match_data) - 1
    resJson = json.dumps([o.dump() for o in tmp_match_data])
    res_obj_json = json.loads(resJson) 
    #comp_result_data.extend(res_obj_json)
    comp_result_data = res_obj_json
    WriteJsonlog(curr_algo_log, comp_result_data) #write the result log to the json file
    
    #check the number of passed entries to delete them and retain the failed ones for the next algo pass
    while(len(tmp_match_data) >= 0 and m < match_ops_rec_cnt):
        match_ops_details = imageops(res_obj_json[k]["image"], res_obj_json[k]["base_img_path"], res_obj_json[k]["runtime_img_path"], res_obj_json[k]["algo"],res_obj_json[k]["expscore"],res_obj_json[k]["original_score"],res_obj_json[k]["result"],res_obj_json[k]["msg"])
        
        # check if the two vars are empty, then assign the base and runtime image paths to the vars for storing only one in the comparison output json file
        if(b_tmp_path=="" or r_tmp_path==""):
            b_tmp_path = match_ops_details.base_img_path
            r_tmp_path = match_ops_details.runtime_img_path
            break
    
        k = k - 1
        m = m + 1

    add_new_BF_baseline_entry()
    generate_BF_baseline_from_dataobject()

    ########create, read and process BF algo baseline json - create the baseline json for the first time from data object###############
    if(len(BF_basetobase_comp_data_model.basetobase_kp_update_list) > 0):
        tmp_b2b_comp_list = BF_basetobase_comp_data_model.basetobase_kp_update_list
        for x in tmp_b2b_comp_list:
            img = x["image"]
            comp_res = x["comp_result"]
            latest_kp = x["basetobase_latest_kp"]
            print("**********basetobase_comp_data*********")
            print(img)
            print(comp_res)
            print(latest_kp)
        
        #print("BF_algo_base_json_object:",BF_algo_base[8]["image"])//For BF_algo_base reference. Working fine - 01-Nov-2020 01:40 AM
    #####end of read and process BF algo baseline json###############
    algo_idx = algo_idx + 1
    res_obj_json[0]["base_img_path"] = b_tmp_path
    res_obj_json[0]["runtime_img_path"] = r_tmp_path
    tmp_match_data = []
    comp_result_data = []
    imageops.image_match_outcome_list = []
    print("========================================================================================================================")

############################# result consolidation #########################
cons_result_file = os.path.join(logpath, "consolidated_result.json")
consolidated_result_list, consolidation_result = consolidate_result(result_file_list)
write_consolidated_result(cons_result_file, consolidated_result_list, True)
############################### end of result consolidation ###################
operation_net_result, failures_with_algos =  analyze_net_result(dt,comp_result_data,operation_net_result)

print("**********+*****+*****+****+*****+*****+****+*****+*****+****")
elapsed_time = time.time() - start_time
print("image operations duration : %s secs" % round(elapsed_time,2))
print("ImageBench v2.0")
print("**********+*****+*****+****+*****+*****+****+*****+*****+****")



'''
for root, dirs, files in os.walk('python/Lib/email'):
    print root, "consumes",
    print sum(getsize(join(root, name)) for name in files),
    print "bytes in", len(files), "non-directory files"
    if 'CVS' in dirs:
        print("Yes-CVS")
        #dirs.remove('CVS')  # don't visit CVS directories

'''


'''
# logging_example.py



# Create a custom logger
logger = logging.getLogger(__name__)

#logging.basicConfig(filename='app.log', filemode='w', format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Create handlers
warning_handler = logging.FileHandler('warning.log')
error_handler = logging.FileHandler('error.log')
info_handler = logging.FileHandler('info.log')
debug_handler = logging.FileHandler('debug.log')
critical_handler = logging.FileHandler('critical.log')
warning_handler.setLevel(logging.WARNING)
info_handler.setLevel(logging.INFO)
debug_handler.setLevel(logging.warning)
critical_handler.setLevel(logging.CRITICAL)
error_handler.setLevel(logging.ERROR)

# Create formatters and add it to handlers
warning_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
info_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
debug_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
critical_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
error_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

warning_handler.setFormatter(warning_format)
info_handler.setFormatter(info_format)
debug_handler.setFormatter(debug_format)
critical_handler.setFormatter(critical_format)
error_handler.setFormatter(error_format)

# Add handlers to the logger
logger.addHandler(warning_handler)
logger.addHandler(info_handler)
logger.addHandler(debug_handler)
logger.addHandler(critical_handler)
logger.addHandler(error_handler)

logger.warning('This is a warning')
logger.info('This is an info')
logger.debug('This is a debugging statement')
logger.critical('This is a critical failure')
logger.error('This is an error. Not blocking, but should be fixed')
'''



'''
# Create a custom logger
logger = logging.getLogger(__name__)

#logging.basicConfig(filename='app.log', filemode='w', format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Create handlers
debug_handler = logging.FileHandler('debug.log')
debug_handler.setLevel(logging.warning)

# Create formatters and add it to handlers
debug_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
debug_handler.setFormatter(debug_format)

# Add handlers to the logger
logger.addHandler(debug_handler)
logger.debug('This is a debugging statement')
'''


''' did not work when called from main or from other functions. Explore later - 26-Oct-2020 01:10 AM
#global imgfile_g
#global message
#global result
# logging_example.py


def log(message):
    # Create a custom logger
    logger = logging.getLogger(__name__)

    #logging.basicConfig(filename='app.log', filemode='w', format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Create handlers
    debug_handler = logging.FileHandler('debug.log')
    debug_handler.setLevel(logging.warning)

    # Create formatters and add it to handlers
    debug_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    debug_handler.setFormatter(debug_format)

    # Add handlers to the logger
    logger.addHandler(debug_handler)
    logger.debug(message)

def autolog(message):
    "Automatically log the current function details."
    import inspect, logging
    # Get the previous frame in the stack, otherwise it would
    # be this function!!!
    func = inspect.currentframe().f_back.f_code
    # Dump the message + the name of this function to the log.
    logging.warning("%s: %s in %s:%i" % (
        message, 
        func.co_name, 
        func.co_filename, 
        func.co_firstlineno
    ))
'''


'''
function to add a new entry to an existing json file - two successful methods

METHOD-1:
--------
# created on 09-Nov-2020 12:40 AM #
def add_new_BF_baseline_entry():
    ########create, read and process BF algo baseline json - add new image entry to the baseline json###############
    if(len(BF_base_data_model.newimgs_baseline_buffer) > 0 and _is_BF_active_algo == True):
        i=0
        print("new imgs are present++++++++++++++++++")
        BF_baseline_file = str(dt["compare_args"]["BRISK_FLANN_parametric"]["BRISK_FLANN_parametric_baseline"])
        if(len(dt["compare_args"]["newimgs_baselinebuffer"]) > 0):
            abc = img_comp_utils.readJson_plain(BF_baseline_file)
           
            # the dt data object for the BF is populated while BF algo operation
            BF_resJson = json.dumps([o.dump() for o in dt["compare_args"]["newimgs_baselinebuffer"]])
            BF_new_entry_obj_json = json.loads(BF_resJson)
   
            while(i < len(BF_new_entry_obj_json)):
                abc.append(BF_new_entry_obj_json[i])
                #BF_res_obj_json.append(BF_new_entry_obj_json[i])
                i = i + 1
   
            write_BF_baseline_Json(BF_baseline_file, abc, "w", False)



METHOD-2:
---------

# created on 09-Nov-2020 12:40 AM #
def add_new_BF_baseline_entry():
    ########create, read and process BF algo baseline json - add new image entry to the baseline json###############
    if(len(BF_base_data_model.newimgs_baseline_buffer) > 0 and _is_BF_active_algo == True):
        i=0
        print("new imgs are present++++++++++++++++++")
        BF_baseline_file = str(dt["compare_args"]["BRISK_FLANN_parametric"]["BRISK_FLANN_parametric_baseline"])
        if(len(dt["compare_args"]["newimgs_baselinebuffer"]) > 0):
            # the dt data object for the BF is populated while BF algo operation
            BF_resJson = json.dumps([o.dump() for o in dt["compare_args"]["newimgs_baselinebuffer"]])
            BF_new_entry_obj_json = json.loads(BF_resJson)

            BF_algo_base = img_comp_utils.readJson_plain(BF_baseline_file)
            BF_algo_base = json.dumps(BF_algo_base)
            BF_algo_base = str(BF_algo_base)
            BF_res_obj_json = json.loads(BF_algo_base)
            while(i < len(BF_new_entry_obj_json)):
                BF_res_obj_json.append(BF_new_entry_obj_json[i])
                i = i + 1
            
            #print("new images are there",BF_res_obj_json)
            write_BF_baseline_Json(BF_baseline_file, BF_res_obj_json, "w", False)

'''