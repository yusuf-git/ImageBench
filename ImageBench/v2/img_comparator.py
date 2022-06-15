#/############################################################
# Author : Yusuf
# Date & Time : 19-Apr-2020 12:00 AM To 06:30 AM, 21-Apr-2020 03:10 AM, 01-May-2020 05:00 AM to 8:00 AM, 16-Aug-2020 07:00 PM, 28-Sep-2020 04:40 PM, 29-Sep-2020 02:45 AM, 30-Sep-2020 02:45 AM, 03-Oct-2020 01:30 AM, 06-Oct-2020 08:15 PM, 11-Oct-2020 02:30 PM to 11:50 PM, 12-Oct-2020 02:00 AM, 13-Oct-2020 04:00 AM, 14-Oct-2020 04:10 AM, 15-Oct-2020 03:20 AM, 16-Oct-2020 02:45 AM, 17-Oct-2020 03:10 AM, 18-Oct-2020 03:45 AM
###############################################################
# create a thumbnail of an image
from genericpath import exists
from os import mkdir
import logging
import time
import sys
from typing import Any
from PIL import Image, ImageEnhance
from skimage.metrics import structural_similarity as ssim
#import matplotlib.pyplot as plt
import numpy as np
import cv2
import imutils
import imagehash
#from skimage.measure import compare_ssim
import os
import pyautogui as pygui
import img_comp_utils
from algos_namelist import comp_algos
from imageops_data_model import imageops
from BF_baseline_data_model import BF_base_data_model
from matplotlib import pyplot as plt

tasks = {}
task = lambda f: tasks.setdefault(f.__name__, f)
#logging.basicConfig()
#logging.getLogger().setLevel(logging.INFO)

# Updates on 02-Oct-2020 09:45 PM, 03-Oct-2020 01:05 AM, 08:00 PM, 18-Oct-2020 03:45 AM, 27-Oct-2020 02:25 AM, 09-Nov-2020 03:30 AM, 12-Nov-2020 02:15 AM  #
#@task
def SSI_compare(dt, result_list, preprocessing_needed=True):
    start_time = time.time()
    logging.info("##############################################################")
    logging.info("active algo : structural similarity index")
    logging.info("##############################################################")
    result_dict = {}
    missing_imgs_added_to_result = False
    diff_path = str(dt["compare_args"]["diff_img_path"])
    debugging = str(dt["compare_args"]["intermediate_output"])
    diff_path = get_diff_path(dt, result_list, "ssi")
    ssi_err_ops = []
    diff_file_cnt = 0
    result_msg = ""
    print('')
    print('')
    print('')
    print("##############################################################")
    print("active algo : structural similarity index")
    print("##############################################################")
    # get expected score for the current algo #
    algo_exp_score = float(img_comp_utils.get_algo_expected_score(comp_algos.ssi,dt))
    img_file1, img_file2, missing_imgs_  = img_comp_utils.preprocess_images(dt, result_list, preprocessing_needed)
    dt["compare_args"]["missing_imgs"] = missing_imgs_
    #dt["compare_args"]["abcde"] = "12345 - test dict entry..."
    #idx, base_img_cnt = get_idx_baseimgcnt(realtime, img_file1, img_file2, result_list)
    #base_path = os.path.dirname(img_file1)
    base_path = str(dt["compare_args"]["workspace_base"])
    runtime_path = str(dt["compare_args"]["workspace_runtime"])
    #runtime_path = os.path.dirname(img_file2)
    print("in SSI - check resultlist:",result_list)
    base_cnt, runtime_cnt = img_comp_utils.get_img_count(base_path,runtime_path,result_list)
    print("baseline imgs path: {0} --> img count:{1}".format(base_path, base_cnt))
    print("runtime imgs path: {0} --> img count:{1}".format(runtime_path, runtime_cnt))
    logging.info("##############################################################")
    logging.info("baseline imgs path: {0} --> img count:{1}".format(base_path, base_cnt))
    logging.info("runtime imgs path: {0} --> img count:{1}".format(runtime_path, runtime_cnt))
    logging.info("##############################################################")
    print("##############################################################")
    #while(idx <= base_img_cnt):
    for filename in os.listdir(base_path):
        img_file1 = os.path.join(base_path, filename)
        img_file2 = os.path.join(runtime_path, filename)
        print("SSI --> current image ::"+img_file1)
        b_tmp_fname = "B_"+os.path.basename(img_file1)
        r_tmp_fname = "R_"+os.path.basename(img_file2)
        #print("Missssssssssssssssssssssssssssssssssssssssssssssss:",missing_imgs_)
        if(b_tmp_fname in missing_imgs_):
            imageops.image_match_outcome_list.append(imageops(b_tmp_fname, os.path.dirname(img_file1), os.path.dirname(img_file2), "SSI", algo_exp_score, "-999.99", False, "missing baseline"))
            missing_imgs_added_to_result = True
            logging.error("missing baseline img:"+b_tmp_fname)
            continue
        if(r_tmp_fname in missing_imgs_):
            imageops.image_match_outcome_list.append(imageops(r_tmp_fname, os.path.dirname(img_file1), os.path.dirname(img_file2), "SSI", algo_exp_score, "-999.99", False, "missing runtime"))
            missing_imgs_added_to_result = True
            logging.error("missing runtime img:"+r_tmp_fname)
            continue
        print("ssi_diff_path:",diff_path)
        diff_file = os.path.join(diff_path,"diff_"+os.path.basename(img_file1))
        
        try:
            img1 = cv2.imread(img_file1)
            img2 = cv2.imread(img_file2)
            grayA = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            grayB = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
            #grayA = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
            #grayB = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)
            #cv2.imshow("base", grayA)
            #cv2.imshow("run", grayB) 
            (score, diff) = ssim(grayA, grayB, multichannel = True, full=True)
            diff = (diff * 250).astype("uint8")
            thresh = cv2.threshold(diff, 0, 255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
            cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)
            for c in cnts:
	            (x, y, w, h) = cv2.boundingRect(c)
	            #cv2.rectangle(img1, (x, y), (x + w, y + h), (0, 0, 155), 2)
	            cv2.rectangle(img2, (x, y), (x + w, y + h), (0, 0, 155), 2)
            if(score < 1.0):
                print("ssi-->diff_file",diff_file)
                cv2.imwrite(diff_file,img2)
                diff_file_cnt = diff_file_cnt + 1
            #cv2.rectangle(diff, (x, y), (x + w, y + h), (0, 0, 155), 2)
            #fileName = str(Path(str(baseline)).resolve())//throws OS error
            #print("diff_"+desc)
            #cv2.imwrite(diff_file,img2)
        except:
            imageops.image_match_outcome_list.append(imageops(os.path.basename(img_file1), str(os.path.dirname(img_file1)).replace("\\","/"), str(os.path.dirname(img_file2)).replace("\\","/"), "SSI", algo_exp_score, "0.0", False, str(sys.exc_info()[0])))
            ssi_err_ops.append("error :: SSI --> Image :" + str(os.path.basename(img_file1)) + ", message : "+ str(sys.exc_info()[0])) 
            print("error :: SSI --> Image :" + str(os.path.basename(img_file1)) + " "+ str(sys.exc_info()[0]))
            img_comp_utils.writeToFile("ssi_err_ops.txt",str(ssi_err_ops))
            logging.error("error :: SSI --> Image :"+ str(os.path.basename(img_file1)) + ", message : "+ str(sys.exc_info()[0]))
            continue

        if(debugging=="true"):
            cv2.imshow("Original", img1)
            cv2.imshow("Runtime", img2)
            cv2.imshow("Diff", diff)
            #cv2.imshow("Thresh", thresh)
            cv2.waitKey(0)
        result_msg = str(score)
        score = round(score,2)
        print("Match similarity (1.0 = 100% match): {}".format(score))
        #algo_exp_score = img_comp_utils.get_algo_expected_score(comp_algos.perceptual_hashing,dt)
        algo_perf_result = img_comp_utils.determine_match_outcome(comp_algos.ssi, score, ">=", dt, "float")
        #print("SSI_compare:algo_perf_result-1:",algo_perf_result)
        result_dict = {os.path.basename(img_file1):{score:algo_perf_result}}
        print("Match result :",result_dict)
        print("##############################################################")
        if(algo_perf_result==False):
            result_msg = result_msg + " - failure"
        else:
            result_msg = result_msg + " - pass"
        imageops.image_match_outcome_list.append(imageops(os.path.basename(img_file1), str(os.path.dirname(img_file1)).replace("\\","/"), str(os.path.dirname(img_file2)).replace("\\","/"), "SSI", algo_exp_score, str(score), algo_perf_result, result_msg))
        result_msg = ""
    if(missing_imgs_added_to_result == False and missing_imgs_ is not None):
        for key in missing_imgs_:
            if key not in missing_imgs_.values():
                imageops.image_match_outcome_list.append(imageops(key, str(os.path.dirname(img_file1)).replace("\\","/"), str(os.path.dirname(img_file2)).replace("\\","/"), "SSI", algo_exp_score, "-999.99", False, "missing image"))
                logging.error("missing img:"+img_file1)
    if(debugging=="true"):        
        print('')
        print('')
        print('')
        print("******************cumulative match result --> algo :: SSI*****************************")
        print('')
        print("Is it correct:?",imageops.image_match_outcome_list)
            #res = [ sub['original_score'] for sub in imageops.image_match_outcome_list] 
        #filt_keys = ['image']
        #res = [imageops.image_match_outcome_list[key] for key in filt_keys] 
        #print("resssssssssssssssssssssssssssssssssssss:",str(res))
        print("****************************************************************************************")
        #print("Exp-result_dict::Final comp result::p-hashing: ",result_dict)
    logging.info("##############################################################")
    elapsed_time = round((time.time() - start_time)/60,2)
    if(diff_file_cnt > 0):
        logging.info("total diff files created for mismatches :{0}".format(diff_file_cnt))
    else:
        logging.info("no mismatches --> no diff files created  :{0}".format(diff_file_cnt))
    logging.info("finished the SSI algo operation in "+ str(elapsed_time) +" minutes...OK")
    logging.info("##############################################################")
    return imageops.image_match_outcome_list, dt

# updates on 28-Oct-2020 01:10 to 02:10 AM, 09-Nov-2020 03:30 AM, 10-Nov-2020 02:30 AM, 04:50 PM, 12-Nov-2020 02:15 AM #
#@task
def perceptual_hash_match(dt, result_list, preprocessing_needed=True):
    start_time = time.time()
    logging.info("##############################################################")
    logging.info("active algo : perceptual hashing")
    logging.info("##############################################################")
    result_dict = {}
    missing_imgs_added_to_result = False
    diff_path = str(dt["compare_args"]["diff_img_path"])
    debugging = str(dt["compare_args"]["intermediate_output"])
    hashsize = str(dt["compare_args"]["p_hash_parametric"]["hash_size"])
    diff_path = get_diff_path(dt, result_list, "p_hash")
    phash_err_ops = []
    diff_file_cnt = 0
    result_msg = ""
    print('')
    print('')
    print('')
    print("##############################################################")
    print("active algo : perceptual hashing")
    print("##############################################################")
    # get expected score for the current algo #
    algo_exp_score = float(img_comp_utils.get_algo_expected_score(comp_algos.perceptual_hashing,dt))
    img_file1, img_file2, missing_imgs_  = img_comp_utils.preprocess_images(dt, result_list, preprocessing_needed)
    dt["compare_args"]["missing_imgs"] = missing_imgs_
    dt["compare_args"]["abcde"] = "12345 - test dict entry..."
    #idx, base_img_cnt = get_idx_baseimgcnt(realtime, img_file1, img_file2, result_list)
    #base_path = os.path.dirname(img_file1)
    base_path = str(dt["compare_args"]["workspace_base"])
    runtime_path = str(dt["compare_args"]["workspace_runtime"])
    #runtime_path = os.path.dirname(img_file2)
    base_cnt, runtime_cnt = img_comp_utils.get_img_count(base_path,runtime_path,result_list)
    print("baseline imgs path: {0} --> img count:{1}".format(base_path, base_cnt))
    print("runtime imgs path: {0} --> img count:{1}".format(runtime_path, runtime_cnt))
    logging.info("##############################################################")
    logging.info("baseline imgs path: {0} --> img count:{1}".format(base_path, base_cnt))
    logging.info("runtime imgs path: {0} --> img count:{1}".format(runtime_path, runtime_cnt))
    logging.info("##############################################################")
    print("##############################################################")
    #while(idx <= base_img_cnt):
    for filename in os.listdir(base_path):
        img_file1 = os.path.join(base_path, filename)
        img_file2 = os.path.join(runtime_path, filename)
        print("p-hash --> current image ::"+img_file1)
        b_tmp_fname = "B_"+os.path.basename(img_file1)
        r_tmp_fname = "R_"+os.path.basename(img_file2)
        if(b_tmp_fname in missing_imgs_):
            imageops.image_match_outcome_list.append(imageops(b_tmp_fname, os.path.dirname(img_file1), os.path.dirname(img_file2), "p-hash", algo_exp_score, "-999.99", False, "missing baseline"))
            missing_imgs_added_to_result = True
            logging.error("missing baseline img:"+b_tmp_fname)
            continue
        if(r_tmp_fname in missing_imgs_):
            imageops.image_match_outcome_list.append(imageops(r_tmp_fname, os.path.dirname(img_file1), os.path.dirname(img_file2), "p-hash", algo_exp_score, "-999.99", False, "missing runtime"))
            missing_imgs_added_to_result = True
            logging.error("missing runtime img:"+r_tmp_fname)
            continue

        diff_file = os.path.join(diff_path,"diff_"+os.path.basename(img_file1))
        try:
            if(len(hashsize) > 0):
                baseline_hash = imagehash.phash(Image.open(img_file1),int(hashsize))
                #print('Original Picture: ' + str(baselineHash))
                actual_hash = imagehash.phash(Image.open(img_file2),int(hashsize))
            else:
                baseline_hash = imagehash.phash(Image.open(img_file1))
                #print('Original Picture: ' + str(baselineHash))
                actual_hash = imagehash.phash(Image.open(img_file2))
            #print('Actual Picture: ' + str(actualHash))
            distance = baseline_hash - actual_hash
        except:
            imageops.image_match_outcome_list.append(imageops(os.path.basename(img_file1), str(os.path.dirname(img_file1)).replace("\\","/"), str(os.path.dirname(img_file2)).replace("\\","/"), "p-hash", algo_exp_score, "0.0", False, str(sys.exc_info()[0])))
            phash_err_ops.append("error :: p-hash --> Image :" + str(os.path.basename(img_file1)) + ", message : "+ str(sys.exc_info()[0])) 
            print("error :: p-hash --> Image :" + str(os.path.basename(img_file1)) + " "+ str(sys.exc_info()[0]))
            #img_comp_utils.writeToFile("ssi_err_ops.txt",str(ssi_err_ops))
            logging.error("error :: p-hash --> Image :"+ str(os.path.basename(img_file1)) + ", message : "+ str(sys.exc_info()[0]))
            continue
        
        if(baseline_hash == actual_hash):
            print("P-hashing ::",os.path.basename(img_file1),": baseline == runtime. Score =",distance)
        else:
            print("P-hashing ::",os.path.basename(img_file1),": baseline != runtime. Score =",distance)
        
        print("p-hash match distance (0 = 100% match): {}".format(distance))
        algo_perf_result = img_comp_utils.determine_match_outcome(comp_algos.perceptual_hashing, distance, "<=", dt)
        result_dict = {os.path.basename(img_file1):{distance:algo_perf_result}}
        print("Match result :",result_dict)
        print("##############################################################")
        if(algo_perf_result==False):
            result_msg = "0"
        else:
            result_msg = "1"
        if(len(hashsize) > 0):
            result_msg = result_msg + " [hs:"+hashsize+"]"
        else:
            result_msg = result_msg + " [hs:def]"
        imageops.image_match_outcome_list.append(imageops(os.path.basename(img_file1), str(os.path.dirname(img_file1)).replace("\\","/"), str(os.path.dirname(img_file2)).replace("\\","/"), "p-hash", algo_exp_score, str(distance), algo_perf_result, result_msg))
        #img_file1, img_file2 = img_comp_utils.getFileName(img_file1, img_file2, idx, result_list)
        
    if(missing_imgs_added_to_result == False and missing_imgs_ is not None):
        for key in missing_imgs_:
            if key not in missing_imgs_.values(): 
                imageops.image_match_outcome_list.append(imageops(key, str(os.path.dirname(img_file1)).replace("\\","/"), str(os.path.dirname(img_file2)).replace("\\","/"), "p_hash", algo_exp_score, "-999.99", False, "missing image"))
                logging.error("missing img:"+img_file1)
    

    if(debugging=="true"):        
        print('')
        print('')
        print('')
        print("******************cumulative match result --> algo :: perceptual-hashing***********************")
        print('')
        print(imageops.image_match_outcome_list)
        print("****************************************************************************************")
    logging.info("##############################################################")
    elapsed_time = round((time.time() - start_time)/60,2)
    logging.info("finished the p-hashing algo operation in "+ str(elapsed_time) +" minutes...OK")
    logging.info("##############################################################")
    return imageops.image_match_outcome_list, dt


# updates on 28-Oct-2020 02:10 AM, 09-Nov-2020 03:25 AM, 12-Nov-2020 02:15 AM #
#@task
def diff_hash_match(dt, result_list, preprocessing_needed=True):
    start_time = time.time()
    logging.info("##############################################################")
    logging.info("active algo : diff hashing")
    logging.info("##############################################################")
    result_dict = {}
    missing_imgs_added_to_result = False
    diff_path = str(dt["compare_args"]["diff_img_path"])
    debugging = str(dt["compare_args"]["intermediate_output"])
    hashsize = str(dt["compare_args"]["d_hash_parametric"]["hash_size"])
    diff_path = get_diff_path(dt, result_list, "d_hash")
    dhash_err_ops = []
    diff_file_cnt = 0
    result_msg = ""
    print('')
    print('')
    print('')
    print("##############################################################")
    print("active algo : diff hashing")
    print("##############################################################")
    # get expected score for the current algo #
    algo_exp_score = float(img_comp_utils.get_algo_expected_score(comp_algos.diff_hashing,dt))
    img_file1, img_file2, missing_imgs_  = img_comp_utils.preprocess_images(dt, result_list, preprocessing_needed)
    dt["compare_args"]["missing_imgs"] = missing_imgs_
    dt["compare_args"]["abcde"] = "12345 - test dict entry..."
    #idx, base_img_cnt = get_idx_baseimgcnt(realtime, img_file1, img_file2, result_list)
    #base_path = os.path.dirname(img_file1)
    base_path = str(dt["compare_args"]["workspace_base"])
    runtime_path = str(dt["compare_args"]["workspace_runtime"])
    #runtime_path = os.path.dirname(img_file2)
    base_cnt, runtime_cnt = img_comp_utils.get_img_count(base_path,runtime_path,result_list)
    print("baseline imgs path: {0} --> img count:{1}".format(base_path, base_cnt))
    print("runtime imgs path: {0} --> img count:{1}".format(runtime_path, runtime_cnt))
    logging.info("##############################################################")
    logging.info("baseline imgs path: {0} --> img count:{1}".format(base_path, base_cnt))
    logging.info("runtime imgs path: {0} --> img count:{1}".format(runtime_path, runtime_cnt))
    logging.info("##############################################################")
    print("##############################################################")
    #while(idx <= base_img_cnt):
    for filename in os.listdir(base_path):
        img_file1 = os.path.join(base_path, filename)
        img_file2 = os.path.join(runtime_path, filename)
        print("d-hash --> current image ::"+img_file1)
        b_tmp_fname = "B_"+os.path.basename(img_file1)
        r_tmp_fname = "R_"+os.path.basename(img_file2)
        if(b_tmp_fname in missing_imgs_):
            imageops.image_match_outcome_list.append(imageops(b_tmp_fname, os.path.dirname(img_file1), os.path.dirname(img_file2), "d-hash", algo_exp_score, "-999.99", False, "missing baseline"))
            missing_imgs_added_to_result = True
            logging.error("missing baseline img:"+b_tmp_fname)
            continue
        if(r_tmp_fname in missing_imgs_):
            imageops.image_match_outcome_list.append(imageops(r_tmp_fname, os.path.dirname(img_file1), os.path.dirname(img_file2), "d-hash", algo_exp_score, "-999.99", False, "missing runtime"))
            missing_imgs_added_to_result = True
            logging.error("missing runtime img:"+r_tmp_fname)
            continue

        diff_file = os.path.join(diff_path,"diff_"+os.path.basename(img_file1))
        try:
            if(len(hashsize) > 0):
                baseline_hash = imagehash.dhash(Image.open(img_file1),int(hashsize))
                #print('Original Picture: ' + str(baselineHash))
                actual_hash = imagehash.dhash(Image.open(img_file2),int(hashsize))
                #print('Actual Picture: ' + str(actualHash))
            else:
                baseline_hash = imagehash.dhash(Image.open(img_file1))
                #print('Original Picture: ' + str(baselineHash))
                actual_hash = imagehash.dhash(Image.open(img_file2))

            distance = baseline_hash - actual_hash
        except:
            imageops.image_match_outcome_list.append(imageops(os.path.basename(img_file1), str(os.path.dirname(img_file1)).replace("\\","/"), str(os.path.dirname(img_file2)).replace("\\","/"), "d-hash", algo_exp_score, "0.0", False, str(sys.exc_info()[0])))
            dhash_err_ops.append("error :: d-hash --> Image :" + str(os.path.basename(img_file1)) + ", message : "+ str(sys.exc_info()[0])) 
            print("error :: d-hash --> Image :" + str(os.path.basename(img_file1)) + " "+ str(sys.exc_info()[0]))
            #img_comp_utils.writeToFile("ssi_err_ops.txt",str(ssi_err_ops))
            logging.error("error :: d-hash --> Image :"+ str(os.path.basename(img_file1)) + ", message : "+ str(sys.exc_info()[0]))
            continue
        
        if(baseline_hash == actual_hash):
            print("diff-hashing ::",os.path.basename(img_file1),": baseline == runtime. Score =",distance)
        else:
            print("diff-hashing ::",os.path.basename(img_file1),": baseline != runtime. Score =",distance)
        
        print("p-hash match distance (0 = 100% match): {}".format(distance))
        algo_perf_result = img_comp_utils.determine_match_outcome(comp_algos.perceptual_hashing, distance, "<=", dt)
        result_dict = {os.path.basename(img_file1):{distance:algo_perf_result}}
        print("Match result :",result_dict)
        print("##############################################################")
        if(algo_perf_result==False):
            result_msg = "0"
        else:
            result_msg = "1"
        if(len(hashsize) > 0):
            result_msg = result_msg + " [hs:"+hashsize+"]"
        else:
            result_msg = result_msg + " [hs:def]"
        imageops.image_match_outcome_list.append(imageops(os.path.basename(img_file1), str(os.path.dirname(img_file1)).replace("\\","/"), str(os.path.dirname(img_file2)).replace("\\","/"), "d-hash", algo_exp_score, str(distance), algo_perf_result, result_msg))
        #img_file1, img_file2 = img_comp_utils.getFileName(img_file1, img_file2, idx, result_list)
        
    if(missing_imgs_added_to_result == False and missing_imgs_ is not None):
        for key in missing_imgs_:
            if key not in missing_imgs_.values(): 
                imageops.image_match_outcome_list.append(imageops(key, str(os.path.dirname(img_file1)).replace("\\","/"), str(os.path.dirname(img_file2)).replace("\\","/"), "d-hash", algo_exp_score, "-999.99", False, "missing image"))
                logging.error("missing img:"+img_file1)
   

    if(debugging=="true"):        
        print('')
        print('')
        print('')
        print("******************cumulative match result --> algo :: perceptual-hashing***********************")
        print('')
        print(imageops.image_match_outcome_list)
        print("****************************************************************************************")
    logging.info("##############################################################")
    elapsed_time = round((time.time() - start_time)/60,2)
    logging.info("finished the diff-hashing algo operation in "+ str(elapsed_time) +" minutes...OK")
    logging.info("##############################################################")
    return imageops.image_match_outcome_list, dt



# updates on 31-Oct-2020 12:30 AM, 01-Nov-2020 04:45 AM, 02-Nov-2020 04:15 AM, 03-Nov-2020 01:30 AM, 09-Nov-2020 03:30 AM, 12-Nov-2020 02:15 AM, 14-Nov-2020 02:05 AM  #
def BRISK_FLANN_match(dt, result_list, preprocessing_needed=True):
    start_time = time.time()
    logging.info("##############################################################")
    logging.info("active algo : BRISK-FLANN")
    logging.info("##############################################################")
    print("result_list:",result_list)
    BRISK_FLANN_err_ops = []
    missing_imgs_added_to_result = False
    diff_path = str(dt["compare_args"]["diff_img_path"])
    debugging = str(dt["compare_args"]["intermediate_output"])
    diff_path = get_diff_path(dt, result_list, "BRISK_FLANN")
    print('')
    print('')
    print('')
    print("##############################################################")
    print("active algo : BRISK-FLANN")
    print("##############################################################")
    # get expected score for the current algo #
    algo_exp_score = float(img_comp_utils.get_algo_expected_score(comp_algos.diff_hashing,dt))
    img_file1, img_file2, missing_imgs_  = img_comp_utils.preprocess_images(dt, result_list, preprocessing_needed)
    dt["compare_args"]["missing_imgs"] = missing_imgs_
    dt["compare_args"]["abcde"] = "12345 - test dict entry..."
    #idx, base_img_cnt = get_idx_baseimgcnt(realtime, img_file1, img_file2, result_list)
    #base_path = os.path.dirname(img_file1)
    base_path = str(dt["compare_args"]["workspace_base"])
    runtime_path = str(dt["compare_args"]["workspace_runtime"])
    #runtime_path = os.path.dirname(img_file2)
    base_cnt, runtime_cnt = img_comp_utils.get_img_count(base_path,runtime_path,result_list)
    print("baseline imgs path: {0} --> img count:{1}".format(base_path, base_cnt))
    print("runtime imgs path: {0} --> img count:{1}".format(runtime_path, runtime_cnt))
    logging.info("##############################################################")
    logging.info("baseline imgs path: {0} --> img count:{1}".format(base_path, base_cnt))
    logging.info("runtime imgs path: {0} --> img count:{1}".format(runtime_path, runtime_cnt))
    logging.info("##############################################################")
    print("##############################################################")
    #while(idx <= base_img_cnt):
    for filename in os.listdir(base_path):
        img_file1 = os.path.join(base_path, filename)
        img_file2 = os.path.join(runtime_path, filename)
        print("BRISK-FLANN --> current image ::"+img_file1)
        b_tmp_fname = "B_"+os.path.basename(img_file1)
        r_tmp_fname = "R_"+os.path.basename(img_file2)
        if(b_tmp_fname in missing_imgs_):
            imageops.image_match_outcome_list.append(imageops(b_tmp_fname, os.path.dirname(img_file1), os.path.dirname(img_file2), "BRISK-FLANN", "", "-999.99", False, "missing baseline"))
            missing_imgs_added_to_result = True
            logging.error("missing baseline img:"+b_tmp_fname)
            continue
        if(r_tmp_fname in missing_imgs_):
            imageops.image_match_outcome_list.append(imageops(r_tmp_fname, os.path.dirname(img_file1), os.path.dirname(img_file2), "BRISK-FLANN", "", "-999.99", False, "missing runtime"))
            missing_imgs_added_to_result = True
            logging.error("missing runtime img:"+r_tmp_fname)
            continue
        kp_1 = 0
        kp_2 = 0
        diff_file = os.path.join(diff_path,"diff_"+os.path.basename(img_file1))
        try:
            img1_arr = cv2.imread(img_file1)
            img2_arr = cv2.imread(img_file2)
            descriptor = cv2.BRISK_create()
            kp_1, desc_1 = descriptor.detectAndCompute(img1_arr, None)
            kp_2, desc_2 = descriptor.detectAndCompute(img2_arr, None)
       
            #kp_1, kp_2, good_points, goodpoints_percent =  FLANNMatch(img1_arr, img2_arr, kp_1, kp_2, desc_1, desc_2, img_file1, dt)
            kp_1, kp_2, good_points, goodpoints_percent =  FLANNMatch(img1_arr, img2_arr, kp_1, kp_2, desc_1, desc_2, diff_file, dt)
        except:
            imageops.image_match_outcome_list.append(imageops(os.path.basename(img_file1), str(os.path.dirname(img_file1)).replace("\\","/"), str(os.path.dirname(img_file2)).replace("\\","/"), "BRISK-FLANN", str(kp_1), str(kp_2), False, str(sys.exc_info()[0])))
            BRISK_FLANN_err_ops.append("error :: BRISK-FLANN --> Image :" + str(os.path.basename(img_file1)) + ", message : "+ str(sys.exc_info()[0])) 
            print("error :: BRISK-FLANN --> Image :" + str(os.path.basename(img_file1)) + " "+ str(sys.exc_info()[0]))
            #img_comp_utils.writeToFile("ssi_err_ops.txt",str(ssi_err_ops))
            logging.error("error :: BRISK-FLANN --> Image :"+ str(os.path.basename(img_file1)) + ", message : "+ str(sys.exc_info()[0]))
            continue

        ###################code block for baseline data generation - to be used in benchmark-util.py right after BRISK-FLANN algo is run########################
        cap_kp_variance = kp_1 - kp_2
        if(str(dt["compare_args"]["BRISK_FLANN_parametric"]["BRISK_FLANN_bl_confirmed_variance_auto_update(disabled)"]).lower() == "true"):
            # for storing into BF baseline json only if the baseline json is not yet generated - 01-Nov-2020 05:30 AM, 02-Nov-2020 11:20 PM #
            # keep the below code block commented until the need arises. Activate it on the need basis - 01-Nov-2020 05:42 AM
            #conf_kp_variance = kp_1 - kp_2
            #BF_base_data_model.BF_algo_baseline_list.append(BF_base_data_model(os.path.basename(img_file1), str(os.path.dirname(img_file1)).replace("\\","/"), str(os.path.dirname(img_file2)).replace("\\","/"), str(kp_1), str(kp_2), str(good_points), str(goodpoints_percent),str(cap_kp_variance),str(kp_1), str(kp_2), str(good_points), str(goodpoints_percent),str(conf_kp_variance),"confirmed_kp_variance - auto updated"))
            pass
        else: #uncomment the next line to capture/process additional info, as the need arises. If enabled, needs update in data model too - 13-Nov-2020 11:50 PM
            #BF_base_data_model.BF_algo_baseline_list.append(BF_base_data_model(os.path.basename(img_file1), str(os.path.dirname(img_file1)).replace("\\","/"), str(os.path.dirname(img_file2)).replace("\\","/"), str(kp_1), str(kp_2), str(good_points), str(goodpoints_percent),str(cap_kp_variance),str(kp_1), str(kp_2), str(good_points), str(goodpoints_percent),"","confirmed_kp_variance - manual update needed"))
            BF_base_data_model.BF_algo_baseline_list.append(BF_base_data_model(os.path.basename(img_file1), str(kp_1), str(kp_2), str(good_points), str(goodpoints_percent),"","est.kp_vari:"+str(cap_kp_variance)+".Need conf."))
        dt["compare_args"]["BF_algo_baseline"] = BF_base_data_model.BF_algo_baseline_list
        ###################end of code block for baseline data generation########################################################

        #######################code block for result evaluation and postings##########################################
        dt["compare_args"]["runtime_conf_kp_variance"] = ""
        dt["compare_args"]["conf_kp_variance"] = ""
        dt["compare_args"]["kp_1"] = kp_1
        dt["compare_args"]["kp_2"] = kp_2
        dt["compare_args"]["good_points"] = good_points
        dt["compare_args"]["goodpoints_percent"] = goodpoints_percent
        confirmed_base_range_kp, base_range_gp, base_range_gpp, res1, dt, BF_result_msg = img_comp_utils.determine_match_within_range(comp_algos.brisk_flann, os.path.basename(img_file1), dt)
        conf_kp_variance = dt["compare_args"]["conf_kp_variance"]
        runtime_conf_kp_variance = dt["compare_args"]["runtime_conf_kp_variance"]
        combined_base_scores = "[kp:confirmed=" + str(confirmed_base_range_kp) + ",current=" + str(kp_1) + "] [gp:" + str(base_range_gp) + "] [gpp:" + str(base_range_gpp) + "] [kp_variance:" + str(conf_kp_variance) + "]"
        combined_act_scores =  "[kp:" + str(kp_2) + "] [gp:" + str(good_points) + "] [gpp:" + str(goodpoints_percent) + "] [kp_variance:" + str(runtime_conf_kp_variance) + "]"
        if(len(imageops.image_match_outcome_list) <= 0):
            imageops.image_match_outcome_list.append(imageops(os.path.basename(img_file1), str(os.path.dirname(img_file1)).replace("\\","/"), str(os.path.dirname(img_file2)).replace("\\","/") ,"BRISK-FLANN", combined_base_scores, combined_act_scores, res1, BF_result_msg))
        else:
            imageops.image_match_outcome_list.append(imageops(os.path.basename(img_file1), "", "", "BF", combined_base_scores, combined_act_scores, res1, BF_result_msg))
        #######################end of code block for result evaluation and postings##########################################
    if(missing_imgs_added_to_result == False and missing_imgs_ is not None):
        for key in missing_imgs_:
            if key not in missing_imgs_.values(): 
                imageops.image_match_outcome_list.append(imageops(key, str(os.path.dirname(img_file1)).replace("\\","/"), str(os.path.dirname(img_file2)).replace("\\","/") ,"BRISK-FLANN", "", "-999.99", False, "missing image"))
                logging.error("missing img:"+img_file1)

    if(debugging=="true"):        
        print('')
        print('')
        print('')
        print("******************cumulative match result --> algo :: BRISK-FLANN***********************")
        print('')
        print(imageops.image_match_outcome_list)
        print("****************************************************************************************")
    logging.info("##############################################################")
    elapsed_time = round((time.time() - start_time)/60,2)
    logging.info("finished the BRISK-FLANN algo operation in "+ str(elapsed_time) +" minutes...OK")
    logging.info("##############################################################")
    return imageops.image_match_outcome_list, dt


# updates on 13-Oct-2020 02:45 AM, 09-Nov-2020 03:30 AM, 11-Nov-2020 12:40 AM #
def FLANNMatch(img_arr1, img_arr2, kp_1, kp_2, desc_1, desc_2, diff_file, dt):
    #index_params = dict(algorithm=0, trees=5)
    FLANN_accuracy = str(dt["compare_args"]["BRISK_FLANN_parametric"]["FLANNmatcher_accuracy"])
    debugging = str(dt["compare_args"]["intermediate_output"])
    #desc_1 = np.float32(desc_1)
    #desc_2 = np.float32(desc_2)
    try:
        FLANN_INDEX_LSH = 6
        #FLANN_INDEX_KDTREE = 0 # returns 0 gp and 0.0 gpp for all images, and hence not recommended for the use in this tool
        index_params= dict(algorithm = FLANN_INDEX_LSH,
                    table_number = 6, # 12
                    key_size = 12,     # 20
                    multi_probe_level = 1) #2
        search_params = dict()
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        knn_matches = flann.knnMatch(desc_1, desc_2, k=2)
        #if(len(knn_matches) <= 0):
        #   return len(kp_1), len(kp_2), 0, 0
        #matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)
        #knn_matches = matcher.knnMatch(desc_1, desc_2, 2)
        #print(knn_matches)
        good_points = []
        #for (m,n) in knn_matches:
        #for i,(m,n) in enumerate(knn_matches): #this too is working fine - 29-Sep-2020 12:51 AM
        for i,m_n in enumerate(knn_matches): #this too is working fine - 29-Sep-2020 12:51 AM
            if len(m_n) != 2:
                continue
            (m,n) = m_n
            if m.distance < float(FLANN_accuracy)*n.distance:
                good_points.append(m)

        # Define how similar they are
        number_keypoints = 0
        if len(kp_1) <= len(kp_2):
            number_keypoints = len(kp_1)
        else:
            number_keypoints = len(kp_2)

        goodpoints_percent = round((len(good_points) / number_keypoints * 100),2)
        print("keypoints baseline image  : ", str(len(kp_1)))
        print("keypoints runtime image   : ", str(len(kp_2)))
        print("good matches              : ", len(good_points))
        print("how good it's the match ? : ", format(goodpoints_percent))
        print("               *****                  ")
        if(len(good_points) > 0):
            result = cv2.drawMatches(img_arr1, kp_1, img_arr2, kp_2, good_points, None)
            cv2.imwrite(diff_file, result)
        if(debugging=="true" and len(good_points > 0)):
            result = cv2.drawMatches(img_arr1, kp_1, img_arr2, kp_2, good_points, None)
            cv2.imshow("result", cv2.resize(result, None, fx=0.9, fy=0.9))
        return len(kp_1), len(kp_2), len(good_points), goodpoints_percent
    except:
        print("exception :: FLANNMatch --> Result : good points = 0)")
        return len(kp_1), len(kp_2), 0, 0.0
###################End of the function####################################


def doperceptualhashing(imageAStr, imageBStr):
    baselineHash = imagehash.phash(Image.open(imageAStr))
    print('Original Picture: ' + str(baselineHash))

    actualHash = imagehash.phash(Image.open(imageBStr))
    print('Actual Picture: ' + str(actualHash))

    if(baselineHash == actualHash):
        print("Perceptual Hashing :: The pictures are perceptually the same !")
    else:
        distance = baselineHash - actualHash
        print("Perceptual Hashing :: The pictures are different, distance: " + str(distance))
###################End of the function####################################


def dodifferencehashing(imageAStr, imageBStr):
    baselineHash = imagehash.dhash(Image.open(imageAStr))
    print('Original Picture: ' + str(baselineHash))

    actualHash = imagehash.dhash(Image.open(imageBStr))
    print('Actual Picture: ' + str(actualHash))

    if(baselineHash == actualHash):
        print("Difference Hashing :: The pictures are perceptually the same !")
    else:
        distance = baselineHash - actualHash
        print("Difference Hashing :: The pictures are different, distance: " + str(distance))
###################End of the function####################################





def compare_images_colored(imageA, imageB, title):
	# compute the mean squared error and structural similarity
	# index for the images
	#m = mse(imageA, imageB)
    # convert the images to grayscale
    grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)
        
    #(score, diff) = ssim(imageA, imageB, multiChannel=True, full=True)/commented for experimental purpose - 23-Apr-2020 03:20 AM
    (score, diff) = ssim(grayA, grayB,  full=True)
    diff = (diff * 250).astype("uint8")
    print("SSIM: {}".format(score))
    #print("l: {}".format(l))
    #print("c: {}".format(c))
    #print("s: {}".format(s))
 
	 # threshold the difference image, followed by finding contours to
    # obtain the regions of the two input images that differ
    
    thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    # loop over the contours
    for c in cnts:
	    # compute the bounding box of the contour and then draw the
	    # bounding box on both input images to represent where the two
	    # images differ
	    (x, y, w, h) = cv2.boundingRect(c)
	    cv2.rectangle(imageA, (x, y), (x + w, y + h), (0, 0, 155), 2)
	    cv2.rectangle(imageB, (x, y), (x + w, y + h), (0, 0, 155), 2)
    #cv2.rectangle(diff, (x, y), (x + w, y + h), (0, 0, 155), 2)
    cv2.imwrite("diff_img_colored.png",diff)
    print("Match similarity:",score)
	# show the output images
    cv2.imshow("Original-Colored", imageA)
    cv2.imshow("Modified-Colored", imageB)
    cv2.imshow("Diff-Colored", diff)
    #cv2.imshow("Thresh", thresh)
    cv2.waitKey(0)

def compare_images_grey(imageA, imageB, title):
	# compute the mean squared error and structural similarity
	# index for the images
	#m = mse(imageA, imageB)
        
    #(score, diff) = ssim(imageA, imageB,full=True)
    (score, diff) = ssim(imageA, imageB, full=True)
    diff = (diff * 250).astype("uint8")
    print("SSIM: {}".format(score))
    #print("l: {}".format(l))
    #print("c: {}".format(c))
    #print("s: {}".format(s))
 
	 # threshold the difference image, followed by finding contours to
    # obtain the regions of the two input images that differ
    
    thresh = cv2.threshold(diff, 0, 255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    # loop over the contours
    for c in cnts:
	    # compute the bounding box of the contour and then draw the
	    # bounding box on both input images to represent where the two
	    # images differ
	    (x, y, w, h) = cv2.boundingRect(c)
	    cv2.rectangle(imageA, (x, y), (x + w, y + h), (0, 0, 155), 2)
	    cv2.rectangle(imageB, (x, y), (x + w, y + h), (0, 0, 155), 2)
    #cv2.rectangle(diff, (x, y), (x + w, y + h), (0, 0, 155), 2)
    cv2.imwrite("diff_img.png",diff)
    print("Match similarity:",score)
	# show the output images
    cv2.imshow("Original-Grey", imageA)
    cv2.imshow("Modified-Grey", imageB)
    cv2.imshow("Diff-Grey", diff)
    #cv2.imshow("Thresh", thresh)
    cv2.waitKey(0)






def dodifferencehashing(imageAStr, imageBStr):
    baselineHash = imagehash.dhash(Image.open(imageAStr))
    print('Original Picture: ' + str(baselineHash))

    actualHash = imagehash.dhash(Image.open(imageBStr))
    print('Actual Picture: ' + str(actualHash))

    if(baselineHash == actualHash):
        print("Difference Hashing :: The pictures are perceptually the same !")
    else:
        distance = baselineHash - actualHash
        print("Difference Hashing :: The pictures are different, distance: " + str(distance))


def dhash(image, hashSize=10):
	# resize the input image, adding a single column (width) so we
	# can compute the horizontal gradient
	resized = cv2.resize(image, (hashSize + 1, hashSize))
	# compute the (relative) horizontal gradient between adjacent
	# column pixels
	diff = resized[:, 1:] > resized[:, :-1]
	# convert the difference image to a hash
	return sum([2 ** i for (i, v) in enumerate(diff.flatten()) if v])

def doOpenCVPerceptualHash(imageAStr, imageBStr):
    imageA = cv2.imread(imageAStr)
    imageB = cv2.imread(imageBStr)
    imageA_grey = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
    baselineHash = dhash(imageA_grey)
    imageB_grey = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)
    actualHash = dhash(imageB_grey)
    if(baselineHash == actualHash):
        print("openCV :: The pictures are perceptually the same !")
    else:
        distance = baselineHash - actualHash
        print("openCV :: The pictures are different, distance: " + str(distance))

# updates on 08-Oct-2020 12:25 AM, 24-Oct-2020 10:25 PM #
def get_idx_baseimgcnt(realtime, img_file1, img_file2, result_list):
    base_img_cnt = 1
    idx = 1
    if(len(result_list) > 0):
        base_img_cnt = len(result_list)
        return idx, base_img_cnt
    img1_path = os.path.dirname(img_file1)
    img2_path = os.path.dirname(img_file2)
    base_img_cnt, runtime_img_cnt = img_comp_utils.get_img_count(img1_path,img2_path)
    return idx, base_img_cnt

 # created on 08-Oct-2020 02:15 AM #
 # updates on 25-Oct-2020 08:45 PM, 26-Oct-2020 02:40 AM, 27-Oct-2020 01:50 AM #
def get_diff_path(dt, result_list, algo_diff_path="ssi"):
    diff_path=""
    algo_diff_path = str(algo_diff_path)
    if(len(result_list) > 0):
        diff_path = os.path.join(result_list[0]["runtime_img_path"])
        return diff_path
    
    diff_path = os.path.join(str(dt["compare_args"]["diff_img_path"]),algo_diff_path)
    if(len(str(dt["compare_args"]["diff_img_path"])) == 0):
        import datetime
        workspace_path = str(dt["compare_args"]["workspace_path"])
        x = datetime.datetime.now()
        dt_part = "{0}{1}{2}_{3}{4}{5}".format(x.day,x.month,x.year,x.hour,x.minute,x.second)
        diff_path = os.path.join(workspace_path,"/image_ops/diffs/",dt_part,algo_diff_path)
    if(os.path.exists(diff_path) == False):
        os.makedirs(diff_path)
    return diff_path















'''
# created on 13-Oct-2020 02:00 AM #
def FLANNMatch_1(img_arr1, img_arr2, kp_1, kp_2, desc_1, desc_2, diff_file, dt):
    #index_params = dict(algorithm=0, trees=5)
    FLANN_accuracy = str(dt["compare_args"]["BRISK_FLANN_parametric"]["FLANNmatcher_accuracy"])
    debugging = str(dt["compare_args"]["intermediate_output"])
    #desc_1 = np.float32(desc_1)
    #desc_2 = np.float32(desc_2)
    desc_1 = desc_1.astype(np.float32)
    desc_2 = desc_2.astype(np.float32)
    #desc_1.convertTo(desc_1, CV_32F)
    #desc_2.convertTo(desc_2, CV_32F)
    # FLANN parameters
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)   # or pass empty dictionary

    flann = cv2.FlannBasedMatcher(index_params,search_params)

    matches = flann.knnMatch(desc_1,desc_2,k=2)

    # Need to draw only good matches, so create a mask
    matchesMask = [[0,0] for i in range(len(matches))]

    # ratio test as per Lowe's paper
    good_points = []
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.7*n.distance:
            matchesMask[i]=[1,0]
            good_points.append(m)

    draw_params = dict(matchColor = (0,255,0),
                   singlePointColor = (255,0,0),
                   matchesMask = matchesMask,
                   flags = 0)

    img3 = cv2.drawMatchesKnn(img_arr1,kp_1,img_arr2,kp_2,matches,None,**draw_params)

    number_keypoints = 0
    if len(kp_1) <= len(kp_2):
        number_keypoints = len(kp_1)
    else:
        number_keypoints = len(kp_2)

    goodpoints_percent = len(good_points) / number_keypoints * 100
    print("keypoints baseline image : " + str(len(kp_1)))
    print("keypoints runtime image  : " + str(len(kp_2)))
    print("good matches:", len(good_points))
    print("how good it's the match ? : ", goodpoints_percent)

    #plt.imshow(img3,),plt.show())
    cv2.imshow("result", cv2.resize(img3, None, fx=0.9, fy=0.9))
    return len(kp_1), len(kp_2), len(good_points), goodpoints_percent
###################End of the function####################################

# created on 13-Oct-2020 01:45 AM #

def FLANNMatch_prev(img_arr1, img_arr2, kp_1, kp_2, desc_1, desc_2, diff_file, dt):
    #index_params = dict(algorithm=0, trees=5)
    FLANN_accuracy = str(dt["compare_args"]["BRISK_FLANN_parametric"]["FLANNmatcher_accuracy"])
    debugging = str(dt["compare_args"]["intermediate_output"])
    #desc_1 = np.float32(desc_1)
    #desc_2 = np.float32(desc_2)
    FLANN_INDEX_LSH = 6
    index_params= dict(algorithm = FLANN_INDEX_LSH,
                    table_number = 6, # 12
                    key_size = 12,     # 20
                    multi_probe_level = 1) #2
    search_params = dict()
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    knn_matches = flann.knnMatch(desc_1, desc_2, k=2)
    #if(len(knn_matches) <= 0):
     #   return len(kp_1), len(kp_2), 0, 0
    #matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)
    #knn_matches = matcher.knnMatch(desc_1, desc_2, 2)
    #print(knn_matches)
    good_points = []
    #for (m,n) in knn_matches:
    #for i,(m,n) in enumerate(knn_matches): #this too is working fine - 29-Sep-2020 12:51 AM
    for i,m_n in enumerate(knn_matches): #this too is working fine - 29-Sep-2020 12:51 AM
        if len(m_n) != 2:
            continue
        (m,n) = m_n
        if m.distance < float(FLANN_accuracy)*n.distance:
            good_points.append(m)

    # Define how similar they are
    number_keypoints = 0
    if len(kp_1) <= len(kp_2):
        number_keypoints = len(kp_1)
    else:
        number_keypoints = len(kp_2)

    goodpoints_percent = len(good_points) / number_keypoints * 100
    print("keypoints baseline image : " + str(len(kp_1)))
    print("keypoints runtime image  : " + str(len(kp_2)))
    print("good matches:", len(good_points))
    print("how good it's the match ? : ", goodpoints_percent)
    if(len(good_points) > 0):
        result = cv2.drawMatches(img_arr1, kp_1, img_arr2, kp_2, good_points, None)
        cv2.imwrite(diff_file, result)
    if(debugging=="true" and len(good_points > 0)):
        cv2.imshow("result", cv2.resize(result, None, fx=0.9, fy=0.9))
    return len(kp_1), len(kp_2), len(good_points), goodpoints_percent
###################End of the function####################################


'''