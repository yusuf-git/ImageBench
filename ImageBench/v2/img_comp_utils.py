##Created on 08-Aug-2020 10:30 AM
##Updates on 08-Aug-2020 01:30 AM, 09-Aug-2020 01:00 AM, 10-Aug-2020 03:10 AM, 11-Aug-2020 10:30 PM, 12-Aug-2020 01:30 AM, 13-Aug-2020 02:30 AM, 14-Aug-2020 02:15 AM, 15-Aug-2020 02:45 AM, 28-Sep-2020 02:20 AM, 29-Sep-2020 01:15 AM, 02-Oct-2020 01:00 AM, 03-Oct-2020 01:15 AM, 04-Oct-2020 12:45 AM, 05-Oct-2020 02:25 AM, 06-Aug-2020 02:45 AM, 07-Aug-2020 03:20 AM, 13-Oct-2020 04:00 AM, 18-Oct-2020 03:45 AM

import argparse
import os.path
import time
import json
import sys
import numpy as np
from PIL import Image  
import cv2
import pyautogui as pygui
from skimage.measure import compare_ssim
import imutils
import os
from pathlib import Path
from shutil import copyfile
import multiprocessing as mp
from multiprocessing import Process, Lock
from algos_namelist import comp_algos
from imageops_data_model import CustomEncoder
from BF_baseline_data_model import BF_base_data_model
from BF_basetobase_comp_data_model import BF_basetobase_comp_data_model
import logging
#from di_container import di_container

def readJson(jsonFile_, mainkey):
    f_ = open(jsonFile_) 
    data_ = json.load(f_) 
    #print(data_)
    for i in data_[mainkey]: 
        #print(i) 
        pass
    f_.close()
    return data_

def readJson_plain(jsonFile_):
    f_ = open(jsonFile_) 
    data_ = json.load(f_) 
    #print(data_) #uncomment later 07-Oct-2020 09:37 PM
    #for i in data_: 
    #    print(i) 
    f_.close()
    return data_

# created on 04-Oct-2020 09:05 PM #
#def writeJson(jsonFile_, data):
 #   with open(jsonFile_, 'w') as fp:
  #      json.dump(data, fp)


def writeResult(result,msg, resPath):
    resFile = os.path.join(resPath, "result.json")
    if(os.path.exists(resFile)):
        os.remove(resFile)
    resultStr = {"result":result,"message":msg}
    resJson = json.dumps(resultStr, indent=4)
    with open(resFile,"w") as json_out:
        json_out.write(resJson)

# created on 06-Aug-2020 02:45 AM #
# updates on 07-Aug-2020 03:20 AM, 03:15 PM, 17-Oct-2020 03:05 AM, 01-Nov-2020 01:35 AM #
def writeJson(resFile, data, stringFormat=False, filemode="w", del_current_file=True):
    #resFile = os.path.join(resPath, "result.json")
    if(os.path.exists(resFile) and del_current_file == True):
        os.remove(resFile)
    #resultStr = {"result":result,"message":msg}
    #print(data)
    #resJson = json.dump(data.__dict__, lambda o: o.__dict__, indent=4)
    #resJson = json.dumps(data, cls = CustomEncoder)
    if(stringFormat==False):
        resJson = json.dumps([o.dump() for o in sorted(data)])
    else:
        data = json.dumps(data)
        resJson = str(data)
    #print("resjson:",resJson) #uncomment later 01-Oct-2020 09:35 PM
    with open(resFile,filemode) as json_out:
        json_out.write(resJson)


# created on 06-Aug-2020 02:45 AM #
# updates on 07-Aug-2020 03:20 AM, 03:15 PM #
def write_dict_Json(resFile, data):
    #resFile = os.path.join(resPath, "result.json")
    if(os.path.exists(resFile)):
        os.remove(resFile)
    #resultStr = {"result":result,"message":msg}
    #print(data)
    #resJson = json.dump(data.__dict__, lambda o: o.__dict__, indent=4)
    #resJson = json.dumps(data, cls = CustomEncoder)
    #resJson = json.dumps([o.dump() for o in data])
    #print("resjson:",resJson) #uncomment later 01-Oct-2020 09:35 PM
    with open(resFile,"w") as json_out:
        json_out.write(str(data))

 # created on 06-Aug-2020 02:45 AM #
 # updates on 07-Aug-2020 01:30 AM, 28-OCt-2020 12:10 AM #
def writelist_toJson(resFile, datalist):
    #resFile = os.path.join(resPath, "result.json")
    if(os.path.exists(resFile)):
        os.remove(resFile)
    #resultStr = {"result":result,"message":msg}
    #print(data)
    #resJson = json.dump(data.__dict__, lambda o: o.__dict__, indent=4)
    resJson = json.dumps([ob.__dict__ for ob in datalist], indent=4)
    #print("resjson:",resJson)
    with open(resFile,"w") as json_out:
        json_out.write(resJson)


def writeToFile(file, contents):
    f = open(file, "a")
    f.write(contents)
    f.close()

def readMaskArea(mask_region_):
    c_x1 = ""
    c_y1 = ""
    c_x2 = ""
    c_y2 = ""
    if(mask_region_ != ""):
        c_x1 = str(mask_region_).split(',')[0]
        c_y1 = str(mask_region_).split(',')[1]
        c_x2 = str(mask_region_).split(',')[2]
        c_y2 = str(mask_region_).split(',')[3]
    return c_x1,c_y1,c_x2,c_y2


# created on 05-Oct-2020 02:10 AM #
# updates on 05-Oct-2020 02:20 AM #
def baseline_apply_mask_excluding_region(baseline_path, runtime_path, mask_area, realtime, showTheMask):
    print("starting to apply mask for baseline images excluding the specified region...")
    c_x1_, c_y1_, c_x2_, c_y2_ = readMaskArea(mask_area)
    img1_path = baseline_path
    img2_path = runtime_path
    print("mask region :",(int(c_x1_), int(c_y1_)), (int(c_x2_), int(c_y2_)))

    #for x in range(1, base_img_cnt+1, 1):
    for filename in os.listdir(img1_path):
        img1_fname = os.path.join(img1_path, filename)
        runtime_ref_file = os.path.join(img2_path, filename)
        image = cv2.imread(img1_fname)
        mask_ = np.zeros(image.shape[:2], dtype = "uint8")
        h,w,c = image.shape
        print("image : ",img1_fname," --> width, height, channel:", w, h, c)
        cv2.rectangle(mask_, (int(c_x1_), int(c_y1_)), (int(c_x2_), int(c_y2_)), (255,0,0), -1)
        masked_ = cv2.bitwise_and(image, image, mask = mask_)
        cv2.imwrite(img1_fname,masked_)
    

# created on 05-Oct-2020 02:05 AM # 
# updates on 05-Oct-2020 02:20 AM #
def runtime_imgs_apply_mask_excluding_region(baseline_path, runtime_path, mask_area, realtime, showTheMask):
    print("starting to apply mask for runtime images excluding the specified region...")
    c_x1_, c_y1_, c_x2_, c_y2_ = readMaskArea(mask_area)
    img1_path = baseline_path
    img2_path = runtime_path
    print("mask region :",(int(c_x1_), int(c_y1_)), (int(c_x2_), int(c_y2_)))

    #for x in range(1, runtime_img_cnt+1, 1):
    for filename in os.listdir(img2_path):
        img2_fname = os.path.join(img2_path, filename)
        runtime_ref_file = os.path.join(img2_path, filename)
        image = cv2.imread(img2_fname)
        mask_ = np.zeros(image.shape[:2], dtype = "uint8")
        h,w,c = image.shape
        print("image : ",img2_fname," --> width, height, channel:", w, h, c)
        cv2.rectangle(mask_, (int(c_x1_), int(c_y1_)), (int(c_x2_), int(c_y2_)), (255,0,0), -1)
        masked_ = cv2.bitwise_and(image, image, mask = mask_)
        cv2.imwrite(img2_fname,masked_)


# updates on 05-Oct-2020 02:00 AM, 02:20 AM #
def baseline_imgs_apply_mask(baseline_path, runtime_path, mask_area, realtime, showTheMask):
    print("starting to apply mask for baseline images...")
    c_x1_, c_y1_, c_x2_, c_y2_ = readMaskArea(mask_area)
    img1_path = baseline_path
    print("mask region :",(int(c_x1_), int(c_y1_)), (int(c_x2_), int(c_y2_)))

    #for x in range(1, base_img_cnt+1, 1):
    for filename in os.listdir(img1_path):
        img1_fname = os.path.join(img1_path, filename)
        #runtime_ref_file = os.path.join(img2_path, filename)
        image = cv2.imread(img1_fname)
        h,w,c = image.shape
        print("image : ",img1_fname," --> width, height, channel:", w, h, c)
        masked_image = image
        masked_image[int(c_y1_):int(c_y2_),int(c_x1_):int(c_x2_)] = (0,0,0)
        cv2.imwrite(img1_fname,masked_image)


# updates on 05-Oct-2020 01:55 AM, 02:20 AM #
def runtime_imgs_apply_mask(baseline_path, runtime_path, mask_area, realtime, showTheMask):
    print("starting to apply mask for runtime images...")
    c_x1_, c_y1_, c_x2_, c_y2_ = readMaskArea(mask_area)
    img2_path = runtime_path
    print("mask region :",(int(c_x1_), int(c_y1_)), (int(c_x2_), int(c_y2_)))

    #for x in range(1, runtime_img_cnt+1, 1):
    for filename in os.listdir(img2_path):
        img2_fname = os.path.join(img2_path, filename)
        image = cv2.imread(img2_fname)
        h,w,c = image.shape
        print("image : ",img2_fname," --> width, height, channel:", w, h, c)
        masked_image = image
        masked_image[int(c_y1_):int(c_y2_),int(c_x1_):int(c_x2_)] = (0,0,0)
        cv2.imwrite(img2_fname,masked_image)


# updates on 04-Oct-2020 09:45 PM, 25-Oct-2020 01:00 AM, 28-Oct-2020 12:15 AM #
def apply_mask_parallel(tmp_img1, tmp_img2, mask_area, realtime, showTheMask):
    start_time = time.time()
    logging.info("starting to mask the specified region....")
    if __name__ == '__main__':
        print("apply_mask_parallel and __name==__main ")
        p1 = Process(target=baseline_imgs_apply_mask, args=(tmp_img1, tmp_img2, mask_area, realtime, showTheMask))
        p1.start()
        p2 = Process(target=runtime_imgs_apply_mask, args=(tmp_img1, tmp_img2, mask_area, realtime, showTheMask))
        p2.start()
        p1.join()
        p2.join()
    else:
        baseline_imgs_apply_mask(tmp_img1, tmp_img2, mask_area, realtime, showTheMask)
        runtime_imgs_apply_mask(tmp_img1, tmp_img2, mask_area, realtime, showTheMask)

    elapsed_time = round((time.time() - start_time)/60,2)
    logging.info("finished masking the specified region in "+ str(elapsed_time) +" minutes....OK")

# updates on 28-Oct-2020 12:15 AM #
def apply_mask_excluding_region_parallel(tmp_img1, tmp_img2, mask_area_excl, realtime, showTheMask):
    start_time = time.time()
    logging.info("starting to mask entire region exlcuding the specified area....")
    if __name__ == '__main__':
        print("invoking apply_mask_excluding_region_parallel and __name==__main... ")
        p1 = Process(target=baseline_apply_mask_excluding_region, args=(tmp_img1, tmp_img2, mask_area_excl, realtime, showTheMask))
        p1.start()
        p2 = Process(target=runtime_imgs_apply_mask_excluding_region, args=(tmp_img1, tmp_img2, mask_area_excl, realtime, showTheMask))
        p2.start()
        p1.join()
        p2.join()
    else:
        print("invoking apply_mask_excluding_region_parallel...")
        baseline_apply_mask_excluding_region(tmp_img1, tmp_img2, mask_area_excl, realtime, showTheMask)
        runtime_imgs_apply_mask_excluding_region(tmp_img1, tmp_img2, mask_area_excl, realtime, showTheMask)
    
    elapsed_time = round((time.time() - start_time)/60,2)
    logging.info("finished masking the entire region exlcuding the specified area in "+ str(elapsed_time) +" minutes....OK")


def resize(imFile,width,height):
    img = Image.open(imFile)
    out = img.resize((width,height))
    print("out------------------:",out.size)
    return out
    #out.save("D:/Automation/DW/SDV/src/test-inputs/images/login/realtime/chrome/baseline/body-resized-1.png")

#def createTempWorkspace(runTimePath):
#    base_tmp_img_path = os.path.join(os.path.dirname(runTimePath),"b_temp")
#    run_tmp_img_path = os.path.join(os.path.dirname(runTimePath),"r_temp")
#    if(os.path.exists(base_tmp_img_path) is False):
#        os.mkdir(base_tmp_img_path)
#    if(os.path.exists(run_tmp_img_path) is False):
#        os.mkdir(run_tmp_img_path)
#    return base_tmp_img_path, run_tmp_img_path

#def getTempFileNames(image1, image2, runTimePath):
#    base_tmp_img_path, run_tmp_img_path = createTempWorkspace(runTimePath)
#    tmp_img1 = os.path.basename(image1)
#    tmp_img2 = os.path.basename(image2)
#    tmp_img1 = os.path.join(base_tmp_img_path,tmp_img1)
#    tmp_img2 = os.path.join(run_tmp_img_path,tmp_img2)
#    return tmp_img1,tmp_img2

# updates on 08-Oct-2020 02:35 AM #
def get_img_count(base_path, runtime_path, result_list=[]):
    #print("basepath1:",base_path)
    #print("runtime_path1:",runtime_path)
    #base_path = os.path.dirname(base_path)
    #runtime_path = os.path.dirname(runtime_path)
    #print("basepath:",base_path)
    #print("runtime_path:",runtime_path)
    if(len(result_list) > 0):
        base_img_cnt = len(result_list)
        runtime_img_cnt = len(result_list)
        return base_img_cnt,runtime_img_cnt

    base_img_cnt = len([name for name in os.listdir(base_path) if os.path.isfile(os.path.join(base_path, name))])
    runtime_img_cnt = len([name for name in os.listdir(runtime_path) if os.path.isfile(os.path.join(runtime_path, name))])
    return base_img_cnt,runtime_img_cnt


                    
# updates on 27-Oct-2020 11:15 PM #
def downsizeAndEqualizeImages(tmp_img1, tmp_img2, diff_img_files_):
    #image1 = cv2.imread(image1)
    #image2 = cv2.imread(image2)
    start_time = time.time()
    logging.info("starting downsizing and equalizing the images....")
    img_size_op = {}
    for key in diff_img_files_:
        if(diff_img_files_[key] is False):
            img_file1 = os.path.join(os.path.dirname(tmp_img1), key)
            img_file2 = os.path.join(os.path.dirname(tmp_img2), key)
       
            image1 = Image.open(img_file1)
            image2 = Image.open(img_file2)
            if(image1.size[0] > image2.size[0] or image1.size[1] > image2.size[1]):
                print("Image1 is resized")
                #imageA = cv2.resize(imageA, (image1.shape[1], image2.shape[0]))
                image1 = image1.resize((image2.size[0], image2.size[1]))
                image1.save(img_file1)
                image2.save(img_file2)
            if(image2.size[0] > image1.size[0] or image2.size[1] > image1.size[1]):
                print("Image2 is resized")
                image2 = image2.resize((image1.size[0], image1.size[1]))
                image1.save(img_file1)
                image2.save(img_file2)
            print("After resize - image1:",image1.size[0],"x",image1.size[1])
            print("After resize - image2:",image2.size[0],"x",image2.size[1])
            if(image1.size == image2.size):
                img_size_op = {key:True}
            else:
                img_size_op = {key:False}
    elapsed_time = round((time.time() - start_time)/60,2)
    logging.info("finished with downsizing and equalizing the images in "+ str(elapsed_time) +" minutes....OK")
    return img_size_op   

# updates on 27-Oct-2020 11:15 PM #
def find_aspect_ratio_and_equalize(base_path, runtime_path, diff_img_files_):
    start_time = time.time()
    logging.info('starting with finding aspect ratio and equalizing the images....')
    img_size_op = {}
    for key in diff_img_files_:
        if(diff_img_files_[key] is False):
            
            img_file1 = os.path.join(base_path, key)
            img_file2 = os.path.join(runtime_path, key)
            print("this is keyyyyyyyyyyyyyyyyyyyyy:",img_file1,"----------",img_file2)
            image1 = Image.open(img_file1)
            image2 = Image.open(img_file2)
            print(image1.size)
            print(image2.size)
            print(image1.size[0],"x", image1.size[1])
            if(image1.size[0] > image2.size[0]):
                print("image1 width resized")
                width_ratio = image2.size[0]/image1.size[0]
                print("ratio:",width_ratio)
                new_height = int(width_ratio * image1.size[1])
                print("new_height",new_height)
                dim = (image2.size[0], new_height)
                print("dim:",dim)
                image1 = image1.resize(dim)
                image1.save(img_file1)
                image2 = resize(img_file2,image1.size[0],image1.size[1])
                image2.save(img_file2)
                print(image1.size)
                print(image2.size)
            if(image1.size[1] > image2.size[1]):
                print("image1 height resized")
                height_ratio = image2.size[1]/image1.size[1]
                new_width = int(height_ratio * image1.size[0])
                dim = (new_width, image2.size[1])
                image1 = image1.resize(dim)
                image1.save(img_file1)
                image2 = resize(img_file2,image1.size[0],image1.size[1])
                image2.save(img_file2)
            if(image2.size[0] > image1.size[0]):
                print("image2 width resized")
                width_ratio = image1.size[0]/image2.size[0]
                new_height = int(width_ratio * image2.size[1])
                dim = (image1.size[0], new_height)
                image2 = image2.resize(dim)
                image2.save(img_file2)
                image1 = resize(img_file1,image2.size[0],image2.size[1])
                image1.save(img_file1)
            if(image2.size[1] > image1.size[1]):
                print("image2 height resized")
                height_ratio = image1.size[1]/image2.size[1]
                new_width = int(height_ratio * image2.size[0])
                dim = (new_width, image1.size[1])
                image2 = image2.resize(dim)
                image2.save(img_file2)
                image1 = resize(img_file1,image2.size[0],image2.size[1])
                image1.save(img_file1)
            print("After resize - image1:",image1.size[0],"x",image1.size[1])
            print("After resize - image2:",image2.size[0],"x",image2.size[1])
            if(image1.size == image2.size):
                img_size_op = {key:True}
            else:
                img_size_op = {key:False}
    elapsed_time = round((time.time() - start_time)/60,2)
    logging.info("finished with finding aspect ratio and equalizing the images in "+ str(elapsed_time) +" minutes....OK")
    return img_size_op

# updates on 08-Oct-2020 02:55 AM #
def getFileName(img1, img2, idx, result_list=[]):
    if(len(result_list) > 0 and idx < len(result_list)):
        #base_ref_file = os.path.join(result_list[idx]["base_img_path"], result_list[idx]["image"])
        #runtime_ref_file = os.path.join(result_list[idx]["runtime_img_path"], result_list[idx]["image"])
        base_ref_file = os.path.join(result_list[0]["base_img_path"], result_list[0]["image"])
        runtime_ref_file = os.path.join(result_list[0]["runtime_img_path"], result_list[0]["image"])
        print("result_list:",base_ref_file," ",runtime_ref_file)
        return base_ref_file, runtime_ref_file

    tmp_ref_file1 = os.path.basename(img1).split('_')
    tmp_img_filepart = tmp_ref_file1[0]
    tmp_img_extpart = tmp_ref_file1[1].split('.')[1]

    tmp_img_filename = tmp_img_filepart+"_"+str(idx)+"."+tmp_img_extpart
    base_ref_file = os.path.join(os.path.dirname(img1),tmp_img_filename)
    runtime_ref_file = os.path.join(os.path.dirname(img2),tmp_img_filename)
    return base_ref_file, runtime_ref_file

# updates on 25-Oct-2020 02:45 AM, 05:00 AM, 05:20 AM, 27-Oct-2020 11:15 PM, 13-Nov-2020 02:30 AM #
def find_diff_sized_realtime_images(base_path, runtime_path):
    import logging
    logging.info('getting started with finding diff sized images....')
    start_time = time.time()
    #logging.basicConfig(filename='app.log', filemode='a', format='%(asctime)s - %(levelname)s - %(message)s')
    idx = 1
    img_files_ = {}
    missing_imgs_ = {}
    err_files = []
    comp_result = True
    img1_path = base_path
    img2_path = runtime_path
    base_ref_file = ""
    runtime_ref_file = ""
    base_img_cnt, runtime_img_cnt = get_img_count(img1_path,img2_path)

    if(base_img_cnt != runtime_img_cnt):
        print("image count :: baseline != runtime :",base_img_cnt,"!=",runtime_img_cnt,"....Not OK ")
    else:
        print("image count :: baseline = runtime :",base_img_cnt,"=",runtime_img_cnt)
    #while(idx <= base_img_cnt):
    for filename in os.listdir(img1_path):
        base_ref_file = os.path.join(img1_path, filename)
        runtime_ref_file = os.path.join(img2_path, filename)
        if(os.path.exists(base_ref_file) and os.path.exists(runtime_ref_file)):
            try:
                image1 = Image.open(base_ref_file)
                image2 = Image.open(runtime_ref_file)
            except:
                err_files.append("image "+str(os.path.basename(base_ref_file))+" read error:"+ str(sys.exc_info()[0])) 
                img_files_[os.path.basename(base_ref_file)] = False
                comp_result = False
                logging.error("image read error:", str(sys.exc_info()[0]))
                continue

            img_files_[os.path.basename(base_ref_file)] = True
        elif(os.path.exists(base_ref_file) is False and os.path.exists(runtime_ref_file)):
            missing_imgs_["B_"+os.path.basename(base_ref_file)] = False
            continue
        elif(os.path.exists(runtime_ref_file) is False and os.path.exists(base_ref_file)):            
            missing_imgs_["R_"+os.path.basename(runtime_ref_file)]  = False
            continue

        if(image1.size != image2.size):
            print(base_ref_file, "  size !=  ", runtime_ref_file,"-->",image1.size,"!=",image2.size)
            comp_result = False
            img_files_[os.path.basename(base_ref_file)]  = False
        
    print(comp_result,img_files_,missing_imgs_)
    elapsed_time = round((time.time() - start_time)/60,2)
    logging.info("diff sized images count :"+str(len(img_files_)))
    logging.info("done with finding diff sized images in "+ str(elapsed_time) +" minutes....OK")
    print("diff sized images count :"+str(len(img_files_)))
    print("done with finding diff sized images in "+ str(elapsed_time) +" minutes....OK")

    return comp_result, img_files_, missing_imgs_


# updates on 24-Oct-2020 11:00 PM, 27-Oct-2020 03:10 AM, 11:15 PM #
def comp_img_size(img1, img2, realtime):
    #logging.info('getting started with finding diff sized images....')
    #start_time = time.time()
    comp_result, diff_img_files_, missing_imgs_ =  find_diff_sized_realtime_images(img1,img2)
    #elapsed_time = round((time.time() - start_time)/60,2)
    #logging.info("finished finding diff sized images in "+ str(elapsed_time) +" minutes....OK")
    return comp_result, diff_img_files_, missing_imgs_

   
 # updates on 08-Oct-2020 02:20 AM, 27-Oct-2020 03:05 AM, 27-Oct-2020 11:15 PM, 12-Nov-2020 02:15 AM #
def preprocess_images(dt, result_list, preprocessing_needed=True):
    logging.info('starting the preprocess operation....')
    start_time = time.time()
    realtime = "true"
    baseline_path = str(dt["compare_args"]["workspace_base"])
    runtime_path = str(dt["compare_args"]["workspace_runtime"])
    mask_area = ""
    mask_area_excl = ""
    if(len(result_list) > 0):
        tmp_img1 = os.path.join(result_list[0]["base_img_path"],result_list[0]["image"])
        tmp_img2 =  os.path.join(result_list[0]["runtime_img_path"],result_list[0]["image"])
        missing_imgs_ = {}
        elapsed_time = round((time.time() - start_time)/60,2)
        logging.info("finished the preprocess operation in "+ str(elapsed_time) +" minutes....OK")
        return tmp_img1, tmp_img2, missing_imgs_
    
    if(preprocessing_needed == False):
        elapsed_time = round((time.time() - start_time)/60,2)
        logging.info("finished the preprocess operation in "+ str(elapsed_time) +" minutes....OK")
        return baseline_path, runtime_path, dt["compare_args"]["missing_imgs"]
    
    if(str(dt["compare_args"]["mask_region"]) != ""):
        mask_area = str(dt["compare_args"]["mask_region"])
    elif(str(dt["compare_args"]["mask_region_excluding"]) != ""):
        mask_area_excl = str(dt["compare_args"]["mask_region_excluding"])
    debug = str(dt["compare_args"]["intermediate_output"])

    #base_cnt, runtime_cnt = get_img_count(img1,img2,realtime)
    img_size_comp_result, diff_img_files_, missing_imgs_  = comp_img_size(baseline_path,runtime_path,realtime)
    if(img_size_comp_result is False):
        #runtime_path = os.path.dirname(img2)
        #runtime_path = runtime_path
        aspect_ratio_cfg = str(dt["compare_args"]["aspect_ratio_required"])
        #tmp_img1, tmp_img2 = getTempFileNames(img1, img2, runtime_path)
        #copyFiles(diff_img_files_, baseline_path, runtime_path, tmp_img1, tmp_img2)

        if(aspect_ratio_cfg.lower() == "false"):
            downsize_op = downsizeAndEqualizeImages(baseline_path,runtime_path,diff_img_files_)
        else:
            print("diff_img_files:",diff_img_files_)
            downsize_op = find_aspect_ratio_and_equalize(baseline_path, runtime_path, diff_img_files_)
    #else:
    #    tmp_img1 = img1
    #    tmp_img2 = img2

    if(mask_area != ""):
        print("masking the defined region....",mask_area)
        apply_mask_parallel(baseline_path, runtime_path, mask_area, realtime, debug)
    elif(mask_area_excl != ""):
        print("masking excluding the defined region....",mask_area_excl)
        apply_mask_excluding_region_parallel( baseline_path, runtime_path, mask_area_excl, realtime, debug)
    
    elapsed_time = round((time.time() - start_time)/60,2)
    logging.info("finished the preprocess operation in "+ str(elapsed_time) +" minutes....OK")
    return baseline_path, runtime_path, missing_imgs_



# Updates on : 28-Sep-2020 03:45 PM, 02-Oct-2020 11:10 PM, 03-Oct-2020 01:05 AM, 07-Oct-2020 03:10 PM, 18-Oct-2020 03:45 AM, 13-Nov-2020 02:20 AM  #
def determine_match_outcome(comp_algo, original_score, eval_operator, dt, exp_score_type="int"):
    if(exp_score_type == "int"):
        algo_exp_score = int(get_algo_expected_score(comp_algo,dt))
    else:
        algo_exp_score = float(get_algo_expected_score(comp_algo,dt))

    print("baseline score :",algo_exp_score)
    print("runtime score  :",original_score)
    act_score = 723, 734
    exp_score = 720, 735
    0.98 >= 0.95
    2 <= 1
    2 <= 1
    723 >= 720 and 723 <= 735
    734 >= 720 and 734 <= 735
    744 >= 720 and 744 <= 735
    719 >= 720 and 719 <= 735

    algo_perf_result_ = False
    if(eval_operator == ">="):
        algo_perf_result_  = bool(original_score >= algo_exp_score)
    elif(eval_operator == "<="):
        algo_perf_result_  = bool(original_score <= algo_exp_score)
    elif(eval_operator == ">"):
        algo_perf_result_  = bool(original_score > algo_exp_score)
    elif(eval_operator == "<"):
        algo_perf_result_  = bool(original_score < algo_exp_score)
    elif(eval_operator == "=="):
        algo_perf_result_  = bool(original_score == algo_exp_score)
    elif(eval_operator == "!="):
        algo_perf_result_  = bool(original_score != algo_exp_score)
    else:
        algo_perf_result_ = False    
    return algo_perf_result_
    
# Updates on : 30-Sep-2020 02:35 AM, 02-Oct-2020 06:15 PM, 02-Nov-2020 03:00 AM, 04:00 AM, 08:00 AM, 01:25 PM, 03-Nov-2020 04:00 AM, 04-Nov-2020 03:10 AM, 10:15 PM, 07-Nov-2020 02:15 AM, 04:25 PM, 11:50 PM, 08-Nov-2020 03:10 AM, 09-Nov-2020 04:00 AM, 14-Nov-2020 02:50 PM #
def determine_match_within_range(comp_algo, imgfile, dt):
    #algo_exp_score = get_algo_expected_score(comp_algo,dt)
    comp_result = False
    BF_algo_net_result = False
    base2base_kp = ""
    baseline_kp = dt["compare_args"]["kp_1"]
    kp = dt["compare_args"]["kp_2"]
    gp = dt["compare_args"]["good_points"]
    gpp = dt["compare_args"]["goodpoints_percent"]
    
    # read from BF baseline json, if present. If not, read from the data object being populated at the time of algo operation
    BF_algo_read_result, BF_base, _is_origin_data_object = read_BF_algo_baseline(dt)
    
    # if the BF-base is False, there's no baseline json available for BF algo
    if not BF_algo_read_result:
        print("Img file:{0} not found in the baseline json or error reading it".format(imgfile))
        return "", "", "", False, dt, "unable to read BRISK-FLANN baseline data or empty records"
    
    # loop through for the supplied image file. Retrieve the corresponding json node details, if the image file is present 
    imgfound, tmp_BF_base, erratic_img, dt = find_image_in_BF_baseline_recs(imgfile, BF_base, dt, False)
     
    # when the specified image is not found in the json file and the previous search scope of the image not happened in the data object, this must be a new image and should be added to the new image buffer in BF_baseline_data_model
    if not imgfound and not _is_origin_data_object:
        n = 0
        BF_resJson = json.dumps([o.dump() for o in dt["compare_args"]["BF_algo_baseline"]])
        BF_base = json.loads(BF_resJson)
        imgfound, tmp_BF_base, erratic_img, dt = find_image_in_BF_baseline_recs(imgfile, BF_base, dt, True)


    # get either the min-max range or the absolute value for the specified image - for the baseline
    base_min_kp, base_max_kp, kp_min_max_truthy, base_range_kp = get_minmax_BF_algo(imgfile, tmp_BF_base["confirmed_baseline_kp"], False)
    base_min_gp, base_max_gp, gp_min_max_truthy, base_range_gp = get_minmax_BF_algo(imgfile, tmp_BF_base["confirmed_good_points"], False)
    base_min_gpp, base_max_gpp, gpp_min_max_truthy, base_range_gpp = get_minmax_BF_algo(imgfile, tmp_BF_base["confirmed_good_points_percent"], True )

    if not imgfound:
        print("Img file {0} not found in the BF baseline".format(imgfile))
        #return base_min_kp, base_max_kp, base_min_gp, base_max_gp, base_min_gpp, base_max_gpp, False, "Img file not found in the BF baseline"
        return base_range_kp, base_range_gp, base_range_gpp, False, dt, "Img file not found in the BF baseline" 
    
    if erratic_img:
        print("Img file {0}. Zero keypoint error".format(imgfile))
        print("captured baseline kp --->",BF_base[0]["confirmed_baseline_kp"])
        return base_range_kp, base_range_gp, base_range_gpp, False, dt, "Zero keypoint error"
    
    # if the confirmed keypoints variance is empty, the baseline values remain unverified and effectively this means there's no authentic baseline
    conf_kp_variance = str(tmp_BF_base["confirmed_kp_variance"])
    if(conf_kp_variance == ""):
        print("FAILURE :: Img file :{0} --> confirmed keypoints variance empty. Baseline min-max and the runtime score :[{1}],[{2}]".format(imgfile,base_range_kp,kp))
        return base_range_kp, base_range_gp, base_range_gpp, False, dt, "kp variance empty"

    # perform a set of comparisons of the baseline with the actual score in the corresponding category
    kp_algo_perf_result_, runtime_conf_kp_variance, msg = evaluate_BF_runtime_score(imgfile, baseline_kp, kp, base_max_kp, kp_min_max_truthy, "kp", "", base_range_kp, base_min_kp, conf_kp_variance, True)
    dt["compare_args"]["conf_kp_variance"] = conf_kp_variance
    dt["compare_args"]["runtime_conf_kp_variance"] = runtime_conf_kp_variance
    gp_algo_perf_result_, runtime_conf_kp_variance, msg = evaluate_BF_runtime_score(imgfile, baseline_kp, gp, base_max_gp, gp_min_max_truthy, "gp", msg, base_range_gp, base_min_gp, conf_kp_variance, False)
    gpp_algo_perf_result_, runtime_conf_kp_variance, msg = evaluate_BF_runtime_score(imgfile, baseline_kp, gpp, base_max_gpp, gpp_min_max_truthy, "gpp", msg, base_range_gpp, base_min_gpp, conf_kp_variance, False)
    if str(dt["compare_args"]["BRISK_FLANN_parametric"]["BRISK_FLANN_gp_gpp_check_enabled"]).lower() == "true":
        BF_algo_net_result = bool(kp_algo_perf_result_ and gp_algo_perf_result_ and gpp_algo_perf_result_)
    elif str(dt["compare_args"]["BRISK_FLANN_parametric"]["BRISK_FLANN_gp_gpp_check_enabled"]).lower() == "false":
        msg = msg + "," + "[gp-gpp check disabled]"
        BF_algo_net_result = bool(kp_algo_perf_result_)
    
    if BF_algo_net_result == False:
        prev_base_range_kp = base_range_kp
        comp_result, base2base_kp =  compare_current_prev_baseline_and_update(imgfile, prev_base_range_kp, baseline_kp, dt)
        if(comp_result == True):
            BF_basetobase_comp_data_model.basetobase_kp_update_list.append(BF_basetobase_comp_data_model(imgfile, comp_result, base2base_kp))    
    return base_range_kp, base_range_gp, base_range_gpp, BF_algo_net_result, dt, msg

# created on 02-Nov-2020 02:20 AM, 13-Nov-2020 02:10 AM #
def get_minmax_BF_algo(imgfile, BF_base_rec, do_roundoff):
    base_score = str(BF_base_rec)
    base_score_min = ""
    base_score_max = ""
    base_score_range = ""
    min_max_truthy = False
    if("-" in base_score):
        base_score_min = str(BF_base_rec).split("-")[0]
        base_score_max = str(BF_base_rec).split("-")[1]
        if(do_roundoff == True):
            base_score_min, base_score_max = roundoff_decimal(base_score_max, base_score_min)
        base_score_range = base_score_min+"-"+base_score_max
        min_max_truthy = True
        logging.info("Img file : {0}. BRISK-FLANN baseline range {1}-{2}:".format(imgfile, base_score_min, base_score_max))
    else:
        base_score_min = ""
        base_score_max = str(BF_base_rec)
        if(do_roundoff == True):
            _, base_score_max = roundoff_decimal(base_score_max, "")
        base_score_range = base_score_max
        min_max_truthy = False
        logging.info("Img file : {0}. BRISK-FLANN baseline value:{1}".format(imgfile, base_score_max))
    print("min-max info:",base_score_min, base_score_max, min_max_truthy, base_score_range)
    return base_score_min, base_score_max, min_max_truthy, base_score_range


# created on 06-Nov-2020 12:40 AM to 03:25 AM #
def compare_current_prev_baseline_and_update(imgfile, prev_base_kp, current_kp, dt):
    base_score = str(prev_base_kp)
    base_score_min = ""
    base_score_max = ""
    base_score_range = ""
    _is_update_done = False
    
    if(str(dt["compare_args"]["BRISK_FLANN_parametric"]["BRISK_FLANN_baseline_metrics_auto_update(disabled)"]).lower() == "false"):
        return False, prev_base_kp
    if str(prev_base_kp) == "0" or str(prev_base_kp) == "" :
        return False, prev_base_kp

    if("-" in base_score):
        base_score_min = str(prev_base_kp).split("-")[0]
        base_score_max = str(prev_base_kp).split("-")[1]
        
        if(current_kp >= base_score_min and current_kp <= base_score_max):
            base_score_range = prev_base_kp
            return False, base_score_range
        base_score_range = current_kp
        logging.info("{0} --> prev baseline kp : {1} updated with {2}".format(imgfile, prev_base_kp, current_kp))
        _is_update_done = True
        return _is_update_done, base_score_range
    else:
        base_score_min = ""
        base_score_max = str(prev_base_kp)
        if(base_score_max == current_kp):
            base_score_range = prev_base_kp
            return False, base_score_range
        if(base_score_max != current_kp):
            _is_update_done = True
            base_score_range = current_kp
            logging.info("{0} --> prev baseline kp : {1} updated with {2}".format(imgfile, prev_base_kp, current_kp))
    return _is_update_done, base_score_range


# created on 04-Nov-2020 09:00 PM #
# updates on 05-Nov-2020 02:30 AM, 14-Nov-2020 02:50 PM #
def find_image_in_BF_baseline_recs(imgfile, BF_base, dt, _shouldbuffer_newimgs):
    n=0
    tmp_BF_base = None
    imgfound = False
    erratic_img = False
    while(n < len(BF_base)):
        if(BF_base[n]["image"] == imgfile):
            imgfound = True
            tmp_BF_base = BF_base[n]

            # if the specified image has been newly added, this needs to be added to baseline json. Hence, it's buffered here for later use
            if _shouldbuffer_newimgs == True:
                dt = buffer_newimgs(imgfile, dt)

            if(BF_base[n]["confirmed_baseline_kp"] == "0" or BF_base[n]["confirmed_runtime_kp"] == "0"):
                erratic_img = True
            break
        n = n + 1
    return imgfound, tmp_BF_base, erratic_img, dt


# created on 04-Nov-2020 11:10 PM #
# updates on 05-Nov-2020 02:30 AM #
def buffer_newimgs(imgfile, dt):
    baseline_kp = dt["compare_args"]["kp_1"]
    kp = dt["compare_args"]["kp_2"]
    gp = dt["compare_args"]["good_points"]
    gpp = dt["compare_args"]["goodpoints_percent"]
    cap_kp_variance = baseline_kp - kp
    # uncomment the below line, when the need arises related to captured and baseline and runtime paths - 14-Nov-2020 2:25 AM
    #BF_base_data_model.newimgs_baseline_buffer.append(BF_base_data_model(os.path.basename(imgfile), "", "", str(baseline_kp), str(kp), str(gp), str(gpp),str(cap_kp_variance),str(baseline_kp), str(kp), str(gp), str(gpp),"","new img -> confirmed_kp_variance - manual update needed"))
    BF_base_data_model.newimgs_baseline_buffer.append(BF_base_data_model(os.path.basename(imgfile), str(baseline_kp), str(kp), str(gp), str(gpp),"","new img:est.kp_vari:"+str(cap_kp_variance)+".Need conf."))
    print(imgfile + "-> new image buffered for BF algo baselining")
    logging.info(imgfile + "-> new image buffered for BF algo baselining")
    dt["compare_args"]["newimgs_baselinebuffer"] = BF_base_data_model.newimgs_baseline_buffer
    return dt

# created on 03-Nov-2020 02:10 AM #
def roundoff_decimal(score_max, score_min=""):
    min_gpp=""
    max_gpp=""
    if score_min != "":
        min_gpp = float(score_min)
        min_gpp = round(min_gpp,2)
    max_gpp = float(score_max)
    max_gpp = round(max_gpp,2)
    return str(min_gpp), str(max_gpp)

# created on 02-Nov-2020 02:20 AM #
# updates on 02-Nov-2020 04:00 AM, 05-Nov-2020 01:20 AM, 07-Nov-2020 02:05 AM, 03:15 PM, 11:50 PM, 08-Nov-2020 03:10 AM, 03:10 PM #
def evaluate_BF_runtime_score(imgfile, baseline_kp, score, max, min_max_truthy, eval_type, eval_result_msg, base_score_range, min="0", conf_kp_variance="-zzz", _is_kp_comparison=True):
    algo_perf_result_ = False
    runtime_conf_kp_variance = ""
    msg = eval_result_msg

    if(conf_kp_variance != "-zzz" and _is_kp_comparison == True):
        runtime_conf_kp_variance = int(baseline_kp) - int(score)
        runtime_conf_kp_variance = str(runtime_conf_kp_variance)
        print("baseline confirmed_kp_variance:",conf_kp_variance,"runtime confirmed_kp_variance:",runtime_conf_kp_variance)
        if(runtime_conf_kp_variance != conf_kp_variance):
            print("{0} --> kp_variance match : FAIL. ".format(imgfile))
            print("baseline conf_kp_variance <> runtime conf_kp_variance : {0} <> {1}".format(conf_kp_variance, runtime_conf_kp_variance))
            msg = "[kp:FAILED-->"
            msg = msg + "{0}<>{1}]".format(conf_kp_variance, runtime_conf_kp_variance)
            return False, runtime_conf_kp_variance, msg
        elif runtime_conf_kp_variance == conf_kp_variance:
            print("{0} --> kp_variance match : PASS. ".format(imgfile))
            print("Baseline kp - runtime kp - kp_variance :[{0}] - [{1}] = [{2}]".format(baseline_kp, score, conf_kp_variance))
            msg = "[kp:PASSED],"
            return True, runtime_conf_kp_variance, msg
        

    if min_max_truthy == True:
        if(eval_type == "gp"):
            score = int(score)
            min = int(min)
            max = int(max)
        elif(eval_type == "gpp"):
            score = float(score)
            min = float(min)
            max = float(max)

        if(score >= min and score <= max):
            algo_perf_result_ = True
            print("{0} --> BRISK-FLANN algo match : PASS".format(imgfile))
            print("min, max and actual score :{0}, {1}, {2}".format(min, max, score))
            if(eval_type == "gp"):
                msg = msg + "[gp:PASSED],"
            elif(eval_type == "gpp"):
                msg = msg + "[gpp:PASSED]"
        else:
            print("{0} --> BRISK-FLANN algo match : FAIL".format(imgfile))
            print("min, max and actual score :{0}, {1}, {2}".format(min, max, score))
            if(eval_type == "gp"):
                msg = msg + "[gp:FAILED-->"
                msg = msg + "{0}<>{1}],".format(base_score_range, score)
            elif(eval_type == "gpp"):
                msg = msg + "[gpp:FAILED-->"
                msg = msg + "{0}<>{1}]".format(base_score_range, score)
            
    if min_max_truthy == False:
        if(str(score) == str(max)):
            algo_perf_result_ = True
            print("Img file : {0} --> BRISK-FLANN algo match : PASS".format(imgfile))
            print("baseline and actual score :{0}={1}".format(max, score))
            if(eval_type == "gp"):
                msg = msg + "[gp:PASSED],"
            elif(eval_type == "gpp"):
                msg = msg + "[gpp:PASSED]"
        else:
            print("Img file : {0} --> BRISK-FLANN algo match : FAIL".format(imgfile))
            print("baseline and actual score :{0}<>{1}".format(max, score))
            if(eval_type == "gp"):
                msg = msg + "[gp:FAILED-->"
                msg = msg + "{0}<>{1}],".format(base_score_range, score)
            elif(eval_type == "gpp"):
                msg = msg + "[gpp:FAILED-->"
                msg = msg + "{0}<>{1}]".format(base_score_range, score)

    return algo_perf_result_, runtime_conf_kp_variance, msg



def read_BF_algo_result(file):
    #f = open(file, "a")
    f = open(file, "r")
    return f.read()
    #print("BF_algo_baseline_read:",f.read())

# created on 01-Nov-2020 10:20 PM #
# updates on 02-Nov-2020 11:45 PM, 04-Nov-2020 02:45 AM, 05:50 PM, 08:50 PM, 11:30 PM, 05-Nov-2020 02:15 AM, 11-Nov-2020 12:30 AM #
def read_BF_algo_baseline(dt):
    tmp_obj = None
    _is_origin_data_object  = False
    base_file = dt["compare_args"]["BRISK_FLANN_parametric"]["BRISK_FLANN_parametric_baseline"]
     # if the data object is empty and the baseline json is not present, then return false with empty object
    if len(str(dt["compare_args"]["BF_algo_baseline"])) <= 0 and not os.path.exists(base_file) :
        return False, tmp_obj, _is_origin_data_object
    
    if(os.path.exists(base_file)):
        BF_algo_base = readJson_plain(base_file)
        BF_algo_base = json.dumps(BF_algo_base)
        BF_algo_base = str(BF_algo_base)
        BF_res_obj_json = json.loads(BF_algo_base)
        return True, BF_res_obj_json, _is_origin_data_object
    
    # if the baseline json is not present, then read the contents from the data object
    BF_resJson = json.dumps([o.dump() for o in dt["compare_args"]["BF_algo_baseline"]])
    BF_res_obj_json = json.loads(BF_resJson)
    _is_origin_data_object = True
 
    return True, BF_res_obj_json, _is_origin_data_object
    

# Updates on 02-Oct-2020 06:30 PM, 03-Oct-2020 07:45 PM #
def get_algo_expected_score(algo,dt):
    algos = dt["compare_args"]["similarity"]
    exp_score = ""
    idx = 0
    for i in algos:
        if(str(list(i.values())[0])==algo):
            exp_score = str(list(i.values())[1])
    if(exp_score==""):
        print("algo : ",algo ," was not found...selecting default : SSI")
        for i in algos:
            if(str(list(i.values())[0])==comp_algos.ssi):
                exp_score = str(list(i.values())[1])
    return exp_score


def resize_pad(image, width,height):
    # read image
    img = cv2.imread(image)
    ht, wd, cc= img.shape
    print("img size: h-",ht," w-",wd)

    # create new image of desired size and color (blue) for padding
    #ww = 1920
    ##hh = 1080
    #(ww,hh) = pygui.size()
    ww = width
    hh = height

    color = (255,0,0)
    result = np.full((hh,ww,cc), color, dtype=np.uint8)

    # compute center offset
    xx = (ww - wd) // 2
    yy = (hh - ht) // 2
    print("width:::::",ww)
    print("height::",hh)
    # copy img image into center of result image
    result[yy:yy+ht, xx:xx+wd] = img

    # view result
    cv2.imshow("result", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

     # save result
    cv2.imwrite("D:/Automation/DW/SDV/src/test-inputs/images/login/realtime/chrome/baseline/resized-pad-1.png", result)




#resiz("D:/Automation/DW/SDV/src/test-inputs/images/login/realtime/chrome/baseline/body_1.png",1200,600)
#resize_pad('D:/Automation/DW/SDV/src/test-inputs/images/login/realtime/chrome/baseline/resized-1.png',448,44)
#file1="D:/Automation/DW/SDV/src/test-inputs/images/login/realtime/chrome/baseline/body_1.png"
#file2="D:/Automation/DW/SDV/src/test-inputs/images/login/realtime/chrome/baseline/body-resized-1.png"

#file1 = "D:/Automation/DW/SDV/src/test-inputs/images/login/realtime/chrome/baseline/104244__B.png"
#file2 = "D:/Automation/DW/SDV/src/test-inputs/images/login/realtime/chrome/baseline/104244__A.png"
#SSI_Compare(file1,file2)



'''
def find_diff_sized_nonrealtime_images(img1, img2):
    img_files_ = {}
    missing_imgs_ = {}
    image1 = None
    image2 = None
    comp_result = True
    images_exist = False
    if(os.path.exists(img1) and os.path.exists(img2)):
        image1 = Image.open(img1)
        image2 = Image.open(img2)
        images_exist = True
        img_files_[os.path.basename(img1)]  = True
    elif(os.path.exists(img1) is False and os.path.exists(img2)):
        missing_imgs_["B_"+os.path.basename(img1)]  = False
    elif(os.path.exists(img1) is False and os.path.exists(img2)):            
        missing_imgs_["R_"+os.path.basename(img2)]  = False

    if(images_exist):
        if(image1.size != image2.size):  
            print(img1, "  size !=  ", img2,"-->",image1.size,"!=",image2.size)
            comp_result = False
            img_files_[os.path.basename(img1)]  = False
    return comp_result, img_files_, missing_imgs_


def copyFiles(img_files_, img1, img2, tmp_img1, tmp_img2):
    #tmp_ref_file1 = os.path.basename(img1).split('_')
    #idx = int(tmp_ref_file1[1].split('.')[0])
    img1_path = os.path.dirname(img1)
    img2_path = os.path.dirname(img2)
    tmp_img1_path = os.path.dirname(tmp_img1)
    tmp_img2_path = os.path.dirname(tmp_img2)
    for key in img_files_:
        print(key, '->', img_files_[key])
        #if(img_files_[key] is False):
        img1_fname = os.path.join(img1_path, key)
        img2_fname = os.path.join(img2_path, key)
        tmp_img1_fname = os.path.join(tmp_img1_path, key)
        tmp_img2_fname = os.path.join(tmp_img2_path, key)
        copyfile(img1_fname, tmp_img1_fname)            
        copyfile(img2_fname, tmp_img2_fname)            
'''