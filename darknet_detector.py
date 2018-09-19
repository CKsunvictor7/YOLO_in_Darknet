#!python3
"""
Python 3 wrapper for identifying objects in images

Requires DLL compilation

Both the GPU and no-GPU version should be compiled; the no-GPU version should be renamed "yolo_cpp_dll_nogpu.dll".

On a GPU system, you can force CPU evaluation by any of:

- Set global variable DARKNET_FORCE_CPU to True
- Set environment variable CUDA_VISIBLE_DEVICES to -1
- Set environment variable "FORCE_CPU" to "true"


To use, either run performDetect() after import, or modify the end of this file.

See the docstring of performDetect() for parameters.

Directly viewing or returning bounding-boxed images requires scikit-image to be installed (`pip install scikit-image`)


Original *nix 2.7: https://github.com/pjreddie/darknet/blob/0f110834f4e18b30d5f101bf8f1724c34b7b83db/python/darknet.py
Windows Python 2.7 version: https://github.com/AlexeyAB/darknet/blob/fc496d52bf22a0bb257300d3c79be9cd80e722cb/build/darknet/x64/darknet.py

@author: Philip Kahn
@date: 20180503
"""
#pylint: disable=R, W0401, W0614, W0703
from ctypes import *
import math
import random
import os
import shutil
import cv2
import time
from PIL import Image, ImageDraw, ImageFont


def sample(probs):
    s = sum(probs)
    probs = [a/s for a in probs]
    r = random.uniform(0, 1)
    for i in range(len(probs)):
        r = r - probs[i]
        if r <= 0:
            return i
    return len(probs)-1

def c_array(ctype, values):
    arr = (ctype*len(values))()
    arr[:] = values
    return arr

class BOX(Structure):
    _fields_ = [("x", c_float),
                ("y", c_float),
                ("w", c_float),
                ("h", c_float)]

class DETECTION(Structure):
    _fields_ = [("bbox", BOX),
                ("classes", c_int),
                ("prob", POINTER(c_float)),
                ("mask", POINTER(c_float)),
                ("objectness", c_float),
                ("sort_class", c_int)]


class IMAGE(Structure):
    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("data", POINTER(c_float))]

class METADATA(Structure):
    _fields_ = [("classes", c_int),
                ("names", POINTER(c_char_p))]



#lib = CDLL("/mnt2/dc_projects/AI_DC_YOLO/libdarknet.so", RTLD_GLOBAL)
#lib = CDLL("darknet.so", RTLD_GLOBAL)
hasGPU = True
if os.name == "nt":
    cwd = os.path.dirname(__file__)
    os.environ['PATH'] = cwd + ';' + os.environ['PATH']
    winGPUdll = os.path.join(cwd, "yolo_cpp_dll.dll")
    winNoGPUdll = os.path.join(cwd, "yolo_cpp_dll_nogpu.dll")
    envKeys = list()
    for k, v in os.environ.items():
        envKeys.append(k)
    try:
        try:
            tmp = os.environ["FORCE_CPU"].lower()
            if tmp in ["1", "true", "yes", "on"]:
                raise ValueError("ForceCPU")
            else:
                print("Flag value '"+tmp+"' not forcing CPU mode")
        except KeyError:
            # We never set the flag
            if 'CUDA_VISIBLE_DEVICES' in envKeys:
                if int(os.environ['CUDA_VISIBLE_DEVICES']) < 0:
                    raise ValueError("ForceCPU")
            try:
                global DARKNET_FORCE_CPU
                if DARKNET_FORCE_CPU:
                    raise ValueError("ForceCPU")
            except NameError:
                pass
            # print(os.environ.keys())
            # print("FORCE_CPU flag undefined, proceeding with GPU")
        if not os.path.exists(winGPUdll):
            raise ValueError("NoDLL")
        lib = CDLL(winGPUdll, RTLD_GLOBAL)
    except (KeyError, ValueError):
        hasGPU = False
        if os.path.exists(winNoGPUdll):
            lib = CDLL(winNoGPUdll, RTLD_GLOBAL)
            print("Notice: CPU-only mode")
        else:
            # Try the other way, in case no_gpu was
            # compile but not renamed
            lib = CDLL(winGPUdll, RTLD_GLOBAL)
            print("Environment variables indicated a CPU run, but we didn't find `"+winNoGPUdll+"`. Trying a GPU run anyway.")
else:
    lib = CDLL("./darknet.so", RTLD_GLOBAL)
lib.network_width.argtypes = [c_void_p]
lib.network_width.restype = c_int
lib.network_height.argtypes = [c_void_p]
lib.network_height.restype = c_int

predict = lib.network_predict
predict.argtypes = [c_void_p, POINTER(c_float)]
predict.restype = POINTER(c_float)

if hasGPU:
    set_gpu = lib.cuda_set_device
    set_gpu.argtypes = [c_int]

make_image = lib.make_image
make_image.argtypes = [c_int, c_int, c_int]
make_image.restype = IMAGE

get_network_boxes = lib.get_network_boxes
get_network_boxes.argtypes = [c_void_p, c_int, c_int, c_float, c_float, POINTER(c_int), c_int, POINTER(c_int), c_int]
get_network_boxes.restype = POINTER(DETECTION)

make_network_boxes = lib.make_network_boxes
make_network_boxes.argtypes = [c_void_p]
make_network_boxes.restype = POINTER(DETECTION)

free_detections = lib.free_detections
free_detections.argtypes = [POINTER(DETECTION), c_int]

free_ptrs = lib.free_ptrs
free_ptrs.argtypes = [POINTER(c_void_p), c_int]

network_predict = lib.network_predict
network_predict.argtypes = [c_void_p, POINTER(c_float)]

reset_rnn = lib.reset_rnn
reset_rnn.argtypes = [c_void_p]

load_net = lib.load_network
load_net.argtypes = [c_char_p, c_char_p, c_int]
load_net.restype = c_void_p

load_net_custom = lib.load_network_custom
load_net_custom.argtypes = [c_char_p, c_char_p, c_int, c_int]
load_net_custom.restype = c_void_p

do_nms_obj = lib.do_nms_obj
do_nms_obj.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

do_nms_sort = lib.do_nms_sort
do_nms_sort.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

free_image = lib.free_image
free_image.argtypes = [IMAGE]

letterbox_image = lib.letterbox_image
letterbox_image.argtypes = [IMAGE, c_int, c_int]
letterbox_image.restype = IMAGE

load_meta = lib.get_metadata
lib.get_metadata.argtypes = [c_char_p]
lib.get_metadata.restype = METADATA

load_image = lib.load_image_color
load_image.argtypes = [c_char_p, c_int, c_int]
load_image.restype = IMAGE

rgbgr_image = lib.rgbgr_image
rgbgr_image.argtypes = [IMAGE]

predict_image = lib.network_predict_image
predict_image.argtypes = [c_void_p, IMAGE]
predict_image.restype = POINTER(c_float)


def array_to_image(arr):
    import numpy as np
    # need to return old values to avoid python freeing memory
    arr = arr.transpose(2,0,1)
    c = arr.shape[0]
    h = arr.shape[1]
    w = arr.shape[2]
    arr = np.ascontiguousarray(arr.flat, dtype=np.float32) / 255.0
    data = arr.ctypes.data_as(POINTER(c_float))
    im = IMAGE(w,h,c,data)
    return im, arr


def classify(net, meta, im):
    out = predict_image(net, im)
    res = []
    for i in range(meta.classes):
        if altNames is None:
            nameTag = meta.names[i]
        else:
            nameTag = altNames[i]
        res.append((nameTag, out[i]))
    res = sorted(res, key=lambda x: -x[1])
    return res


def detect(net, meta, image, thresh=.5, hier_thresh=.5, nms=.45, debug= False):
    """
    Performs the meat of the detection
    """
    #pylint: disable= C0321
    im = load_image(image, 0, 0)
    #import cv2
    #custom_image_bgr = cv2.imread(image) # use: detect(,,imagePath,)
    #custom_image = cv2.cvtColor(custom_image_bgr, cv2.COLOR_BGR2RGB)
    #custom_image = cv2.resize(custom_image,(lib.network_width(net), lib.network_height(net)), interpolation = cv2.INTER_LINEAR)
    #import scipy.misc
    #custom_image = scipy.misc.imread(image)
    #im, arr = array_to_image(custom_image)		# you should comment line below: free_image(im)
    if debug: print("Loaded image")
    num = c_int(0)
    if debug: print("Assigned num")
    pnum = pointer(num)
    if debug: print("Assigned pnum")
    predict_image(net, im)
    if debug: print("did prediction")
    #dets = get_network_boxes(net, custom_image_bgr.shape[1], custom_image_bgr.shape[0], thresh, hier_thresh, None, 0, pnum, 0) # OpenCV
    dets = get_network_boxes(net, im.w, im.h, thresh, hier_thresh, None, 0, pnum, 0)
    if debug: print("Got dets")
    num = pnum[0]
    if debug: print("got zeroth index of pnum")
    if nms:
        do_nms_sort(dets, num, meta.classes, nms)
    if debug: print("did sort")
    res = []
    if debug: print("about to range")
    for j in range(num):
        if debug: print("Ranging on "+str(j)+" of "+str(num))
        if debug: print("Classes: "+str(meta), meta.classes, meta.names)
        for i in range(meta.classes):
            if debug: print("Class-ranging on "+str(i)+" of "+str(meta.classes)+"= "+str(dets[j].prob[i]))
            if dets[j].prob[i] > 0:
                b = dets[j].bbox
                if altNames is None:
                    nameTag = meta.names[i]
                else:
                    nameTag = altNames[i]
                if debug:
                    print("Got bbox", b)
                    print(nameTag)
                    print(dets[j].prob[i])
                    print((b.x, b.y, b.w, b.h))
                res.append((nameTag, dets[j].prob[i], (b.x, b.y, b.w, b.h)))
    if debug: print("did range")
    res = sorted(res, key=lambda x: -x[1])
    if debug: print("did sort")
    free_image(im)
    if debug: print("freed image")
    free_detections(dets, num)
    if debug: print("freed detections")
    return res


def corner_calculater(rect, img_size):
    """
    :param rect:
    :param img_size: = (h, w)
    :return:
    """
    (x, y, h, w) = rect
    (img_h, img_w) = img_size
    left_up_corner_x = int(x - w/2) if int(x - w/2) > 0 else 0
    left_up_corner_y = int(y - h / 2) if int(y - h / 2) > 0 else 0

    right_bottm_corner_x = int(x + w/2) if int(x + w/2) < img_w else img_w
    right_bottm_corner_y = int(y + h/2) if int(y + h/2) < img_h else img_h

    return (left_up_corner_x, left_up_corner_y), (right_bottm_corner_x, right_bottm_corner_y)


netMain = None
metaMain = None
altNames = None


def performDetect(image_list, thresh= 0.25, configPath = "./cfg/yolov3.cfg", weightPath = "yolov3.weights", metaPath= "./data/coco.data", showImage= True, makeImageOnly = False, initOnly= False):
    """
    Convenience function to handle the detection and returns of objects.

    Displaying bounding boxes requires libraries scikit-image and numpy

    Parameters
    ----------------
    imagePath: str
        Path to the image to evaluate. Raises ValueError if not found

    thresh: float (default= 0.25)
        The detection threshold

    configPath: str
        Path to the configuration file. Raises ValueError if not found

    weightPath: str
        Path to the weights file. Raises ValueError if not found

    metaPath: str
        Path to the data file. Raises ValueError if not found

    showImage: bool (default= True)
        Compute (and show) bounding boxes. Changes return.

    makeImageOnly: bool (default= False)
        If showImage is True, this won't actually *show* the image, but will create the array and return it.

    initOnly: bool (default= False)
        Only initialize globals. Don't actually run a prediction.

    Returns
    ----------------------


    When showImage is False, list of tuples like
        ('obj_label', confidence, (bounding_box_x_px, bounding_box_y_px, bounding_box_width_px, bounding_box_height_px))
        The X and Y coordinates are from the center of the bounding box. Subtract half the width or height to get the lower corner.

    Otherwise, a dict with
        {
            "detections": as above
            "image": a numpy array representing an image, compatible with scikit-image
            "caption": an image caption
        }
    """
    # Import the global variables. This lets us instance Darknet once, then just call performDetect() again without instancing again
    global metaMain, netMain, altNames #pylint: disable=W0603
    assert 0 < thresh < 1, "Threshold should be a float between zero and one (non-inclusive)"
    if not os.path.exists(configPath):
        raise ValueError("Invalid config path `"+os.path.abspath(configPath)+"`")
    if not os.path.exists(weightPath):
        raise ValueError("Invalid weight path `"+os.path.abspath(weightPath)+"`")
    if not os.path.exists(metaPath):
        raise ValueError("Invalid data file path `"+os.path.abspath(metaPath)+"`")
    if netMain is None:
        netMain = load_net_custom(configPath.encode("ascii"), weightPath.encode("ascii"), 0, 1)  # batch size = 1
    if metaMain is None:
        metaMain = load_meta(metaPath.encode("ascii"))
    if altNames is None:
        # In Python 3, the metafile default access craps out on Windows (but not Linux)
        # Read the names file and create a list to feed to detect
        try:
            with open(metaPath) as metaFH:
                metaContents = metaFH.read()
                import re
                match = re.search("names *= *(.*)$", metaContents, re.IGNORECASE | re.MULTILINE)
                if match:
                    result = match.group(1)
                else:
                    result = None
                try:
                    if os.path.exists(result):
                        with open(result) as namesFH:
                            namesList = namesFH.read().strip().split("\n")
                            altNames = [x.strip() for x in namesList]
                except TypeError:
                    pass
        except Exception:
            pass
    if initOnly:
        print("Initialized detector")
        return None
    #if not os.path.exists(imagePath):
    #    raise ValueError("Invalid image path `"+os.path.abspath(imagePath)+"`")
    # Do the detection
    #detections = detect(netMain, metaMain, imagePath, thresh)	# if is used cv2.imread(image)

    # remove previous result imgs in food_samples
    # shutil.rmtree('food_samples', ignore_errors=True)
    # os.mkdir('food_samples')

    for imagePath in image_list:
        print('--------------- Detecting {} ---------------'.format(imagePath))
        s_time = time.time()
        """
        bboxes is a list, each element is a list: 
        ('obj_label', confidence, (bounding_box_x_px, bounding_box_y_px, bounding_box_width_px, bounding_box_height_px)
        """
        bboxes = detect(netMain, metaMain, imagePath.encode("ascii"), thresh)
        print('detection time = ', time.time() - s_time)
        print('nb of bbox = ', len(bboxes))

        with open(os.path.join(eval_config['SAVE_DETECTION_DIR'], str(os.path.splitext(os.path.basename(imagePath))[0]) + '.txt'), 'a') as w:
            for rect in bboxes:
                if rect[1] > eval_config['SAVE_DETECTION_THRESHOLD']:
                    # output format = category confidence box
                    w.write('{} {} {} {} {} {}\n'.format(
                            rect[0], rect[1], rect[2][0], rect[2][1], rect[2][2], rect[2][3]))
                    if eval_config['DETAIL']:
                        print('category:{} confidence:{} (x, y, w, d)=({}, {}, {}, {})'.format(
                            rect[0], rect[1], rect[2][0], rect[2][1], rect[2][2], rect[2][3]))

                if eval_config['SAVE_LABEL'] and rect[1] > eval_config['SAVE_LABEL_THRESHOLD']:
                    with open(os.path.join(eval_config['SAVE_LABEL_DIR'], str(os.path.splitext(os.path.basename(imagePath))[0]) + '.txt'), 'a') as ww:
                        ww.write('{} {} {} {} {} \n'.format(
                            rect[0], rect[2][0], rect[2][1], rect[2][2], rect[2][3]))
                """
                # only draw rect with larger than 0.5
                if eval_config['SAVE_IMG'] and rect[1] > eval_config['SAVE_IMG_THRESHOLD']:
                    left_up_corner, right_bottm_corner = corner_calculater((rect[2][0], rect[2][1], rect[2][2], rect[2][3]),
                                                                           (img.shape[0], img.shape[1]))
                    # print(left_up_corner)
                    # print(right_bottm_corner)
                    # image, up-left, bottom-right, color code, thickness ,
                    cv2.rectangle(img, left_up_corner, right_bottm_corner, (0, 255, 0), 4)
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    text = '{}:{:0.2f}'.format(rect[0], float(rect[1]))
                    # img, text to add/ left_up_corner/ font, font size, color, thickness
                    # due to sometimes left_up_corner's location is bad for Text,
                    # put Text in the middle of bbox
                    tect_location = (left_up_corner[0],
                                     int((left_up_corner[1]+right_bottm_corner[1])/2))
                    cv2.putText(img, text, tect_location, font, 1, (0, 0, 0), 2)
                    cv2.imwrite(os.path.join(eval_config['SAVE_IMG_DIR'], os.path.basename(imagePath)), img)
                """
        """
        if eval_config['SAVE_IMG'] and bboxes:
            # remove the highly overlapped bboxes
            r_bboxs = [bboxes[0]]
            for bbox in bboxes[1:]:
                for r_bbox in r_bboxs:
                    print('bbox = ', bbox)
                    if overlapped_ratio(bbox[2:], r_bbox[2:]) > eval_config['overlapped_ratio_threshold']:
                        break
                    else:
                        r_bboxs.append(bbox)

            vis_path = os.path.join(eval_config['SAVE_IMG_DIR'],
                                    os.path.basename(imagePath))
            img = Image.open(imagePath)
            draw_bbox(img, filename=vis_path, bboxes=r_bboxs)
        """
    """
    if showImage:
        try:
            from skimage import io, draw
            import numpy as np
            image = io.imread(imagePath)
            print("*** "+str(len(detections))+" Results, color coded by confidence ***")
            imcaption = []
            for detection in detections:
                label = detection[0]
                confidence = detection[1]
                pstring = label+": "+str(np.rint(100 * confidence))+"%"
                imcaption.append(pstring)
                print(pstring)
                bounds = detection[2]
                shape = image.shape
                # x = shape[1]
                # xExtent = int(x * bounds[2] / 100)
                # y = shape[0]
                # yExtent = int(y * bounds[3] / 100)
                yExtent = int(bounds[3])
                xEntent = int(bounds[2])
                # Coordinates are around the center
                xCoord = int(bounds[0] - bounds[2]/2)
                yCoord = int(bounds[1] - bounds[3]/2)
                boundingBox = [
                    [xCoord, yCoord],
                    [xCoord, yCoord + yExtent],
                    [xCoord + xEntent, yCoord + yExtent],
                    [xCoord + xEntent, yCoord]
                ]
                # Wiggle it around to make a 3px border
                rr, cc = draw.polygon_perimeter([x[1] for x in boundingBox], [x[0] for x in boundingBox], shape= shape)
                rr2, cc2 = draw.polygon_perimeter([x[1] + 1 for x in boundingBox], [x[0] for x in boundingBox], shape= shape)
                rr3, cc3 = draw.polygon_perimeter([x[1] - 1 for x in boundingBox], [x[0] for x in boundingBox], shape= shape)
                rr4, cc4 = draw.polygon_perimeter([x[1] for x in boundingBox], [x[0] + 1 for x in boundingBox], shape= shape)
                rr5, cc5 = draw.polygon_perimeter([x[1] for x in boundingBox], [x[0] - 1 for x in boundingBox], shape= shape)
                boxColor = (int(255 * (1 - (confidence ** 2))), int(255 * (confidence ** 2)), 0)
                draw.set_color(image, (rr, cc), boxColor, alpha= 0.8)
                draw.set_color(image, (rr2, cc2), boxColor, alpha= 0.8)
                draw.set_color(image, (rr3, cc3), boxColor, alpha= 0.8)
                draw.set_color(image, (rr4, cc4), boxColor, alpha= 0.8)
                draw.set_color(image, (rr5, cc5), boxColor, alpha= 0.8)
            if not makeImageOnly:
                io.imshow(image)
                io.show()
            detections = {
                "detections": detections,
                "image": image,
                "caption": "\n<br/>".join(imcaption)
            }
        except Exception as e:
            print("Unable to show image: "+str(e))
    """
    return None


gpu6_exp11 = {
    'Model_PATH':"/mnt/YOLO_AB_weights/UEC256_exp11_25000.weights",
    ".cfg":"exps/UEC256_exp11.cfg",
    ".data":"exps/UEC256_exp11.data",

}

eval_finc_config = {
    "sample_dir":'/mnt/finc_data/155',
    'DETAIL':False,      # whether print details
    'SAVE_IMG':True,
    'SAVE_IMG_THRESHOLD':0.3,
    'SAVE_IMG_DIR':'/mnt/results/exp11_vis_155',
    'SAVE_DETECTION_DIR':'/mnt/results/exp11_detection_155',
    'SAVE_DETECTION_THRESHOLD':0.1,
    'SAVE_LABEL':True,   # whether output initial info for labeling
    'SAVE_LABEL_DIR':'/mnt/results/exp11_155',
    'SAVE_LABEL_THRESHOLD':0.5,
}

eval_UEC_config = {
    'sample_dir':'',
    'DETAIL':False,
    'SAVE_IMG':True,
    'SAVE_IMG_THRESHOLD':0.3,
    'SAVE_IMG_DIR':'/mnt/UEC_results/exp11_vis',
    'SAVE_DETECTION_DIR':'/mnt/UEC_results/exp11_detection',
    'SAVE_DETECTION_THRESHOLD':0.1,
    'SAVE_LABEL':True,   # whether output initial info for labeling
    'SAVE_LABEL_DIR':'/mnt/UEC_results/exp11',
    'SAVE_LABEL_THRESHOLD':0.5,
}



gpu4_exp11 = {
    'Model_PATH':"/mnt2/models/yolov3/UEC256_exp12_25000.weights",
    ".cfg":"exps/UEC256_exp12.cfg",
    ".data":"exps/UEC_exp12.data",
    "sample_dir":"/mnt2/DB/samples/"
}

gpu7_exp14 = {
    'Model_PATH': "/mnt2/model/exp14_2/exp14_2_7000.weights",
    ".cfg": "exps/exp14_2.cfg",
    ".data": "exps/exp14_2.data",

    'sample_dir':'',    # sample_dir or sample_txt, must be only one active
    'sample_txt':'exps/train_val/exp14_val.txt',
    'DETAIL':False,
    'SAVE_IMG':False,
    'SAVE_IMG_THRESHOLD':0.3,
    'overlapped_ratio_threshold':0.6,
    'SAVE_IMG_DIR':'/mnt2/results/exp14_vis',
    'SAVE_DETECTION_DIR':'/mnt2/results/exp14_detections',
    'SAVE_DETECTION_THRESHOLD':0.1,
    'SAVE_LABEL':False,   # whether output initial info for labeling
    'SAVE_LABEL_DIR':'-',
    'SAVE_LABEL_THRESHOLD':0.5,
}

gpu6_exp13 = {
    'Model_PATH': "/mnt/UEC_backup_AB/UEC256_exp13_35000.weights",
    ".cfg": "exps/UEC256_exp13.cfg",
    ".data": "exps/UEC256_exp13.data",

    'sample_dir':'/mnt/155', #'/mnt/155',    # sample_dir or sample_txt, must be only one active
    'sample_txt':'', #'exps/train_val/exp13_val.txt',
    'SAVE_DETECTION_THRESHOLD':0.2,
    'SAVE_DETECTION_DIR':'/mnt/results/exp13_detection_155',
    'DETAIL':False,
    'SAVE_IMG':False,
    'SAVE_IMG_THRESHOLD':0.3,
    'overlapped_ratio_threshold':0.6,
    'SAVE_IMG_DIR':'/mnt/results/exp13_vis',
    'SAVE_LABEL':False,   # whether output initial info for labeling
    'SAVE_LABEL_DIR':'-',
    'SAVE_LABEL_THRESHOLD':0.5,
}

gpu6_exp15 = {
    'Model_PATH': "/mnt/exp15_backup/exp15_40000.weights",
    ".cfg": "exps/exp15.cfg",
    ".data": "exps/exp15.data",
    'sample_dir':'/mnt/156', #'/mnt/155',    # sample_dir or sample_txt, must be only one active
    'sample_txt':'', #'exps/train_val/exp13_val.txt',
    'SAVE_DETECTION_THRESHOLD':0.2,
    'SAVE_DETECTION_DIR':'/mnt/results/exp15_detection_156',
    'DETAIL':False,
    'SAVE_IMG':False,
    'SAVE_IMG_THRESHOLD':0.3,
    'overlapped_ratio_threshold':0.6,
    'SAVE_IMG_DIR':'-',
    'SAVE_LABEL':False,   # whether output initial info for labeling
    'SAVE_LABEL_DIR':'-',
    'SAVE_LABEL_THRESHOLD':0.5,
}


eval_config = gpu6_exp15


def overlapped_ratio(box1, box2):
    """
    :param area_1 & param area_2: [xmin,  ymin, xmax, ymax]
    :return: overlapped ratio
    """
    print(box1)
    xmin_1, ymin_1, xmax_1, ymax_1 = float(box1[0]), float(box1[1]), float(box1[2]), float(box1[3])
    xmin_2, ymin_2, xmax_2, ymax_2 = float(box2[0]), float(box2[1]), float(box2[2]), float(box2[3])
    overlapped_area = (min(xmax_1, xmax_2) - max(xmin_1, xmin_2))*(min(ymax_1, ymax_2) - max(ymin_1, ymin_2))
    area_1 = (xmax_1 - xmin_1)*(ymax_1 - ymin_1)
    area_2 = (xmax_2 - xmin_2)*(ymax_2 - ymin_2)
    """
        area_1 = [float(area_1[0]), float(area_1[1]), float(area_1[2]), float(area_1[3])]
        area_2 = [float(area_2[0]), float(area_2[1]), float(area_2[2]), float(area_2[3])]
        overlapped_area = (min([area_1[3], area_2[3]]) - max([area_1[1], area_2[1]]))*(min([area_1[2], area_2[2]]) - max([area_1[0], area_2[0]]))
        if overlapped_area > .0:
            overlapped_area_1 = overlapped_area/((area_1[2]-area_1[0])*(area_1[3]-area_1[1]))
            overlapped_area_2 = overlapped_area/((area_2[2]-area_2[0])*(area_2[3]-area_2[1]))
            # print('{:.4f}  {:.4f}'.format(overlapped_area_1, overlapped_area_2))
            return max([overlapped_area_1, overlapped_area_2])
        else:
            return .0
    """

    if overlapped_area > 0:
        return overlapped_area/(area_1 + area_2 - overlapped_area)
    else:
        return 0


def draw_bbox(img, filename, bboxes):
    """
    bboxes = name xmin ymin xmax ymax
    """
    color_list= [(0,255,255), (255,255,0), (255,0,255), (255,255,255),  (0,255,0), (0,0,255)]
    for idx, box in enumerate(bboxes):
        #LU_corner = (float(box[1][0]) - float(box[1][2])/2, float(box[1][1]) - float(box[1][3])/2)
        #RD_corner = (float(box[1][0]) + float(box[1][2])/2, float(box[1][1]) + float(box[1][3])/2)
        LU_corner = (box[1], box[2])
        RD_corner = (box[3], box[4])
        if idx < len(color_list):
            rect_color = color_list[idx]
            font_color = color_list[idx]
        else:
            rect_color = (0, 0, 0)
            font_color = (0, 0, 0)

        xSize, ySize = img.size
        # ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeMono.ttf", 28, encoding="unic")
        fnt = ImageFont.truetype(font="/System/Library/Fonts/SFNSText.ttf", size=min(xSize, ySize) // 10)

        drawimg = ImageDraw.Draw(img)
        drawimg.rectangle((LU_corner, RD_corner),fill=None,outline=rect_color)
        drawimg.text(LU_corner, box[0], fill=font_color, font = fnt)
    #img.show()
    img.save(filename)


def main():
    #server = gpu7_exp14

    # 2 input method
    # 1). listdir
    if eval_config['sample_dir']:
        img_list = [os.path.join(eval_config['sample_dir'], f) for f in os.listdir(eval_config['sample_dir']) if f.endswith('.jpg')]
    # 2). read from .txt
    else:
        img_list = []
        with open(eval_config['sample_txt'], 'r') as r:
            for line in r.readlines():
                img_list.append(line.rstrip())

    performDetect(image_list=img_list, thresh= 0.01, configPath=eval_config['.cfg'], weightPath=eval_config['Model_PATH'],
                  metaPath=eval_config['.data'], showImage= False, makeImageOnly = False, initOnly= False)


if __name__ == "__main__":
    main()
