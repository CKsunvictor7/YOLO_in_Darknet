#!python3
"""
Data Cleaner using YOLOv3 models using YOLO models, based on darknet_detector.py

do data cleaning on all the images in 'raw_data_dir':

1). remove the bad quality images
2). classify images into nonfood, single_food, multiple_food
    and save images, annotations(yolo format or xmin, ymin, xmax, ymax) separately:
    nofood ; single & single_annos ;  multiple & multiple_annos

3). reindex the images
"""
from ctypes import *
import random
import os, shutil
import time
from PIL import Image, ImageDraw, ImageFont
from DB_tools import get_file_list

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


def performDetect(image_list, configPath = "./cfg/yolov3.cfg", weightPath = "yolov3.weights", metaPath= "./data/coco.data", initOnly= False):
    """
    Convenience function to handle the detection and returns of objects.

    Displaying bounding boxes requires libraries scikit-image and numpy

    Parameters
    ----------------
    imagePath: str
        Path to the image to evaluate. Raises ValueError if not found

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
    assert 0 < config['thresh'] < 1, "Threshold should be a float between zero and one (non-inclusive)"
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

    id = config['id_starts_from']

    # make directories
    no_food_dir = os.path.join(config['dst_dir'], 'nofood')
    single_food_dir = os.path.join(config['dst_dir'], 'single')
    single_food_annos_dir = os.path.join(config['dst_dir'], 'single_annos')
    mulitple_food_dir = os.path.join(config['dst_dir'], 'multiple')
    mulitple_food_annos_dir = os.path.join(config['dst_dir'], 'multiple_annos')
    for dir in [no_food_dir, single_food_dir, mulitple_food_dir,
                single_food_annos_dir, mulitple_food_annos_dir]:
        if not os.path.exists(dir):
            print('{} does not exist, create new one'.format(dir))
            os.mkdir(dir)
    print('len(image_list)=', len(image_list))
    for imagePath in image_list:
        print('--------------- Detecting {} ---------------'.format(imagePath))
        s_time = time.time()
        """
        bboxes is a list, each element is a list: 
        ('obj_label', confidence, (bounding_box_x_px, bounding_box_y_px, bounding_box_width_px, bounding_box_height_px)
        """
        bboxes = detect(netMain, metaMain, imagePath.encode("ascii"), config['thresh'])
        print('detection time = ', time.time() - s_time)

        if len(bboxes) == 0:
            print('no food detected@{}'.format(imagePath))
        elif len(bboxes) == 1:
            print('nb of bbox = {} @{}'.format(len(bboxes), imagePath))
            # move image
            shutil.copyfile(imagePath, os.path.join(single_food_dir, '{}{}'.format(id, os.path.splitext(os.path.basename(imagePath))[1])))
            # write annotations
            with open(os.path.join(single_food_annos_dir, '{}.txt'.format(id)), 'w') as w:
                for rect in bboxes:
                    if config['annos_format'] == 'YOLO':
                        w.write('{} {} {} {} {} \n'.format(
                        rect[0], rect[2][0], rect[2][1], rect[2][2], rect[2][3]))
                    elif config['annos_format'] == 'TF':
                        #TODO
                        pass
            id += 1
        else:
            print('nb of bbox = {} @{}'.format(len(bboxes), imagePath))
            # move image
            shutil.copyfile(imagePath, os.path.join(mulitple_food_dir, '{}{}'.format(id, os.path.splitext(os.path.basename(imagePath))[1])))
            # write annotations
            with open(os.path.join(mulitple_food_annos_dir, '{}.txt'.format(id)), 'w') as w:
                for rect in bboxes:
                    if config['annos_format'] == 'YOLO':
                        w.write('{} {} {} {} {} \n'.format(
                            rect[0], rect[2][0], rect[2][1], rect[2][2],
                            rect[2][3]))
                    elif config['annos_format'] == 'TF':
                        # TODO
                        pass
            id += 1
    return None




gpu7_exp14 = {
    'raw_img_dir':'/mnt2/DB/test_samples', # should be a copy from S3 server
    'dst_dir':'/mnt2/DB/clean_DB',
    'Model_PATH': "/mnt2/models/YOLO/exp14_2/exp14_2_7000.weights",
    ".cfg": "exps/exp14_2.cfg",
    ".data": "exps/exp14_2.data",

    'min_w':448, 'min_h':448,
    'thresh':0.25,  # The detection threshold, default = 0.25
    'id_starts_from':10000,
    'annos_format':'YOLO', # YOLO or TF
}

config = gpu7_exp14


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


def bad_quality_remover(dir):
    img_list = get_file_list(dir, extensions=('.jpg', 'jpeg', '.png', '.bmp', '.JPG', 'JPEG', '.PNG', '.BMP'))
    print('len(img_list))=', len(img_list))
    good_img_list = []
    for img_path in img_list:
        img = Image.open(img_path)
        if img.size[0] < config['min_w'] or img.size[1] < config['min_h']:
            print('the quality of {} is not good enough, skip'.format(img_path))
        else:
            good_img_list.append(img_path)

    return good_img_list


def main():
    # 1. read file list & remove the one with bad quality
    print('remove bad quality images, which w < {} than or h < {}'.
          format(config['min_w'], config['min_h']))
    good_img_list = bad_quality_remover(dir=config['raw_img_dir'])
    # 2. do detection
    performDetect(image_list=good_img_list, configPath=config['.cfg'], weightPath=config['Model_PATH'],
                  metaPath=config['.data'], initOnly= False)


if __name__ == "__main__":
    main()
