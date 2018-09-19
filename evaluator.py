"""
evaluate the detection results
by
1) visualization

2) mAP :

3) Acc (do not care about bbox location)

4) Confusion matrix

"""
from PIL import Image, ImageDraw, ImageFont
import os
from YOLODB_tools import get_file_list

UEC_config = {
    'GT_DIR':'-',
    'SAVE_DETECTION_DIR':'/Users/fincep004/Desktop/evaluation/YOLO/exp13_detection_UEC',
    'CONFIDENCE_THRESHOLD':0.3,
    'IMG_DIR':'/Users/fincep004/Documents/UEC256_images',
    'SAVE_BBOX_IMG_PATH':'/Users/fincep004/Desktop/evaluation/YOLO/exp13_vis_UEC',
    'overlapped_ratio_threshold':0.5
}

finc_data_config = {
    'GT_DIR':'-',
    'SAVE_DETECTION_DIR':'/Users/fincep004/Desktop/evaluation/YOLO/exp15_detection_156',
    'CONFIDENCE_THRESHOLD':0.3,
    'IMG_DIR':'/Users/fincep004/Desktop/finc_food_db/156',
    'SAVE_BBOX_IMG_PATH':'/Users/fincep004/Desktop/evaluation/YOLO/exp15_vis_156',
    'overlapped_ratio_threshold':0.5
}

config = finc_data_config


def isDigit(x):
    try:
        float(x)
        return True
    except ValueError:
        return False


def overlapped_ratio(box1, box2):
    """
    :param area_1 & param area_2: [xmin,  ymin, xmax, ymax]
    :return: overlapped ratio
    """
    xmin_1, ymin_1, xmax_1, ymax_1 = float(box1[0]), float(box1[1]), float(box1[2]), float(box1[3])
    xmin_2, ymin_2, xmax_2, ymax_2 = float(box2[0]), float(box2[1]), float(box2[2]), float(box2[3])

    width = min(xmax_1, xmax_2) - max(xmin_1, xmin_2)
    height = min(ymax_1, ymax_2) - max(ymin_1, ymin_2)
    if width < 0 or height < 0:
        return 0

    overlapped_area = width*height
    area_1 = (xmax_1 - xmin_1)*(ymax_1 - ymin_1)
    area_2 = (xmax_2 - xmin_2)*(ymax_2 - ymin_2)

    if overlapped_area > 0:
        return overlapped_area/(area_1 + area_2 - overlapped_area)
    else:
        return 0




def draw_bbox(img, name_to_save, bboxes):
    """
    bboxes = a list of [name xmin ymin xmax ymax]
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
        fnt = ImageFont.truetype(font="/System/Library/Fonts/SFNSText.ttf", size=min(xSize, ySize) // 20)

        drawimg = ImageDraw.Draw(img)
        drawimg.rectangle((LU_corner, RD_corner),fill=None,outline=rect_color)
        drawimg.text((box[1], (box[2]+box[4])/2), box[0], fill=font_color, font = fnt)
    #img.show()
    img.save(name_to_save)







def main():

    """
    bboxs = []
    with open(os.path.join('/Users/fincep004/Desktop/finc_food_db/exp11_detection_155/308645.txt'), 'r') as r:
        for line in r.readlines():
            name = ''
            for x in line.rstrip().split(' '):
                if isDigit(x):
                    break
                name = name + ' ' + x
            # format = name(may with blank) x y w h
            coord = line.rstrip().split(' ')[-4:]
            xmin = float(coord[0]) - float(coord[2]) / 2
            ymin = float(coord[1]) - float(coord[3]) / 2
            xmax = float(coord[0]) + float(coord[2]) / 2
            ymax = float(coord[1]) + float(coord[3]) / 2
            bboxs.append([name, xmin, ymin, xmax, ymax])

    exit()
    """
    # UEC v3 format:  name(may with blank) confidence x y w h
    anno_list = get_file_list(config['SAVE_DETECTION_DIR'], extensions=('txt'))
    for f in anno_list:
        print(f)
        base_name = os.path.splitext(os.path.basename(f))[0]

        bboxes = []
        with open(os.path.join(f), 'r') as r:
            for line in r.readlines():
                pieces = line.rstrip().split(' ')
                # get name
                conf = 0.0
                name = ''
                for x in pieces:
                    if isDigit(x):
                        conf = float(x)
                        break
                    name = name + ' ' + x

                if conf < config['CONFIDENCE_THRESHOLD']:
                    break

                # format = name(may with blank) x y w h
                coord = pieces[-4:]
                xmin = float(coord[0]) - float(coord[2]) / 2
                ymin = float(coord[1]) - float(coord[3]) / 2
                xmax = float(coord[0]) + float(coord[2]) / 2
                ymax = float(coord[1]) + float(coord[3]) / 2
                bboxes.append([name, xmin, ymin, xmax, ymax])

        if not bboxes:
            print('empty detection @', f)
            continue

        # remove the highly overlapped bboxes
        r_bboxs = [bboxes[0]]
        for bbox in bboxes[1:]:
            for r_bbox in r_bboxs:
                if overlapped_ratio(bbox[1:], r_bbox[1:]) > config['overlapped_ratio_threshold']:
                    break
                else:
                    r_bboxs.append(bbox)

        # draw bounding box
        img_path = os.path.join(config['IMG_DIR'], base_name + '.jpg')
        vis_path = os.path.join(config['SAVE_BBOX_IMG_PATH'], base_name + '.jpg')
        img = Image.open(img_path)
        draw_bbox(img, name_to_save=vis_path, bboxes=r_bboxs)


if __name__ == '__main__':
    main()



"""
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import time

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
from distutils.version import StrictVersion

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")
from object_detection.utils import ops as utils_ops
from utils import visualization_utils, label_map_util

if StrictVersion(tf.__version__) < StrictVersion('1.9.0'):
  raise ImportError('Please upgrade your TensorFlow installation to v1.9.* or later!')
  
  
def get_file_list(dir_path, extensions):
    file_list = []
    for f in os.listdir(dir_path):
        path = os.path.join(dir_path, f)
        if os.path.isdir(path):
            file_list = file_list + get_file_list(path, extensions)
        elif f.endswith(extensions):
            file_list.append(path)

    return file_list


def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)


exp13 = {
    'NUM_CLASSES':110,
    # Path to model
    'PATH_TO_CKPT':os.path.join(os.path.sep, '/mnt2/model/FRCNN/exp13', 'frozen_inference_graph.pb'),
    # List of the strings that is used to add correct label for each box.
    'PATH_TO_LABELS':os.path.join(os.path.sep, '/mnt2/projects/TF_Obj_Detection/label_maps/','exp13_label_map.pbtxt'),
    #'IMG_PATH':os.path.join(os.path.sep, '/mnt2/DB/155'),
    'SAVE_FIG':True,
    'PATH_OF_SAVE_FIG':'/mnt2/results/FRCNN/exp13_vis',
    # whether show the info during detection
    'SHOW_INFO':False,
}
config = exp13

def runner(TEST_IMAGE_PATHS):
    print('loading models')
    s_time = time.time()
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(config['PATH_TO_CKPT'], 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    print('loading model done, took {} sec'.format( time.time() - s_time))

    # convert id to category name
    label_map = label_map_util.load_labelmap(config['PATH_TO_LABELS'])
    categories = label_map_util.convert_label_map_to_categories(
        label_map, max_num_classes=config['NUM_CLASSES'], use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    with detection_graph.as_default():
        with tf.Session() as sess:
            # Get handles to input and output tensors
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            for key in [
                'num_detections', 'detection_boxes', 'detection_scores',
                'detection_classes', 'detection_masks']:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[
                        key] = tf.get_default_graph().get_tensor_by_name(
                        tensor_name)

            for image_path in TEST_IMAGE_PATHS:
                s_time = time.time()
                print(image_path)
                image = Image.open(image_path)
                # the array based representation of the image will be used later in order to prepare the
                # result image with boxes and labels on it.
                image_np = load_image_into_numpy_array(image)
                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                img_batch = np.expand_dims(image_np, 0)
                #img_batch = image_np

                # example code of using batch images
                BATCH = False
                if BATCH:
                    # batch-input, the shape should be (nb_batch, w, h, nb_channel)
                    # ex: (578, 432, 3) -> (1, 578, 432, 3)
                    # TODO: need to resize images to same size
                    # TODO: decode the batch output
                    img_batch = np.vstack((image_np, image_np))

                if 'detection_masks' in tensor_dict:
                    # The following processing is only for single image
                    detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
                    detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
                    # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
                    real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
                    detection_boxes = tf.slice(detection_boxes, [0, 0],
                                               [real_num_detection, -1])
                    detection_masks = tf.slice(detection_masks, [0, 0, 0],
                                               [real_num_detection, -1, -1])
                    detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                        detection_masks, detection_boxes, image.shape[0],
                        image.shape[1])
                    detection_masks_reframed = tf.cast(
                        tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                    # Follow the convention by adding back the batch dimension
                    tensor_dict['detection_masks'] = tf.expand_dims(
                        detection_masks_reframed, 0)

                image_tensor = tf.get_default_graph().get_tensor_by_name(
                    'image_tensor:0')

                # Run inference
                output_dict = sess.run(tensor_dict, feed_dict={image_tensor: img_batch})

                # all outputs are float32 numpy arrays, so convert types as appropriate
                output_dict['num_detections'] = int(output_dict['num_detections'][0])
                output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.uint8)
                output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
                output_dict['detection_scores'] = output_dict['detection_scores'][0]
                if 'detection_masks' in output_dict:
                    output_dict['detection_masks'] = output_dict['detection_masks'][0]

                print('one iteration took {}'.format(time.time() - s_time))


                if config['SHOW_INFO']:
                    print('num_detections = ', output_dict['num_detections'])
                    print('top 3 boxes:')
                    for i in range(3):
                        print('bbox-{} = {}, score={}, bbox={}'.format(i+1,
                        output_dict['detection_classes'][i], output_dict['detection_scores'][i], output_dict['detection_boxes'][i]))

                    print('detection_classes = ', output_dict['detection_classes'])
                    print('detection_scores = ', output_dict['detection_scores'])
                    for i in output_dict['detection_boxes']:
                        print(i)

                if config['SAVE_FIG']:
                    visualization_utils.visualize_boxes_and_labels_on_image_array(
                        image_np,
                        np.asarray(output_dict['detection_boxes']), #np.squeeze(boxes),
                        np.squeeze(output_dict['detection_classes']).astype(np.int32),
                        np.squeeze(output_dict['detection_scores']), category_index,
                        use_normalized_coordinates=True,
                        line_thickness=8)
                    im = Image.fromarray(image_np)
                    im.save(os.path.join(config['PATH_OF_SAVE_FIG'], image_path.split(os.path.sep)[-1]))
    print('Done, please check result@{}'.format(config['PATH_OF_SAVE_FIG']))


from matplotlib import pyplot as plt
# This is needed to display the images.
%matplotlib inline
# adjust size if not clear
IMAGE_SIZE=(14, 10)

vis_imgs = get_file_list(config['PATH_OF_SAVE_FIG'], ('.jpg', 'jpeg', '.png', '.bmp', '.JPG', 'JPEG', '.PNG', '.BMP'))

for img_path in vis_imgs:
    img = Image.open(img_path)
    img_np = load_image_into_numpy_array(img)
    plt.figure(figsize=IMAGE_SIZE)
    plt.imshow(img_np)
"""