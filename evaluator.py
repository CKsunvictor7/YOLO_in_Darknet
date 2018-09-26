"""
evaluate the detection results
by
1) visualization: visualize the detection results from 'darknet_detector.py'

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