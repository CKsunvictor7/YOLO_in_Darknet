"""
This code contains tools to build a YOLO DB for training


"""

import os
import shutil
from PIL import Image
from sklearn.model_selection import StratifiedKFold, KFold
import numpy as np
import operator
import pandas as pd
import operator


def get_file_list(dir_path, extensions):
    """
    return abs_path of all files who ends with __ in all sub-directories as a list
    extensions: a tuple to specify the file extension
    ex: ('.jpg', 'jpeg', '.png', '.bmp', '.JPG', 'JPEG', '.PNG', '.BMP')
    """
    file_list = []
    for f in os.listdir(dir_path):
        path = os.path.join(dir_path, f)
        if os.path.isdir(path):
            file_list = file_list + get_file_list(path, extensions)
        elif f.endswith(extensions):
            file_list.append(path)

    return file_list


def split_by_KFold(data, nb_splits=3):
    kfold = KFold(n_splits=nb_splits, shuffle=True)

    index_folds = []
    for train_index, test_index in kfold.split(data):
        index_folds.append([train_index, test_index])

    train_data = [data[k] for k in index_folds[0][0]]
    val_data = [data[k] for k in index_folds[0][1]]
    return train_data, val_data


def split_by_StratifiedKFold(data, labels, nb_splits=3):
    """
    split data into training and validation set,
    and make sure the distribution of training is same as validation

    :param data: np.array, data
    :param labels: np.array, label
    :param nb_splits:  Number of folds. Must be at least 2.
    :return: train_data, train_labels, val_data, val_labels of one k-fold
    """
    skf = StratifiedKFold(n_splits=nb_splits, shuffle=True)

    index_folds = []
    for train_index, test_index in skf.split(data, labels):
        index_folds.append([train_index, test_index])

    # here we only return the first k-fold
    train_data = [data[k] for k in index_folds[0][0]]
    train_labels = [[labels[k] for k in index_folds[0][0]]]
    val_data = [data[k] for k in index_folds[0][1]]
    val_labels = [[labels[k] for k in index_folds[0][1]]]
    return train_data, train_labels, val_data, val_labels


def BBox_TO_YOLO(size, box):
    """
    convert bbox format to YOLO format
    :param size: size of img (w, h)
    :param box: bounding box: (xmin, xmax, ymin, ymax)
    :return: YOLO format
    """
    dw = 1. / float(size[0])
    dh = 1. / float(size[1])
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)


def UEC_BBox_TO_YOLO(annotation_dir, img_dir, new_annotation_dir):
    """
    to convert all annotation file in annotation_dir to YOLO format,
    and write the annotation(same name) under new_annotation_dir

    :param annotation_dir: ex:''/Users/fincep004/Documents/UEC_ground_truth''
    :param img_dir: ex: '/Users/fincep004/Documents/UEC256all''
    :paran new_annotation_dir: '/Users/fincep004/Documents/UEC256_annotations_YOLO'
    :return: None, it writes
    """
    annotation_list = [f for f in os.listdir(annotation_dir) if f.endswith('txt') and f != 'classes.txt']
    for f in annotation_list:
        try:
            img = Image.open(os.path.join(img_dir, os.path.splitext(f)[0] + '.jpg'))
            with open(os.path.join(annotation_dir, f), 'r') as r:
                for l in r.readlines():
                    pieces = l.rstrip().split(' ')
                    (x, y, w, h) = BBox_TO_YOLO(img.size, (
                    float(pieces[1]), float(pieces[2]), float(pieces[3]), float(pieces[4])))

                    if x < 0 or x > 1 or y < 0 or y > 1 or w < 0 or w > 1 or h < 0 or h > 1:
                        print('error, range overflow @{} {} {} {} {}\n'.format(f, x, y, w, h))

                    # print('{} {} {} {} {}'.format(pieces[0], x, y, w, h))
                    with open(os.path.join(new_annotation_dir, f), 'a') as writer:
                        # int(pieces[0])-1: to map UEC256(1,N) -> YOLO(0,N-1)
                        writer.write('{} {} {} {} {}\n'.format(int(pieces[0])-1, x, y, w, h))
        except:
            print('sth wrong @', f)

    print('convert successfully')


def renamer():
    dir = '/Users/fincep004/Desktop/OneDrive - FiNC Co.Ltd/Annotations/finc-food-furuta/annotations_v2/annotations_chukadon'
    #dir = '/Users/fincep004/Desktop/web_food_imgs/bowl_of_rice_and_fried_fish'
    #dir = '/Users/fincep004/Desktop/web_food_imgs_f/chukadon/2'
    for f in os.listdir(dir):
        #shutil.move(os.path.join(dir, f), os.path.join(dir, f + '.txt'))
        new_name = f.replace("chukadon", "chuka_don")
        print(new_name)
        shutil.move(os.path.join(dir, f), os.path.join(dir, new_name))


def file_mover(src_dir, dst_dir):
    img_list = get_file_list(src_dir, extensions=('.jpg', 'jpeg', '.png', '.bmp', '.JPG', 'JPEG', '.PNG', '.BMP'))
    print('nb of img = ', len(img_list))
    for c, f in enumerate(img_list):
        shutil.copyfile(f, os.path.join(dst_dir, os.path.basename(f)))


def same_img(p1, p2):
    """
    judge if p1, p2 are same image using size & mean value of R,G,B channel
    :param p1: path
    :param p2: path
    :return: True or False
    """
    img1 = np.array(Image.open(p1))
    img2 = np.array(Image.open(p2))

    if img1.shape == img2.shape and np.mean(img1[:,:,0])==np.mean(img2[:,:,0]) and np.mean(img1[:,:,1])==np.mean(img2[:,:,1]) and np.mean(img1[:,:,2])==np.mean(img2[:,:,2]):
        return True
    else:
        return False


def YOLODB_check(train_list):
    """
    check whether this DB is good to train by the logic of how YOLO read training data
    1. check the if image file exists or not
    2. check there is one annotation file for each image file in train/val list
       by the path rules of YOLO Darknet
    3. YOLO annotation format check: length & is float

    print the bad paths
    """
    count = 0
    with open(train_list, 'r') as r:
        for l in r.readlines():
            # check anno
            img_path = l.rstrip() #.replace( 'images', 'labels')

            if not os.path.exists(img_path):
                print('error, this img is missed: ', img_path)

            anno_path = str.replace(img_path, '.jpg', '.txt')
            #anno_path = str.replace(anno_path, '.JPG', '.txt')
            #anno_path = str.replace(anno_path, '.png', '.txt')
            #anno_path = str.replace(anno_path, '.JPEG', '.txt')
            if os.path.exists(anno_path):
                with open(anno_path, 'r') as r:
                    for line in  r.readlines():
                        piece = line.rstrip().split(' ')
                        assert len(piece) == 5, 'error, too many or less anno @{}'.format(anno_path)
                        for p in piece:
                            try:
                                float(p)
                            except:
                                print('error, not digital @{} and {}'.format(anno_path, p))
                count += 1
                continue
            else:
                print('error, this anno is missed: ', anno_path)
    print('check {} annotations completed'.format(count))



def DB_info_id_version(anno_dir):
    category_list = []
    with open('exps/category.names', 'r') as r:
        for line in r.readlines():
            category_list.append(line.rstrip())


    category_dict = {}
    anno_list = get_file_list(anno_dir, '.txt')
    for f in anno_list:
        with open(f, 'r') as r:
            for l in r.readlines():
                label = int(l.rstrip().split(' ')[0])
                if label not in category_dict:
                    category_dict[label] = 1
                else:
                    category_dict[label] += 1
    for k,v in sorted(category_dict.items(), key=operator.itemgetter(1), reverse=True):
        print('{}:{}'.format(category_list[k], v))
    print('total {} categories'.format(len(category_dict)))


def DB_info_name_version(anno_dirs):
    """
    show the info of anno_dir
    :param anno_dirs: a list of annotations []
    return anno_dict{'category name':nb}
    """
    anno_list = []
    for src_dir in anno_dirs:
        anno_list += get_file_list(src_dir, extensions=('txt'))
    #anno_list = get_file_list(anno_dir, extensions=('txt'))
    anno_dict = {}
    for f in anno_list:
        with open(f, 'r') as r:
            for line in r.readlines():
                food = line.rstrip().split(' ')[0]
                if food == 'nan':
                    print('nan @', f)
                # TODO: more pythonic
                if food not in anno_dict:
                    anno_dict[food] = 1
                else:
                    anno_dict[food] += 1

    print('total nb of category = ', len(anno_dict))

    for k, v in sorted(anno_dict.items(), key=operator.itemgetter(1),
                       reverse=True):
        print(k, v)

    return anno_dict


def id_to_name_conversion(src_dir, dst_dir):
    """
    convert annotation from id to name & save to dst_dir with same name.
    for old version labeling annotations: 'UEC256_annotations_YOLO' and '__'
    """
    table = pd.read_csv('labelsheet_v2.csv', encoding='utf-8')
    anno_list = get_file_list(src_dir, extensions=('txt'))
    print('nb of data = ', len(anno_list))
    for f in anno_list:
        dst_path = os.path.join(dst_dir, os.path.basename(f))
        with open(f, 'r') as r:
            with open(dst_path, 'w') as w:
                for line in r.readlines():
                    pieces = line.rstrip().split(' ')

                    if len(pieces) < 5:
                        print('sth wrong @', f)
                        break

                    # notice: label range start from 0 or 1
                    matching_name = table[table['label'] == (int(pieces[0]))]['level3_en'].values[0]
                    #print(matching_name)
                    if str(matching_name) != 'nan':
                        w.write('{} {} {} {} {}\n'.format(matching_name, pieces[1], pieces[2], pieces[3], pieces[4]))
    print('conversion done')


"""
need to check list:
Wrong label: /mnt/UEC256_images/990.txt - j = 0, x = 0.500000, y = 0.500000, width = 1.000000, height = 1.070000 
 loaded 	 image: 10430    box: 15341

Wrong label: /mnt/UEC256_images/14650.txt - j = 0, x = 0.325000, y = 0.500000, width = 0.650000, height = 1.012900 
 loaded 	 image: 13338 	 box: 19542

Wrong label: /mnt/UEC256_images/12466.txt - j = 0, x = 0.500000, y = 0.500000, width = 1.000000, height = 1.025000 
 loaded 	 image: 18803 	 box: 27540


nan@ /Users/fincep004/Documents/UEC256_annotations_YOLO_v2_c/248286.txt
nan@ /Users/fincep004/Documents/UEC256_annotations_YOLO_v2_c/15830.txt
nan@ /Users/fincep004/Documents/UEC256_annotations_YOLO_v2_c/91701.txt
nan@ /Users/fincep004/Documents/UEC256_annotations_YOLO_v2_c/5745.txt
nan@ /Users/fincep004/Documents/UEC256_annotations_YOLO_v2_c/6264.txt
nan@ /Users/fincep004/Documents/UEC256_annotations_YOLO_v2_c/287622.txt
nan@ /Users/fincep004/Documents/UEC256_annotations_YOLO_v2_c/3081.txt
nan@ /Users/fincep004/Documents/UEC256_annotations_YOLO_v2_c/26343.txt
nan@ /Users/fincep004/Documents/UEC256_annotations_YOLO_v2_c/249416.txt
nan@ /Users/fincep004/Documents/UEC256_annotations_YOLO_v2_c/16690.txt
nan@ /Users/fincep004/Documents/UEC256_annotations_YOLO_v2_c/65871.txt
nan@ /Users/fincep004/Documents/UEC256_annotations_YOLO_v2_c/7153.txt
nan@ /Users/fincep004/Documents/UEC256_annotations_YOLO_v2_c/15366.txt
nan@ /Users/fincep004/Documents/UEC256_annotations_YOLO_v2_c/216460.txt
nan@ /Users/fincep004/Documents/UEC256_annotations_YOLO_v2_c/15370.txt
nan@ /Users/fincep004/Documents/UEC256_annotations_YOLO_v2_c/15358.txt
nan@ /Users/fincep004/Documents/UEC256_annotations_YOLO_v2_c/9020.txt
nan@ /Users/fincep004/Documents/UEC256_annotations_YOLO_v2_c/291137.txt
nan@ /Users/fincep004/Documents/UEC256_annotations_YOLO_v2_c/67334.txt
nan@ /Users/fincep004/Documents/UEC256_annotations_YOLO_v2_c/3726.txt
nan@ /Users/fincep004/Documents/UEC256_annotations_YOLO_v2_c/5357.txt
nan@ /Users/fincep004/Documents/UEC256_annotations_YOLO_v2_c/196018.txt
nan@ /Users/fincep004/Documents/UEC256_annotations_YOLO_v2_c/3875.txt
nan@ /Users/fincep004/Documents/UEC256_annotations_YOLO_v2_c/235290.txt
nan@ /Users/fincep004/Documents/UEC256_annotations_YOLO_v2_c/288088.txt
nan@ /Users/fincep004/Documents/UEC256_annotations_YOLO_v2_c/249820.txt
over range@ /Users/fincep004/Documents/UEC256_annotations_YOLO_v2_c/2164.txt
over range@ /Users/fincep004/Documents/UEC256_annotations_YOLO_v2_c/15573.txt
nan@ /Users/fincep004/Documents/UEC256_annotations_YOLO_v2_c/291640.txt
nan@ /Users/fincep004/Documents/UEC256_annotations_YOLO_v2_c/15765.txt
nan@ /Users/fincep004/Documents/UEC256_annotations_YOLO_v2_c/6854.txt
nan@ /Users/fincep004/Documents/UEC256_annotations_YOLO_v2_c/5839.txt
nan@ /Users/fincep004/Documents/UEC256_annotations_YOLO_v2_c/14644.txt
over range@ /Users/fincep004/Documents/UEC256_annotations_YOLO_v2_c/271164.txt
nan@ /Users/fincep004/Documents/UEC256_annotations_YOLO_v2_c/3680.txt
nan@ /Users/fincep004/Documents/UEC256_annotations_YOLO_v2_c/3694.txt
over range@ /Users/fincep004/Documents/UEC256_annotations_YOLO_v2_c/15176.txt
nan@ /Users/fincep004/Documents/UEC256_annotations_YOLO_v2_c/12141.txt
nan@ /Users/fincep004/Documents/UEC256_annotations_YOLO_v2_c/3696.txt
nan@ /Users/fincep004/Documents/UEC256_annotations_YOLO_v2_c/266962.txt
nan@ /Users/fincep004/Documents/UEC256_annotations_YOLO_v2_c/15217.txt
over range@ /Users/fincep004/Documents/UEC256_annotations_YOLO_v2_c/2172.txt
over range@ /Users/fincep004/Documents/UEC256_annotations_YOLO_v2_c/2172.txt
over range@ /Users/fincep004/Documents/UEC256_annotations_YOLO_v2_c/2172.txt
nan@ /Users/fincep004/Documents/UEC256_annotations_YOLO_v2_c/13276.txt
nan@ /Users/fincep004/Documents/UEC256_annotations_YOLO_v2_c/13276.txt
nan@ /Users/fincep004/Documents/UEC256_annotations_YOLO_v2_c/15310.txt
nan@ /Users/fincep004/Documents/UEC256_annotations_YOLO_v2_c/57686.txt
nan@ /Users/fincep004/Documents/UEC256_annotations_YOLO_v2_c/14797.txt
nan@ /Users/fincep004/Documents/UEC256_annotations_YOLO_v2_c/155423.txt
nan@ /Users/fincep004/Documents/UEC256_annotations_YOLO_v2_c/4565.txt
nan@ /Users/fincep004/Documents/UEC256_annotations_YOLO_v2_c/134156.txt
nan@ /Users/fincep004/Documents/UEC256_annotations_YOLO_v2_c/113528.txt
nan@ /Users/fincep004/Documents/UEC256_annotations_YOLO_v2_c/3195.txt
nan@ /Users/fincep004/Documents/UEC256_annotations_YOLO_v2_c/3340.txt
nan@ /Users/fincep004/Documents/UEC256_annotations_YOLO_v2_c/267952.txt
nan@ /Users/fincep004/Documents/UEC256_annotations_YOLO_v2_c/3208.txt
nan@ /Users/fincep004/Documents/UEC256_annotations_YOLO_v2_c/174174.txt
nan@ /Users/fincep004/Documents/UEC256_annotations_YOLO_v2_c/14409.txt
nan@ /Users/fincep004/Documents/UEC256_annotations_YOLO_v2_c/6833.txt
nan@ /Users/fincep004/Documents/UEC256_annotations_YOLO_v2_c/222836.txt
nan@ /Users/fincep004/Documents/UEC256_annotations_YOLO_v2_c/14384.txt
nan@ /Users/fincep004/Documents/UEC256_annotations_YOLO_v2_c/15925.txt
nan@ /Users/fincep004/Documents/UEC256_annotations_YOLO_v2_c/287247.txt
nan@ /Users/fincep004/Documents/UEC256_annotations_YOLO_v2_c/12079.txt
nan@ /Users/fincep004/Documents/UEC256_annotations_YOLO_v2_c/14805.txt
nan@ /Users/fincep004/Documents/UEC256_annotations_YOLO_v2_c/12496.txt
nan@ /Users/fincep004/Documents/UEC256_annotations_YOLO_v2_c/58000.txt
nan@ /Users/fincep004/Documents/UEC256_annotations_YOLO_v2_c/13997.txt
nan@ /Users/fincep004/Documents/UEC256_annotations_YOLO_v2_c/287900.txt
nan@ /Users/fincep004/Documents/UEC256_annotations_YOLO_v2_c/237033.txt
over range@ /Users/fincep004/Documents/UEC256_annotations_YOLO_v2_c/15060.txt
nan@ /Users/fincep004/Documents/UEC256_annotations_YOLO_v2_c/15921.txt
nan@ /Users/fincep004/Documents/UEC256_annotations_YOLO_v2_c/14425.txt
nan@ /Users/fincep004/Documents/UEC256_annotations_YOLO_v2_c/187013.txt
nan@ /Users/fincep004/Documents/UEC256_annotations_YOLO_v2_c/269548.txt
nan@ /Users/fincep004/Documents/UEC256_annotations_YOLO_v2_c/15842.txt
over range@ /Users/fincep004/Documents/UEC256_annotations_YOLO_v2_c/12485.txt
nan@ /Users/fincep004/Documents/UEC256_annotations_YOLO_v2_c/18989.txt
nan@ /Users/fincep004/Documents/UEC256_annotations_YOLO_v2_c/291620.txt
nan@ /Users/fincep004/Documents/UEC256_annotations_YOLO_v2_c/4011.txt
nan@ /Users/fincep004/Documents/UEC256_annotations_YOLO_v2_c/14368.txt
nan@ /Users/fincep004/Documents/UEC256_annotations_YOLO_v2_c/16557.txt
nan@ /Users/fincep004/Documents/UEC256_annotations_YOLO_v2_c/14354.txt
nan@ /Users/fincep004/Documents/UEC256_annotations_YOLO_v2_c/143244.txt
over range@ /Users/fincep004/Documents/UEC256_annotations_YOLO_v2_c/215131.txt
nan@ /Users/fincep004/Documents/UEC256_annotations_YOLO_v2_c/787.txt
nan@ /Users/fincep004/Documents/UEC256_annotations_YOLO_v2_c/13026.txt
nan@ /Users/fincep004/Documents/UEC256_annotations_YOLO_v2_c/14173.txt
nan@ /Users/fincep004/Documents/UEC256_annotations_YOLO_v2_c/15286.txt
nan@ /Users/fincep004/Documents/UEC256_annotations_YOLO_v2_c/12503.txt
nan@ /Users/fincep004/Documents/UEC256_annotations_YOLO_v2_c/16759.txt
nan@ /Users/fincep004/Documents/UEC256_annotations_YOLO_v2_c/290309.txt
nan@ /Users/fincep004/Documents/UEC256_annotations_YOLO_v2_c/1992.txt
nan@ /Users/fincep004/Documents/UEC256_annotations_YOLO_v2_c/254822.txt
nan@ /Users/fincep004/Documents/UEC256_annotations_YOLO_v2_c/143470.txt
nan@ /Users/fincep004/Documents/UEC256_annotations_YOLO_v2_c/287878.txt
nan@ /Users/fincep004/Documents/UEC256_annotations_YOLO_v2_c/15454.txt
nan@ /Users/fincep004/Documents/UEC256_annotations_YOLO_v2_c/92261.txt
nan@ /Users/fincep004/Documents/UEC256_annotations_YOLO_v2_c/92261.txt
nan@ /Users/fincep004/Documents/UEC256_annotations_YOLO_v2_c/14170.txt
nan@ /Users/fincep004/Documents/UEC256_annotations_YOLO_v2_c/332190.txt
over range@ /Users/fincep004/Documents/UEC256_annotations_YOLO_v2_c/6147.txt
nan@ /Users/fincep004/Documents/UEC256_annotations_YOLO_v2_c/12065.txt
nan@ /Users/fincep004/Documents/UEC256_annotations_YOLO_v2_c/11593.txt
nan@ /Users/fincep004/Documents/UEC256_annotations_YOLO_v2_c/202734.txt
nan@ /Users/fincep004/Documents/UEC256_annotations_YOLO_v2_c/16000.txt
nan@ /Users/fincep004/Documents/UEC256_annotations_YOLO_v2_c/14603.txt
nan@ /Users/fincep004/Documents/UEC256_annotations_YOLO_v2_c/290493.txt
nan@ /Users/fincep004/Documents/UEC256_annotations_YOLO_v2_c/3348.txt
nan@ /Users/fincep004/Documents/UEC256_annotations_YOLO_v2_c/290652.txt
nan@ /Users/fincep004/Documents/UEC256_annotations_YOLO_v2_c/15657.txt
nan@ /Users/fincep004/Documents/UEC256_annotations_YOLO_v2_c/12100.txt
nan@ /Users/fincep004/Documents/UEC256_annotations_YOLO_v2_c/12100.txt
nan@ /Users/fincep004/Documents/UEC256_annotations_YOLO_v2_c/14565.txt
nan@ /Users/fincep004/Documents/UEC256_annotations_YOLO_v2_c/91221.txt
nan@ /Users/fincep004/Documents/UEC256_annotations_YOLO_v2_c/6963.txt
nan@ /Users/fincep004/Documents/UEC256_annotations_YOLO_v2_c/14015.txt
nan@ /Users/fincep004/Documents/UEC256_annotations_YOLO_v2_c/13020.txt
nan@ /Users/fincep004/Documents/UEC256_annotations_YOLO_v2_c/636.txt
nan@ /Users/fincep004/Documents/UEC256_annotations_YOLO_v2_c/12060.txt
nan@ /Users/fincep004/Documents/UEC256_annotations_YOLO_v2_c/6817.txt
nan@ /Users/fincep004/Documents/UEC256_annotations_YOLO_v2_c/270833.txt
nan@ /Users/fincep004/Documents/UEC256_annotations_YOLO_v2_c/280040.txt
nan@ /Users/fincep004/Documents/UEC256_annotations_YOLO_v2_c/248770.txt
nan@ /Users/fincep004/Documents/UEC256_annotations_YOLO_v2_c/243802.txt
nan@ /Users/fincep004/Documents/UEC256_annotations_YOLO_v2_c/15691.txt
nan@ /Users/fincep004/Documents/UEC256_annotations_YOLO_v2_c/7315.txt
nan@ /Users/fincep004/Documents/UEC256_annotations_YOLO_v2_c/269596.txt
nan@ /Users/fincep004/Documents/UEC256_annotations_YOLO_v2_c/16615.txt
nan@ /Users/fincep004/Documents/UEC256_annotations_YOLO_v2_c/62619.txt
nan@ /Users/fincep004/Documents/UEC256_annotations_YOLO_v2_c/62619.txt
nan@ /Users/fincep004/Documents/UEC256_annotations_YOLO_v2_c/91.txt
nan@ /Users/fincep004/Documents/UEC256_annotations_YOLO_v2_c/92677.txt
nan@ /Users/fincep004/Documents/UEC256_annotations_YOLO_v2_c/92677.txt
nan@ /Users/fincep004/Documents/UEC256_annotations_YOLO_v2_c/3206.txt
nan@ /Users/fincep004/Documents/UEC256_annotations_YOLO_v2_c/14639.txt
nan@ /Users/fincep004/Documents/UEC256_annotations_YOLO_v2_c/3366.txt
nan@ /Users/fincep004/Documents/UEC256_annotations_YOLO_v2_c/16614.txt
nan@ /Users/fincep004/Documents/UEC256_annotations_YOLO_v2_c/12880.txt
nan@ /Users/fincep004/Documents/UEC256_annotations_YOLO_v2_c/14573.txt
nan@ /Users/fincep004/Documents/UEC256_annotations_YOLO_v2_c/269540.txt
nan@ /Users/fincep004/Documents/UEC256_annotations_YOLO_v2_c/12171.txt
nan@ /Users/fincep004/Documents/UEC256_annotations_YOLO_v2_c/288295.txt
nan@ /Users/fincep004/Documents/UEC256_annotations_YOLO_v2_c/288295.txt
nan@ /Users/fincep004/Documents/UEC256_annotations_YOLO_v2_c/14919.txt
nan@ /Users/fincep004/Documents/UEC256_annotations_YOLO_v2_c/12549.txt
nan@ /Users/fincep004/Documents/UEC256_annotations_YOLO_v2_c/13327.txt
nan@ /Users/fincep004/Documents/UEC256_annotations_YOLO_v2_c/175983.txt
nan@ /Users/fincep004/Documents/UEC256_annotations_YOLO_v2_c/60478.txt
nan@ /Users/fincep004/Documents/UEC256_annotations_YOLO_v2_c/269485.txt
nan@ /Users/fincep004/Documents/UEC256_annotations_YOLO_v2_c/269485.txt
nan@ /Users/fincep004/Documents/UEC256_annotations_YOLO_v2_c/90371.txt
nan@ /Users/fincep004/Documents/UEC256_annotations_YOLO_v2_c/13736.txt
nan@ /Users/fincep004/Documents/UEC256_annotations_YOLO_v2_c/13246.txt
nan@ /Users/fincep004/Documents/UEC256_annotations_YOLO_v2_c/13246.txt
nan@ /Users/fincep004/Documents/UEC256_annotations_YOLO_v2_c/4525.txt
nan@ /Users/fincep004/Documents/UEC256_annotations_YOLO_v2_c/202742.txt
nan@ /Users/fincep004/Documents/UEC256_annotations_YOLO_v2_c/15741.txt
nan@ /Users/fincep004/Documents/UEC256_annotations_YOLO_v2_c/197722.txt
nan@ /Users/fincep004/Documents/UEC256_annotations_YOLO_v2_c/16260.txt
over range@ /Users/fincep004/Documents/UEC256_annotations_YOLO_v2_c/205207.txt
nan@ /Users/fincep004/Documents/UEC256_annotations_YOLO_v2_c/288136.txt
nan@ /Users/fincep004/Documents/UEC256_annotations_YOLO_v2_c/6864.txt
nan@ /Users/fincep004/Documents/UEC256_annotations_YOLO_v2_c/4040.txt
nan@ /Users/fincep004/Documents/UEC256_annotations_YOLO_v2_c/6125.txt
nan@ /Users/fincep004/Documents/UEC256_annotations_YOLO_v2_c/13325.txt
nan@ /Users/fincep004/Documents/UEC256_annotations_YOLO_v2_c/16705.txt
over range@ /Users/fincep004/Documents/UEC256_annotations_YOLO_v2_c/2141.txt
over range@ /Users/fincep004/Documents/UEC256_annotations_YOLO_v2_c/2141.txt
nan@ /Users/fincep004/Documents/UEC256_annotations_YOLO_v2_c/7763.txt
nan@ /Users/fincep004/Documents/UEC256_annotations_YOLO_v2_c/122211.txt
nan@ /Users/fincep004/Documents/UEC256_annotations_YOLO_v2_c/12177.txt
nan@ /Users/fincep004/Documents/UEC256_annotations_YOLO_v2_c/249430.txt
nan@ /Users/fincep004/Documents/UEC256_annotations_YOLO_v2_c/14923.txt
nan@ /Users/fincep004/Documents/UEC256_annotations_YOLO_v2_c/14300.txt
nan@ /Users/fincep004/Documents/UEC256_annotations_YOLO_v2_c/15963.txt
nan@ /Users/fincep004/Documents/UEC256_annotations_YOLO_v2_c/186819.txt
nan@ /Users/fincep004/Documents/UEC256_annotations_YOLO_v2_c/186819.txt
nan@ /Users/fincep004/Documents/UEC256_annotations_YOLO_v2_c/95353.txt
over range@ /Users/fincep004/Documents/UEC256_annotations_YOLO_v2_c/6042.txt
nan@ /Users/fincep004/Documents/UEC256_annotations_YOLO_v2_c/15357.txt
nan@ /Users/fincep004/Documents/UEC256_annotations_YOLO_v2_c/271184.txt
nan@ /Users/fincep004/Documents/UEC256_annotations_YOLO_v2_c/255152.txt
nan@ /Users/fincep004/Documents/UEC256_annotations_YOLO_v2_c/16266.txt
nan@ /Users/fincep004/Documents/UEC256_annotations_YOLO_v2_c/3701.txt
nan@ /Users/fincep004/Documents/UEC256_annotations_YOLO_v2_c/14699.txt
nan@ /Users/fincep004/Documents/UEC256_annotations_YOLO_v2_c/271185.txt
nan@ /Users/fincep004/Documents/UEC256_annotations_YOLO_v2_c/15342.txt
nan@ /Users/fincep004/Documents/UEC256_annotations_YOLO_v2_c/2999.txt
nan@ /Users/fincep004/Documents/UEC256_annotations_YOLO_v2_c/13243.txt
nan@ /Users/fincep004/Documents/UEC256_annotations_YOLO_v2_c/15168.txt
nan@ /Users/fincep004/Documents/UEC256_annotations_YOLO_v2_c/14504.txt
"""
def id_to_name_conversion_for_intern_UEC(src_dir, dst_dir):
    """
    convert annotation from id to name & save to dst_dir with same name.
    for old version labeling annotations: 'UEC256_annotations_YOLO' and '__'
    """
    table = pd.read_csv('UEC256_2_v3.csv', encoding='utf-8')
    table_2 = pd.read_csv('labelsheet_v2.csv', encoding='utf-8')

    empty_count = 0
    anno_list = get_file_list(src_dir, extensions=('.txt'))
    print('nb of data = ', len(anno_list))
    for f in anno_list:
        annos = []
        with open(f, 'r') as r:
            for line in r.readlines():
                pieces = line.rstrip().split(' ')
                # TODO: format checker
                id = (int(pieces[0]) + 1)
                # notice: label range start from 0 or 1

                if id <= 256:
                    matching_name = table[table['label'] == id]['v3'].values[0]
                else:  #
                    if id > 743:
                        print('over range@', f)
                        continue
                    id -= 257
                    matching_name = table_2[table_2['label'] == id]['level3_en'].values[0]

                # if matching_name == nan, means those images label may have issue, should check
                if str(matching_name) == 'nan' or str(matching_name) == 'skip':
                    print('nan@', f)
                else:
                    annos.append('{} {} {} {} {}\n'.format(matching_name, pieces[1], pieces[2], pieces[3], pieces[4]))

        if annos:
            dst_path = os.path.join(dst_dir, os.path.basename(f))
            with open(dst_path, 'w') as w:
                for anno in annos:
                    w.write(anno)
        else:
            print('empty annotation@', f)
            empty_count += 1

    print('conversion done')
    print('empty_count = ', empty_count)


def v3_JP_to_EN_conversion(src_dir, dst_dir):
    DEBUG = True

    table = pd.read_csv('labelsheet_v3.csv', encoding='utf-8')
    anno_list = get_file_list(src_dir, extensions=('.txt'))

    try:
        anno_list.remove(os.path.join(src_dir, 'classes.txt'))
    except:
        print('no classes to remove')

    print('nb of data = ', len(anno_list))
    for f in anno_list:
        if DEBUG: print(f)
        dst_path = os.path.join(dst_dir, os.path.basename(f))
        with open(f, 'r') as r:
            with open(dst_path, 'w') as w:
                for line in r.readlines():
                    pieces = line.rstrip().split(' ')

                    # break if empty line
                    if len(pieces) < 2:
                        print('sth wrong @', line)
                        print('@', f)
                        break
                    if DEBUG: print(pieces[0])

                    matching_name = table[table['JP_name'] == pieces[0]]['En_name'].values[0]
                    # print(matching_name)
                    if str(matching_name) == 'nan':
                        print('sth wrong @'.format(f))
                    else:
                        w.write('{} {} {} {} {}\n'.format(matching_name, pieces[1],
                                                      pieces[2], pieces[3],
                                                      pieces[4]))
    print('conversion done')


def finc_v0_conversion(src, dst, img_dir):
    """
    1. convert coordinate to yolo format
    2. id -1 to start from 0
    """
    annotation_list = get_file_list(src, extensions=('.txt'))
    for f in annotation_list:

        # !! notice the format of image
        img = Image.open(os.path.join(img_dir, os.path.splitext(os.path.basename(f))[0] + '.png'))
        try:
            with open(f, 'r') as r:
                for l in r.readlines():
                    pieces = l.rstrip().split(' ')
                    (x, y, w, h) = BBox_TO_YOLO(img.size, (
                    float(pieces[1]), float(pieces[3]), float(pieces[2]), float(pieces[4])))

                    if x < 0 or x > 1 or y < 0 or y > 1 or w < 0 or w > 1 or h < 0 or h > 1:
                        print('error, range overflow @{} {} {} {} {}\n'.format(f, x, y, w, h))

                    # print('{} {} {} {} {}'.format(pieces[0], x, y, w, h))
                    with open(os.path.join(dst, os.path.basename(f)), 'a') as writer:
                        # int(pieces[0])-1: to map UEC256(1,N) -> YOLO(0,N-1)
                        writer.write('{} {} {} {} {}\n'.format(int(pieces[0])-1, x, y, w, h))
        except:
            print('sth wrong @', f)

    print('convert successfully')




def category_name_checker(name):
    assert len(name.rstrip().split(' ')) == 1, \
        'error@ {}, category should not have blank'.format(name.rstrip())


def Img_info():
    print('')






def main():
    DB_info_name_version(['/mnt/UEC256_annotations_YOLO_v3_c', '/mnt/155_v3', '/mnt/web_imgs_annotations_v3'])
    exit()
    YOLODB_check('exps/train_val/exp15_train.txt')
    YOLODB_check('exps/train_val/exp15_val.txt')
    #DB_info_name_version('exps/train_val/exp15_train.txt')
    exit()

    for dir in os.listdir('/Users/fincep004/Desktop/OneDrive - FiNC Co.Ltd/Annotations/finc-food-furuta/annotations_v2/'):
        src_path = os.path.join('/Users/fincep004/Desktop/OneDrive - FiNC Co.Ltd/Annotations/finc-food-furuta/annotations_v2/', dir)
        if os.path.isdir(src_path):
            id_to_name_conversion(src_path,
            '/Users/fincep004/Desktop/OneDrive - FiNC Co.Ltd/Annotations/finc-food-furuta/annotations_v3')
    exit()
    id_to_name_conversion('/Users/fincep004/Desktop/OneDrive - FiNC Co.Ltd/Annotations/finc-food-sasaki/155_v2',
                          '/Users/fincep004/Desktop/OneDrive - FiNC Co.Ltd/Annotations/finc-food-sasaki/155_v3')
    exit()
    id_to_name_conversion_for_intern_UEC(
        '/Users/fincep004/Documents/UEC256_annotations_YOLO_v2_c',
        '/Users/fincep004/Documents/UEC256_annotations_YOLO_v3_c')
    exit()

    v3_JP_to_EN_conversion('/Users/fincep004/Desktop/finc_food_db/annotations_157','/Users/fincep004/Desktop/finc_food_db/annotations_157_en')
    exit()

    # step 0. convert id format(v2) to name format (v3)


    #DB_info_name_version('/mnt2/DB/finc_data/155_v3')
    #exit()


    #DB_info_name_version('/Users/fincep004/Desktop/OneDrive - FiNC Co.Ltd/Annotations/finc-food-sasaki/155_v3')

    #id_to_name_conversion_for_intern_UEC('/mnt/UEC256_annotations_YOLO_v2_c',
    #                                     '/mnt/UEC256_annotations_YOLO_v3_c')



    #DB_info_name_version('/mnt/UEC256_annotations_YOLO_v3_c')
    """
    [0]. use tools to process data
    [1]. make self-defined YOLO DB
    [2]. check the DB is good to train or not by YOLODB_check()
    """
    #YOLODB_check('/mnt/UEC256_train.txt')
    #YOLODB_check('/mnt/UEC256_val.txt')
    #DB_info('/mnt/UEC256_labels')




if __name__ == '__main__':
    main()

"""
difficulty list of web_img
/Users/fincep004/Desktop/finc_food_db/web_imgs/almond_junket_10.jpg
/Users/fincep004/Desktop/finc_food_db/web_imgs/almond_junket_104.jpg
/Users/fincep004/Desktop/finc_food_db/web_imgs/almond_junket_164.jpg
/Users/fincep004/Desktop/finc_food_db/web_imgs/almond_junket_187.jpg
/Users/fincep004/Desktop/finc_food_db/web_imgs/almond_junket_197.jpg
/Users/fincep004/Desktop/finc_food_db/web_imgs/almond_junket_253.jpg
/Users/fincep004/Desktop/finc_food_db/web_imgs/almond_junket_83.jpg
/Users/fincep004/Desktop/finc_food_db/web_imgs/anpan_232.jpg
/Users/fincep004/Desktop/finc_food_db/web_imgs/apple_122.jpg
/Users/fincep004/Desktop/finc_food_db/web_imgs/apple_149.jpg
/Users/fincep004/Desktop/finc_food_db/web_imgs/apple_187.jpg
/Users/fincep004/Desktop/finc_food_db/web_imgs/apple_264.jpg
/Users/fincep004/Desktop/finc_food_db/web_imgs/apple_321.jpg
/Users/fincep004/Desktop/finc_food_db/web_imgs/apple_91.jpg
/Users/fincep004/Desktop/finc_food_db/web_imgs/avocado_and_raw_vegetable_salad_112.jpg
/Users/fincep004/Desktop/finc_food_db/web_imgs/avocado_and_raw_vegetable_salad_17.jpg
/Users/fincep004/Desktop/finc_food_db/web_imgs/avocado_and_raw_vegetable_salad_174.jpg
/Users/fincep004/Desktop/finc_food_db/web_imgs/avocado_and_raw_vegetable_salad_180.jpg
/Users/fincep004/Desktop/finc_food_db/web_imgs/avocado_and_raw_vegetable_salad_340.jpg
/Users/fincep004/Desktop/finc_food_db/web_imgs/avocado_and_raw_vegetable_salad_39.jpg
/Users/fincep004/Desktop/finc_food_db/web_imgs/banana_116.jpg
/Users/fincep004/Desktop/finc_food_db/web_imgs/banana_161.jpg
/Users/fincep004/Desktop/finc_food_db/web_imgs/banana_275.jpg
/Users/fincep004/Desktop/finc_food_db/web_imgs/beans_and_vegetable_salad_164.jpg
"""