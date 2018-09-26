"""
tools to build a YOLO DB for training and validation
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
    return abs_path of all files who ends with extensions in all sub-directories as a list
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
    """
    split data into training and validation set randomly

    :param data: np.array, data
    :param nb_splits: Number of folds. Must be at least 2.
    :return: train_data, val_data of one k-fold
    """
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
    :return: None
    """
    annotation_list = [f for f in os.listdir(
        annotation_dir) if f.endswith('txt') and f != 'classes.txt']
    for f in annotation_list:
        try:
            img = Image.open(
                os.path.join(
                    img_dir,
                    os.path.splitext(f)[0] +
                    '.jpg'))
            with open(os.path.join(annotation_dir, f), 'r') as r:
                for l in r.readlines():
                    pieces = l.rstrip().split(' ')
                    (x, y, w, h) = BBox_TO_YOLO(img.size, (float(pieces[1]), float(
                        pieces[2]), float(pieces[3]), float(pieces[4])))

                    if x < 0 or x > 1 or y < 0 or y > 1 or w < 0 or w > 1 or h < 0 or h > 1:
                        print(
                            'error, range overflow @{} {} {} {} {}\n'.format(
                                f, x, y, w, h))

                    # print('{} {} {} {} {}'.format(pieces[0], x, y, w, h))
                    with open(os.path.join(new_annotation_dir, f), 'a') as writer:
                        # int(pieces[0])-1: to map UEC256(1,N) -> YOLO(0,N-1)
                        writer.write('{} {} {} {} {}\n'.format(
                            int(pieces[0]) - 1, x, y, w, h))
        except BaseException:
            print('sth wrong @', f)

    print('convert successfully')


def renamer(dir):
    """
    to rename all files under the folder
    :return: None
    """
    for f in os.listdir(dir):
        #shutil.move(os.path.join(dir, f), os.path.join(dir, f + '.txt'))
        new_name = f.replace("chukadon", "chuka_don")
        print(new_name)
        shutil.move(os.path.join(dir, f), os.path.join(dir, new_name))


def file_mover(src_dir, dst_dir):
    img_list = get_file_list(
        src_dir,
        extensions=(
            '.jpg',
            'jpeg',
            '.png',
            '.bmp',
            '.JPG',
            'JPEG',
            '.PNG',
            '.BMP'))
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

    if img1.shape == img2.shape and np.mean(img1[:, :, 0]) == np.mean(img2[:, :, 0]) and np.mean(
            img1[:, :, 1]) == np.mean(img2[:, :, 1]) and np.mean(img1[:, :, 2]) == np.mean(img2[:, :, 2]):
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
            img_path = l.rstrip()  # .replace( 'images', 'labels')

            if not os.path.exists(img_path):
                print('error, this img is missed: ', img_path)

            anno_path = str.replace(img_path, '.jpg', '.txt')
            #anno_path = str.replace(anno_path, '.JPG', '.txt')
            #anno_path = str.replace(anno_path, '.png', '.txt')
            #anno_path = str.replace(anno_path, '.JPEG', '.txt')
            if os.path.exists(anno_path):
                with open(anno_path, 'r') as r:
                    for line in r.readlines():
                        piece = line.rstrip().split(' ')
                        assert len(piece) == 5, 'error, too many or less anno @{}'.format(
                            anno_path)
                        for p in piece:
                            try:
                                float(p)
                            except BaseException:
                                print(
                                    'error, not digital @{} and {}'.format(
                                        anno_path, p))
                count += 1
                continue
            else:
                print('error, this anno is missed: ', anno_path)
    print('check {} annotations completed'.format(count))


def DB_info_id_version(anno_dir):
    """
    show the info of anno_dir
    :param anno_dirs: a list of annotations []
    return anno_dict{'category name':nb}
    """
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
    for k, v in sorted(category_dict.items(),
                       key=operator.itemgetter(1), reverse=True):
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
                    matching_name = table[table['label'] == (
                        int(pieces[0]))]['level3_en'].values[0]
                    # print(matching_name)
                    if str(matching_name) != 'nan':
                        w.write(
                            '{} {} {} {} {}\n'.format(
                                matching_name,
                                pieces[1],
                                pieces[2],
                                pieces[3],
                                pieces[4]))
    print('conversion done')


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
                    matching_name = table_2[table_2['label']
                                            == id]['level3_en'].values[0]

                # if matching_name == nan, means those images label may have
                # issue, should check
                if str(matching_name) == 'nan' or str(matching_name) == 'skip':
                    print('nan@', f)
                else:
                    annos.append(
                        '{} {} {} {} {}\n'.format(
                            matching_name,
                            pieces[1],
                            pieces[2],
                            pieces[3],
                            pieces[4]))

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
    """
    since the annotations are Japanese, use this func to convert them to English
    :param src_dir:
    :param dst_dir:
    :return:
    """
    DEBUG = True

    table = pd.read_csv('labelsheet_v3.csv', encoding='utf-8')
    anno_list = get_file_list(src_dir, extensions=('.txt'))

    try:
        anno_list.remove(os.path.join(src_dir, 'classes.txt'))
    except BaseException:
        print('no classes to remove')

    print('nb of data = ', len(anno_list))
    for f in anno_list:
        if DEBUG:
            print(f)
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
                    if DEBUG:
                        print(pieces[0])

                    matching_name = table[table['JP_name']
                                          == pieces[0]]['En_name'].values[0]
                    # print(matching_name)
                    if str(matching_name) == 'nan':
                        print('sth wrong @'.format(f))
                    else:
                        w.write(
                            '{} {} {} {} {}\n'.format(
                                matching_name,
                                pieces[1],
                                pieces[2],
                                pieces[3],
                                pieces[4]))
    print('conversion done')


def finc_v0_conversion(src, dst, img_dir):
    """
    special version of v0
    1. convert coordinate to yolo format
    2. id -1 to start from 0
    """
    annotation_list = get_file_list(src, extensions=('.txt'))
    for f in annotation_list:

        # !! notice the format of image
        img = Image.open(
            os.path.join(
                img_dir,
                os.path.splitext(
                    os.path.basename(f))[0] +
                '.png'))
        try:
            with open(f, 'r') as r:
                for l in r.readlines():
                    pieces = l.rstrip().split(' ')
                    (x, y, w, h) = BBox_TO_YOLO(img.size, (float(pieces[1]), float(
                        pieces[3]), float(pieces[2]), float(pieces[4])))

                    if x < 0 or x > 1 or y < 0 or y > 1 or w < 0 or w > 1 or h < 0 or h > 1:
                        print(
                            'error, range overflow @{} {} {} {} {}\n'.format(
                                f, x, y, w, h))

                    # print('{} {} {} {} {}'.format(pieces[0], x, y, w, h))
                    with open(os.path.join(dst, os.path.basename(f)), 'a') as writer:
                        # int(pieces[0])-1: to map UEC256(1,N) -> YOLO(0,N-1)
                        writer.write('{} {} {} {} {}\n'.format(
                            int(pieces[0]) - 1, x, y, w, h))
        except BaseException:
            print('sth wrong @', f)

    print('convert successfully')


def category_name_checker(name):
    """
    check the name of category follows rules or not
    :param name:
    :return:
    """
    assert len(name.rstrip().split(' ')) == 1, \
        'error@ {}, category should not have blank'.format(name.rstrip())


def main():
    DB_info_name_version(['/mnt/UEC256_annotations_YOLO_v3_c',
                          '/mnt/155_v3', '/mnt/web_imgs_annotations_v3'])
    exit()
    YOLODB_check('exps/train_val/exp15_train.txt')
    YOLODB_check('exps/train_val/exp15_val.txt')

    """
    usage history
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
    """


if __name__ == '__main__':
    main()
