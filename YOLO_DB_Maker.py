"""
TO build a YOLO DB:
Prepare data
->read the src annotation files
-> decide the categories of the DB,
-> output the annotations and train/val txt


Preparation:
1) labelsheet_v3.csv: the category file with col 'EN_name', 'JP_name', 'parent', 'merge', 'country'
2) 'IMG_DIR': the folder contains all images
3) 'src_anno_dir': list of annotations in the format: name x y w h
4) 'dst_anno_dir': destination of new annotations

Steps: @main()

Output:
1) category list stored at 'category.names_path'
2) annotations stored in 'dst_anno_dir'
3) training, validation file list@'src_anno_dir' & dst_anno_dir'
"""
import os
import shutil
from PIL import Image
from YOLODB_tools import get_file_list, DB_info_name_version, split_by_KFold, \
    id_to_name_conversion, id_to_name_conversion_for_intern_UEC, DB_info_id_version, \
    YOLODB_check
import numpy as np
import operator
import pandas as pd


local = {
    'src_anno_dir':"/Users/fincep004/Documents/UEC256_annotations_YOLO_v2_name_c",
    'dst_anno_dir':"/Users/fincep004/Documents/UEC256_annotations_YOLO_v3_c",
    'IMG_DIR':''
}

gpu7 = {
    'src_anno_dir':'/mnt2/DB/finc_data/155_v3',
    'dst_anno_dir':"/mnt2/DB/finc_data/exp_labels_tmp",
    'IMG_DIR':'/mnt2/DB/finc_data/155'
}

gpu6_exp0 = {
    'dst_anno_dir':"/mnt/finc_db_v0_annotations_yolo",
    'IMG_DIR':'/mnt/finc_food_db_v0'
}

gpu8_exp0 = {
    'dst_anno_dir':"/mnt2/DB/finc_db_v0_annotations_yolo",
    'IMG_DIR':'/mnt2/DB/finc_food_db_v0'
}

gpu6_exp13 = {
    'src_anno_dir':['/mnt/UEC256_annotations_YOLO_v3_c'], #UEC256_annotations_YOLO_v3_c
    'dst_anno_dir':"/mnt/UEC256_labels_exp/", #UEC256_labels_exp
    'IMG_DIR':'/mnt/UEC256_images/'
}

gpu6_exp15 = {
    'src_anno_dir':['/mnt/UEC256_annotations_YOLO_v3_c', '/mnt/155_v3', '/mnt/web_imgs_annotations_v3'], #UEC256_annotations_YOLO_v3_c
    'dst_anno_dir':"/mnt/exp15_annos/", #UEC256_labels_exp
    'IMG_DIR':'/mnt/exp15_db/'
}

gpu6_exp16 = {
    'src_anno_dir':['/mnt/UEC256_annotations_YOLO_v3_c', '/mnt/155_v3', '/mnt/web_imgs_annotations_v3'], #UEC256_annotations_YOLO_v3_c
    'dst_anno_dir':"/mnt/exp16_annos/", #UEC256_labels_exp
    'IMG_DIR':'/mnt/m_db/'
}

gpu6_exp16_new = {
    'src_anno_dir':['/mnt/UEC256_annotations_YOLO_v3_c', '/mnt/155_v3', '/mnt/web_imgs_annotations_v3'], #UEC256_annotations_YOLO_v3_c
    'dst_anno_dir':"/mnt/exp16_annos/", #UEC256_labels_exp
    'IMG_DIR':'/mnt/m_db/',
    'category.names_path':'exps/exp16.names',
    'training_list_path':'exps/train_val/exp16_train.txt',
    'val_list_path':'exps/train_val/exp16_val.txt',
}


server = gpu6_exp16_new


# skip if the nb of category is less than Min_num_list
Min_num_list = 50
DO_MERGE = True
skip_country_list = ['southeast_asia', 'Chinese', 'Hawaii', 'Korea', 'Taiwan']
skip_list = ['nonfood',  'skip']



table = pd.read_csv('labelsheet_v3.csv', encoding='utf-8')
# remove NaN rows
table.dropna(how='all', inplace=True)
table.reset_index(drop=True, inplace=True)


def decide_category():
    """
    make category list by 'merge' or 'country'
    and store as 'exps/category.names'

    input: dict: {k(category_name):nb}
    :return: list of category
    """

    anno_list = []
    for src_dir in server['src_anno_dir']:
        anno_list += get_file_list(src_dir, extensions=('txt'))

    # recording the category:nb of bboxes
    anno_dict = {}
    for f in anno_list:
        with open(f, 'r') as r:
            for line in r.readlines():
                food = line.rstrip().split(' ')[0]
                if food == 'nan':
                    print('nan @', f)
                    continue

                if food == 'skip' or food in skip_list:
                    print('skip {}@{}'.format(food, f))
                    continue

                # country filter
                if skip_country_list:
                    try:
                        country = table[table['En_name'] == food]['country'].values[0]
                    except:
                        print('[country]error @', f)
                        print('[country]error @', line)
                        exit()
                    if str(country) != 'nan' and country in skip_country_list:
                        continue

                # do merge
                if DO_MERGE:
                    try:
                        new_name = table[table['En_name'] == food]['merge'].values[0]
                    except:
                        print('[merge]error @', f)
                        print('[merge]error @', line)
                        exit()
                    if str(new_name) != 'nan':
                        # TODO: more pythonic
                        if new_name not in anno_dict:
                            anno_dict[new_name] = 1
                        else:
                            anno_dict[new_name] += 1
                    else:
                        # TODO: more pythonic
                        if food not in anno_dict:
                            anno_dict[food] = 1
                        else:
                            anno_dict[food] += 1
                else:
                    if food not in anno_dict:
                        anno_dict[food] = 1
                    else:
                        anno_dict[food] += 1

    print('total nb of category = ', len(anno_dict))

    category_list = []

    for k, v in sorted(anno_dict.items(), key=operator.itemgetter(1), reverse=True):
        if v < Min_num_list:
            print('discard the category which num is less than ', Min_num_list)
            break
        print(k, v)
        category_list.append(k)

    print('nb of new categories = ', len(category_list))

    with open(server['category.names_path'], 'w') as w:
        for c in category_list:
            w.write('{}\n'.format(c))

    return category_list


def make_YOLO_DB(src_anno_dir, dst_anno_dir, category_list):
    """
    make YOLO anno DB as dst_anno_dir from src_anno_dir by category_list,
    convert the category_name of src_anno_dir to label(num) and store in dst_anno_dir,
    according to category_list, which is generated by Make_category, a list contains category in this task
    :return:
    """
    anno_list = []
    for src_dir in server['src_anno_dir']:
        anno_list += get_file_list(src_dir, extensions=('txt'))

    c = 0
    for f in anno_list:
        desired_annos = []
        #print(f)
        with open(f) as r:
            for line in r.readlines():
                pieces = line.rstrip().split(' ')

                if len(pieces) < 2:
                    print('sth wrong @', line)
                    print('@', f)
                    break

                # category
                v3_name = pieces[0].rstrip()
                #print('v3_name@',v3_name)

                if v3_name in skip_list:
                    continue

                if DO_MERGE:
                    # if this category is one of the category
                    if v3_name in category_list:
                        desired_annos.append('{} {} {} {} {}'.format(
                            category_list.index(v3_name), pieces[1], pieces[2], pieces[3], pieces[4]))
                        continue
                    new_name = table[table['En_name'] == v3_name]['merge'].values[0]
                    #print('new_name = ', new_name)
                    # if the merged category is one of the category
                    if str(new_name) != 'nan' and str(new_name) in category_list:
                        #print('merged: {} -> {}'.format(v3_name, new_name))
                        desired_annos.append('{} {} {} {} {}'.format(category_list.index(new_name), pieces[1], pieces[2], pieces[3], pieces[4]))
                else:
                    if v3_name not in category_list:
                        continue
                    #print('v3_name', v3_name)
                    desired_annos.append('{} {} {} {} {}'.format(category_list.index(v3_name), pieces[1], pieces[2], pieces[3], pieces[4]))
        if desired_annos:
            c += 1
            with open(os.path.join(dst_anno_dir, os.path.basename(f)), 'w') as w:
                for a in desired_annos:
                    #print(a)
                    w.write('{}\n'.format(a))
    print('nb of DB =', c)


def make_train_val_list(dst_anno_dir):
    """
    make train & val list for YOLO, and store as UEC256_train_test.txt, UEC256_val_test.txt
    """
    anno_list = get_file_list(dst_anno_dir, extensions=('.txt'))
    print(len(anno_list))
    train_list, val_list = split_by_KFold(anno_list, nb_splits=10)
    print(len(train_list))
    print(len(val_list))

    c = 0
    with open(server['training_list_path'], 'w') as w:
        for f in train_list:
            id = os.path.splitext(os.path.basename(f))[0]
            # TODO: test for other types image
            img_path = os.path.join(server['IMG_DIR'], id + '.jpg')
            if os.path.exists(img_path):
                w.write('{}\n'.format(img_path))
                c += 1
            else:
                print('{} does not exist, skip'.format(img_path))
    print('finally, {} training data'.format(c))

    c = 0
    with open(server['val_list_path'], 'w') as w:
        for f in val_list:
            id = os.path.splitext(os.path.basename(f))[0]
            # TODO: test for other types image
            img_path = os.path.join(server['IMG_DIR'], id + '.jpg')
            if os.path.exists(img_path):
                w.write('{}\n'.format(img_path))
                c += 1
            else:
                print('{} does not exist, skip'.format(img_path))
    print('finally, {} val data'.format(c))
    # !! after created, cp all files of UEC256_labels_exp to UEC256_images


def maker():
    # step1. check the DB info of src_anno_dir and get anno_dict
    # & decide new category list for new task
    new_category_list = decide_category()

    # step2. make new anno DB by converting name of src_anno_dir -> label(num)
    make_YOLO_DB(server['src_anno_dir'], server['dst_anno_dir'],
                 new_category_list)

    # step3. based on dst_anno_dir to build DB
    make_train_val_list(server['dst_anno_dir'])

    """
    step4. check the category.names＠'category.names_path' 
    step5. check training / val txt 
    """
    exit()

    # step6. check the DB by YOLODB_check
    YOLODB_check(server['training_list_path'])
    YOLODB_check(server['val_list_path'])

    # step7. download & use labelImg tool to check


def make_food_only_DB(src_anno_dir, dst_anno_dir):
    anno_list = get_file_list(src_anno_dir, extensions=('.txt'))

    for f in anno_list:
        with open(f) as r:
            with open(os.path.join(dst_anno_dir, os.path.basename(f)), 'w') as w:
                for l in r.readlines():
                    pieces = l.rstrip().split(' ')
                    if pieces[0] == 'nonfood':
                        #w.write('{} {} {} {} {}\n'.format(0, pieces[1], pieces[2],pieces[3], pieces[4]))
                        print('nonfood, skip')
                    else:
                        w.write('{} {} {} {} {}\n'.format(0, pieces[1], pieces[2], pieces[3], pieces[4]))


def food_nonfood_maker():
    make_food_only_DB(server['src_anno_dir'], server['dst_anno_dir'])


def finc_v0_maker():
    make_train_val_list(server['dst_anno_dir'])


def main():
    maker()
    #finc_v0_maker()



if __name__ == '__main__':
    main()









