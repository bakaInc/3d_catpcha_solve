# -*- coding: utf-8 -*-

import argparse
import json
import string
import os
import shutil
import uuid
from captcha.image import ImageCaptcha
import codecs
import itertools
import random
from PIL import Image



def _gen_captcha(get_data, save_dir):
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    img_list = os.listdir(get_data)
    print('load img files', len(img_list))
    i = 0
    for img_file in img_list:
        _img = Image.open(os.path.join(get_data, img_file))
        #width = _img.size[0]
        #height = _img.size[1]

        #_img = _img.crop((0, 0, width, height - 12))
        fn = os.path.join(save_dir, '%s_%s.png' % (img_file[:5], uuid.uuid4()))
        #print(img_file, fn)
        _img.save(fn)
        i += 1
        if i > 25000:
            break


    _img = Image.open(os.path.join(get_data, img_list[0]))
    print('generated and save')
    print('size', _img.size[0], _img.size[1])
    return _img.size[0], _img.size[1]

def build_file_path(data_dir, npi, n_epoch, x):
    return os.path.join(data_dir, 'char-%s-epoch-%s' % (npi, n_epoch), x)


def gen_dataset():
    npi = 5
    n_epoch = 2

    data_dir = r'E:\\Python\\base_captcha-tensorflow\\datasets\\images'

    train_path = build_file_path(data_dir, npi, n_epoch, 'train')
    test_path = build_file_path(data_dir, npi, n_epoch, 'test')
    print(train_path, test_path)

    #get_data = 'E:\\Program Files\\XamppPHP\\htdocs\\kcaptcha\\data'
    get_test = 'E:\\Program Files\\XamppPHP\\htdocs\\kcaptcha\\test'

    #width, height = _gen_captcha(get_data=get_data, save_dir=train_path)
    width, height = _gen_captcha(get_data=get_test, save_dir=test_path)

    STRING_DATA = '23456789abcdegkmnpqsuvxyz'
    META_FILENAME = 'meta.json'
    meta = {
        'num_per_image': npi,
        'label_size': len(STRING_DATA),
        'label_choices': STRING_DATA,
        'n_epoch': n_epoch,
        'width': width,
        'height': height,
    }

    meta_filename = build_file_path(data_dir, npi, n_epoch, META_FILENAME)

    print(meta)
    with codecs.open(meta_filename, 'w', encoding='UTF-8') as f:
        json.dump(meta, f, indent=4)
    print('write meta info in %s' % meta_filename)


if __name__ == '__main__':

    gen_dataset()
