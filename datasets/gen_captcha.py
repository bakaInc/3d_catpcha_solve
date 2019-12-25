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

META_FILENAME = 'meta.json'
STRING_DATA = 'абвгдеёжзийклмнопрстуфхцчшщъыьэюя'
FLAGS = None

def _gen_captcha(img_dir, n, width, height):
    #print('dir ' + img_dir)
    if os.path.exists(img_dir):
        shutil.rmtree(img_dir)
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)

    print('n', n)
    font_files = ['63.ttf', '5751.ttf']
    image = ImageCaptcha(width=width, height=height, fonts=font_files)

    with open('russian.txt', 'r',  encoding='UTF-8') as file:
        file_data = file.read().split('\n')

    for _ in range(n):
        for i in file_data:
            captcha = ''.join(i)
            fn = os.path.join(img_dir, '%s_%s.png' % (captcha, uuid.uuid4()))
            image.write(captcha, fn)

    choices = STRING_DATA

    data = list(itertools.permutations(choices, 4))
    print(len(data))
    length = 10
    for _ in range(n):
        for i in range(length):#num_per_image
            x = random.choice(data)
            captcha = ''.join(x)
            fn = os.path.join(img_dir, '%s_%s.png' % (captcha, uuid.uuid4()))
            image.write(captcha, fn)


def build_file_path(data_dir, npi, n_epoch, x):
    return os.path.join(data_dir, 'char-%s-epoch-%s' % (npi, n_epoch), x)


def gen_dataset(data_dir, n_epoch, npi, test_ratio):
    width = 40 + 20 * npi#40 + x * 20
    height = 100#100

    # meta info
    meta = {
        'num_per_image': npi,
        'label_size': len(STRING_DATA),
        'label_choices': STRING_DATA,
        'n_epoch': n_epoch,
        'width': width,
        'height': height,
    }
    #print(meta)

    train_path = build_file_path(data_dir, npi, n_epoch, 'train')
    test_path = build_file_path(data_dir, npi, n_epoch, 'test')
    print(train_path, test_path)

    _gen_captcha(train_path, n_epoch, width, height)
    _gen_captcha(test_path, max(1, int(n_epoch * test_ratio)), width, height)

    meta_filename = build_file_path(data_dir, npi, n_epoch, META_FILENAME)

    print(meta)
    with codecs.open(meta_filename, 'w', encoding='UTF-8') as f:
        json.dump(meta, f, indent=4)
    print('write meta info in %s' % meta_filename)


if __name__ == '__main__':
    data_dir = 'E:\\Python\\captcha-tensorflow\\datasets\\images'
    n_epoch = 2
    nip = 4
    ratio = 0.2

    gen_dataset(data_dir, n_epoch, nip, ratio)
