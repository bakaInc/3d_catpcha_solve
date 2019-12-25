
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

choices = u'абвгдеёжзийклмнопрстуфхцчшщъыьэюя'
data = list(itertools.permutations(choices, 4))
print(len(data))

with open('interdata.txt', 'w', encoding='UTF-8') as file:
    for i in data:
        file.write(str(''.join(i)) + '\n')
print('end')