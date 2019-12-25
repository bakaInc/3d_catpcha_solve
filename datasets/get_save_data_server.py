from multiprocessing.synchronize import Lock

import requests

from multiprocessing import Process, Lock
import random
from requests.auth import AuthBase
from bs4 import BeautifulSoup
import re

import time


import threading
import math

_index = 0

def sendLock(lock):
    global _index

    data = requests.get('http://localhost/kcaptcha/')

    lock.acquire()
    try:
        _index += 1
    finally:
        lock.release()


def startThread(data, lock):
    global _index
    while _index < 30000:
        try:
            sendLock(lock)
        except Exception as exp:
            print('err')


if __name__ == '__main__':

    start = time.time()
    lock: Lock = Lock()
    pool = []
    for i in range(100):
        t = threading.Thread(target=startThread, args=('start', lock))
        pool.append(t)
        t.start()
    for i in pool:
        i.join()
    end = time.time()
    print(end - start)




