#!/usr/bin/env python
import shutil
import os
import random

#os.mkdir('/dev')
#os.mkdir('/test')
#os.mkdir('/train')

def make_dir(mode):
    while True:
        try:
            os.makedirs(mode)
            break
        except OSError, e:
            if e.errno != os.errno.EEXIST:
                raise
            # time.sleep might help here
            pass

dev = make_dir('dev')
test = make_dir('test')
train = make_dir('train')
files = [f for f in os.listdir('.') if os.path.isfile(f)]
random.shuffle(files)
#print len(files)
num_articles = len(files)/2

sublist = [f[:f.find('.')] for f in files if f.endswith(".txt") and 'label' not in f]
import pdb; pdb.set_trace()

dev = random.sample(sublist, num_articles * 2 / 10)
for index, article in enumerate(dev):
    art = article + '.txt'
    shutil.move(art, 'dev/')
    sublist.remove(article)
    label = article + "_label.txt"
    shutil.move(label, 'dev/')



test = random.sample(sublist, num_articles * 1 / 10)
for index, article in enumerate(test):
    art = article + '.txt'
    shutil.move(art, 'test/')
    sublist.remove(article)
    label = article + "_label.txt"
    shutil.move(label, 'test/')

train = sublist
for article in train:
    art = article + '.txt'
    #sublist.remove(article)
    shutil.move(art, 'train/')
    label = article + "_label.txt"
    shutil.move(label, 'train/')
