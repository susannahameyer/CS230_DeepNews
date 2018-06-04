#!/usr/bin/env python

import subprocess

ls = subprocess.Popen(['ls', '230_tokenize', '-F'], stdout=subprocess.PIPE)
next = subprocess.Popen(['grep', '-v', '/'], stdin=ls.stdout, stdout=subprocess.PIPE)
n_articles = subprocess.check_output(['wc', 'l'], stdin=next.stdout)
ls.wait()
print n_articles
n_dev = n_articles * 2 / 10
n_test = n_articles * 1 / 10
n_train = n_articles * 7 / 10

subprocess.call(["sort", "-R", "CS230_tokenize"])
subprocess.Popen(["mv" "'ls"], stdout=subprocess.PIPE)
subprocess.Popen(['head', -1*n_dev, '', 'dev/'], stdin=n_articles)

subprocess.call(["sort", "-R", "CS230_tokenize"])
subprocess.Popen(["mv" "'ls"], stdout=subprocess.PIPE)
subprocess.Popen(['head', -1*n_test, '', 'test/'], stdin=n_articles)
)
subprocess.call(["mv", ".", "train/"])

print n_dev, n_test, n_train, n_articles
