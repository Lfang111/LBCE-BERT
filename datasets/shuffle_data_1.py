#!/usr/bin/python
# -*- coding: UTF-8 -*-

import random
from sys import *

if len(argv) != 3:
    exit('Usage: python %s infile outfile' % argv[0])

data_all = open(argv[1], 'r').readlines()
random.shuffle(data_all)
output = open(argv[2],'w')
for line in data_all:
    output.write('%s' % line)
output.close()
