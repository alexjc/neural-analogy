#!/usr/bin/env python3
"""                          _                     _                    
  _ __   ___ _   _ _ __ __ _| |   __ _ _ __   __ _| | ___   __ _ _   _  
 | '_ \ / _ \ | | | '__/ _` | |  / _` | '_ \ / _` | |/ _ \ / _` | | | | 
 | | | |  __/ |_| | | | (_| | | | (_| | | | | (_| | | (_) | (_| | |_| | 
 |_| |_|\___|\__,_|_|  \__,_|_|  \__,_|_| |_|\__,_|_|\___/ \__, |\__, | 
                                                           |___/ |___/  
"""
#
# Copyright (c) 2017, creative.ai
#
# Neural Analogy is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General
# Public License version 3. This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
# without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#

__version__ = '0.1'

import os
import argparse


# Configure all options to be passed in from the command-line.
parser = argparse.ArgumentParser(description='Transform one image into another and back again by computing analogies.',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
add_arg = parser.add_argument
add_arg('first',                type=str,                help='First input image, usually called A.')
add_arg('second',               type=str,                help='Second input image, usually named B’.')
args = parser.parse_args()


#----------------------------------------------------------------------------------------------------------------------

# Color coded output helps visualize the information a little better, plus it looks cool!
class ansi:
    WHITE = '\033[0;97m'
    WHITE_B = '\033[1;97m'
    YELLOW = '\033[0;33m'
    YELLOW_B = '\033[1;33m'
    RED = '\033[0;31m'
    RED_B = '\033[1;31m'
    BLUE = '\033[0;94m'
    BLUE_B = '\033[1;94m'
    CYAN = '\033[0;36m'
    CYAN_B = '\033[1;36m'
    ENDC = '\033[0m'

def error(message, *lines):
    string = "\n{}ERROR: " + message + "{}\n" + "\n".join(lines) + ("{}\n" if lines else "{}")
    print(string.format(ansi.RED_B, ansi.RED, ansi.ENDC))
    sys.exit(-1)

def warn(message, *lines):
    string = "\n{}WARNING: " + message + "{}\n" + "\n".join(lines) + "{}\n"
    print(string.format(ansi.YELLOW_B, ansi.YELLOW, ansi.ENDC))

print("""{}   {}\nTransformation and resynthesis of images powered by Deep Learning!{}
  - Code licensed as AGPLv3, models under CC BY-NC-SA.{}""".format(ansi.CYAN_B, __doc__, ansi.CYAN, ansi.ENDC))

# Scientific & Imaging Libraries
import numpy as np
import scipy.ndimage, scipy.misc

# Numeric Computing (GPU)
import torch


#======================================================================================================================
# Convolution Network
#======================================================================================================================
class Model(object):

    def __init__(self):
        pass


#======================================================================================================================
# Algorithm & Processing
#======================================================================================================================
class NeuralAnalogy(object):

    def __init__(self):
        print('\n{}Processing the image analogies specified on the command-line.{}'\
              .format(ansi.BLUE_B, ansi.BLUE))

        self.model = Model()
        print('{}'.format(ansi.ENDC))

    def process(self, first, second):
        return scipy.misc.toimage(first, cmin=0, cmax=255), scipy.misc.toimage(second, cmin=0, cmax=255)


if __name__ == "__main__":
    first_input = scipy.ndimage.imread(args.first, mode='RGB')
    second_input = scipy.ndimage.imread(args.second, mode='RGB')

    analogy = NeuralAnalogy()
    first_output, second_output = analogy.process(first_input, second_input)

    first_output.save(os.path.splitext(args.first)[0]+'_na.png')
    second_output.save(os.path.splitext(args.second)[0]+'_na.png')
