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
import torch, torch.autograd, torchvision.models, torchvision.transforms


#======================================================================================================================
# Convolution Network
#======================================================================================================================
class Model(object):

    def __init__(self):
        """Loads the pre-trained VGG19 convolution layers from the PyTorch vision module.
        """
        self.vgg19 = torchvision.models.vgg19(pretrained=False)
        del self.vgg19.classifier
        self.vgg19.load_state_dict(torch.load('vgg19_conv.pth'))

    def extract(self, image, layers={1,6,11,20,29}):
        """Preprocess an image to be compatible with pre-trained model, and return required features.
        """
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        variable = torch.autograd.Variable(transform(image)).view(1, -1, image.shape[0], image.shape[1])
        current, output = variable, []
        for i in range(max(layers)+1):
            current = self.vgg19.features[i].forward(current)
            if i not in layers:
                continue

            norms = torch.sum(torch.abs(current), dim=1).expand_as(current)
            yield (current / norms).detach()

    def patches_score(self, first, y, x, second, v, u):
        """Compute the match score of a patch in one image at (y,x) compared to the patch of second image at (v,u).
        """
        score = 0.0
        for j, i in [(-1,-1),(-1,0),(-1,+1),(0,-1),(0,0),(0,+1),(+1,-1),(+1,0),(+1,+1)]:
            score += torch.sum((first.input[0,:,y+j,x+i] - second.repro[0,:,1+v+j,1+u+i]) ** 2.0) \
                   + torch.sum((first.repro[0,:,y+j,x+i] - second.input[0,:,1+v+j,1+u+i]) ** 2.0)
        return score.detach().data[0] # perf?

    def patches_initialize(self, first, second):
        """Compute the scores for matching all patches based on the pre-initialized indices.
        """
        for v in range(first.indices.size(0)):
            for u in range(first.indices.size(1)):
                y, x = first.indices[v,u]
                first.scores[v,u] = self.patches_score(first, y, x, second, v, u)

    def patches_propagate(self, first, second, i):
        """Propagate all indices either towards the top-left or bottom-right, and update patch scores that are better.
        """
        even, indices = bool((i%2)==0), first.indices
        for b in range(0, indices.size(0)) if even else range(indices.size(0)-1, -1, -1):
            for a in range(0, indices.size(1)) if even else range(indices.size(1)-1, -1, -1):
                for offset in [(0, -1 if even else +1), (-1 if even else +1, 0)]:
                    i1, i2 = indices[min(indices.size(0)-1, max(b+offset[0], 0)), min(indices.size(1)-1, max(a+offset[1], 0))]\
                                    - torch.from_numpy(np.array(offset, dtype=np.int32))
                    i1, i2 = min(first.input.size(2)-2, max(i1, 1)), min(first.input.size(3)-2, max(i2, 1))
                    score = self.patches_score(first, i1, i2, second, b, a)
                    if score < first.scores[b,a]:
                        first.scores[b,a] = score
                        indices[b,a] = torch.from_numpy(np.array([i1, i2], dtype=np.int32))

    def patches_search(self, first, second, k):
        """Iteratively search out from each index pair, updating the patches found that match better.
        """
        indices = first.indices
        for b in range(indices.size(0)):
            for a in range(indices.size(1)):
                for radius in range(k+1, 0, -1):
                    w, i1, i2 = 2 ** (radius-1), *indices[b,a]
                    i1 = min(first.input.size(2)-2, max(i1 + np.random.randint(-w, +w), 1))
                    i2 = min(first.input.size(3)-2, max(i2 + np.random.randint(-w, +w), 1))
                    score = self.patches_score(first, i1, i2, second, b, a)
                    if score < first.scores[b,a]:
                        first.scores[b,a] = score
                        indices[b,a] = torch.from_numpy(np.array([i1, i2], dtype=np.int32))


#======================================================================================================================
# Algorithm & Processing
#======================================================================================================================

class Buffer(object):

    def __init__(self, features):
        self.input = features
        self.repro = features
        self.output = None

        indices = np.zeros((features.size(2) - 2, features.size(3) - 2, 2), dtype=np.int32)
        indices[:,:,0] = np.random.randint(low=1, high=features.size(2)-1, size=indices.shape[:2])
        indices[:,:,1] = np.random.randint(low=1, high=features.size(3)-1, size=indices.shape[:2])
        self.indices = torch.from_numpy(indices)

        scores = np.zeros((features.size(2) - 2, features.size(3) - 2), dtype=np.float32)
        self.scores = torch.from_numpy(scores)
    
    def merge(self, buffer):
        if buffer is None:
            return

        import scipy.ndimage.interpolation
        padded = np.pad(buffer.indices.numpy() * 2, ((1,1), (1,1), (0,0)), mode='edge')
        zoomed = scipy.ndimage.interpolation.zoom(padded, zoom=(2,2,1), order=0)
        cropped = zoomed[1:-1,1:-1]

        assert self.indices.size() == cropped.shape
        self.indices[:,:] = torch.from_numpy(cropped[:,:])


class NeuralAnalogy(object):

    def __init__(self):
        self.model = Model()

    def extract(self, image, label):
        features, output, total = self.model.extract(image), [], 0
        for i, feature in enumerate(features):
            shape, memory = tuple(feature.size()[1:]), (feature.numel() * feature.element_size()) // 1024
            print(f'\r  - Layer {i} with {memory:,}kb features as {shape} array.', end='')
            total += memory
            output.append(Buffer(feature))
        print(f'\r  - Extracted {len(output)} layers using total {total:,}kb memory from {label} image.')
        return reversed(output)

    def process(self, first_image, second_image):
        print('\n{}Processing the image analogies specified on the command-line.{}'\
              .format(ansi.BLUE_B, ansi.BLUE))

        first_buffers = self.extract(first_image, label='first')
        second_buffers = self.extract(second_image, label='second')
        print('{}'.format(ansi.ENDC))

        first_previous, second_previous = None, None
        for first, second in zip(first_buffers, second_buffers):
            first.merge(first_previous)
            second.merge(second_previous)

            self.match_patches(first, second)
            first_previous, second_previous = first, second

        return scipy.misc.toimage(first_image, cmin=0, cmax=255), scipy.misc.toimage(second_image, cmin=0, cmax=255)

    def match_patches(self, first, second):
        print('Patches initializing...')
        self.model.patches_initialize(first, second)

        for i in range(2):
            print('Patches propagating...', first.scores.mean())
            self.model.patches_propagate(first, second, i)
            print('Patches searching...', first.scores.mean())
            self.model.patches_search(first, second, 2)


if __name__ == "__main__":
    first_input = scipy.ndimage.imread(args.first, mode='RGB')
    second_input = scipy.ndimage.imread(args.second, mode='RGB')

    analogy = NeuralAnalogy()
    first_output, second_output = analogy.process(first_input, second_input)

    first_output.save(os.path.splitext(args.first)[0]+'_na.png')
    second_output.save(os.path.splitext(args.second)[0]+'_na.png')
