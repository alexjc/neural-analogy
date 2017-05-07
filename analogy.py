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

# Built-in Python Modules
import os
import argparse

# Scientific & Imaging Libraries
import numpy as np
import scipy.ndimage, scipy.misc

# Numeric Computing (GPU)
import torch, torch.autograd, torchvision.models, torchvision.transforms


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
            yield torch.nn.functional.pad(current, pad=(1, 1, 1, 1), mode='replicate').detach()

    def patches_score(self, first, y, x, second, v, u):
        """Compute the match score of a patch in one image at (y,x) compared to the patch of second image at (v,u).
        """
        score = 0.0
        for j, i in [(-1,-1),(-1,0),(-1,+1),(0,-1),(0,0),(0,+1),(+1,-1),(+1,0),(+1,+1)]:
            score += torch.sum((first.repro[0,:,1+y+j,1+x+i] - second.input[0,:,1+v+j,1+u+i]) ** 2.0) \
                   + torch.sum((first.input[0,:,1+y+j,1+x+i] - second.repro[0,:,1+v+j,1+u+i]) ** 2.0)
        return score.detach().data[0] # perf?

    def patches_initialize(self, first, second):
        """Compute the scores for matching all patches based on the pre-initialized indices.
        """
        for y in range(first.indices.size(0)):
            for x in range(first.indices.size(1)):
                v, u = first.indices[y,x]
                first.scores[y,x] = self.patches_score(first, y, x, second, v, u)

    def patches_propagate(self, first, second, i):
        """Propagate all indices either towards the top-left or bottom-right, and update patch scores that are better.
        """
        even, indices = bool((i%2)==0), first.indices
        for y in range(0, indices.size(0)) if even else range(indices.size(0)-1, -1, -1):
            for x in range(0, indices.size(1)) if even else range(indices.size(1)-1, -1, -1):
                for offset in [(0, -1 if even else +1), (-1 if even else +1, 0)]:
                    v, u = indices[min(indices.size(0)-1, max(y+offset[0], 0)), min(indices.size(1)-1, max(x+offset[1], 0))]\
                                    - torch.from_numpy(np.array(offset, dtype=np.int32))
                    v, u = min(indices.size(0)-1, max(v, 0)), min(indices.size(1)-1, max(u, 0))
                    score = self.patches_score(first, y, x, second, v, u)
                    if score < first.scores[y,x]:
                        first.scores[y,x] = score
                        indices[y,x] = torch.from_numpy(np.array([v, u], dtype=np.int32))

    def patches_search(self, first, second, times, radius):
        """Iteratively search out from each index pair, updating the patches found that match better.
        """
        indices, scores = first.indices, first.scores
        for y in range(indices.size(0)):
            for x in range(indices.size(1)):
                for _ in range(times):
                    v, u = first.indices[y,x]
                    v = min(indices.size(0)-1, max(v + np.random.randint(-radius, +radius), 0))
                    u = min(indices.size(1)-1, max(u + np.random.randint(-radius, +radius), 0))
                    score = self.patches_score(first, y, x, second, v, u)
                    if score < scores[y,x]:
                        scores[y,x] = score
                        indices[y,x] = torch.from_numpy(np.array([v, u], dtype=np.int32))


#======================================================================================================================
# Algorithm & Processing
#======================================================================================================================

class Buffer(object):

    def __init__(self, features, *, weight, radius):
        norms = torch.sum(torch.abs(features), dim=1)
        self.input = features / norms.expand_as(features)
        self.repro = features / norms.expand_as(features)
        self.origin = None

        norms = (norms - norms.mean().expand_as(norms)) / norms.std().expand_as(norms)
        self.weights = weight / (1.0 + torch.exp(-10*(norms-0.5)))
        self.weight = weight
        self.radius = radius

        indices = np.zeros((features.size(2) - 2, features.size(3) - 2, 2), dtype=np.int32)
        indices[:,:,0] = np.random.randint(low=0, high=features.size(2)-2, size=indices.shape[:2])
        indices[:,:,1] = np.random.randint(low=0, high=features.size(3)-2, size=indices.shape[:2])
        self.indices = torch.from_numpy(indices)

        scores = np.zeros((features.size(2) - 2, features.size(3) - 2), dtype=np.float32)
        self.scores = torch.from_numpy(scores)
    
    def merge(self, parent, other):
        if parent is None:
            return

        padded = parent.indices.numpy() * 2
        zoomed = scipy.ndimage.interpolation.zoom(padded, zoom=(2,2,1), order=0)
        assert self.indices.size() == zoomed.shape
        self.indices[:,:] = torch.from_numpy(zoomed)

        size = np.array(self.input.size(), dtype=np.int32) - [0, 0, 2, 2]
        warped_features = np.zeros(size, dtype=np.float32)
        for y in range(warped_features.shape[2]):
            for x in range(warped_features.shape[3]):
                v, u = zoomed[y,x]
                warped_features[0,:,y,x] = other.input[0,:,1+v,1+u].data.numpy()

        repro = torch.autograd.Variable(torch.from_numpy(warped_features))
        repro = torch.nn.functional.pad(repro, pad=(1, 1, 1, 1), mode='replicate')
        assert self.repro.size() == repro.size()

        w = self.weights.expand_as(self.repro)
        self.repro = self.repro * w + (1.0 - w) * repro


class NeuralAnalogy(object):

    def __init__(self):
        self.model = Model()

    def extract(self, image, label):
        features, output, total = self.model.extract(image), [], 0
        weights = [0.1, 0.6, 0.7, 0.8, 1.0]
        radii = [4, 4, 6, 6, 10]
        for i, feature in enumerate(features):
            shape, memory = tuple(feature.size()[1:]), (feature.numel() * feature.element_size()) // 1024
            print(f'\r  - Layer {i} with {memory:,}kb features as {shape} array.', end='')
            total += memory
            output.append(Buffer(feature, weight=weights[i], radius=radii[i]))
        print(f'\r  - Extracted {len(output)} layers using total {total:,}kb memory from {label} image.')
        return reversed(output)

    def process(self, first_image, second_image):
        print('\n{}Processing the image analogies specified on the command-line.{}'\
              .format(ansi.BLUE_B, ansi.BLUE))

        first_buffers = self.extract(first_image, label='first')
        second_buffers = self.extract(second_image, label='second')
        print('{}'.format(ansi.ENDC))

        first_previous, second_previous = None, None
        for i, (first, second) in enumerate(zip(first_buffers, second_buffers)):
            first.merge(first_previous, second)
            first.origin = first_image

            second.merge(second_previous, first)
            second.origin = second_image

            self.compute_flow(first, second, layer=f'{i}.f')
            # self.compute_output(first, second)

            self.compute_flow(second, first, layer=f'{i}.s')
            # self.compute_output(second, first)
            first_previous, second_previous = first, second

        return scipy.misc.toimage(first_image, cmin=0, cmax=255), scipy.misc.toimage(second_image, cmin=0, cmax=255)

    def compute_flow(self, first, second, *, layer):
        print('Computing warp via patch matching...')
        self.model.patches_initialize(first, second)

        last = 1.0
        for i in range(16):
            score = first.scores.mean()
            if score == 0.0 or score == last or (score/last) > 0.99:
                break

            if i % 2 == 0:
                print('  - Propagating best matches.', int(100 * score / last), score)
                self.model.patches_propagate(first, second, i // 2)
            if i % 2 == 1:
                print(f'  - Searching in radius {first.radius}.', int(100 * score / last), score)
                self.model.patches_search(first, second, times=4, radius=first.radius)
            last = score

        print('  - Warping resulting image.', first.scores.mean())
        indices = first.indices.numpy()
        zoom = first.origin.shape[0] // indices.shape[0]

        zoomed_field = np.zeros(first.origin.shape[:2]+(3,), dtype=np.float32)
        warped_image = np.zeros(first.origin.shape[:2]+(4,), dtype=np.float32)
        weight_array = np.zeros(first.origin.shape[:2], dtype=np.float32)
        scores_array = np.zeros(first.origin.shape[:2], dtype=np.float32)

        weights = first.weights.data.numpy()
        scores = first.scores.numpy()

        for y in range(warped_image.shape[0]):
            for x in range(warped_image.shape[1]):
                v, u = indices[y // zoom, x // zoom]
                warped_image[y,x,:3] = second.origin[v * zoom + y % zoom, u * zoom + x % zoom]
                warped_image[y,x,3] = 192 if (zoom>1 and ((x//zoom) % 2) ^ ((y//zoom) % 2)) else 256

                zoomed_field[y,x,0] = indices[y // zoom, x // zoom, 0] * 255.0 / indices.shape[0]
                zoomed_field[y,x,2] = indices[y // zoom, x // zoom, 1] * 255.0 / indices.shape[1]

                weight_array[y,x] = weights[0,:,1+y // zoom, 1+x // zoom]
                scores_array[y,x] = scores[y // zoom, x // zoom]

        scipy.misc.toimage(zoomed_field.clip(0.0, 255.0), cmin=0, cmax=255).save(f'frames/{layer}_field.png')
        scipy.misc.toimage(warped_image.clip(0.0, 255.0), cmin=0, cmax=255).save(f'frames/{layer}_output.png')
        scipy.misc.toimage(weight_array * 255.0, cmin=0, cmax=255).save(f'frames/{layer}_weight.png')
        scipy.misc.toimage((scores_array - scores_array.min()) * 255.0 / (scores_array.max() - scores_array.min()), cmin=0, cmax=255).save(f'frames/{layer}_scores.png')


if __name__ == "__main__":
    first_input = scipy.ndimage.imread(args.first, mode='RGB')
    second_input = scipy.ndimage.imread(args.second, mode='RGB')

    analogy = NeuralAnalogy()
    first_output, second_output = analogy.process(first_input, second_input)

    first_output.save(os.path.splitext(args.first)[0]+'_na.png')
    second_output.save(os.path.splitext(args.second)[0]+'_na.png')
