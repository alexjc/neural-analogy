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
import math
import time
import argparse
import itertools

# Scientific & Imaging Libraries
import numpy as np
import scipy.ndimage, scipy.misc
import sklearn.feature_extraction

# Numeric Computing (GPU)
import torch, torch.autograd, torchvision.models, torchvision.transforms


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
        self.vgg19.cuda()

    def extract(self, image, layers={1, 6, 11, 20, 29}):
        """Preprocess an image to be compatible with pre-trained model, and return required features.
        """
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        data = transform(image).cuda()
        current = torch.autograd.Variable(data).view(1, -1, image.shape[0], image.shape[1])
        for i in range(max(layers)+1):
            current = self.vgg19.features[i].forward(current)
            if i in layers:
                yield current.detach()

    def patches_score(self, first, y, x, second, v, u, padding):
        """Compute the match score of a patch in one image at (y,x) compared to the patch of second image at (v,u).
        """
        return 0.5 * (torch.sum(first.patches_repro[y, x] * second.patches_orign[v, u]) \
                    + torch.sum(first.patches_orign[y, x] * second.patches_repro[v, u]))

    def patches_initialize(self, first, second, padding=1):
        """Compute the scores for matching all patches based on the pre-initialized indices.
        """
        first_shape, second_shape = first.indices.shape, second.indices.shape
        for y in range(first_shape[0]):
            for x in range(first_shape[1]):
                v, u = first.indices[y, x]
                v, u = min(second_shape[0]-1, max(v, 0)), min(second_shape[1]-1, max(u, 0))
                first.scores[y, x] = self.patches_score(first, y, x, second, v, u, padding)

    def patches_propagate(self, first, second, i, padding=1):
        """Propagate all indices either towards the top-left or bottom-right, and update patch scores that are better.
        """
        even, first_shape, second_shape = bool((i%2) == 0), first.indices.shape, second.indices.shape
        for y in range(0, first_shape[0]) if even else range(first_shape[0] - 1, -1, -1):
            for x in range(0, first_shape[1]) if even else range(first_shape[1] - 1, -1, -1):
                for offset in [(0, -1 if even else +1), (-1 if even else +1, 0)]:
                    v, u = first.indices[min(first_shape[0] - 1, max(y + offset[0], 0)),
                                         min(first_shape[1] - 1, max(x + offset[1], 0))] - np.array(offset, dtype=np.int32)
                    v, u = min(second_shape[0] - 1, max(v, 0)), min(second_shape[1] - 1, max(u, 0))
                    score = self.patches_score(first, y, x, second, v, u, padding)
                    if score > first.scores[y, x]:
                        first.scores[y, x] = score
                        first.indices[y, x] = [v, u]
                    # if score > second.scores[v, u]:
                    #     second.scores[v, u] = score
                    #     second.indices[v, u] = [y, x]

    def patches_search(self, first, second, times, radius, padding=1):
        """Iteratively search out from each index pair, updating the patches found that match better.
        """
        first_shape, second_shape = first.indices.shape, second.indices.shape
        for y in range(first_shape[0]):
            for x in range(first_shape[1]):
                for _ in range(times):
                    if radius > 0:
                        v, u = first.indices[y, x]
                        v = min(second_shape[0] - 1, max(v + np.random.randint(-radius, radius+1), 0))
                        u = min(second_shape[1] - 1, max(u + np.random.randint(-radius, radius+1), 0))
                    else:
                        v = np.random.randint(0, second_shape[0])
                        u = np.random.randint(0, second_shape[1])

                    score = self.patches_score(first, y, x, second, v, u, padding)
                    if score > first.scores[y, x]:
                        first.scores[y, x] = score
                        first.indices[y, x] = [v, u]
                    # if score > second.scores[v, u]:
                    #     second.scores[v, u] = score
                    #     second.indices[v, u] = [y, x]


#======================================================================================================================
# Algorithm & Processing
#======================================================================================================================

class Buffer(object):

    def __init__(self, features, patches, *, weight, radius, padding):
        norms = torch.sqrt(torch.sum(torch.pow(features, 2.0), dim=1))
        self.features_orign = features

        patches_norm = np.sqrt(np.sum(patches ** 2.0, axis=(2,), keepdims=True))
        # patches_norm = np.sum(np.abs(patches), axis=(2,), keepdims=True)
        # patches_norm = 1.0
        self.patches_orign = torch.from_numpy(patches / patches_norm).cuda()
        self.patches_repro = self.patches_orign

        self.origin = None

        norms = (norms - norms.mean().expand_as(norms)) / norms.std().expand_as(norms)
        self.weights = weight / (1.0 + torch.exp(-10*(norms-0.5)))
        self.weight = weight
        self.radius = radius
        self.padding = padding

        indices = np.zeros((features.size(2), features.size(3), 2), dtype=np.int32)
        self.indices = indices
        self.scores = np.zeros((features.size(2), features.size(3)), dtype=np.float32)


class NeuralAnalogy(object):

    def __init__(self):
        self.model = Model()

    def extract(self, image, label):
        features, output, total = self.model.extract(image), [], 0
        weights = [0.1, 0.6, 0.7, 0.8, 1.0]
        radii = [4, 4, 6, 6, -1]
        padding = [2, 2, 1, 1, 1]
        for i, feature in enumerate(features):
            shape, memory = tuple(feature.size()[1:]), (feature.numel() * feature.element_size()) // 1024
            print(f'  - Layer {i} with {memory:,}kb features as {shape} array, padding {padding[i]}.    ', end='\r')
            total += memory

            patches = self.extract_patches(feature.data.cpu().numpy(), padding=padding[i])
            output.append(Buffer(feature, patches, weight=weights[i], radius=radii[i], padding=padding[i]))

        print(f'  - Extracted {len(output)} layers using total {total:,}kb memory from {label} image.')
        return reversed(output)

    def extract_patches(self, feature, padding):
        # feature_norm = np.sqrt(np.sum(feature ** 2.0, axis=(1,), keepdims=True))
        # feature = feature / feature_norm

        p = padding
        padded = np.pad(feature[0].transpose((1, 2, 0)), ((p, p), (p, p), (0, 0)), mode='edge')
        patches = sklearn.feature_extraction.image.extract_patches_2d(padded, patch_size=(p*2+1, p*2+1))
        patches = patches.reshape(feature.shape[2:]+(-1,)).astype(np.float32)
        return patches

    def reconstruct_patches(self, patches, padding):
        p = padding
        res = (patches.shape[0]+p*2, patches.shape[1]+p*2)
        patches = patches.reshape((np.product(patches.shape[:2]), p*2+1, p*2+1, -1))
        result = sklearn.feature_extraction.image.reconstruct_from_patches_2d(patches, (res)+(patches.shape[-1],))
        result = result[p:-p,p:-p].transpose((2,0,1))[np.newaxis]
        return result

    def warp_features(self, array, indices, padding):
        patches = self.extract_patches(array, padding)
        patches_warpd = np.zeros(indices.shape[:2]+patches.shape[2:], dtype=np.float32)
        # array_warpd = np.zeros(array.shape, dtype=np.float32)
        for y in range(indices.shape[0]):
            for x in range(indices.shape[1]):
                v, u = indices[y, x]
                patches_warpd[y, x] = patches[v, u]
                # array_warpd[0, :, y, x] = array[0, :, v, u]
        return self.reconstruct_patches(patches_warpd, padding)

    def process(self, first_image, second_image):
        print('\n{}Processing the image analogies specified on the command-line.{}'\
              .format(ansi.BLUE_B, ansi.BLUE))

        first_buffers = self.extract(first_image, label='first')
        second_buffers = self.extract(second_image, label='second')
        print('{}'.format(ansi.ENDC))

        first_previous, second_previous = None, None
        for i, (first, second) in enumerate(zip(first_buffers, second_buffers)):
            self.merge_flow(first, first_previous, second)
            first.origin = first_image

            self.merge_flow(second, second_previous, first)
            second.origin = second_image

            print(16-i*4)
            for _ in range(16-i*4):
                self.compute_flow(first, second, layer=f'{i}.f')
                self.compute_flow(second, first, layer=f'{i}.s')
            first_previous, second_previous = first, second

        return scipy.misc.toimage(first_image, cmin=0, cmax=255), scipy.misc.toimage(second_image, cmin=0, cmax=255)

    def compute_flow(self, first, second, *, layer):
        """
        indices = first.indices
        zoom = first.origin.shape[0] // indices.shape[0]
        warped_image = np.zeros(first.origin.shape[:2]+(4,), dtype=np.float32)
        for y in range(warped_image.shape[0]):
            for x in range(warped_image.shape[1]):
                v, u = indices[y // zoom, x // zoom]
                warped_image[y, x, :3] = second.origin[v * zoom + y % zoom, u * zoom + x % zoom]
                warped_image[y, x, 3] = 192 if (zoom > 1 and ((x//zoom) % 2) ^ ((y//zoom) % 2)) else 256
        scipy.misc.toimage(warped_image.clip(0.0, 255.0), cmin=0, cmax=255).save(f'frames/{layer}_scaled.png')
        """

        print('Computing warp via patch matching...')
        self.model.patches_initialize(first, second, padding=first.padding)
        score = first.scores.mean()

        print('  - Initialized with score', score)
        for i in range(4):
            last, start = score, time.time()
            if i % 2 == 0:
                self.model.patches_propagate(first, second, i // 2, padding=first.padding)
                print('  - Propagating best matches.', end='')
            if i % 2 == 1:
                r = first.radius if first.radius > 0 else 12
                self.model.patches_search(first, second, times=r//2, radius=first.radius, padding=first.padding)
                print(f'  - Searching radius={first.radius} times={r//2}.', end='')

            score = first.scores.mean() * 0.8 + 0.2 * score
            progress, elapsed = 100.0 * math.pow(last / score, 50.0), time.time() - start
            print(f' score={score} progress={progress:3.1f}% elapsed={elapsed}s')

        indices = first.indices
        zoom = first.origin.shape[0] // indices.shape[0]

        zoomed_field = np.zeros(first.origin.shape[:2]+(3,), dtype=np.float32)
        warped_image = np.zeros(first.origin.shape[:2]+(4,), dtype=np.float32)
        weight_array = np.zeros(first.origin.shape[:2], dtype=np.float32)
        scores_array = np.zeros(first.origin.shape[:2], dtype=np.float32)

        weights = first.weights.data.cpu().numpy()
        scores = first.scores

        for y in range(warped_image.shape[0]):
            for x in range(warped_image.shape[1]):
                v, u = indices[y // zoom, x // zoom]
                # warped_image[y, x, :3] = second.origin[v * zoom, u * zoom]
                warped_image[y, x, :3] = second.origin[v * zoom + y % zoom, u * zoom + x % zoom]
                warped_image[y, x, 3] = 192 if (zoom > 1 and ((x//zoom) % 2) ^ ((y//zoom) % 2)) else 256

                zoomed_field[y, x, 0] = indices[y // zoom, x // zoom, 0] * 255.0 / indices.shape[0]
                zoomed_field[y, x, 2] = indices[y // zoom, x // zoom, 1] * 255.0 / indices.shape[1]

                weight_array[y, x] = weights[0, :, y // zoom, x // zoom]
                scores_array[y, x] = scores[y // zoom, x // zoom]

        scipy.misc.toimage(zoomed_field.clip(0.0, 255.0), cmin=0, cmax=255).save(f'frames/{layer}_field.png')
        scipy.misc.toimage(warped_image.clip(0.0, 255.0), cmin=0, cmax=255).save(f'frames/{layer}_output.png')
        scipy.misc.toimage(weight_array * 255.0, cmin=0, cmax=255).save(f'frames/{layer}_weight.png')
        scipy.misc.toimage(first.origin, cmin=0, cmax=255).save(f'frames/{layer}_target.png')
        scipy.misc.toimage((scores_array - scores_array.min()) * 255.0 / (scores_array.max() - scores_array.min()), cmin=0, cmax=255).save(f'frames/{layer}_scores.png')

    def merge_flow(self, this, parent, other):
        if parent is None:
            # 1) Random search, this happens immediately after initalization anyway.
            # this.indices[:, :, 0] = np.random.randint(low=0, high=other.indices.shape[0], size=this.indices.shape[:2])
            # this.indices[:, :, 1] = np.random.randint(low=0, high=other.indices.shape[1], size=this.indices.shape[:2])

            # 2) Identity grid, this is harder to reproduce with random search.
            this.indices[:, :, 0] = np.array([(y*other.indices.shape[0])//this.indices.shape[0] for y in range(this.indices.shape[0])], dtype=np.int32).reshape((-1, 1))
            this.indices[:, :, 1] = np.array([(x*other.indices.shape[1])//this.indices.shape[1] for x in range(this.indices.shape[1])], dtype=np.int32).reshape((1, -1))
            return

        indices, p = parent.indices * 2, this.padding
        zoomed = scipy.ndimage.interpolation.zoom(indices, zoom=(2, 2, 1), order=0)
        assert this.indices.shape == zoomed.shape
        this.indices[:, :] = zoomed

        features_other = other.features_orign.data.cpu().numpy()
        features_warpd = self.warp_features(features_other, zoomed, p)

        w = this.weights.data.cpu().numpy()
        f = this.features_orign.data.cpu().numpy()
        features_repro = f * w + (1.0 - w) * features_warpd
        patches_repro = self.extract_patches(features_repro, padding=p)
        patches_norm = np.sqrt(np.sum(patches_repro ** 2.0, axis=(2,), keepdims=True))
        # patches_norm = np.sum(np.abs(patches_repro), axis=(2,), keepdims=True)
        # patches_norm = 1.0
        this.patches_repro = torch.from_numpy(patches_repro / patches_norm).cuda()


def main():
    # Configure all options to be passed in from the command-line.
    parser = argparse.ArgumentParser(description='Transform one image into another and back again by computing analogies.',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_arg = parser.add_argument
    add_arg('first',                type=str,                help='First input image, usually called A.')
    add_arg('second',               type=str,                help='Second input image, usually named B’.')
    args = parser.parse_args()

    # Load the images from disk, always converting to 3 channels.
    first_input = scipy.ndimage.imread(args.first, mode='RGB')
    second_input = scipy.ndimage.imread(args.second, mode='RGB')

    # Run the main algorithm to generate the output images.
    analogy = NeuralAnalogy()
    first_output, second_output = analogy.process(first_input, second_input)

    # Save the results to disk with a specific suffix.
    first_output.save(os.path.splitext(args.first)[0]+'_na.png')
    second_output.save(os.path.splitext(args.second)[0]+'_na.png')


if __name__ == "__main__":
    main()
