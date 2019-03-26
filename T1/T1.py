#!/usr/bin/env python
# coding: utf-8
import numpy as np


def center_of_mass(image):
    x_sum = np.sum([np.sum(np.flatnonzero(x)) for x in image])
    y_sum = np.sum([np.sum(np.flatnonzero(x)) for x in np.rot90(image)])
    mass = np.sum(image)
    return (x_sum / mass, y_sum / mass)


def get_boundaries(image):
    left = np.min([np.min(np.flatnonzero(x), initial=10e10) for x in image])
    top = np.min([np.min(np.flatnonzero(x), initial=10e10) for x in np.rot90(image)])
    right = np.max([np.max(np.flatnonzero(x), initial=-10e10) for x in image])
    bottom = np.max([np.max(np.flatnonzero(x), initial=-10e10) for x in np.rot90(image)])
    return top, right, bottom, left


def get_corners(boundaries):
    top, right, bottom, left = boundaries
    return (left, top), (right, top), (left, bottom), (right, bottom)


def get_sizes(boundaries):
    top, right, bottom, left = boundaries
    return right - left, bottom - top


def get_surrounding_region(point, expansion=1):
    x, y = int(point[0]), int(point[1])
    region = []
    for i in range(x - expansion, x + expansion + 1):
        for j in range(y - expansion, y + expansion + 1):
            region.append((i, j))
    return region


def image_to_vector(image, label):
    # get center of mass
    center = center_of_mass(image)
    # get center of mass surroindings (9x9 square)
    region = np.array(get_surrounding_region(center))
    values = image[region[:, 1], region[:, 0]]
    # get 'painted' proportion
    boundaries = get_boundaries(image)
    size = get_sizes(boundaries)
    mass = np.sum(image)
    proportion = mass / (size[0] * size[1])
    return [proportion, *values, label]


def reconocedorSC(X):
    vector = image_to_vector(X, None)
    return int(round(np.dot(vector[:-1], learned_parameters)))


# obtained after training
learned_parameters = np.array([1.52370782e-17, 1.11111111e-01, 1.11111111e-01,
                               1.11111111e-01, 1.11111111e-01, 1.11111111e-01,
                               1.11111111e-01, 1.11111111e-01, 1.11111111e-01,
                               1.11111111e-01])


if __name__ == '__main__':
    print(reconocedorSC(np.load('S01.npy')))
