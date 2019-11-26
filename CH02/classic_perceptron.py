# _*_ coding: utf-8 _*_

"""The classic perception to classify data set"""

import numpy as np
import random
# import os


def loadData(file_dir):
    try:
        trainingset = []
        with open(file_dir) as data1:
            temp = data1.readlines()
            noOfpoint = len(temp)   # the number of data points
            for data in temp:
                data = data.strip().split(' ')
                for val in data:
                    trainingset.append(float(val))
        noOffeature = len(trainingset) // noOfpoint
        return trainingset, noOfpoint, noOffeature
    except TypeError:
        print('Please input the directory of the data in string form')


def cal(trainingset):
    global omega, b, n
    for trainingpoint in trainingset:
        res = 0
        for i in range(n-1):
            res += trainingpoint[i] * omega[i]
        res += b
        res *= trainingpoint[-1]
        if res <= 0:
            update(trainingpoint)
            return res
        else:
            continue
    if res > 0:
        return res


def update(trainingpoint):
    # update omega and b
    global omega, b, n
    for i in range(n-1):
        omega[i] += trainingpoint[i] * trainingpoint[-1]
    b += trainingpoint[-1]


def check(trainingset, m):
    Done = False
    while not Done:
        if cal(trainingset) > 0:
            Done = True
        else:
            continue
    return


if __name__ == '__main__':
    # m: the number of data points
    # n: the number of features
    trainingSet, m, n = loadData('data1.txt')
    trainingSet = np.array(trainingSet).reshape(m, n)

    # initial parameters
    omega = [0] * (n-1)
    b = 0
    eta = 1

    check(trainingSet, m)
    print(omega, b)
