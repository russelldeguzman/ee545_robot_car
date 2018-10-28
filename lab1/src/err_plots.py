#!/usr/bin/env python

import sys
import csv
import numpy as np
import matplotlib.pyplot as plt

CSV_PATH = "/home/car-user/racecar_ws/src/ee545_robot_car/lab1/src/"
CSV_FILE_LIST = ["pid-error.csv", "pid-error2.csv"]

err_sets = []
# with open(CSV_PATH + CSV_FILE, mode="r") as err_file:\
for file in CSV_FILE_LIST:
    data = np.genfromtxt(CSV_PATH + file, delimiter=" ")
    err_sets.append(data)
# print(np.size(data))

for idx, err_data in enumerate(err_sets):
    plt.plot(err_data[:, 0], err_data[:, 1], "Error {}".format(idx+1))
plt.legend()
plt.show()
