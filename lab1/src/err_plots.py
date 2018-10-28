#!/usr/bin/env python

import sys
import csv
import numpy as np
import matplotlib.pyplot as pyplot

CSV_PATH = "/home/car-user/racecar_ws/src/ee545_robot_car/lab1/src/"
CSV_FILE = "pid-error.csv"

# with open(CSV_PATH + CSV_FILE, mode="r") as err_file:
data = np.genfromtxt(CSV_PATH + CSV_FILE, delimiter=" ")
print(np.size(data))

