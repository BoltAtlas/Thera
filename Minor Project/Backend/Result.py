import random
import numpy
import seaborn
import pandas as pd
import matplotlib.pyplot as plt
import csv
from csv import reader
import matplotlib
matplotlib.use('Agg')


def save_plot(d):
    plt.clf()
    labels = ['Happy', 'Angry', 'Surprise', 'Sad', 'Fear']
    sizes = []
    color = ["green", "red", "yellow", "blue", "yellowgreen"]

    for x, y in d.items():
        sizes.append(y)

    plt.pie(sizes, colors=color)
    patches, texts = plt.pie(sizes, startangle=90)
    # plt.legend(patches, labels, loc="best")
    plt.axis('equal')
    # plt.legend()
    plt.savefig(
        r'C:\Users\trish\.vscode\Projects\Minor Project\Frontend\Resource folder\User_data.png')
    # plt.show()


# save_plot({'Happy': 1, 'Angry': 1,
#            'Surprise': 1, 'Sad': 1, 'Fear': 1})
