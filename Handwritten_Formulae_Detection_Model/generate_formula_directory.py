# B. Tech Project
# HANDWRITTEN FORMULAE DETECTION
# Semester 7
# December 2020
# Dr. Gaurav Harit
# Shashwat Kathuria - B17CS050
# Satya Prakash Sharma - B17CS048

# Importing required libraries
import os
from tkinter import *
from PIL import Image, ImageDraw, ImageTk
from collections import defaultdict

FORMULAS_DIRECTORY = 'Handwritten_Formulas/'

# Creating directory if it doe not exist
try:
    os.mkdir('Handwritten_Formula_Images')
except FileExistsError:
    pass

fileCounter = 0
# Looping through formulas images
for index, filename in enumerate(os.listdir(FORMULAS_DIRECTORY)):
    filepath = formulaFilePath(filename)
    # Adding the png images
    if '.png' in filepath:
        fileCounter += 1
        # Initializing image object
        img = Image.open(FORMULAS_DIRECTORY + filename)
        # Saving into output directory
        img.save('Handwritten_Formula_Images/' + str(fileCounter) + '.png')

    if fileCounter >= 3000:
        break
