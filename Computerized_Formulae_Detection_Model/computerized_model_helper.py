# B. Tech Project
# HANDWRITTEN FORMULAE DETECTION
# Semester 7
# December 2020
# Dr. Gaurav Harit
# Shashwat Kathuria - B17CS050
# Satya Prakash Sharma - B17CS048

# Importing required library
import os

# Defining constants required
ANNOTATIONS_DIRECTORY = 'Annotations_Modified/'
IMAGES_DIRECTORY = 'Images_Modified/'

def getCoordinatesDictList(filename):
    '''Function to read the filename and get all bounding box coordinates as dicts in a list.'''

    # Defining list to store all annotations as dicts
    # in the list
    coordinatesDictList = []

    # Opening annotation input file
    file = open( ANNOTATIONS_DIRECTORY + filename + '.txt' )
    # Storing coordinates as a list in the outer list
    # Order => X1, Y1, X2, Y2
    rectangularCoordinatesList = [coordinates.strip('\n').split(' ') for coordinates in file.readlines()]

    # Looping through the coordinates already read
    for coordinates in rectangularCoordinatesList:
        # Getting x1, y1, x2, y2 values
        x1, y1, x2, y2 = coordinates[0], coordinates[1], coordinates[2], coordinates[3]
        # Appending coordinates dict to list
        coordinatesDictList.append({ 'x1': int(x1), 'x2': int(x2), 'y1': int(y1), 'y2': int(y2) })

    # Returning list with all annotations as dicts
    return coordinatesDictList

def getSortedFilenames():
    '''Function to return list of sorted annotations filenames.'''

    # Returning sorted filenamess
    return [ filename.strip('.txt') for filename in sorted(list(os.listdir(ANNOTATIONS_DIRECTORY))) ]

def getModifiedImagePath(filename):
    '''Function to get relative path of image filename.'''

    # Returning image filename path
    return IMAGES_DIRECTORY + filename + '.png'

def getModifiedAnnotationsPath(filename):
    '''Function to get relative path of annotation filename.'''

    # Returning annotation filename path
    return ANNOTATIONS_DIRECTORY + filename + '.txt'
