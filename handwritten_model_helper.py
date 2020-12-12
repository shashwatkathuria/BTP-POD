import os

# HOW TO READ A LINE
# file = open('Annotations_Modified/POD_0001.txt')
# lines = [line.strip('\n').split(' ') for line in file.readlines()]
# Order => X1, Y1, X2, Y2

ANNOTATIONS_DIRECTORY = 'Handwritten_Images_Formula_Annotations/'
IMAGES_DIRECTORY = 'Handwritten_Images/'

def getSortedFilenames():
    return [ filename.strip('.txt') for filename in sorted(list(os.listdir(ANNOTATIONS_DIRECTORY))) ]

def getCoordinatesDictList(filename):
    coordinatesDictList = []

    # Order => X1, Y1, X2, Y2
    file = open( ANNOTATIONS_DIRECTORY + filename + '.txt' )
    rectangularCoordinatesList = [coordinates.strip('\n').split(' ') for coordinates in file.readlines()]

    for coordinates in rectangularCoordinatesList:
        x1, y1, x2, y2 = coordinates[0], coordinates[1], coordinates[2], coordinates[3]
        coordinatesDictList.append({ 'x1': int(x1), 'x2': int(x2), 'y1': int(y1), 'y2': int(y2) })

    return coordinatesDictList

def getModifiedImagePath(filename):
    return IMAGES_DIRECTORY + filename + '.png'

def getModifiedAnnotationsPath(filename):
    return ANNOTATIONS_DIRECTORY + filename + '.txt'

if __name__ == '__main__':
    main()
