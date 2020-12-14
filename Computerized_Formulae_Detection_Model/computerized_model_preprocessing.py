# B. Tech Project
# HANDWRITTEN FORMULAE DETECTION
# Semester 7
# December 2020
# Dr. Gaurav Harit
# Shashwat Kathuria - B17CS050
# Satya Prakash Sharma - B17CS048

# Importing required libraries
from PIL import Image, ImageDraw
import os
import xml.etree.ElementTree as etree

def main():
    '''Main function.'''

    # calling function to create directories
    createDirectories()

    # Initializing variables required
    filenameList = [ filename.strip('.xml') for filename in sorted(list(os.listdir('Annotations/')))]
    dict = {}

    # Looping through filenames
    for filename in filenameList:
        print('On file', filename)
        # Creating image
        img = Image.open('Images/' + filename + '.bmp')
        if img.size[0] == 1061 and img.size[1] == 1373:
            # Parsing coordinates
            coordinatesDictList = parseCoordinates('Annotations/', filename)
            # Saving image if only there are >= 1 formula annotations
            if len(coordinatesDictList) > 0:
                # Resizing and saving image
                newImg = img.resize(img.size)
                newImg.save('Images_Modified/' + filename + '.png', 'png')



def parseCoordinates(path, filename):
    '''Function to parse the coordinates in the ICDAR 2017 Dataset.'''

    # Initializing XML tree
    tree = etree.parse(path + filename + '.xml')
    root = tree.getroot()

    # Initializing variables required
    rectangularCoordinatesList = []
    coordinatesDictList = []

    # Looping through all the nodes
    for node in root:
        # If the node is a formulaRegion
        if node.tag=='formulaRegion':
            # Looping through the children
            for elem in node.iter():
                if elem.tag=='Coords':
                    # Reading and storing the coordinates
                    cor = elem.attrib['points'].split(' ')
                    x1 = int(cor[0].split(',')[0])
                    y1 = int(cor[0].split(',')[1])
                    x2 = int(cor[3].split(',')[0])
                    y2 = int(cor[3].split(',')[1])
                    # Appending to list
                    rectangularCoordinatesList.append([x1, y1, x2, y2])

    # If >= 1 formulaRegion exists, save the annotations of them
    # Order => X1, Y1, X2, Y2
    if len(rectangularCoordinatesList) > 0:

        # Saving annotations in a new directory and file
        modifiedFile = open('Annotations_Modified/' + filename + '.txt', 'w')

        # Looping though coordinates
        for coordinates in rectangularCoordinatesList:
            # Storing coordinates and writing to file, and appending to list
            x1, y1, x2, y2 = coordinates[0], coordinates[1], coordinates[2], coordinates[3]
            modifiedFile.write( str(x1) + ' ' + str(y1) + ' ' + str(x2) + ' ' + str(y2) + '\n' )
            coordinatesDictList.append({ 'x1': x1, 'x2': x2,'y1': y1,'y2': y2})

        # Closing file
        modifiedFile.close()

    # Returning list
    return coordinatesDictList

def createDirectories():
    '''Function to create directories required. Ignoring if they already exist.'''

    # Initializing list of directories to be created
    directoriesToCreate = [
        'Annotations_Modified',
        'Images_Modified'
    ]

    # Looping through list
    for directoryName in directoriesToCreate:
        # Create directory if does not exist
        try:
            os.mkdir(directoryName)
        # If directory exists, then continue
        except FileExistsError:
            pass

# Callling main function
if __name__ == '__main__':
    main()
