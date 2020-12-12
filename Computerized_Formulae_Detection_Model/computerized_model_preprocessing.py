from PIL import Image, ImageDraw
import os
import xml.etree.ElementTree as etree

# HOW TO READ A LINE
# file = open('Annotations_Modified/POD_0001.txt')
# lines = [line.strip('\n').split(' ') for line in file.readlines()]
# Order => X1, Y1, X2, Y2

def main():
    try:
        os.mkdir('Annotations_Modified')
    except FileExistsError:
        pass
    try:
        os.mkdir('Images_Modified')
    except FileExistsError:
        pass

    filenameList = [ filename.strip('.xml') for filename in sorted(list(os.listdir('Annotations/')))]

    dict = {}
    print(filenameList)

    for filename in filenameList:
        img = Image.open('Images/' + filename + '.bmp')
        if img.size[0] == 1061 and img.size[1] == 1373:
            coordinatesDictList = parseCoordinates('Annotations/', filename)
            print(coordinatesDictList)
            if len(coordinatesDictList) > 0:
                newImg = img.resize(img.size)
                newImg.save('Images_Modified/' + filename + '.png', 'png')



def parseCoordinates(path, filename):
    tree = etree.parse(path + filename + '.xml')
    root = tree.getroot()

    rectangularCoordinatesList = []
    coordinatesDictList = []

    for node in root:
        if node.tag=='formulaRegion':
            for elem in node.iter():
                if elem.tag=='Coords':
                    cor = elem.attrib['points'].split(' ')
                    x1 = int(cor[0].split(',')[0])
                    y1 = int(cor[0].split(',')[1])
                    x2 = int(cor[3].split(',')[0])
                    y2 = int(cor[3].split(',')[1])
                    rectangularCoordinatesList.append([x1, y1, x2, y2])

    if len(rectangularCoordinatesList) > 0:
        modifiedFile = open('Annotations_Modified/' + filename + '.txt', 'w')
        for coordinates in rectangularCoordinatesList:
            x1, y1, x2, y2 = coordinates[0], coordinates[1], coordinates[2], coordinates[3]
            modifiedFile.write( str(x1) + ' ' + str(y1) + ' ' + str(x2) + ' ' + str(y2) + '\n' )
            coordinatesDictList.append({ 'x1': x1, 'x2': x2,'y1': y1,'y2': y2})
        modifiedFile.close()

    return coordinatesDictList

if __name__ == '__main__':
    main()
