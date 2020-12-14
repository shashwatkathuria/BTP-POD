# B. Tech Project
# HANDWRITTEN FORMULAE DETECTION
# Semester 7
# December 2020
# Dr. Gaurav Harit
# Shashwat Kathuria - B17CS050
# Satya Prakash Sharma - B17CS048

# Importing required libraries
import os, random, time
from tkinter import *
from PIL import Image, ImageDraw, ImageTk
from collections import defaultdict

# Defining constants required
# Temp and output directories
TEMP_POSTSCRIPT_IMAGE_DIRECTORY = 'Modified_Text_PDF_Images/'
TEXT_ANNOTATIONS_DIRECTORY = 'Handwritten_Images_Text_Annotations/'
FORMULA_ANNOTATIONS_DIRECTORY = 'Handwritten_Images_Formula_Annotations/'
IMAGES_DIRECTORY = 'Handwritten_Images/'

# Input directories
FORMULAS_DIRECTORY = 'Handwritten_Formula_Images/'
TEXT_DIRECTORY = 'Handwritten_Sentences/'

# Some constants for output images
LEFT_MARGIN = 20
RIGHT_MARGIN = 20
BOTTOM_MARGIN = 40
PDF_HEIGHT = 1000
PDF_WIDTH = 700
PARA_GAP = 90
FORMULA_GAP = 45
LINE_GAP = 13

def main():
    '''Main function.'''

    # Calling function to create output and temp directories
    createDirectories()

    # Calling function to get input formula images
    formulasArray = getInputFormulaImages()

    # Calling function to get input text images
    textImagesDict = getInputTextImages()

    root = Tk()

    # Initializing empty white canvas with specififed width and height
    cv = Canvas(root, width = PDF_WIDTH, height = PDF_HEIGHT, bg = 'white')

    # Converting all formula images in list to tkinter PhotoImage object
    # Initializing temp list
    tempFormulasArray = []
    # Looping through array
    for formulaImage in formulasArray:
        # Converting to PhotoImage object
        tkImg = ImageTk.PhotoImage(formulaImage)
        # Appending PhotoImage object and image size to array
        tempFormulasArray.append([tkImg, formulaImage.size])
    formulasArray = tempFormulasArray

    # Generating synthetic pdf images with text and formula

    # Looping through keyed common handwritings
    for key in textImagesDict:
        # 50 images per handwriting
        for pageNo in range(50):
            # Random seed
            random.seed(time.time())

            # Initializing text and formula annotation file to write
            textAnnotationFile = open(TEXT_ANNOTATIONS_DIRECTORY + key + str(pageNo) + '.txt', 'w')
            formulaAnnotationFile = open(FORMULA_ANNOTATIONS_DIRECTORY + key + str(pageNo) + '.txt', 'w')

            # Converting all text images in list to tkinter PhotoImage object
            # Initializing temp list
            textImagesArray = []
            # Looping through array
            for image in textImagesDict[key]:
                # Converting to PhotoImage object
                tkImg = ImageTk.PhotoImage(image)
                # Appending PhotoImage object and image size to array
                textImagesArray.append([tkImg, image.size])

            # Shuffling array
            random.shuffle(textImagesArray)

            # Initializing variables required
            paraCount = 1
            numberOfParas = random.randint(4, 6)

            # Initial height, and x1 and y1(=height)
            height = 20
            x1 = LEFT_MARGIN
            y1 = height

            # Add text and formulas until the page is not completely filled
            while height < PDF_HEIGHT - BOTTOM_MARGIN:
                # Random seed
                random.seed(time.time())

                # Randomly shuffling arrays
                random.shuffle(textImagesArray)
                random.shuffle(formulasArray)

                # Initializing variables required
                # Sentence starts from LEFT_MARGIN
                width = LEFT_MARGIN
                # Variable for maxHeight of whole sentence
                maxHeight = textImagesArray[0][1][1]
                # Keeping track of number of images added
                counter = 0

                # Looping through textImagesArray to add text and formulas
                for element in textImagesArray:

                    # Randomly adding formula to sentence
                    if random.random() > 0.99:
                        isFormula = True
                        imageElement = random.sample(formulasArray, 1)[0]
                        imggg = imageElement[0]
                        imgggSize = imageElement[1]
                    # Else adding plain text
                    else:
                        isFormula = False
                        imggg = element[0]
                        imgggSize = element[1]

                    # If element goes out of page width, skip and go on to next one
                    if width + imgggSize[0] > PDF_WIDTH - RIGHT_MARGIN:
                        continue
                    # If element goes out of page height, skip and go on to next one
                    if height + imgggSize[1] > PDF_HEIGHT - BOTTOM_MARGIN:
                        continue
                    # Incrementing counter as now image will be added
                    counter += 1
                    # Adding image to canvas
                    cv.create_image(width, height, anchor = NW, image = imggg)

                    # If a formula, add annotation also
                    if isFormula:
                        # Formula bounding box annotation
                        formulaX1 = width
                        formulaX2 = width + imgggSize[0]
                        formulaY1 = height
                        formulaY2 = height + imgggSize[1]

                        # Adding annotation to file
                        formulaAnnotationFile.write(str(formulaX1) + ' ' + str(formulaY1) + ' ' + str(formulaX2) + ' ' + str(formulaY2) + '\n')

                        # Uncomment below line to see the annotations in action in tkinter canvas
                        # cv.create_line(formulaX1, formulaY1, formulaX2, formulaY2, fill="blue")

                    # Incrementing width and max height of sentence elements accordingly
                    width += imgggSize[0]
                    maxHeight = max(maxHeight, imgggSize[1])

                # If images added, add max height and line gap
                if counter > 0:
                    height += maxHeight + LINE_GAP
                # If no image added, add line gap
                else:
                    height += LINE_GAP

                # Randomly adding paragraph breaks in text with maximum as numberOfParas
                if random.random() > 0.8 and paraCount < numberOfParas:

                    # Adding text annotation as para ends here
                    x2 = PDF_WIDTH - RIGHT_MARGIN
                    y2 = height
                    # Adding annotation to file
                    textAnnotationFile.write(str(x1) + ' ' + str(y1) + ' ' + str(x2) + ' ' + str(y2) + '\n')
                    # Uncomment below line to see the annotations in action in tkinter canvas
                    # cv.create_line(x1, y1, x2, y2, fill = "red")

                    # Initializing variables required
                    height += FORMULA_GAP
                    numberOfFormulas = random.randint(1, 3)

                    # Add horizontal formula alignment randomly if even number of formulas
                    if random.random() > 0.3 and numberOfFormulas == 2:

                        # Initializing variable required
                        maxHeight = 0
                        factor = int((PDF_WIDTH - LEFT_MARGIN - RIGHT_MARGIN) / 4)
                        counter = 0

                        # Looping
                        for i in [1, 3]:
                            # Randomly chosse a formula from list
                            formula = random.sample(formulasArray, 1)[0]
                            formulaImage = formula[0]
                            formulaImageSize = formula[1]

                            # If element goes out of page width, skip and go on to next one
                            if LEFT_MARGIN + formulaImageSize[0] > PDF_WIDTH - RIGHT_MARGIN:
                                continue
                            # If element goes out of page height, skip and go on to next one
                            if height + formulaImageSize[1] > PDF_HEIGHT - BOTTOM_MARGIN:
                                continue

                            # Incrementing counter as now image will be added
                            counter += 1

                            # Adding bounding box coordinates accordingly
                            initialWidth = int( (factor * i) - (formulaImageSize[0] / 2) )
                            formulaX1 = initialWidth
                            formulaX2 = initialWidth + formulaImageSize[0]
                            formulaY1 = height
                            formulaY2 = height + formulaImageSize[1]
                            # Adding annotation to file
                            formulaAnnotationFile.write(str(formulaX1) + ' ' + str(formulaY1) + ' ' + str(formulaX2) + ' ' + str(formulaY2) + '\n')
                            # Adding image to canvas
                            cv.create_image(initialWidth, height, anchor = NW, image = formulaImage)

                            # Uncomment below line to see the annotations in action in tkinter canvas
                            # cv.create_line(formulaX1, formulaY1, formulaX2, formulaY2, fill="blue")

                            # Updating formula line max height
                            maxHeight = max(maxHeight, formulaImageSize[1])

                        # If images added, add max height and line gap
                        if counter > 0:
                            height += maxHeight + PARA_GAP
                        # If images added, add line gap
                        else:
                            height += PARA_GAP
                    # Vertical formula alignment
                    else:

                        # Looping through number of formulas to be added
                        for i in range(numberOfFormulas):
                            # Randomly chosse a formula from list
                            formula = random.sample(formulasArray, 1)[0]
                            formulaImage = formula[0]
                            formulaImageSize = formula[1]

                            # If element goes out of page width, skip and go on to next one
                            if LEFT_MARGIN + formulaImageSize[0] > PDF_WIDTH - RIGHT_MARGIN:
                                continue
                            # If element goes out of page height, skip and go on to next one
                            if height + formulaImageSize[1] > PDF_HEIGHT - BOTTOM_MARGIN:
                                continue

                            # Adding bounding box coordinates accordingly
                            initialWidth = int( (PDF_WIDTH - LEFT_MARGIN - RIGHT_MARGIN - formulaImageSize[0]) / 2)
                            formulaX1 = initialWidth
                            formulaX2 = initialWidth + formulaImageSize[0]
                            formulaY1 = height
                            formulaY2 = height + formulaImageSize[1]

                            # Adding annotation to file
                            formulaAnnotationFile.write(str(formulaX1) + ' ' + str(formulaY1) + ' ' + str(formulaX2) + ' ' + str(formulaY2) + '\n')

                            # Adding image to canvas
                            cv.create_image(initialWidth, height, anchor = NW, image = formulaImage)

                            # Uncomment below line to see the annotations in action in tkinter canvas
                            # cv.create_line(formulaX1, formulaY1, formulaX2, formulaY2, fill="blue")

                            # Updating height
                            height += formulaImageSize[1] + 20

                    # Uncomment below line to see the annotations in action in tkinter canvas
                    # cv.create_line(0,height,PDF_WIDTH,height, fill="blue")

                    # New para starts, updating values of x1 and y1
                    paraCount += 1
                    x1 = LEFT_MARGIN
                    y1 = height

            # End of text para, updating values
            x2 = PDF_WIDTH - RIGHT_MARGIN
            y2 = height
            # cv.create_line(x1, y1, x2, y2, fill="red")

            cv.pack()
            cv.update()
            # Adding postscript output to file
            cv.postscript(colormode = 'color', file = TEMP_POSTSCRIPT_IMAGE_DIRECTORY + key + str(pageNo) + '.eps', pagewidth=PDF_WIDTH -1, pageheight=PDF_HEIGHT-1)
            # Generating image object from above postscript output
            img = Image.open(TEMP_POSTSCRIPT_IMAGE_DIRECTORY + key + str(pageNo) + '.eps')
            # Saving generated image as png
            img.save(IMAGES_DIRECTORY + key + str(pageNo) + '.png', 'png')
            # Clearing canvas for next page generation
            cv.delete('all')

    # Root mainloop
    root.mainloop()

def getInputTextImages():
    '''Function to get processed text images dict keyed according to common handwritings.'''

    # Initializing dict (keyed according to common handwritings) and scaling factor
    # Default element in textImagesDict is an empty list
    textImagesDict = defaultdict(list)
    scalingFactor = 3.25

    # Looping through all the text images
    for index, filename in enumerate(sorted(os.listdir(TEXT_DIRECTORY))[:3000]):
        # Initializing image object
        img = Image.open(TEXT_DIRECTORY + filename)
        # Resizing according to scaling factor
        img = img.resize((int(img.size[0] / scalingFactor), int(img.size[1] / scalingFactor)), Image.ANTIALIAS)
        if filename[:3] not in ['b06', 'c03', 'c06', 'g06', 'd06']: # 'do6'
            # Appending to array of keyed element in dict
            textImagesDict[filename[:3]].append(img)

    # Returning dict
    return textImagesDict

def getInputFormulaImages():
    '''Function to get processed formula images array.'''

    # Initializing list and scaling factor
    formulasArray = []
    scalingFactor = 4.25

    # Looping through all the formulas images
    for index, filename in enumerate(sorted(os.listdir(FORMULAS_DIRECTORY))):
        # Initializing image object
        img = Image.open(FORMULAS_DIRECTORY + filename)
        if img.size[0] == 523:
            if img.size[1] > 120 and img.size[1] < 200:
                # Resizing according to scaling factor
                img = img.resize((int(img.size[0] / scalingFactor), int(img.size[1] / scalingFactor)), Image.ANTIALIAS)
                # Appending to array
                formulasArray.append(img)

    # Returning array
    return formulasArray

def createDirectories():
    '''Function to create directories required. Ignoring if already created.'''

    # Initializing directories to be created
    directoriesToCreate = [
        TEMP_POSTSCRIPT_IMAGE_DIRECTORY,
        IMAGES_DIRECTORY,
        TEXT_ANNOTATIONS_DIRECTORY,
        FORMULA_ANNOTATIONS_DIRECTORY
    ]

    # Looping through list
    for directoryName in directoriesToCreate:
        # Create directory
        try:
            os.mkdir(directoryName)
        # Ignore if already exists
        except FileExistsError:
            pass

# Calling main function
if __name__ == "__main__":
    main()
