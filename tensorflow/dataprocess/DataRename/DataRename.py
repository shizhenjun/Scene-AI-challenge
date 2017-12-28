import csv
import os


def readLabel(csvPath):
    labelDict = {}
    if not os.path.exists(csvPath):
        return labelDict
    f = open(csvPath, 'r')
    r = csv.reader(f)
    for row in r:
        labelDict[int(row[0])] = row[-1].replace('/', '_')
    return labelDict


def renameClass(dataPath, labelDict):
    if not os.path.exists(dataPath):
        return
    for classname in os.listdir(dataPath):
        classPath = os.path.join(dataPath, classname)
        if os.path.isdir(classPath):
            classpath_new = os.path.join(
                dataPath, classname + '.' + str(labelDict[int(classname.split('.')[0])]))
            print(classPath)
            print(classpath_new)
            os.rename(classPath, classpath_new)
            count = 0
            for imageName in os.listdir(classpath_new):
                imagePath = os.path.join(classpath_new, imageName)
                # print(imagePath)
                imagePath_new = os.path.join(
                    classpath_new, classname + '_' + str(count + 1) + '.jpg')
                # print(imagePath_new)
                os.rename(imagePath, imagePath_new)
                count += 1

if __name__ == '__main__':
    labelDict = readLabel('scene_classes.csv')
    renameClass('validation/', labelDict)
