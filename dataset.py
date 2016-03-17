import sys
import os
import random


def rename(parent):
    count = 0
    fileList = os.listdir(parent)
    for filename in fileList:
        oldpath = os.path.join(parent, filename)
        arr = os.path.splitext(filename)
        if arr[1].lower() not in ['.jpg', '.jpeg', '.png']:
            continue
        newpath = os.path.join(parent, str(count) + arr[1].lower())
        # print newpath
        os.rename(oldpath, newpath)
        count = count + 1
        # break
    print count
    return


def genDataset():
    fileList = []
    exteriorDir = 'data/exterior/'
    interiorDir = 'data/interior/'
    exFiles = os.listdir(exteriorDir)
    for filename in exFiles:
        fileList.append(('exterior/' + filename, 0))
    inFiles = os.listdir(interiorDir)
    for filename in inFiles:
        fileList.append(('interior/' + filename, 1))
    # shuffle to split filelist into trainset, valset, testset
    random.shuffle(fileList)
    valList = []
    testList = []
    for i in range(1000):
        randomId = int(random.uniform(0, len(fileList)))
        valList.append(fileList[randomId])
        del[fileList[randomId]]
    for i in range(1000):
        randomId = int(random.uniform(0, len(fileList)))
        testList.append(fileList[randomId])
        del[fileList[randomId]]

    fh = open('train.txt', 'w')
    for line in fileList:
        fh.write(line[0] + ' ' + str(line[1]) + '\n')
    fh.close()
    fh = open('val.txt', 'w')
    for line in valList:
        fh.write(line[0] + ' ' + str(line[1]) + '\n')
    fh.close()
    fh = open('test.txt', 'w')
    for line in testList:
        fh.write(line[0] + ' ' + str(line[1]) + '\n')
    fh.close()
    return

if __name__ == '__main__':
    exteriorDir = 'data/exterior/'
    interiorDir = 'data/interior/'
    # rename(exteriorDir)
    # rename(interiorDir)
    genDataset()
