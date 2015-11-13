from PIL import Image
from PIL import ImageFilter
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pickle
import re
import sys
import os

# ans -> projLignes + projColonnes
def projection(img):
    matrice = cv2.Canny(img, 150, 255, apertureSize=3)
    height, width = matrice.shape
    ans = np.zeros(height + width)
    for i in range(1, height-1):
        for j in range(1, width-1):
            ans[i] += matrice[i][j]
            ans[j + height] += matrice[i][j]
    return ans


def projection2(img):
    matrice = cv2.Canny(img, 150, 255, apertureSize=3)
    height, width = matrice.shape
    ans = np.zeros(height + width)
    incr = 0.001
    coef = incr
    for i in range(1, height-1):
        for j in range(1, width-1):
            if matrice[i][j] == 0:
                coef = 0
            else:
                if coef < 1:
                    coef += incr
                    pass
            pass
            ans[i] += matrice[i][j] * coef
            ans[j + height] += matrice[i][j] * coef
    return ans


    #incrTmp = 0.001
    # for i in xrange(1, height-1):
    #     for j in xrange(1, width-1):
    #         if matrice[i][j] == 0:
    #             tmp = 0
    #         else:
    #             if tmp < 1:
    #                 tmp += incrTmp
    #                 pass
    #         pass
    #         projLignes2[i] += matrice[i][j]*tmp
    #         projColonnes2[j] += matrice[i][j]*tmp
    #     pass


def main():

    # img = cv2.imread('/home/jean-baptiste/Desktop/corePy/pgm/100/HPC-T4-2013-GearsAndSprockets-GB/HPC-T4-2013-GearsAndSprockets-GB-031.pgm')
    # fichier = open("test.pickle",'w')
    # pickle.dump( projection(img) , fichier , protocol=2 )
    # fichier.close()

    # img = cv2.imread('HPC-T4-2013-GearsAndSprockets-GB-038.pgm')
    # projection2(img)

    # if len(sys.argv) != 2:
    #     print "ERROR : too few arguments"
    #     exit()
    #     pass
    # path = "/home/jean-baptiste/Desktop/corePy/pgm/100/HPC-T4-2013-GearsAndSprockets-GB"
    # directoryWithPgm = os.listdir(path)

    # for x in range(0,len(directoryWithPgm)):
    #     print directoryWithPgm[x]
    #     img = cv2.imread(path + "/" + directoryWithPgm[x])
    #     fichier = open(path + "/" + re.sub('.pgm', "", directoryWithPgm[x]) + ".pickle",'w')
    #     pickle.dump( projection(img) , fichier , protocol=2 )
    #     fichier.close()
    #     pass

    # image = Image.open("/Users/jean-baptiste/Desktop/Cafe/test/prefix-056.png")
    # image = image.convert("L")
    # image = image.filter(ImageFilter.FIND_EDGES)
    # # On convert l'image en array 2D
    # width, height = image.size
    # data = np.array(image.getdata())
    # matrice = np.zeros((height, width))
    # for i in xrange(0, height):
    #     for j in xrange(0, width):
    #         matrice[i][j] = data[i*width+j]
    #         pass
    #    pass

    # img = cv2.imread('hpc_low/prefix-023.pgm')
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # matrice = cv2.Canny(gray, 150, 255, apertureSize=3)
    # height, width = matrice.shape

    # string = "hpc-415-c1.png"
    # tmp = re.findall('c[01]', string)
    # number = re.sub('c', "", tmp[0])
    # tmp = re.findall('-.*-', string)
    # page = re.sub('-', "", tmp[0])

    # # On sommes les colonnes et les lignes (on ne prend pas en compte les bords)
    # projLignes = np.zeros(height)
    # projColonnes = np.zeros(width)
    # for i in range(1, height-1):
    #     for j in range(1, width-1):
    #         projLignes[i] += matrice[i][j] * 100.0 / (255*(width-2))
    #         projColonnes[j] += matrice[i][j] * 100.0 / (255*(height-2))
    #         pass
    #     pass
    # plt.plot(projLignes)
    # plt.show()


    # # On fais une moyenne mobile sur projLignes
    # tailleFenetre = 100
    # projLignesMM = np.zeros(len(projLignes))
    # for x in xrange(tailleFenetre, len(projLignes)-tailleFenetre):
    #     for y in xrange(-tailleFenetre, tailleFenetre+1):
    #         projLignesMM[x] += projLignes[x+y]
    #         pass
    #     pass
    # coef = max(projLignes)/max(projLignesMM)
    # for x in xrange(0, len(projLignesMM)):
    #     projLignesMM[x] *= coef
    #     pass
    # plt.plot(projLignesMM)

    # # Nouvelle projection test
    # tmp = 0
    # projLignes2 = np.zeros(height)
    # projColonnes2 = np.zeros(width)
    # incrTmp = 0.001
    # for i in xrange(1, height-1):
    #     for j in xrange(1, width-1):
    #         if matrice[i][j] == 0:
    #             tmp = 0
    #         else:
    #             if tmp < 1:
    #                 tmp += incrTmp
    #                 pass
    #         pass
    #         projLignes2[i] += matrice[i][j]*tmp
    #         projColonnes2[j] += matrice[i][j]*tmp
    #     pass
    # maxWidthProj2 = 0
    # tmp = 0
    # for x in xrange(1,width-1):
    #     if tmp < 1:
    #         tmp += incrTmp
    #         pass
    #     maxWidthProj2 += 255*tmp
    #     pass
    # # coef = max(projLignes)/max(projLignes2)
    # # for x in xrange(0, len(projLignes2)):
    # #     projLignes2[x] *= coef
    # #     pass
    # for x in xrange(0, len(projLignes2)):
    #     projLignes2[x] *= 100.0 / maxWidthProj2
    #     pass
    # plt.plot(projLignes2)

    #image.show()
    pass

if __name__ == '__main__':
    main()
