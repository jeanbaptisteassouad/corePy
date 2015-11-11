from PIL import Image
from PIL import ImageFilter
import numpy as np
import cv2
import matplotlib.pyplot as plt

def main():
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

    img = cv2.imread('/Users/jean-baptiste/Desktop/Cafe/test/prefix-056.png')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    matrice = cv2.Canny(gray, 150, 255, apertureSize=3)
    height, width = matrice.shape

    # On sommes les colonnes et les lignes (on ne prend pas en compte les bords)
    projLignes = np.zeros(height)
    projColonnes = np.zeros(width)
    for i in xrange(1, height-1):
        for j in xrange(1, width-1):
            projLignes[i] += matrice[i][j] * 100.0 / (255*(width-2))
            projColonnes[j] += matrice[i][j] * 100.0 / (255*(height-2))
            pass
        pass
    plt.plot(projLignes)
    plt.show()
    #print projLignes
    #print projColonnes
    # for x in xrange(0, len(projLignes)):
    #     projLignes[x] *= 100.0 / (255*(width-2))
    #     pass
    # plt.plot(projLignes)
    #plt.show()

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
