import cv2
# import numpy as np
import subprocess

from Content import Content
from TableFactory import TableFactory

from PIL import Image


def ocr():
    contenu = Content()
    table = TableFactory("core/table2.png",contenu.paris_dichotomie)
    table.tree_of_content.remove_useless_leaf()
    table.tree_of_content.sort_subtree_by_frame()
    # get leafs
    leafs = []
    table.tree_of_content.return_leafs(leafs)

    for elt in leafs:
        subimage = table.image[elt.frame[0][1]:elt.frame[1][1],elt.frame[0][0]:elt.frame[1][0]]
        cv2.imwrite("tmp.tiff",subimage)
        # Subprocess
        subprocess.call(['tesseract', 'tmp.tiff', 'output'])
        with open('output.txt') as f:
            text = f.read()
        print("#####")
        print(text)

        cv2.imshow("image",subimage)
        cv2.waitKey()

    pass

if __name__ == '__main__':
    ocr()