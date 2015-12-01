import cv2
# import numpy as np

from Content import Content
from TableFactory import TableFactory

from PIL import Image
from pytesseract import image_to_string

def ocr():
    contenu = Content()
    table = TableFactory("core/table.png",contenu.paris_dichotomie)
    table.tree_of_content.remove_useless_leaf()
    table.tree_of_content.sort_subtree_by_frame()
    # get leafs
    leafs = []
    table.tree_of_content.return_leafs(leafs)

    for elt in leafs:
        subimage = table.image[elt.frame[0][1]:elt.frame[1][1],elt.frame[0][0]:elt.frame[1][0]]
        print(image_to_string(subimage))
        cv2.imshow("image",subimage)
        cv2.waitKey()

    pass

def test_ocr():
    # tesseract trial
    img = Image.open("core/table.png")
    string = (image_to_string(img))
    print(" text : "+string)
    exit()
    pass



if __name__ == '__main__':
    test_ocr()