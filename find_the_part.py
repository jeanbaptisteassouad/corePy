import os
import cv2
import subprocess
import re



def process_one_tiff(ans,path):
    subprocess.call(['tesseract',path, 'output'])
    with open('output.txt') as f:
        content = f.read()

    tmp = re.sub('\n',' ',content)
    tmp = re.split(' *' , tmp)
    for x in range(0,len(tmp)):
        if x > 9:
            break
        print tmp[x]
        shouldAdd = True
        for y in range(0,len(ans)):
            if tmp[x] == ans[y][1]:
                ans[y][0] += 1
                shouldAdd = False
                break

        if shouldAdd:
            ans.append([1,tmp[x]])
            pass

    return ans


def main():
    listNameFile = []
    for filename in os.listdir("png/hd"):
        listNameFile.append( filename )

    listNameFile.sort()

    ans = []

    for x in range(0,len(listNameFile)):
        print x
        process_one_tiff(ans,"png/hd/"+listNameFile[x])

    ans.sort()
    ans.reverse()

    print ans


if __name__ == '__main__':
    main()

