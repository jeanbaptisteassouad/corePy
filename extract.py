import cv2
import numpy as np
import matplotlib.pyplot as plt


def preprocess(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, img = cv2.threshold(img,150,255,cv2.THRESH_BINARY)
    img = cv2.Laplacian(img, cv2.CV_8U)
    # dilate and erode - helps getting a clearer layout view of the image
    element = cv2.getStructuringElement(cv2.MORPH_CROSS,(2,2))
    img = cv2.erode(cv2.dilate(img, element), element)
    return img


def local_projection(img):
    h, w = img.shape
    x = np.zeros(h)
    y = np.zeros(w)
    for i in range(h):
        for j in range(w):
            if img[i][j] == 255:
                x[i] += 1
                y[j] += 1
    return x, y


def remove_lines(x,y, mean_percentile, max_percentile):
    mean_h, max_h = (mean_percentile*np.mean(x),max_percentile*np.max(x))
    mean_w, max_w = (mean_percentile*np.mean(y),max_percentile*np.max(y))
    # remove lines under mean and above maximal value
    for i in range(len(x)):
        if x[i] < mean_h or x[i] > max_h:
            x[i] = 0
    for i in range(len(y)):
        if y[i] < mean_w or y[i] > max_w:
            y[i] = 0
    return x, y


def clear_thin_lines(layout, seuil):
    start = 0
    stop = 0
    for i in range(len(layout)):
        if layout[i] != 0 and start == 0:
            start = i
        if layout[i] == 0 and start != 0:
            stop = i
        val = (stop - start)
        if val > 0:
            if val <= seuil:
                for j in range(start,stop):
                    layout[j] = 0
            start = 0
            stop = 0
    return layout


def clean_img(img,x,y):
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if x[i] == 0 or y[j] == 0:
                img[i][j] = 0
    return img


def cutting_values(layout, seuil):
    vals = np.nonzero(layout)[0]
    if len(vals) is not 0:
        cutting_vals = [vals[0]]
    else:
        return []
    for i in range(len(vals)-1):
        if vals[i+1] - vals[i] >= seuil:
            cutting_vals.append(vals[i])
            cutting_vals.append((vals[i+1]))
    cutting_vals.append(vals[len(vals)-1])
    return cutting_vals


def blank_cutting_values(layout,seuil):
    zerovals = np.where(layout == 0)[0]
    if len(zerovals) is not 0:
        cutting_vals = [zerovals[0]]
    else:
        return []
    for i in range(len(zerovals)-1):
        if zerovals[i+1] - zerovals[i] >= 2:
            cutting_vals.append(zerovals[i])
            cutting_vals.append((zerovals[i+1]))
    cutting_vals.append(zerovals[len(zerovals)-1])
    cutting_vals = np.unique(cutting_vals)
    #print(cutting_vals)
    result = []
    for i in range(len(cutting_vals)/2 ):
        if cutting_vals[i*2+1] - cutting_vals[i*2] >= seuil:
            #print(str(cutting_vals[i*2+1])+" ,"+str(cutting_vals[i*2]))
            result.append(cutting_vals[i*2])
            result.append(cutting_vals[i*2+1])
        else:
            pass
    #print(result)
    return result

def find_empty_spaces(layout, seuil):
    start, stop = 0, 0
    res = []
    for i in range(len(layout)):
        if layout[i] != 0 and start != 0:
            stop = i
        if layout[i] == 0 and start == 0:
            start = i
        val = (stop - start)
        if val > 0:
            if val >= seuil:
                res.append((start,stop))
            start, stop = 0, 0
    return res



def process_each_chunk(img, output, x, y, cutting_value_seuil, blank_cutting_value_seuil):
    cutting_vals = cutting_values(x,cutting_value_seuil)
    #print(cutting_vals)
    for i in range(int(len(cutting_vals)/2)):
        # compute local columns projection
        local_x, local_y = local_projection(img[cutting_vals[i*2]:cutting_vals[i*2+1]+1][:])
        local_x, local_y = remove_lines(local_x, local_y, 0, 0.8)
        #print(local_y)
        print(i)
        #blank_cutvals = blank_cutting_values(local_y,blank_cutting_value_seuil)
        blank_cutvals = find_empty_spaces(local_y,blank_cutting_value_seuil)
        print(blank_cutvals)
        # draw columns
        for j in range(int(len(blank_cutvals))):
            line_y = (blank_cutvals[j][0] + blank_cutvals[j][1] +1)/2
            for k in range(cutting_vals[i*2],cutting_vals[i*2+1]+1):
                output[k][line_y] = 255
    # Draw lines
    blank_cutvals_lines = blank_cutting_values(x,5)
    for i in range(int(len(blank_cutvals_lines)/2)):
        line_x = (blank_cutvals_lines[i*2] + blank_cutvals_lines[i*2+1] +1)/2
        output[line_x][:] = 255
    return output


def projection_histogram(img):
    features = np.zeros(5000*2)
    height, width = img.shape
    cpt_features = 0
    # Ligne Horizontal and Vertical not in the same histo
    ans = np.zeros(height + width)
    for i in range(0, height):
        for j in range(0, width):
            ans[i] += img[i][j]
            ans[j + height] += img[i][j]
    hist = np.histogram(ans[0:height], bins=5000, range=(255, 255*width))[0]
    for x in range(0, len(hist)):
        features[cpt_features] = 100*hist[x]/float(width)
        cpt_features += 1
        pass
    hist = np.histogram(ans[height:height + width], bins=5000, range=(255, 255*height))[0]
    for x in range(0, len(hist)):
        features[cpt_features] = 100*hist[x]/float(height)
        cpt_features += 1
        pass
    return features


def main():
    base = "core/table"
    ext="png"
    filename = base+"."+ext

    img = preprocess(cv2.imread(filename))
    w,h = img.shape
    # Compute simple projection here
    x, y = local_projection(img)
    x, y = remove_lines(x,y,0,0.8)
    img = clean_img(img, x, y)
    # Recompute simple projections after cleaning up existing lines or columns
    local_x, local_y = local_projection(img)

    # looking fo columns :
    # Compute mean horizontal empty spacing
    vertical_white_spacing = find_empty_spaces(local_y,1)
    print(vertical_white_spacing)

    mean = 0
    for i in range(len(vertical_white_spacing)):
        mean += vertical_white_spacing[i][1] - vertical_white_spacing[i][0]
    mean/=len(vertical_white_spacing)
    print(mean)
    # keep only vertical spacings above 'mean' value
    columns = []
    for i in range(len(vertical_white_spacing)):
        if vertical_white_spacing[i][1] - vertical_white_spacing[i][0] >= mean:
            columns.append(vertical_white_spacing[i])
    print(columns)
    # draw columns
    for j in range(int(len(columns))):
        line_y = (columns[j][0] + columns[j][1] +1)/2
        for k in range(w):
            img[k][line_y] = 255

    # looking for lines :
    horizontal_white_spacing = find_empty_spaces(local_x,1)
    mean = 0
    for i in range(len(horizontal_white_spacing)):
        mean += horizontal_white_spacing[i][1] - horizontal_white_spacing[i][0]
    mean/=len(horizontal_white_spacing)
    print(mean)
    lines = []
    for i in range(len(horizontal_white_spacing)):
        if horizontal_white_spacing[i][1] - horizontal_white_spacing[i][0] >= mean:
            lines.append(horizontal_white_spacing[i])
    print(lines)
    # draw lines
    for j in range(int(len(lines))):
        line_x = (lines[j][0] + lines[j][1] +1)/2
        for k in range(h):
            img[line_x][k] = 255

    cv2.imshow("image",img)
    cv2.waitKey()

    # modify last parameter to set minimum empty space between columns
    #table_layout =  process_each_chunk(img, img, x, y, 50, 24)
    #cv2.imshow("image",table_layout)
    #cv2.waitKey()


if __name__ == '__main__':
    main()