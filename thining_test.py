import os
import cv2

def thinning(filepath , fpnumber, outputpath, separated_path):
    
    os.chdir(filepath)
    
    if fpnumber.endswith('.png'):
        fpnumber = fpnumber[:-4]
    
    img_name = fpnumber + '_stair_bnd2.png'
    img = cv2.imread(img_name)
    gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    ret, image = cv2.threshold(gray,127,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    BW_Original = image/255

    def pre_thinning1(image):
        rows, columns = image.shape               # x for rows, y for columns
        for x in range(1, rows - 1):                     # No. of  rows
            for y in range(1, columns - 1):            # No. of columns
                P1,P2,P3,P4,P5,P6,P7,P8 = n = neighbours(x, y, image)
                B_odd = P1 + P3 + P5 + P7
                if B_odd < 2:
                    image[y][x] = 0
                elif B_odd > 2:
                    image[y][x] = 1
                else:
                    image[y][x] = image[y][x]
        return image

    def neighbours(x,y,image):
        """Return 8-neighbours of image point P(x,y), in a clockwise order
        P4 P3 P2
        P5  P  P1
        P6 P7 P8
        """
        img = image
        x_1, y_1, x1, y1 = x-1, y-1, x+1, y+1
        plist = [ img[y][x1], img[y_1][x1], img[y_1][x], img[y_1][x_1], img[y][x_1], img[y1][x_1], img[y1][x], img[y1][x1] ] #P1 ~ P8
        return plist

    def transitions(neighbours):
        "No. of 0,1 patterns (transitions from 0 to 1) in the ordered sequence"
        n = neighbours + neighbours[0:1]      # P1, P2, ... , P7, P8, P1
        return sum( (n1, n2) == (0, 1) for n1, n2 in zip(n, n[1:]) )  # (P1,P2), (P2,P3), ... , (P7,P8), (P8,P1)

    def modified(image):
        "modified Thinning Algorithm"
        Image_Thinned = image.copy()  # deepcopy to protect the original image
        rows, columns = Image_Thinned.shape               # x for rows, y for columns
        changing1 = changing2 = 1        #  the points to be removed (set as 0)
        count = 0
        one_list = []
        tmp_list = []
        for x in range(1, rows - 1):
            for y in range(1, columns - 1):
                if Image_Thinned[y][x] == 1:
                    one_list.append((x, y))

        while changing1 or changing2:   #  iterates until no further changes occur in the image
            # Step 1
            changing1 = []

            for x, y in one_list:
                P1, P2, P3, P4, P5, P6, P7, P8 = n = neighbours(x, y, Image_Thinned)
                if (Image_Thinned[y][x] == 1 and    # Condition 0: Point P in the object regions
                    2 <= sum(n) <= 6 and    # Condition 1: 2<= N(P1) <= 6
                    transitions(n) == 1 and # Condition 2:
                    P1 * P3 * P7 == 0 and   # Condition 3
                    P1 * P5 * P7 == 0):     # Condition 4
                    changing1.append((x, y))
                else:
                    tmp_list.append((x, y))

            for x, y in changing1:
                Image_Thinned[y][x] = 0

            one_list = tmp_list
            tmp_list = []

            # Step 2
            changing2 = []

            for x, y in one_list:
                P1, P2, P3, P4, P5, P6, P7, P8 = n = neighbours(x, y, Image_Thinned)
                if (Image_Thinned[y][x] == 1     and    # Condition 0: Point P in the object regions
                    2 <= sum(n) <= 6   and    # Condition 1: 2<= N(P1) <= 6
                    transitions(n) == 1 and    # Condition 2:
                    P1 * P5 * P7 == 0  and    # Condition 3
                    P3 * P5 * P7 == 0):         # Condition 4
                    changing2.append((x, y))
                else:
                    tmp_list.append((x, y))

            one_list = tmp_list
            tmp_list = []

            for x, y in changing2:
                Image_Thinned[y][x] = 0

            count = count + 1
            if count > 20:
                break
        return Image_Thinned
    "Apply the algorithm on images"
    BW_Skeleton = 255 - modified(pre_thinning1(BW_Original))*255
    
        
    cv2.imwrite(filepath + fpnumber + '_stair_thin.bmp' , BW_Skeleton)
    return BW_Skeleton


mainpath = 'C:/Users/user/Desktop/module3_vec/module3/'   
filepath = 'C:/Users/user/Desktop/real/'
outputpath = 'C:/Users/user/Desktop/module3_vec/module3/test_output/'
separated_path = 'C:/Users/user/Desktop/module3_vec/module3/separated_output/'
fpnumber = 't900-3.png'
fpnumberr = 't900-3'

thinning(filepath, fpnumber, outputpath, separated_path)



