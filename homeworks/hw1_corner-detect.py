import cv2
import numpy as np
from random import randint
from matplotlib import pyplot as plt

# Kernel matrixes
#---------------------------------------------------
top_l =  [[0,0,-3,-1,0],
            [0,-1,-6,-1,-1],
            [-3,-6,0,0,0],
            [-1,-1,0,1,1],
            [0,-1,0,1,0]]

 
top_r =  [[0,-1,-3,0,0],
            [-1,-1,-6,-1,0],
            [0,0,0,-6,-3],
            [1,1,0,-1,-1],
            [0,1,0,-1,0],]

bottom_l =     [[0,-1,0,1,0],
            [-1,-1,0,1,1],
            [-3,-6,0,0,0],
            [0,-1,-6,-1,-1],
            [0,0,-3,-1,0]]

bottom_r =     [[0,1,0,-1,0],
            [1,1,0,-1,-1],
            [0,0,0,-6,-3],
            [-1,-1,-6,-1,0],
            [0,-1,-3,0,0]]



degree_30 = [    [0,	0,	0,	0,	0,	-1,	-1],
    [0,	0,	0,	-1,	-1,	-1,	0],
    [0,	-3,	-1,	-1,	0,	0,	1],
    [-1,	-6,	0,	0,	1,	1,	1],
    [-1,	0,	0,	0,	0,	0,	0],
    [-1,	-6,	-1,	-1,	-1,	-1,	-1],
    [0,	0,	-3,	-1,	0,	0,	0]
]

#---------------------------------------------------
def gen_kernal(m_list):
    '''From a list of lists returns a kernal'''
   
    matrix = np.zeros((len(m_list),len(m_list[0])))
    for row,l in enumerate(m_list):
        for col,num in enumerate(l):
            matrix[row,col] = num
    #print matrix
    return matrix
  
    
t_l = gen_kernal(top_l)
t_r = gen_kernal(top_r)
b_l = gen_kernal(bottom_l)
b_r = gen_kernal(bottom_r)
d_30 = gen_kernal(degree_30)

#---------------------------------------------------
def corner(image, times, kernel):
    '''Applies kernal to image'''
    #print kernel
    for i in range(0,times):
        image = cv2.filter2D(image, -1, kernel)   
    return image

#---------------------------------------------------
def salt_n_pepper(img, percentage):
    ''' Generates salt and pepper noise in an 
    image'''
    if img == None:
        print "Image Not Found"
        return 
    
    height, width = img.shape[:2]
    size = img.size
    
    percentage = percentage*.01
    num_salt_pep = int(percentage*size)
    
    for i in range (num_salt_pep):
        black_or_white = randint(0, 1)
        
        if black_or_white == 0:
            pixel = [0,0,0]
        else:
            pixel = [255,255,255]
            
        x = randint(0,width-1)
        y = randint(0, height-1)
        
        img[y,x] = pixel
    
    return img

#---------------------------------------------------
def all_corners(img):
    ''' Generates the image with all the rectangels 
    in it'''
    image1 = corner(img, 1, b_l)
    image2 = corner(img, 1, b_r)
    image3 = corner(img, 1, t_l)
    image4 = corner(img, 1, t_r)

    image5 = image1 + image2 + image3 + image4
    
    return image5

#---------------------------------------------------
def blur(image, scale, times):
    ''' Median blurs an image based on the scale and
    number of times'''
    
    # Kernal Matrix
    kernel = np.ones((scale, scale),
                     np.float32)/(scale*scale)

    for i in range(0, times):
        image = cv2.filter2D(image, -1, kernel)
    
    return image




def get_whites(img):
    
    nums = []
    for row,i in enumerate(img):
        for col,j in enumerate(i):
            if j[0] > 150 and j[1] > 150 and j[2] > 150:
                nums.append([row,col])
        
    return nums

def threshold_image(img, inverse=False):
    if (not(inverse)):
        ret,img = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
    else:
        ret,img = cv2.threshold(img,127,255,cv2.THRESH_BINARY_INV)
    return img

def corners(image):
    '''Returns a dictionary of possible 4 corners of a rectangle'''
    image = threshold_image(image)
    
    corners = {}
    image1 = corner(image, 1, b_l)
    #display_image(image)
    #display_image(image1)
   # image1_i = corner(image, 1, b_l)
    corners['bottom_left'] = get_whites(threshold_image(image1))
    #corners['bottom_left'][0:0] = get_whites(threshold_image(image1_i))
    
    image2 = corner(image, 1, b_r)
    #display_image(image2)
    
    #image2_i = corner(image, 1, b_l)
    corners['bottom_right'] = get_whites(threshold_image(image2))
    #corners['bottom_right'][0:0] = get_whites(threshold_image(image2_i))
    
    
    image3 = corner(image, 1, t_l)
    #display_image(image3)
    
    #image3_i = corner(image, 1, b_l)
    corners['top_left'] = get_whites(threshold_image(image3))
    #corners['top_left'][0:0] = get_whites(threshold_image(image3_i))
    

    image4 = corner(image, 1, t_r)    
    #display_image(image4)
    
    #image4_i = corner(image, 1, b_l)
    corners['top_right'] = get_whites(threshold_image(image4))
    #corners['top_right'][0:0] = get_whites(threshold_image(image4_i))
    
    return corners

def display_image(image):
    # Display
    cv2.imshow("blurp", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()    

def poss_corner_right(point, pts_below, dof=3):
    ''' finds a possible corner below the point'''
    if point == None:
            return None    
    
    # Sorts inplace by first parameter in inner list
    pts_below.sort(key=lambda x: x[0])
    
    # Searches for the closest point below
    for pp in pts_below:
        if pp[0] <= point[0]+dof and pp[0] >= point[0]-dof and point[1] < pp[1]:
            return pp

def poss_corner_below(point, pts_below, dof=3):
    ''' finds a possible corner below the point'''
    if point == None:
        return None
    
    # Sorts inplace by first parameter in inner list
    pts_below.sort(key=lambda x: x[1])
    
    # Searches for the closest point below
    for pp in pts_below:
        if pp[1] <= point[1]+dof and pp[1] >= point[1]-dof and point[0] < pp[0]:
            return pp
    
def find_rectangle(pt, point_dict):
    ''' finds a rectangle given a top_left point'''
    
    rectangle = {}
    

    rectangle["top_left"] = pt
    rectangle["bottom_left"] = poss_corner_below(pt, point_dict["bottom_left"])
    rectangle["top_right"] = poss_corner_right(pt, point_dict["top_right"])      
    rectangle["bottom_right"] = poss_corner_below(rectangle["top_right"], point_dict["bottom_right"])
    
    for c in rectangle:
        if rectangle[c] == None:
            return None, point_dict
    
    # Removal from list
    point_dict["bottom_left"].remove(rectangle["bottom_left"])
    point_dict["bottom_right"].remove(rectangle["bottom_right"])
    point_dict["top_right"].remove(rectangle["top_right"])
    
    
    
    return rectangle, point_dict
    

def draw_rectangle(image, rect, i):
    rect = flip_xy(rect)
    
    cv2.line(image, tuple(rect["top_left"]), tuple(rect["top_right"]), (0,255,0), 1)
    cv2.line(image, tuple(rect["top_left"]), tuple(rect["bottom_left"]), (0,255,0), 1)
    cv2.line(image, tuple(rect["top_right"]), tuple(rect["bottom_right"]), (0,255,0), 1)
    cv2.line(image, tuple(rect["bottom_left"]), tuple(rect["bottom_right"]), (0,255,0), 1)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    placement = list(rect["bottom_left"])
    placement[0] = placement[0] + 5 
    placement[1] = placement[1] - 5
    cv2.putText(image,str(i), tuple(placement), font, .2, (0,255,0))    
    
def flip_xy(rect):
    for key in rect:
        rect[key] = (rect[key][1],rect[key][0])
    return rect

def check_rectangularity(rect,dof=1):
    rect["top_left"]
    rect["top_right"]
    rect["bottom_left"]
    rect["bottom_right"]
    
    if (rect["top_right"][0] <= rect["top_left"][0] + dof) and (rect["top_right"][0] >= rect["top_left"][0] - dof):
        if (rect["bottom_left"][1] <= rect["top_left"][1] + dof) and (rect["bottom_left"][1] >= rect["top_left"][1] - dof):
            if (rect["bottom_right"][0] <= rect["bottom_left"][0] + dof) and (rect["bottom_right"][0] >= rect["bottom_left"][0] - dof): 
                if (rect["bottom_right"][1] <= rect["top_right"][1] + dof) and (rect["bottom_right"][1] >= rect["top_right"][1] - dof):
                    return True
                
    return False
    

def all_rectangles(img):
    '''Gets all of the rectangles from a set of points'''
    c = corners(img)
    rects = []    
    
    print c

    c["top_left"].sort(key=lambda x: x[1])
        
    for tl in c["top_left"]:
        r, c =  find_rectangle(tl, c)

        
        if not(r == None):
            if check_rectangularity(r):
                rects.append(r)
            
        

    print len(rects)
    return rects

def draw_all_rectangles(img, rects):
    ''' Draws all of the rectangles to the image'''
    for i,r in enumerate(rects):
        draw_rectangle(img, r, i)
    return img
        
def detect_rectangles(img):
    ''' Detects, draws, and then desplays all of the rectangles found in an image'''
    r = all_rectangles(img)
    img_i = threshold_image(img, inverse=True)
    r[0:0] = all_rectangles(img_i)
    return draw_all_rectangles(img, r)

def detect_30_degrees(img):
    ''' Detects all 30 degree, dispalys image with white area where 30 degree angle is'''
    img2 = threshold_image(img)
    img2 = corner(img2, 1, d_30)
    return img2   
        
def hist_equalization(img):  
    ''' Preforms histogram equalization on an image '''
    ''' code taken from:  http://docs.opencv.org/master/doc/py_tutorials/py_imgproc/py_histograms/py_histogram_equalization/py_histogram_equalization.html'''
    
    hist,bins = np.histogram(img.flatten(),256,[0,256])
        
    cdf = hist.cumsum()
    cdf_m = np.ma.masked_equal(cdf,0)
    cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
    cdf = np.ma.filled(cdf_m,0).astype('uint8')    
    img2 = cdf[img]
    
    return img2    
        
if __name__== "__main__":    
    
    # images
    #img = cv2.imread('../images/rectangles.jpg')
    #img = cv2.imread('../images/rectangles2.jpg')
    #img = cv2.imread('../images/ovechkin-skillscomp.jpg')
    #img = cv2.imread('../images/skyscrapers.jpg')
    #img = cv2.imread('../images/checkerboard.jpg')
    #img = cv2.imread('../images/checkerboard2.jpg')
    #img = cv2.imread('../images/squareb.jpg')
    #img = cv2.imread('../images/squarew.jpg') 
    #img = cv2.imread('../images/sq.jpg')  
    #img = cv2.imread('../images/checkbb.jpg')
    #img = cv2.imread('../images/planner.jpg')
    #img = cv2.imread('../images/redcheck.jpg')
    #img = cv2.imread('../images/triangle30.jpg')
    #img = cv2.imread('../images/angle30.jpg')
    #img = cv2.imread('../images/dark.jpg')
    img = cv2.imread('../images/building.jpg')
    
    
    #img = cv2.medianBlur(img,1)
    
    
    detect_rectangles(img)
       
    #img = salt_n_pepper(img,3)
    #img = blur(img, 5, 2)
    
    
    #img = rectangles(img)
    
    
    

    
   
    

    
