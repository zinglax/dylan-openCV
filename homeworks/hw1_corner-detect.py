import cv2
import numpy as np
from random import randint



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


#---------------------------------------------------
def gen_kernal(m_list):
    '''From a list of lists returns a kernal 5x5'''
    matrix = np.zeros((5,5))
    for row,l in enumerate(m_list):
        for col,num in enumerate(l):
            matrix[row,col] = num
    #print matrix
    return matrix
  
    
t_l = gen_kernal(top_l)
t_r = gen_kernal(top_r)
b_l = gen_kernal(bottom_l)
b_r = gen_kernal(bottom_r)

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
def rectangles(img):
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
    #image_inverse = threshold_image(image, True)
    
    corners = {}
    image1 = corner(image, 1, b_l)
   # image1_i = corner(image, 1, b_l)
    corners['bottom_left'] = get_whites(threshold_image(image1))
    #corners['bottom_left'][0:0] = get_whites(threshold_image(image1_i))
    
    image2 = corner(img, 1, b_r)
    #image2_i = corner(image, 1, b_l)
    corners['bottom_right'] = get_whites(threshold_image(image2))
    #corners['bottom_right'][0:0] = get_whites(threshold_image(image2_i))
    
    
    image3 = corner(img, 1, t_l)
    #image3_i = corner(image, 1, b_l)
    corners['top_left'] = get_whites(threshold_image(image3))
    #corners['top_left'][0:0] = get_whites(threshold_image(image3_i))
    

    image4 = corner(img, 1, t_r)    
    #image4_i = corner(image, 1, b_l)
    corners['top_right'] = get_whites(threshold_image(image4))
    #corners['top_right'][0:0] = get_whites(threshold_image(image4_i))
    
    return corners

def display_image(image):
    # Display
    cv2.imshow("blurp", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()    

def poss_corner_right(point, pts_below):
    ''' finds a possible corner below the point'''
    if point == None:
            return None    
    
    # Sorts inplace by first parameter in inner list
    pts_below.sort(key=lambda x: x[0])
    
    # Searches for the closest point below
    for pp in pts_below:
        if pp[0] <= point[0]+1 and pp[0] >= point[0]-1 and point[1] < pp[1]:
            return pp

def poss_corner_below(point, pts_below):
    ''' finds a possible corner below the point'''
    if point == None:
        return None
    
    # Sorts inplace by first parameter in inner list
    pts_below.sort(key=lambda x: x[1])
    
    # Searches for the closest point below
    for pp in pts_below:
        if pp[1] <= point[1]+1 and pp[1] >= point[1]-1 and point[0] < pp[0]:
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
    

def draw_rectangle(image, rect):
    rect = flip_xy(rect)
    
    cv2.line(image, tuple(rect["top_left"]), tuple(rect["top_right"]), (0,255,0), 1)
    cv2.line(image, tuple(rect["top_left"]), tuple(rect["bottom_left"]), (0,255,0), 1)
    cv2.line(image, tuple(rect["top_right"]), tuple(rect["bottom_right"]), (0,255,0), 1)
    cv2.line(image, tuple(rect["bottom_left"]), tuple(rect["bottom_right"]), (0,255,0), 1)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image,"hello", tuple(rect["bottom_left"]), font, .2, (0,255,0))    
    
def flip_xy(rect):
    for key in rect:
        rect[key] = (rect[key][1],rect[key][0])
    return rect
    
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
    img = cv2.imread('../images/sq.jpg')  
       
    #img = salt_n_pepper(img,.03)
    #img = blur(img, 2, 1)
    #img = rectangles(img)
    
    
    c = corners(img)
    #img = corner(img, 1, b_r)
    img = rectangles(img)

    rects = []
    count = 0
    
    d = c
    print c
    
    c["top_left"].sort(key=lambda x: x[1])
    
    
    rects = []
    for tl in c["top_left"]:
        r, d =  find_rectangle(tl, d)

        
        if not(r == None):
            rects.append(r)


    print len(rects)
    for r in rects:
        draw_rectangle(img, r)
        
        #if count == 15:
            #display_image(img)
            ##break
        #count = count + 1 
        
    
    print count

    display_image(img)
    
    
    
   
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    #print c['top_left'][0:3]
    #print c['bottom_left'][0:3]
    #print c['top_right'][0:3]
    #print c['bottom_right'][0:3]
    
    
    #rects = {}
    #for tl in c['top_left']:
        
        ## Possible Bottom Lefts
        #poss_bl = []
        #for bl in c['bottom_left']:
            
            ## if the bottom left col value is +- 3 away from top left
            #if bl[1] < tl[1]+3 and bl[1] > tl[1]-3:
                #poss_bl.append(bl)
                
        
        ## Sorts the list of list on the first value of inner list
        #poss_bl.sort(key=lambda x: x[0])
        
        ## Finds all of the other points possible corners in the local area
        #minimum = poss_bl[0][0]
        #for bl in poss_bl:
            #if not(bl[0] >= minimum and bl[0] <= 6):
                #poss_bl.remove(bl)
        
        ## Finding all of the possible bottom lefts for this rectangle
        
        
        ## TODO
        ## Need to find minimum row value and then add trim all the ones that are not +-3 from this value
        ## This basically gets all of the possible bottom lefts directly below this top left
        ##
        ## Next Do this with the top Rights in the same fashion 
        ## Finally From all of your top rights and bottom lefts find your bottom rights
        ## 
        ## Might want to create a function that finds all of the possible points above, below, left, or right of a given point
        ##
        ## Once you have all of the possible points for this rectangle, draw them...
        
        
        #rects[tuple(tl)] = {'poss_bl':poss_bl}
        
        
        
        
        ## Possible Top Rights
    
    
        
    #print rects
    