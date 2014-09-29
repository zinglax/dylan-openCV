import cv2
import numpy as np
from random import randint

# images
#img = cv2.imread('../images/rectangles.jpg')
#img = cv2.imread('../images/rectangles2.jpg')
#img = cv2.imread('../images/ovechkin-skillscomp.jpg')
#img = cv2.imread('../images/skyscrapers.jpg')
img = cv2.imread('../images/checkerboard.jpg')
#img = cv2.imread('../images/squareb.jpg')
#img = cv2.imread('../images/squarew.jpg')

# Kernel matrixes
#---------------------------------------------------
#bottom_r =  [[-1,-1,-1,-1,-1],
            #[-1,-2,-2,-2,-2],
            #[-1,-2,0,0,0],
            #[-1,-2,0,2,2],
            #[-1,-2,0,2,1]]

#bottom_l =  [[-1,-1,-1,-1,-1],
            #[-2,-2,-2,-2,-1],
            #[0,0,0,-2,-1],
            #[2,2,0,-2,-1],
            #[1,2,0,-2,-1],]

#top_r =     [[-1,-2,0,2,1],
            #[-1,-2,0,2,2],
            #[-1,-2,0,0,0],
            #[-1,-2,-2,-2,-2],
            #[-1,-1,-1,-1,-1]]

#top_l =     [[1,2,0,-2,-1],
            #[2,2,0,-2,-1],
            #[0,0,0,-2,-1],
            #[-2,-2,-2,-2,-1],
            #[-1,-1,-1,-1,-1]]


 
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
    
    whites = []
    
    height, width = img.shape[:2]
    for x in range(width-1):
        for y in range(height-1):
            print "x: "+ str(x) +" y: "+ str(y) + " value: " + str(img[x,y])
            if str(img[y,x]) == '[255 255 255]':
                whites.append([y,x])

    return whites

# Rectangle finder
#image = salt_n_pepper(img,.5)
#image = blur(image, 1, 1)
#image = rectangles(img)

image = corner(img, 1, t_l)

#whites = get_whites(image)
#for i in whites:
    #print i

#print np.all(image == '[255 255 255]',-1)

ret,image = cv2.threshold(image,127,255,cv2.THRESH_BINARY)

nums = []
for row,i in enumerate(image):
    for col,j in enumerate(i):
        if j[0] > 150 and j[1] > 150 and j[2] > 150:
            nums.append([row,col])
    
print nums



# Display
cv2.imshow("blurp", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
