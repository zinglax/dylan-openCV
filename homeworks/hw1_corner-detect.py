import cv2
import numpy as np

# image
#img = cv2.imread('../images/rectangles.jpg')
img = cv2.imread('../checkerboard.jpg')

# Kernel matrixes
top_left = np.zeros((5,5))
top_right = np.zeros((5,5))
bottom_left = np.zeros((5,5))
bottom_right = np.zeros((5,5))


bottom_r =  [[-1,-1,-1,-1,-1],
            [-1,-2,-2,-2,-2],
            [-1,-2,0,0,0],
            [-1,-2,0,2,2],
            [-1,-2,0,2,1]]

bottom_l =  [[-1,-1,-1,-1,-1],
            [-2,-2,-2,-2,-1],
            [0,0,0,-2,-1],
            [2,2,0,-2,-1],
            [1,2,0,-2,-1],]

top_r =     [[-1,-2,0,2,1],
            [-1,-2,0,2,2],
            [-1,-2,0,0,0],
            [-1,-2,-2,-2,-2],
            [-1,-1,-1,-1,-1]]

top_l =     [[1,2,0,-2,-1],
            [2,2,0,-2,-1],
            [0,0,0,-2,-1],
            [-2,-2,-2,-2,-1],
            [-1,-1,-1,-1,-1]]

def gen_kernal(m_list):
    matrix = np.zeros((5,5))
    for row,l in enumerate(m_list):
        for col,num in enumerate(l):
            matrix[row,col] = num
    print matrix
    return matrix
    
t_l = gen_kernal(top_l)
t_r = gen_kernal(top_r)
b_l = gen_kernal(bottom_l)
b_r = gen_kernal(bottom_r)



#---------------------------------------------------
top_left[0,0] = -1
top_left[0,1] = -1
top_left[0,2] = -1
top_left[0,3] = -1
top_left[0,4] = -1

top_left[1,0] = -1
top_left[2,0] = -1
top_left[3,0] = -1
top_left[4,0] = -1

top_left[1,1] = -2
top_left[1,2] = -2
top_left[1,3] = -2
top_left[1,4] = -2

top_left[2,1] = -2
top_left[3,1] = -2
top_left[4,1] = -2

top_left[3,3] = 2
top_left[3,4] = 2
top_left[4,3] = 2

top_left[4,4] = 1


#---------------------------------------------------
top_right[0,0] = -1
top_right[0,1] = -1
top_right[0,2] = -1
top_right[0,3] = -1
top_right[0,4] = -1

top_right[1,4] = -1
top_right[2,4] = -1
top_right[3,4] = -1
top_right[4,4] = -1

top_right[1,0] = -2
top_right[1,1] = -2
top_right[1,2] = -2
top_right[1,3] = -2

top_right[2,3] = -2
top_right[3,3] = -2
top_right[4,3] = -2

top_right[3,0] = 2
top_right[3,1] = 2
top_right[4,1] = 2

top_right[4,0] = 1

#---------------------------------------------------

bottom_left[0,0] = -1
bottom_left[1,0] = -1
bottom_left[2,0] = -1
bottom_left[3,0] = -1
bottom_left[4,0] = -1

bottom_left[4,1] = -1
bottom_left[4,2] = -1
bottom_left[4,3] = -1
bottom_left[4,4] = -1

bottom_left[0,1] = -2
bottom_left[1,1] = -2
bottom_left[2,1] = -2
bottom_left[3,1] = -2

bottom_left[3,2] = -2
bottom_left[3,3] = -2
bottom_left[3,4] = -2

bottom_left[0,3] = 2
bottom_left[1,3] = 2
bottom_left[1,4] = 2

bottom_left[0,4] = 1

#---------------------------------------------------
# [0,0] [0,1]
# [1,0] [1,1]

bottom_right[0,0] = 1

bottom_right[1,1] = 2
bottom_right[1,0] = 2
bottom_right[0,1] = 2

bottom_right[3,0] = -2
bottom_right[3,1] = -2
bottom_right[3,2] = -2
bottom_right[3,3] = -2

bottom_right[0,3] = -2
bottom_right[1,3] = -2
bottom_right[2,3] = -2

bottom_right[0,4] = -1
bottom_right[1,4] = -1
bottom_right[2,4] = -1
bottom_right[3,4] = -1
bottom_right[4,4] = -1

bottom_right[4,0] = -1
bottom_right[4,1] = -1
bottom_right[4,2] = -1
bottom_right[4,3] = -1

#---------------------------------------------------


def corner(image, times, kernel):
    print kernel
    
    for i in range(0,times):
        image = cv2.filter2D(image, -1, kernel)
        
    return image

image = corner(img, 1, t_l)
#image = corner(image, 1, top_right)

# Display
cv2.imshow("blurp", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
