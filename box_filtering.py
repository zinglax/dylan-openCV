import cv2
import numpy as np

# Ovechkin image is 368x600, Height x Width
img = cv2.imread('./images/ovechkin-skillscomp.jpg')


def blur(image, times):

    # Kernal Matrix
    kernel = np.ones((5,5), np.float32)/25

    for i in range(0, times):
        image = cv2.filter2D(image, -1, kernel)
    
    return image

def emboss(image, times):
    
    # Kernel matrix
    kernel = np.zeros((5,5))
    
    kernel[0,0] = -1
    kernel[1,1] = -2
    kernel[2,2] = 3
    
    print kernel
    
    for i in range(0, times):
        image = cv2.filter2D(image, -1, kernel)
        
    return image    

def sharpen(image, times):
    
    # Kernel
    kernel = np.zeros((3,3))
    kernel[0,1] = -1
    kernel[1,0] = -1
    kernel[2,1] = -1
    kernel[1,2] = -1
    kernel[1,1] = 5
    
    for i in range(0, times):
        image = cv2.filter2D(image, -1, kernel)
            
    return image      


def diagonal_Prewitt(image,times):
    # Kernel
    kernel = np.zeros((3,3))   
    kernel[1,0] = 1
    kernel[2,0] = 1
    kernel[2,1] = 1
    kernel[0,1] = -1
    kernel[0,2] = -1
    kernel[1,2] = -1
    
    for i in range(0, times):
        image = cv2.filter2D(image, -1, kernel)
            
    return image      
    
    


def horizontal_frei_chen(image,times):
    
    # Kernel
    kernel = np.zeros((3,3))   
    kernel[2,0] = -1
    kernel[0,0] = -1
    kernel[1,0] = -1.4142
    kernel[1,2] = -1.4142
    kernel[0,2] = 1
    kernel[2,2] = 1
    
    for i in range(0, times):
        image = cv2.filter2D(image, -1, kernel)
            
    return image        

image = emboss(img,1)
#image = sharpen(image,1)
#image = blur(image, 1)
#image = diagonal_Prewitt(img, 1)
#image = emboss(image, 1)
#image = horizontal_frei_chen(image, 1)


# Display
cv2.imshow("blurp", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
