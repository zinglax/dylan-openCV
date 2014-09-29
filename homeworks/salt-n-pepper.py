import cv2
import numpy as np
from random import randint

def salt_n_pepper(img, percentage):
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


## Testing  
#img = cv2.imread('../images/checkerboard.jpg')
#image = salt_n_pepper(img,25)

## Display
#cv2.imshow("blurp", image)
#cv2.waitKey(0)
#cv2.destroyAllWindows()