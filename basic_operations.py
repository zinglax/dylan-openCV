import cv2
import numpy as np


# Ovechkin image is 368x600, Height x Width
img = cv2.imread('./images/ovechkin-skillscomp.jpg')

# Splitting the image
b = img[:,:,0]
g = img[:,:,1]
r = img[:,:,2]

cv2.imwrite("./images/red_ov.jpg", r)

img2 = cv2.imread('./images/red_ov.jpg')
img2[:,:,0]=125
img2[:,:,1]=125




def only_red(image):
    h,w = get_height_and_width(image)
    
    for x,y in np.ndenumerate(image):
        print x
        #if (not(px[2] > px[1]) and not(px[2]  > px[0])):
            #px = [0,0,0]
        
    

def get_height_and_width(image):
    height = image.shape[0]
    width = image.shape[1]
    return height, width
    
only_red(img)

# Show Image
cv2.imshow("OV", img)
cv2.waitKey(0)

# Show Image
cv2.imshow("OV2", img2)
cv2.waitKey(0)

cv2.destroyAllWindows()
