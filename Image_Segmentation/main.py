# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 00:20:49 2021

@author: punit
"""

#imports
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture as GMM


def resize_image(img,Scale):
    """Reshape the image to the scale of the image"""
    #new width
    width = int(data_img.shape[1] * Scale/ 100)
    
    #new height
    height = int(data_img.shape[0] * Scale/ 100)
    
    #new dimension
    dimension = (width, height)   
    
    # return resized new image
    return cv2.resize(img, dimension, interpolation = cv2.INTER_AREA)

#variables
Name = "cat"
File_Name = Name+".jpg"
Result_Name = Name+"-Segmented.jpg"


#import image
data_img = cv2.imread("image/"+File_Name,cv2.IMREAD_COLOR)

#resized image
new_img = resize_image(data_img, 30)



#image resize to lower-dimension for segentation
res_image = new_img.reshape((-1,3))


#implement the model
#coavriance_type can be used a.tied, b.full, c.diag, d. spherical
n_comp = 2
seg_model = GMM(n_components=n_comp,covariance_type="full")

#fit the  resahped image
seg_model.fit(res_image)

#prdicted labels of gmm
seg_labels = seg_model.predict(res_image)
print(seg_labels.shape)

seg_temp = seg_labels

#size of the image
Shape = new_img.shape
print("Shape of the resized image : ",Shape)

image_segmented = seg_labels.reshape(Shape[0],Shape[1])


#copy resize image
for i in range(n_comp):
    temp = res_image.copy()
    for j in range(len(seg_temp)):
        
        if(seg_temp[j]==i):
            temp[j] = np.array([255,255,255]) 
    temp = np.reshape(temp,(Shape[0],Shape[1],3))
    cv2.imshow(Name+str(i),temp)
    cv2.imwrite("Result/"+Name+str(i)+".jpg",temp)

#plot
plt.imshow(image_segmented)



#save file
plt.tight_layout()
plt.savefig("Result/"+Result_Name,dpi = 500, bbox_inches = 'tight')
plt.show()


#read stored file
Data = cv2.imread("Result/"+Result_Name,0)
Data = resize_image(Data,60)

# show image
cv2.imshow("Resized-Tiger",new_img)
cv2.imshow("Segemented-image",Data)
cv2.waitKey(0)
cv2.destroyAllWindows()








