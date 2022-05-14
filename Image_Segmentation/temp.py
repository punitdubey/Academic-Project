# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 00:55:56 2021

@author: punit
"""

import cv2
File_Name = "Tiger.jpg"
data_img = cv2.imread("image/"+File_Name,cv2.IMREAD_COLOR)
cv2.imshow("Tiger",data_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

