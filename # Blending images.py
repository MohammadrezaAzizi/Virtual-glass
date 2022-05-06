# Blending images
from PIL import Image
import cv2
import numpy as np
from numpy import eye
from torch import int16
Lens = cv2.imread("D:/pic/WP craft PC/flower_plant_petals_background_91800_1920x1080.jpg")
Lens = cv2.cvtColor( imgfg,cv2.COLOR_BGR2GRAY)
eye = cv2.imread("D:/pic/WP craft PC/lamborghini_aventador_lp_750_106615_1920x1080.jpg")
eye = cv2.cvtColor( imgbg,cv2.COLOR_BGR2GRAY)
### Color detection
#height, width, depth = img.shape
#circle_img = np.zeros((height, width), np.uint8)
#mask = cv2.circle(circle_img, (int(width / 2), int(height / 2)), 1, 1, thickness=-1)
#masked_img = cv2.bitwise_and(img, img, mask=circle_img)
#circle_locations = mask == 1
#bgr = img[circle_locations]
#rgb = bgr[..., ::-1]
#column_sums = (rgb.sum(axis=0)/5).round()


#Lens_shape = Lens.shape
#eye_shape = eye.shape
### Image blending
#resized_fg = cv2.resize(imgfg, ())
#resized_bg = cv2.resize(imgbg, ())
#blend = cv2.addWeighted(imgbg, 0.5, imgfg, 0.8, 0.0)
colorRGBALens = (Lens[,])
colorRGBAeye = (eye[,])
class image_combine:
    # class attribute
    Lens = "bird"

    # instance attribute
    def __init__(self, name, age):
        self.name = name
        self.age = age
def get_color(colorRGBALens, colorRGBAeye):
    alpha = 255 - ((255 - colorRGBALens[3]) * (255 - colorRGBAeye[3]) / 255)
    red   = (colorRGBALens[0] * (255 - colorRGBAeye[3]) + colorRGBAeye[0] * colorRGBAeye[3]) / 255
    green = (colorRGBALens[1] * (255 - colorRGBAeye[3]) + colorRGBAeye[1] * colorRGBAeye[3]) / 255
    blue  = (colorRGBALens[2] * (255 - colorRGBAeye[3]) + colorRGBAeye[2] * colorRGBAeye[3]) / 255
    rred = int(red)
    ggreen = int(green)
    bblue = int(blue)
    aalpha = int(alpha)
    return rred, ggreen, bblue, aalpha

def implmnt_clr(colorRGBALens, colorRGBAeye):
    get_color(colorRGBALens, colorRGBAeye)
    eye_image = eye
    eye_image[0] = rred
    eye_image[1] = ggreen
    eye_image[2] = bblue
    eye_image[3] = aalpha


    



