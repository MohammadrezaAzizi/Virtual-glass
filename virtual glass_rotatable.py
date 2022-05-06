import numpy as np
import cv2
import mediapipe as mp
import math
from PIL import Image






# Mediapipe Virtual glass model

sunglasses = cv2.imread('D:/MY DOC/career/Chashmyar/Virtual glass/Glasses/3367_Gcut.jpg')
#Choosing your preferred pair of sungalasses
def virtual_glass():

    mp_face_detection = mp.solutions.face_detection
    mp_drawing = mp.solutions.drawing_utils
    face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5, model_selection=1) #choosing my preferred model
    #model_selection = 0 ; for near distances (minimum of 30cm and maximum of 80cm - optimum distance: 45cm)
    #model_selection = 1 ; for farther distances (I used model_selection = 1)
    cap = cv2.VideoCapture(0) #Defining a name for videocapture function designed in opencv
    #there was this other way of defining a webcam using imutils.video that I didn't find functional. so cv2 was the choice :)

    while 1: # As I wanted to make sure my webcam is going to be open untill I shut the kernel down.
        success, image = cap.read()
        image = cv2.flip(image, 1) #To make interaction with my model easier. since at first I had it not flipped and it was a headache :)
        image_width, image_height = image.shape[1], image.shape[0] # As coordination of features is in percentage on mediapipe and I wanted
        #to have coordinations in pixel unit

    
        results = face_detection.process(image) #Assigning the attribute "process" from face_detection function to "result"
        if results.detections:  # If there is any face detected by the model (in the image):
            detections = results.detections # Assigning all the detections done by model (list) to "detections"
            for detection in detections:   #Iterating on each detection existing in "detections" list
                bbox = detection.location_data.relative_bounding_box # As it wasn't possible to slice features of interest from the
                # so called "list", I searched around about the code to call the feature of interest (in this case: relative bbox)
                # So I did the same exact thing for other features like eyes, nose

                x,y = int(bbox.xmin * image_width),int(bbox.ymin * image_height)    #Coordination of (x,y) corresponds to starting-
                # coordination of face (TOP left)
                w,h = int(bbox.width * image_width),int(bbox.height * image_height) #Width and height of face bounding box
                faces = x,y,w,h #Gathering all the coordinations corresponding to topleft coordination of face and it's width and height
                #I wrote the above line just to print it and make sure that all coordinations are delivered by model.
                # Also did the same thing for eyes too. I deleted the "print" part just to make the code clean.


                #Right Eye 
                righteye = detection.location_data.relative_keypoints[0] 
                RE_x,RE_y = int(righteye.x * image_width),int(righteye.y * image_height) 
            
                #Left Eye
                lefteye = detection.location_data.relative_keypoints[1] 
                LE_x,LE_y = int(lefteye.x * image_width),int(lefteye.y * image_height) 
            
            
                #nose
                nose = detection.location_data.relative_keypoints[2] 
                nose_x, nose_y = int(nose.x * image_width),int(nose.y * image_height) 
            
                width = int(2.8*(LE_x-(RE_x+8))) #As in the previous codes ("haarcascade" model for example), there was this 
                #sovereignity for left eye, which biased the model towards collapsing in situation where detecting left eye was impossible
                # To overcome that challenge, I set the width of sunglasses to be a combination of both eyes' X coordination.
                # The height of sunglasses was not equal to height of face (like we didn't know that :] ) so I had to use a fraction of 
                # the height of face.
                height = int(h/1.7)
                Eyes_coord = LE_x,LE_y,width # (Just) a simple trial to check the capability of the face detection model
            
                img_roi = image[ int(1.1*y):int(1.1*y)+height,x:x+width] # The region of interest to switch face region with exact dimensions
                # of sunglasses [ delete face in regions that sunglasses overlay, and show sunglasses image instead ]
                #sunglasses_small = cv2.resize(sunglasses, (width,height),  interpolation=cv2.INTER_AREA) 
                #Resizing the sunglasses in (Width,height) dimension.
                ####################
                sunglasses_small =  Image.open("D:/MY DOC/career/Chashmyar/Virtual glass/Glasses/3367_Gcut.jpg")
                sunglasses_small = sunglasses_small.resize((width,height), resample=Image.LANCZOS, reducing_gap=None)
                # Rotate the image by 60 degrees counter clockwise
                # Angle is in degrees counter clockwise
                angle_elements = ((LE_y-RE_y)/(LE_x-RE_x))
                #angle = np.angle(angle_elements)
                angle = math.atan(angle_elements)
                angle = math.degrees((-1)*angle)
                result = sunglasses_small.rotate(angle)
                #result = cv2.imread(result)
                
                #print(angle)

                #image_center = tuple(np.array(sunglasses_small.shape[1::-1]) / 2)
                #rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
                #result = cv2.warpAffine(sunglasses_small, rot_mat, sunglasses_small.shape[1::-1], flags=cv2.INTER_LINEAR)
                ####################

                gray_sunglasses = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY) 
                #The shadow-wise image of sunglasses
                ret, mask = cv2.threshold(gray_sunglasses, 230, 255,  cv2.THRESH_BINARY_INV) 
                #To make sure that sunglasses image is more like shadow.
                # Mask is the whole image of sunglasses not including the glasses. we don't need that part, to be more specific we just 
                # need the sunglasses and not the surroundings of it.
                mask_inv = cv2.bitwise_not(mask)
                #Mask_inv is not mask (in every sense of the word) It's just the glasses and not the surroundings of them unlike Mask

                masked_face = cv2.bitwise_and(result, result, mask=mask)
                # The unity of "sunglasses_small" image with the "mask" image (sunglasses region without sunglasses)
                # So now there's gonna be an image, with a black background and a pair of glasses in center
             
                masked_frame = cv2.bitwise_and(img_roi,  img_roi, mask=mask_inv)
                image[int(1.1*y):int(1.1*y)+height,x:x+width] = cv2.add(masked_face,  masked_frame)

            cv2.imshow('image', image)
            k = cv2.waitKey(30) & 0xff
            if k == 27:
                break
    cap.release()            
if __name__ == '__main__':
    virtual_glass()