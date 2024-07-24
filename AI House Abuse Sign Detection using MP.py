#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install mediapipe')


# In[ ]:


# import the opencv library 
import mediapipe as mp
import numpy
import cv2
import numpy as np
import uuid
import os
fingerFlag=False
thumbFlag=False
HouseAbuseDetected=False


# In[ ]:


mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands


# In[ ]:


cap = cv2.VideoCapture(0)

with mp_hands.Hands(min_detection_confidence=0.9, min_tracking_confidence=0.5) as hands: 
    while cap.isOpened():
        ret, frame = cap.read()
        
        # BGR 2 RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Flip on horizontal
        image = cv2.flip(image, 1)
        
        # Set flag
        image.flags.writeable = False
        
        # Detections
        results = hands.process(image)
        
        # Set flag to true
        image.flags.writeable = True
        
        # RGB 2 BGR
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Detections
        #print(results)
        
        # Rendering results
        if results.multi_hand_landmarks:
            for num, hand in enumerate(results.multi_hand_landmarks):
                mp_drawing.draw_landmarks(image, hand, mp_hands.HAND_CONNECTIONS) 
#                                         mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
#                                         mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2),
                if get_label(num, hand, results):
                    text, coord = get_label(num, hand, results)
                    cv2.putText(image, text, coord, cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2,cv2.LINE_AA)
            #Draw Agnles
            draw_finger_angles(image, results, joint_list)
            #Curling
            cv2.rectangle(image,(0,0),(200,220),(255,255,255),-1)
            cv2.putText(image,"Fingers Curled ",(15,14),
                        cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),1,cv2.LINE_AA)
            cv2.putText(image,str(fingerFlag),(12,40),
                        cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),1,cv2.LINE_AA)
            
            
            cv2.putText(image,"Thumb Curled ",(15,70),
                        cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),1,cv2.LINE_AA)
            cv2.putText(image,str(thumbFlag),(12,100),
                        cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),1,cv2.LINE_AA)
            
            
            cv2.putText(image,"Abuse Detected ",(15,130),
                        cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),1,cv2.LINE_AA)
            cv2.putText(image,str(HouseAbuseDetected),(12,160),
                        cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),1,cv2.LINE_AA)
            
            
            if get_angle(image, results, joint=[11,10,9]) > 100 and  get_angle(image, results, joint=[15,14,13]) > 100 and get_angle(image, results, joint=[19,18,17]) > 100:
                fingerFlag=False
                
                
            if get_angle(image, results, joint=[3,2,1])>140:
                thumbFlag=False 
            if fingerFlag==False and thumbFlag==False :
                HouseAbuseDetected=False
                
            if get_angle(image, results, joint=[11,10,9]) < 100 and  get_angle(image, results, joint=[15,14,13]) < 100 and get_angle(image, results, joint=[19,18,17]) < 100:
                fingerFlag=True
                print("Fingers curled ",fingerFlag)
                
            if get_angle(image, results, joint=[3,2,1])<140:
                thumbFlag=True
                print("Thumb Curled ",thumbFlag)
                
              
                
            if fingerFlag and thumbFlag :
                HouseAbuseDetected=True
                print("HouseAbuseDetected : ",HouseAbuseDetected)
            
                #HouseAbuseDetected=False
        cv2.imshow('House Abuse Detection', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()


# In[ ]:


def get_label(index, hand, results):
    output = None
    for idx, classification in enumerate(results.multi_handedness):
        if classification.classification[0].index == index:
            #process results
            label = classification.classification[0].label
            score = classification.classification[0].score
            text = '{} {}'.format(label, round(score, 2))
            
            #extract coords
            coords = tuple(np.multiply(
            np.array((hand.landmark[mp_hands.HandLandmark.WRIST].x, hand.landmark[mp_hands.HandLandmark.WRIST].y)),
                [640,480]).astype(int))
            
            output = text, coords
    return output


# In[ ]:


from matplotlib import pyplot as plt  


# In[ ]:


joint_list = [[11,10,9],[19,18,17],[3,2,1]]


# In[ ]:


def draw_finger_angles(image, results, joint_list):
    for hand in results.multi_hand_landmarks:
        for joint in joint_list:
            a = np.array([hand.landmark[joint[0]].x, hand.landmark[joint[0]].y])
            b = np.array([hand.landmark[joint[1]].x, hand.landmark[joint[1]].y])
            c = np.array([hand.landmark[joint[2]].x, hand.landmark[joint[2]].y])
            
            radians = np.arctan2(c[1] - b[1], c[0]-b[0]) - np.arctan2(a[1] - b[1], a[0]-b[0])
            angle = np.abs(radians*180.0/np.pi)
            
            if angle > 180.0:
                angle = 360 - angle
            
            cv2.putText(image, str(round(angle,2)),tuple(np.multiply(b, [640,480]).astype(int)),
                                                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255),2,cv2.LINE_AA)
    return image


# In[ ]:





# In[ ]:


plt.imshow(cv2.cvtColor(test,cv2.COLOR_BGR2RGB))


# In[ ]:


def get_angle(image, results, joint):
    for hand in results.multi_hand_landmarks:
            a = np.array([hand.landmark[joint[0]].x, hand.landmark[joint[0]].y])
            b = np.array([hand.landmark[joint[1]].x, hand.landmark[joint[1]].y])
            c = np.array([hand.landmark[joint[2]].x, hand.landmark[joint[2]].y])
            
            radians = np.arctan2(c[1] - b[1], c[0]-b[0]) - np.arctan2(a[1] - b[1], a[0]-b[0])
            angle = np.abs(radians*180.0/np.pi)
            
            if angle > 180.0:
                angle = 360 - angle
            
            cv2.putText(image, str(round(angle,2)),tuple(np.multiply(b, [640,480]).astype(int)),
                                                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255),2,cv2.LINE_AA)
    return angle


# In[ ]:


b=draw_finger_angless(image, results, joint=[7,6,5])
print(b)


# In[ ]:




