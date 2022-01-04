import cv2
import time
import numpy as np
import hand_detector
from autopy import screen
from scipy.signal import savgol_filter
import mouse



#######################
camera_width, camera_height = 1200, 720
screen_width, screen_height = screen.size()
camera_number = 0
######################
detector  = hand_detector.HandDetector(min_detection_confidence=0.7)
cap = cv2.VideoCapture(camera_number)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, camera_width) 
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_height)
#######################
xp_history = []
yp_history = []
#######################




def smooth_pointer(xp, yp):
    xp_history.append(xp)
    yp_history.append(yp)

    if len(xp_history)>1000 and len(xp_history)>1000:
        del xp_history[:500]
        del yp_history[:500]

    if len(xp_history)>60 and len(xp_history)>60:  #start smothing after 60 frames 
        xp_soomthed = savgol_filter(xp_history, 13, 1)
        yp_soomthed = savgol_filter(yp_history, 13, 1)
        return xp_soomthed[-2], yp_soomthed[-2] #get the position in the previous 4 frames as it will be smoothed depending on the following movement (this will cause a delay around 50 ms )
    
    return xp, yp
        
      

def get_screen_coordinates(xp, yp, camera_width, screen_width):
    #change from camera coordinates (xp,yp) to return screen coordinates (xp_screen, yp_screen)

    # xp_screen = np.interp(xp, (0, camera_width), (0, screen_width))
    # yp_screen = np.interp(yp, (0, camera_height), (0, screen_height))
    #changing the range of the screen to avoide entering the camera deadzone to reach screen edges
    xp_screen = np.interp(xp, (0, camera_width), (0-screen_width*0.1, screen_width+screen_width*0.1))  
    yp_screen = np.interp(yp, (0, camera_height), (0-screen_height*0.1, screen_height+screen_height*0.1))

    xp_screen = screen_width-xp_screen #using screen_width-xp , because the camera image is mirrored 

    xp_screen, yp_screen = smooth_pointer(xp_screen, yp_screen)

    if xp_screen < 1:
        xp_screen = 1
    if xp_screen > screen_width-1:
        xp_screen = screen_width-1
    if yp_screen < 1:
        yp_screen = 1
    if yp_screen > screen_height-1:
        yp_screen = screen_height-1
    
    return xp_screen, yp_screen




prev_time = 0
while True:
    success, img = cap.read()
    img = detector.find_hands(img)
    
    if detector.find_position(img, draw=False):
        landmarks_list = detector.find_position(img, draw=False)


        #thumb finger
        fing_1 = x1, y1 = landmarks_list[4][1], landmarks_list[4][2]
        #index finger
        fing_2 = x2, y2 = landmarks_list[8][1], landmarks_list[8][2]
        cv2.line(img, fing_1, fing_2, (255,0,255), 3)

        #pointer
        pointer_position= xp, yp = detector.find_midpoint("thumb_finger", "index_finger", landmarks_list)
        cv2.circle(img, pointer_position, 15, (255,0,255), cv2.FILLED)

        cv2.circle(img, fing_1, 15, (255,0,0), cv2.FILLED)
        cv2.circle(img, fing_2, 15, (255,0,0), cv2.FILLED)


        index_touch_thumb = detector.fingers_are_touching("thumb_finger", "index_finger", landmarks_list)
        middle_touch_thumb = detector.fingers_are_touching("thumb_finger", "middle_finger", landmarks_list)
        ring_touch_thumb = detector.fingers_are_touching("thumb_finger", "ring_finger", landmarks_list)

        
        # if index_touch_thumb and not middle_touch_thumb:
        #     #left mouse click case
        #     if mouse.is_pressed():
        #         xp_screen, yp_screen = get_screen_coordinates(xp, yp, camera_width, screen_width)
        #         mouse.move(xp_screen, yp_screen)
        #     else:
        #         cv2.circle(img, pointer_position, 15, (255,0,0), cv2.FILLED)
        #         mouse.press()
        #         print("case 2")
        # else:
        #     mouse.release()


        if not (index_touch_thumb or middle_touch_thumb or ring_touch_thumb):#move the pointer in the pointer position

            #get coordinates
            print("case 1" + str(time.time()))
            xp_screen, yp_screen = get_screen_coordinates(xp, yp, camera_width, screen_width)
            mouse.move(xp_screen, yp_screen)

        
        elif index_touch_thumb and not (middle_touch_thumb or ring_touch_thumb):
            #light figher press will be one left click, with little bit more force will be left click
            mouse.click()
            time.sleep(0.2)
            print("case 2")

        elif middle_touch_thumb and not (index_touch_thumb or ring_touch_thumb):
            #mouse right click case
            mouse.right_click()
            time.sleep(0.5)
            print("case 3")

        elif ring_touch_thumb:
            #scroll
            xp_screen, yp_screen = get_screen_coordinates(xp, yp, camera_width, screen_width)
            if yp_screen > screen_height/2:
                mouse.wheel(-0.5)
            else:
                mouse.wheel(0.5)
                
            print("case 4")
            

       
    #displaying fps
    cur_time = time.time()
    fps = int(1/(cur_time-prev_time))
    prev_time = cur_time
    cv2.putText(img, str(fps), (10,70), cv2.FONT_HERSHEY_COMPLEX, 3, (255, 0, 0), 3)

    cv2.imshow("Img", img)
    cv2.waitKey(1)

    #smooth