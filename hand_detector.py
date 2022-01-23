import cv2
import mediapipe as mp
import math



class HandDetector():
    def __init__(self, static_image_mode=False,model_complexity=1, max_num_hands=2,  min_detection_confidence=0.5, min_tracking_confidence=0.5):
        
        self.mod = static_image_mode
        self.max_hands = max_num_hands
        self.complexity = model_complexity
        self.detection_conf = min_detection_confidence
        self.tracking_conf = min_tracking_confidence

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(self.mod, self.max_hands,self.complexity, self.detection_conf, self.tracking_conf)
        self.mp_draw = mp.solutions.drawing_utils

        self.finger_tips_idx = {
        "thumb_finger":4,
        "index_finger":8,
        "middle_finger":12,
        "ring_finger":16,
        "pinky_finger":20}



    def find_hands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #gets rgb image as hands object only uses RGB images
        self.results = self.hands.process(imgRGB) #this process the frame for us and give us the results

        if self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:
                if draw:
                    self.mp_draw.draw_landmarks(img, hand_landmarks, self.mp_hands.HAND_CONNECTIONS) #we need to display it on img as we are displaying image, mpHands.HAND_CONNECTIONS for the line between teh dots        

        return img
    
    def find_position(self, img, hand_idx=0, draw=True):

        if not self.results.multi_hand_landmarks:
            return False

        landmarks_list = []
        the_hand = self.results.multi_hand_landmarks[hand_idx]

        for finger_indx, land_mark in enumerate(the_hand.landmark): #id refers to index number of our fingers
            height, width, cannels = img.shape
            position_x, position_y = int(land_mark.x*width), int(land_mark.y*height)
            landmarks_list.append([finger_indx, position_x, position_y])
            
            if draw:
                cv2.circle(img, (position_x,position_y), 8, (255, 0, 0), cv2.FILLED)  
        return landmarks_list

    
    def touching_dist(self, landmarks_list): 
    #the distance between two fingers touching each other. touching distance is considered to be equal to 1.5*distance between idx 9 and 13 
        #index 13
        x13, y13 = landmarks_list[13][1], landmarks_list[13][2] 
        #index 17
        x17, y17 = landmarks_list[17][1], landmarks_list[17][2]

        touching_distance = 1.5*math.hypot(x13-x17, y13-y17)
        return touching_distance


    def fingers_are_touching(self, finger1_name, finger2_name, landmarks_list):  

        x1, y1 = landmarks_list[self.finger_tips_idx[finger1_name]][1], landmarks_list[self.finger_tips_idx[finger1_name]][2]
        x2, y2 = landmarks_list[self.finger_tips_idx[finger2_name]][1], landmarks_list[self.finger_tips_idx[finger2_name]][2]

        distance_between = math.hypot(x2-x1, y2-y1)
        
        if distance_between < self.touching_dist(landmarks_list):
            return True
        
        return False
    
    def find_midpoint(self, finger1_name, finger2_name, landmarks_list):  

        x1, y1 = landmarks_list[self.finger_tips_idx[finger1_name]][1], landmarks_list[self.finger_tips_idx[finger1_name]][2]
        x2, y2 = landmarks_list[self.finger_tips_idx[finger2_name]][1], landmarks_list[self.finger_tips_idx[finger2_name]][2]

        xp, yp = int((x1+x2)/2), int((y1+y2)/2)
        return xp, yp
        
