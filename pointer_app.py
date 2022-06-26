import cv2
import time
import numpy as np
import hand_detector
from autopy import screen
from scipy.signal import savgol_filter
import mouse


class PointerApp:
    def __init__(
        self, camera_number: int = 0, camera_width: int = 1200, camera_height: int = 720
    ) -> None:
        self.camera_width = camera_width
        self.camera_height = camera_height
        self.screen_width, self.screen_height = screen.size()

        self.camera_number = camera_number
        self.detector = hand_detector.HandDetector(min_detection_confidence=0.7)
        self.xp_history = []
        self.yp_history = []

    def _smooth_pointer(self, xp: float, yp: float) -> tuple:
        """
        This function store the the coordinates for using later in smoothing the pointer.
        It uses savgol_filter to smooth a coordinates-history graph and return the last point.
        This smoothes the pointer movement and reduce the vibration of the pointer caused by human hand-shakes.\
        Args:
            xp (float): the x-cordinate of the pointer
            yp (float): the y-cordinate of the pointer
        Return:
            (tuple): the new pointer coordinates after filteration applied
        """
        self.xp_history.append(xp)
        self.yp_history.append(yp)

        if len(self.xp_history) > 1000 and len(self.xp_history) > 1000:
            del self.xp_history[:500]
            del self.yp_history[:500]

        if (
            len(self.xp_history) > 60 and len(self.xp_history) > 60
        ):  # start smothing after 60 frames
            xp_soomthed = savgol_filter(self.xp_history, 13, 1)
            yp_soomthed = savgol_filter(self.yp_history, 13, 1)
            return (
                xp_soomthed[-2],
                yp_soomthed[-2],
            )  # get the position in the previous 4 frames as it will be smoothed depending on the following movement (this will cause a delay around 50 ms )

        return (xp, yp)

    def _get_screen_coordinates(self, xp: float, yp: float) -> tuple:
        """
        Changes from camera coordinates to screen coordinates
        Args:
            xp (float): the y-cordinate position on camera
            yp (float): the y-cordinate position on camera
        Returns:
            (tuple): the X-Y-coordintaes of the pointer on screen which is equivilant to the camera X-Y-coordinates
        """
        width_offset = self.screen_width * 0.1
        height_offset = self.screen_height * 0.1

        # changing the range of the screen to avoide entering the camera deadzone to reach screen edges
        xp_screen = np.interp(
            xp,
            (0, self.camera_width),
            (0 - width_offset, self.screen_width + width_offset),
        )
        yp_screen = np.interp(
            yp,
            (0, self.camera_height),
            (0 - height_offset, self.screen_height + height_offset),
        )

        xp_screen = (
            self.screen_width - xp_screen
        )  # using self.screen_width-xp , because the camera image is mirrored
        xp_screen, yp_screen = self._smooth_pointer(xp_screen, yp_screen)

        if xp_screen < 1:
            xp_screen = 1
        if xp_screen > self.screen_width - 1:
            xp_screen = self.screen_width - 1
        if yp_screen < 1:
            yp_screen = 1
        if yp_screen > self.screen_height - 1:
            yp_screen = self.screen_height - 1

        return (xp_screen, yp_screen)

    def start(self):
        """This Function is called to start the"""
        self.cap = cv2.VideoCapture(self.camera_number)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.camera_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camera_height)
        prev_time = 0
        while True:
            _, img = self.cap.read()
            img = self.detector.find_hands(img)
            if self.detector.find_position(img, draw=False):
                landmarks_list = self.detector.find_position(img, draw=False)

                # thumb finger
                fing_1 = (landmarks_list[4][1], landmarks_list[4][2])
                # index finger
                fing_2 = (landmarks_list[8][1], landmarks_list[8][2])
                cv2.line(img, fing_1, fing_2, (255, 0, 255), 3)

                # pointer
                pointer_position = xp, yp = self.detector.find_midpoint(
                    "thumb_finger", "index_finger", landmarks_list
                )
                cv2.circle(img, pointer_position, 15, (255, 0, 255), cv2.FILLED)

                cv2.circle(img, fing_1, 15, (255, 0, 0), cv2.FILLED)
                cv2.circle(img, fing_2, 15, (255, 0, 0), cv2.FILLED)

                index_touch_thumb = self.detector.fingers_are_touching(
                    "thumb_finger", "index_finger", landmarks_list
                )
                middle_touch_thumb = self.detector.fingers_are_touching(
                    "thumb_finger", "middle_finger", landmarks_list
                )
                ring_touch_thumb = self.detector.fingers_are_touching(
                    "thumb_finger", "ring_finger", landmarks_list
                )

                if not (index_touch_thumb or middle_touch_thumb or ring_touch_thumb):
                    # move the pointer to the fingers position
                    print("case 1" + str(time.time()))
                    xp_screen, yp_screen = self._get_screen_coordinates(
                        xp, yp, self.camera_width, self.screen_width
                    )
                    mouse.move(xp_screen, yp_screen)

                elif index_touch_thumb and not (middle_touch_thumb or ring_touch_thumb):
                    # light figher press will be one left click, with little bit more force will be left click
                    mouse.click()
                    time.sleep(0.2)
                    print("case 2")

                elif middle_touch_thumb and not (index_touch_thumb or ring_touch_thumb):
                    # mouse right click case
                    mouse.right_click()
                    time.sleep(0.5)
                    print("case 3")

                elif ring_touch_thumb:
                    # scroll
                    xp_screen, yp_screen = self._get_screen_coordinates(
                        xp, yp, self.camera_width, self.screen_width
                    )
                    if yp_screen > self.screen_height / 2:
                        mouse.wheel(-0.5)
                    else:
                        mouse.wheel(0.5)
                    print("case 4")

            # displaying fps
            cur_time = time.time()
            fps = int(1 / (cur_time - prev_time))
            prev_time = cur_time
            cv2.putText(
                img, str(fps), (10, 70), cv2.FONT_HERSHEY_COMPLEX, 3, (255, 0, 0), 3
            )

            cv2.imshow("Img", img)
            cv2.waitKey(1)


if __name__ == "__main__":
    pointer_app = PointerApp()
    pointer_app.start()
