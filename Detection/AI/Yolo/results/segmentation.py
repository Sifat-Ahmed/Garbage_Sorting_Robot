import cv2
import numpy as np
from imutils import perspective
from collections import OrderedDict


class SegmentRock:
    def __init__(self, cfg, binary_threshold=30):
        self._cfg = cfg
        self._binary_threshold = binary_threshold

    def __get_threshold_image(self, image):
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret, thresh_image = cv2.threshold(gray_image, self._binary_threshold, 255, 0)
        # cv2.imshow("", thresh_image)
        # cv2.waitKey(100)
        return thresh_image

    def __get_contour(self, thresh_image):
        contours, hierarchy = cv2.findContours(image=thresh_image, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
        # cv2.imshow("",
        #            cv2.drawContours(thresh_image.copy(), contours ,-1,(255,0,0),5))
        return contours

    def measure_theta(self, image):
        thresh_image = self.__get_threshold_image(image)
        contours = self.__get_contour(thresh_image)
        contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
        if len(contours) < 1:
            return None
        rect = cv2.minAreaRect(contours[0])
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        box = perspective.order_points(box)
        theta = (rect[-1])
        return {
            "theta": theta,
        }

    def is_pickable(self, rock, to_x, to_y, segmented_image):
        x1 = (rock['start'].x - to_x) if (rock['start'].x - to_x) >= 0 else 0
        y1 = (rock['start'].y - to_y) if (rock['start'].y - to_y) >= 0 else 0

        x2 = (rock['end'].x + to_x) if (rock['end'].x + to_x) <= self._cfg.image_size[1] else self._cfg.image_size[1]
        y2 = (rock['end'].y + to_y) if (rock['end'].y + to_y) <= self._cfg.image_size[0] else self._cfg.image_size[0]

        ## top
        top_image = segmented_image[rock['end'].y: y2, rock['start'].x: rock['end'].x]
        ## left side
        left_image = segmented_image[y1: rock['start'].y, x1: rock['start'].x]
        ## right side
        right_image = segmented_image[rock['end'].y: y2, rock['end'].x: x2]
        ## ##bottom
        bottom_image = segmented_image[y1: rock['start'].y, x1: rock['start'].x]

        sides = [top_image, bottom_image, left_image, right_image]
        names = ['top', 'bottom', 'left', 'right']

        side_details = OrderedDict()

        for name, side in zip(names, sides):
            if side.shape[0] == 0 or side.shape[1] == 0:
                side_details[name] = -1
                continue
            gray = self.__get_threshold_image(side)
            contours = self.__get_contour(gray)
            contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
            if len(contours) < 1:
                side_details[name] = 0
            else:
                if cv2.contourArea(contours[0]) >= self._cfg.rock_area:
                    side_details[name] = 1
                else:
                    side_details[name] = 1

        return side_details
        # try:
        #     debug_image = segmented_image[y1: y2, x1:x2]
        #     cv2.imshow('', debug_image)
        #     cv2.waitKey(100)
        # except Exception as e:
        #     print(x1, y1, x2, y2)


