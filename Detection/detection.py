# import socket
# import time
# import sys
# from tkinter import S
# from Detection.Camera.Zed.zed import zed_stream
# # from Detection.AI.Yolo.object_detector import ObjectDetector
# from Detection.config import Config
# import numpy as np
# from Detection.AI.Yolo.SingleDetector import Detector
# from Detection.AI.Yolo.results.coordinates import Rock
# from Detection.AI.Yolo.results.segmentation import SegmentRock
# from Detection.Camera.Zed import cmake_example as cv_demo1
# import cv2
# import math


# class Detection():
#     def __init__(self):
#         self.cfg = Config()
#         self.zed = zed_stream()
#         self.object_detector = Detector(self.cfg)
#         self.rock_properties = Rock()
#         self.segmentor = SegmentRock(self.cfg)


#     def get_zed_image(self):
#         image = self.zed.get_image()
#         #image_3D = self.zed.get_3D_data()
#         # image_3D = np.zeros((256, 256, 1), dtype = "uint8")
#         return image[0],image[2],image[1]

#     def get_zed_bbox(self):
#         image_result = self.get_zed_image()
#         segmented_image = image_result[1]
#         rocks_in_the_image = list()
#         color = (255, 255, 0)
#         image = image_result[0]
        
#         bounding_boxes_in_the_image = self.object_detector.get_bbox(image_result[0])
#         for bbox in bounding_boxes_in_the_image:
#             rock = self.rock_properties.unpack_coordinates(bbox)
#             cv_demo1.z(rgb_image,Image_3D)
#             ## extracting the theta
#             # data = self.segmentor.measure_theta(
#             #     segmented_image[rock["start"].y: rock["end"].y, rock["start"].x: rock["end"].x])
#             # if data is None:
#             #     continue
#             image =

#         # AI_Result = [[],[],[]]
#         return rocks_in_the_image,image,image_result[1] #AI_Result, image_result[1],image_result[2]
#     def get_bbox_video(self,image_result):
#         AI_Result = self.object_detector.get_bbox(image_result, self.item)
#         return AI_Result




import socket
import time
import sys
from tkinter import S
from Detection.Camera.Zed.zed import zed_stream
# from Detection.AI.Yolo.object_detector import ObjectDetector
from Detection.config import Config
import numpy as np
from Detection.AI.Yolo.SingleDetector import Detector
from Detection.AI.Yolo.results.coordinates import Rock
from Detection.AI.Yolo.results.segmentation import SegmentRock
from Detection.Camera.Zed import cmake_example as cv_demo1
import cv2
import math


class Detection():
    def __init__(self):
        self.cfg = Config()
        self.zed = zed_stream()
        self.object_detector = Detector(self.cfg)
        self.rock_properties = Rock()
        self.segmentor = SegmentRock(self.cfg)


    def get_zed_image(self):
        image = self.zed.get_image()
        # image_3D = self.zed.get_3D_data()
        # image_3D = np.zeros((256, 256, 1), dtype = "uint8")
        return image[0],image[1],image[2]

    def get_zed_bbox(self):
        image_result = self.get_zed_image()
        segmented_image = image_result[2]
        rocks_in_the_image = list()
        color = (255, 255, 0)
        image = image_result[0]
        
        bounding_boxes_in_the_image = self.object_detector.get_bbox(image_result[0])
        for bbox in bounding_boxes_in_the_image:
            rock = self.rock_properties.unpack_coordinates(bbox)
            ## extracting the theta
            data = self.segmentor.measure_theta(
                segmented_image[rock["start"].y: rock["end"].y, rock["start"].x: rock["end"].x])
            if data is None:
                continue
            image = cv2.rectangle(
                image_result[0], (rock["start"].x, rock["start"].y),
                (rock["end"].x, rock["end"].y),
                color=color, thickness=2)
            image = cv2.putText(
                image_result[0], "ROCK: {}".format(data["theta"] ),
                (rock["start"].x, rock["start"].y),
                fontFace=cv2.FONT_HERSHEY_DUPLEX,
                fontScale=0.7, color=color)
            # print(data)
            
            point_x = math.floor(rock["start"].x + rock["width"] /2) 
            point_y = math.floor(rock["start"].y + rock["height"] /2)
            distance = 60
            start_point = (point_x, point_y)

            angle_point_x = point_x + distance * math.cos(data["theta"] * math.pi / 180)
            angle_point_y = point_y + distance * math.sin(data["theta"] * math.pi / 180)
            
            end_point = (math.floor(angle_point_x), math.floor(angle_point_y))


            image = cv2.line(image, start_point, end_point, color=color, thickness=2)

            angle_point_x = point_x - distance * math.cos(data["theta"] * math.pi / 180)
            angle_point_y = point_y - distance * math.sin(data["theta"] * math.pi / 180)
            
            end_point = (math.floor(angle_point_x), math.floor(angle_point_y))

            image = cv2.line(image, start_point, end_point, color=color, thickness=2)


            rock.update(data)
            rocks_in_the_image.append(rock)

        # AI_Result = [[],[],[]]
        return rocks_in_the_image,image,image_result[2] 
    def get_bbox_video(self,image_result):
        AI_Result = self.object_detector.get_bbox(image_result, self.item)
        return AI_Result


