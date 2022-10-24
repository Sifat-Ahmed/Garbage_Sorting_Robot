
import sys
import pyzed.sl as sl
import cv2
from datetime import datetime
import numpy as np
now = datetime.now()
import time
import math
# sys.path.insert(0, '/home/eiismoke/scara_robot/src/Detection/Camera/Zed/')
from Detection.Camera.Zed import cmake_example as cv_demo1

class zed_stream():
    def __init__(self):
        self.zed = sl.Camera()
        self.input_type = sl.InputType()
        if len(sys.argv) >= 2 :
            self.input_type.set_from_svo_file(sys.argv[1])
        self.init = sl.InitParameters(input_t=self.input_type)
        self.init.camera_resolution = sl.RESOLUTION.HD720
        self.init.depth_mode = sl.DEPTH_MODE.QUALITY
        self.init.coordinate_units = sl.UNIT.MILLIMETER
        self.init.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
        self.init.depth_maximum_distance = 700
        self.init.depth_minimum_distance = 100
        
        err = self.zed.open(self.init)
        if err != sl.ERROR_CODE.SUCCESS :
            print(repr(err))
            self.zed.close()
            exit(1)

        self.zed.set_camera_settings(sl.VIDEO_SETTINGS.BRIGHTNESS, 4)
        self.zed.set_camera_settings(sl.VIDEO_SETTINGS.CONTRAST, 4)
        self.zed.set_camera_settings(sl.VIDEO_SETTINGS.HUE, 0)
        self.zed.set_camera_settings(sl.VIDEO_SETTINGS.SATURATION, 4)
        self.zed.set_camera_settings(sl.VIDEO_SETTINGS.SHARPNESS, 6)
        self.zed.set_camera_settings(sl.VIDEO_SETTINGS.GAMMA, 48)
        self.zed.set_camera_settings(sl.VIDEO_SETTINGS.GAIN, 48)
        self.zed.set_camera_settings(sl.VIDEO_SETTINGS.EXPOSURE, 27)
        

        self.res = sl.Resolution()
        self.res.width = 640
        self.res.height = 480
        self.image_zed = sl.Mat(self.res.width, self.res.height, sl.MAT_TYPE.U8_C4)
        self.point_cloud = sl.Mat(self.res.width, self.res.height, sl.MAT_TYPE.U8_C3, sl.MEM.CPU)
        self.point_cloud = sl.Mat(self.res.width, self.res.height, sl.MAT_TYPE.U8_C3, sl.MEM.CPU)

    def get_image(self):
        if self.zed.grab() == sl.ERROR_CODE.SUCCESS:
            self.zed.retrieve_image(self.image_zed, sl.VIEW.LEFT, sl.MEM.CPU, self.res)
            self.zed.retrieve_measure(self.point_cloud, sl.MEASURE.XYZRGBA,sl.MEM.CPU, self.res)
            color_image = self.image_zed.get_data()
            point_image = self.point_cloud.get_data()
            
            rgb_image = cv2.cvtColor(color_image, cv2.COLOR_BGRA2BGR)
            Image_3D = cv2.cvtColor(point_image, cv2.COLOR_BGRA2BGR)

            segmented_image = cv_demo1.test_rgb_to_gray(rgb_image,Image_3D)
            return color_image, point_image ,segmented_image

    def get_3D_data(self):
        if self.zed.grab() == sl.ERROR_CODE.SUCCESS:
            self.zed.retrieve_measure(self.point_cloud, sl.MEASURE.XYZRGBA,sl.MEM.CPU, self.res)
                
        return self.point_cloud


        