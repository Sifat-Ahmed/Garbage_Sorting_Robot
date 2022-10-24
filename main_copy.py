# FOR GUI
from threading import Thread
import PySimpleGUI as sg
import time

#For main

import numpy
from Control.controll import Controll
from Control.Communication.TCP_IP.HRSS_server import TCPServer
from Detection.detection import Detection
from Decision.decision import decision
import cv2
import random
import pyzed.sl as sl
import math
from threading import Thread
import statistics
import PySimpleGUI as sg
import time
# from config import Config

def convert(theta):
    old_min = 1 
    old_max = 90 
    new_min = 130
    new_max = 40 
    new_value = ( (old_value - old_min) / (old_max - old_min) ) * (new_max - new_min) + new_min
    return new_value

def main(args=None):
    which_ = input("CAM = 1 , VID = 2, Dummy = 3, Which?:")
    TCP_Conection = TCPServer('192.168.1.3',3002)
    TCP_Conection.connect()
    detection_ = Detection()
    decision_ = decision()
    control = Controll(TCP_Conection)
    control = Controll(TCP_Conection)
    # t1 = Thread(target = TCP_Conection.readData)
    # t1.setDaemon(True)
    # t1.start()
    theta_ = [100,120,130,140,150]

    while True:
        try:
            start = time.time()
            if(which_ == "3"):
                #intput_x     =  input(" Select x: ")
                #intput_y     =  input(" Select y: ")
                #intput_theta =  input(" Select theta: ")
                intput_x     = 0
                intput_y     = 0
                intput_theta = input(" Select theta: ")
                object       = 1
                

                control.sender_test(intput_x,intput_y,intput_theta,object)  
            
            elif(which_ == "1"):
                start_def = start - time.time()
                image_result = detection_.get_zed_bbox()
                AI = start - time.time()
                pick_result = decision_.getBestItems(image_result[0])
                pick_result_time = start - time.time()
                
                object = 1
                xofset = 10
                timing = 0
                if(pick_result[0]):
                    for item in pick_result[0]:
                        z = abs(pick_result[1][timing])
                        if ()
                        if(z >= 650 and z <= 740):
                            object = 1
                            item[0] = item[0] + 30
                        elif(z >= 610 and z <= 650):
                            object = 3
                            # item[0] = item[0] - 12
                        elif(z >= 560 and z <= 610):
                            object = 4
                            # item[0] = item[0] - 14
                        elif(z >= 520 and z <= 560):
                            object = 5
                            # item[0] = item[0] - 16
                        elif(z >= 470 and z <= 520):
                            object = 6
                            # item[0] = item[0] + 40
                        elif(z >= 440 and z <= 470):
                            object = 7
                            # item[0] = item[0] + 20
                        elif(z >= 200 and z <= 440):
                            object = 8
                            #item[0] = item[0] - 30
                        #Item 0 = x, item 1 = y item 4 = theta
                        #Default theta value is 130
                        # control.sender(item[0],item[1] - 5,13a0,object)  
                        control.sender(item[0],item[1] - 5,item[4],object)  
                        timing =+ timing + 1
                    pause(0.4)
                # else:
                #     pause(0.4)
                
                total_time = start - time.time()

                # print("start time:{},AI time:{},pick_time:{},total:{}".format(start_def,AI,pick_result_time,total_time))
                xx = 70
                cv2.line(image_result[1], pt1=(490,xx), pt2=(560,xx), color=(0,0,255), thickness=10)
                xxx = 330
                cv2.line(image_result[1], pt1=(490,xxx), pt2=(560,xxx), color=(0,0,255), thickness=10)

                # cv2.namedWindow('Zed')
                cv2.imshow('Zed', image_result[1])
                # cv2.imshow('Zed3', image_result[1])
                cv2.waitKey(10)
                
        except KeyboardInterrupt:
            #t1.join()
            print("Ending Program")
        pass


if __name__ == '__main__':
    main()

