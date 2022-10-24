import time
import sys
from tkinter import S
from threading import Thread

import numpy
from Control.controll import Controll
from Control.Communication.TCP_IP.HRSS_server import TCPServer
from Detection.detection import Detection
from Decision.decision import decision
import cv2
import random
import pyzed.sl as sl
import math


def main(args=None):
    which_ = input("CAM = 1 , VID = 2, Dummy = 3, Which?:")
    TCP_Conection = TCPServer('192.168.1.8',3002)
    TCP_Conection.connect()
    detection_ = Detection()
    decision_ = decision()
    control = Controll(TCP_Conection)
    control = Controll(TCP_Conection)
    t1 = Thread(target = TCP_Conection.readData)
    t1.setDaemon(True)
    t1.start()
    theta_ = [100,120,130,140,150]

    while True:
        try:
            start = time.time()
            if(which_ == "3"):
                intput_x     =  input(" Select x: ")
                intput_y     =  input(" Select y: ")
                intput_theta =  input(" Select theta: ")
                object       =  input(" item #: ")
                control.sender_test(intput_x,intput_y,intput_theta,object)  
            elif(which_ == "2"):
                
                cap = cv2.VideoCapture('realsense6.mp4')
                while(cap.isOpened()):
                    ret, frame = cap.read()
                    color_frame = frame

                    image_result = detection_.get_bbox_video(color_frame)
                    pick_result = decision_.getBestItems(image_result[0][1])
                    if(pick_result):
                        for item in pick_result:
                            print(item[0],item[1],130)
                            control.sender(-20.0,item[1],130)  
                    else:
                        print("Item not found")
                    cv2.namedWindow('RealSense')
                    cv2.imshow('RealSense', image_result[0])
                    cv2.waitKey(3)
                    stop = time.time()
                    #time.sleep(4)
                    print(start - stop)

            elif(which_ == "1"):

                image_result = detection_.get_zed_bbox()
                # print(type(image_result))
                # image_result = detection_.get_realsense_bbox()
                pick_result = decision_.getBestItems(image_result[0][1])
                point_cloud = image_result[1]
                object = 1
                xofset = 10
                if(pick_result):
                    for item in pick_result:
                        #print(300,item[1],130)
                        # control.sender(20,20,130)  
                        #random_number = random.randint(1, 4)
                        #control.sender(1,item[1],theta_[random_number]) 
                        stop = time.time() 
                        z = point_cloud.get_value(item[3],item[2])
                        run_time = abs(start - stop)
                        item[0] = item[0] + (xofset * run_time   - 7.5)
                        # if(run_time >= 0.2 and run_time <= 0.3):
                        #     item[0] = item[0] + 10
                        # elif(run_time >= 0.3 and run_time <= 0.4):
                        #     item[0] = item[0] + 20
                        # elif(run_time >= 0.5 ):
                        #     item[0] = item[0] + 30
                        # elif(z[1][2] >= -590 and z[1][2] <= -640):
                        #     object = 2
                        # elif(z[1][2] >= -560 and z[1][2] <= -640):
                        #     object = 3
                        # elif(z[1][2] >= -500 and z[1][2] <= -640):
                        #     object = 4
                        # elif(z[1][2] >= -450 and z[1][2] <= -640):
                        #     object = 5
                        print("Z: ",z[1][2])
                        control.sender(item[0],item[1],130,object)  
                        # time.sleep(1)
                # else:
                #     print("Item not found")

                xx = 70
                cv2.line(image_result[0][0], pt1=(490,xx), pt2=(560,xx), color=(0,0,255), thickness=10)
                xxx = 330
                cv2.line(image_result[0][0], pt1=(490,xxx), pt2=(560,xxx), color=(0,0,255), thickness=10)
                # cv2.namedWindow('RealSense')
                # cv2.imshow('RealSense', image_result[0])
                # cv2.waitKey(10)
                cv2.namedWindow('Zed')
                cv2.imshow('Zed', image_result[0][0])
                cv2.waitKey(10)
                stop = time.time()
                print()
                print(start - stop)
                
        except KeyboardInterrupt:
            t1.join()
        pass


if __name__ == '__main__':
    main()


# from Control.Util.relayControl import RelayControl
# import socket
# import time
# import sys

# HOST = '192.168.1.8' # This is the IP of our PC and will need to be set on caterpillars vision setting
# PORT = 3002          # The port will be set on caterpillars vision setting
# s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# s.bind((HOST, PORT))
# s.listen(1)
# conn, addr = s.accept()
# print('Connected by', addr)
# relay = RelayControl()
# while True:
#     data= conn.recv(1024)
#     print(data)
#     if(data != b''):
#         input("Press Enter to continue...")
#         success = conn.sendall("{1,1,NA,1,1,1,130}".encode("utf8")) # The strong will need to be sent as a byte
#         print(success)
#         relay.turnOn(1)  # will turn channel 1 of the relay on
#         time.sleep(0.5)
#         relay.turnOff(1) # will turn channel 1 of the relay off
# conn.close()