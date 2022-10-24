#!usr/bin/env python
# -*- coding: utf-8 -*-

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

mode = False
which_ = ""
init = False
close = False
keyboardinterrupt = False

def GUI():
    global close
    global mode
    global which_
    global init
    global keyboardinterrupt

    sg.theme('Dark')
    sg.set_options(element_padding=(0, 0))

    options = {'size':(60, 2), 'background_color':'blue', 'justification':'c',}
    layout = [[sg.Text('EII', **options, pad=(4, 4),font=('Any 16')) ],
            [sg.Button('Start', button_color=('white', 'green'), key='-Start-')],
            [sg.Button('Stop', button_color=('white', 'red'), key='-Stop-')]]
    # Create the window
    window = sg.Window("EII", layout,
                default_element_size=(150, 20),
            text_justification='r',
            auto_size_text=False,
            auto_size_buttons=False,
            default_button_element_size=(24, 12),
            finalize=True)

    while True:
        event, values = window.read()
    
        if event == sg.WIN_CLOSED:
            break
        elif event == '-Start-' and mode == False :
            which_ = "1"
            mode = True
            print("start cam")

        elif event == '-Stop-' or keyboardinterrupt == True:
            print("Stop")
            break

    close = True

    window.close()
def convert(theta):
    old_min = 1 
    old_max = 90 
    new_min = 130
    new_max = 80 
    new_value = ( (theta - old_min) / (old_max - old_min) ) * (new_max - new_min) + new_min
    return new_value
def pause(seconds):
    start = time.time()
    while time.time() < start + seconds:
        pass
def main(args=None):
    global close
    global mode
    global which_
    global init
    global keyboardinterrupt
    t2 = Thread(target = GUI)
    t2.setDaemon(True)
    t2.start()
    time.sleep(0.5)

    while close == False:
        try:
            start = time.time()

            
            if mode == True and init == False:
                # time.sleep
                TCP_Conection = TCPServer('192.168.1.3',3002)
                detection_ = Detection()
                decision_ = decision()
                control = Controll(TCP_Conection)
                keyboard_interupt = TCP_Conection.connect()
                if (keyboard_interupt):
                    close = True
                #   print("Program Ready")
                init = True
            elif(which_ == "1" and init == True):
                start_def = start - time.time()
                image_result = detection_.get_zed_bbox()
                AI = start - time.time()
                pick_result = decision_.getBestItems(image_result[0],image_result[1])
                pick_result_time = start - time.time()
                
                object = 1
                xofset = 10
                timing = 0
                if(pick_result[0]):
                    for item in pick_result[0]:
                        z = abs(pick_result[1][timing])

                        theta = convert(item[4])

                        if(z >= 650 and z <= 740):
                            object = 1
                            # item[0] = item[0] - 30
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
                        print(z)
                        # print("x: {}, y: {}, theta: {}".format(-50,item[1],theta))
                        # control.sender(1,item[1] - 5,theta,object)  
                        control.sender(1,item[1] - 5,130,object)  
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
            
            keyboardinterrupt = True
            TCP_Conection.diconnectSocket()

            break
    

if __name__ == '__main__':
    main()

