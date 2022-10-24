import socket
import time
import sys
from tkinter import S
#import _thread
from threading import Thread
from matplotlib.pyplot import connect
from .Util.relayControl import RelayControl
from .Communication.TCP_IP.HRSS_server import TCPServer

class Controll():
    def __init__(self,TCP_Conection):
        self.TCP_Conection = TCP_Conection
        self.Relay = RelayControl()
        # self.t1 = Thread(target = self.TCP_Conection.readData)
        # self.t1.setDaemon(True)
        # self.t1.start()


    def sender(self,x,y,theta,object):
        # print("sending")
        self.TCP_Conection.sendDataToClient(x,y,theta,object)
        self.Relay.turnOn(1) 
        time.sleep(0.05) 
        self.Relay.turnOff(1)
        self.TCP_Conection.sendDataToClient(x,y,theta,object)
        time.sleep(0.05) 
        self.Relay.turnOn(1) 
        time.sleep(0.05) 
        self.Relay.turnOff(1)
    def sender_test(self, x,y,theta,object):
        # print("sending")
        self.TCP_Conection.sendDataToClient_test(x,y,theta,object)
        self.Relay.turnOn(1) 
        time.sleep(0.05) 
        self.Relay.turnOff(1)
        self.TCP_Conection.sendDataToClient_test(x,y,theta,object)
        time.sleep(0.05) 
        self.Relay.turnOn(1) 
        time.sleep(0.05) 
        self.Relay.turnOff(1)

