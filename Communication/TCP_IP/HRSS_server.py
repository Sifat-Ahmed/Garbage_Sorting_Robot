import socket
from ssl import SOL_SOCKET
import time
from tkinter import S

class TCPServer():
    def __init__(self, host, port):
        self.HOST = host
        self.PORT = port
        self.s = None
        self.conn = None
        self.addr = None

    def connect(self):
        try:
            self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            # print(str(self.HOST) + " + " + str(self.PORT))
            
            ## SOL_SOCKET - protocol independent 
            self.s.setsockopt(SOL_SOCKET, socket.SO_KEEPALIVE, 1)
            self.s.settimeout(None)
            self.s.bind((self.HOST, self.PORT))
            self.s.settimeout(None)
            self.s.listen(5)
            print("Listening for client . . .")
            self.conn, self.addr = self.s.accept()
            print("Connected to client at: {}".format(self.addr))                    
            ##
            # self.conn.setsockopt(SOL_SOCKET, socket.SO_KEEPALIVE, 1)
            
            # self.readData()
        except KeyboardInterrupt:
            print("True")
            return True
        finally:
            self.diconnectSocket()

    def diconnectSocket(self):
        self.s.close()
        #self.conn.shutdown(socket.SHUT_RDWR)
        # self.conn.close()
    
    
    def sendDataToClient(self,x,y,theta,object):
        # print("{" + "1,2,NA,1,{},{},{}".format(int(x),int(y),int(theta)) + "}")
        if int(y) >= 371:
            print("over reach")
            return 
        # try:
        xyz_cord = "{" + "1,2,NA,{},{},{},{}".format(object,int(x),int(y),int(theta)) + "}"
        self.conn.sendall(xyz_cord.encode("utf8"))
        # print(xyz_cord)
        # except:

    def sendDataToClient_test(self,x,y,theta,object):
        xyz_cord = "{" + "1,2,NA,{},{},{},{}".format(object,int(x),int(y),int(theta)) + "}"
        self.conn.sendall(xyz_cord.encode("utf8"))
        # print(xyz_cord)

    # def readData(self):
    #     while True:
    #         try:
                
    #             data = self.conn.recv(1024)
    #             if data:
    #                 xyz_cord = "True"
    #                 self.conn.sendall(xyz_cord.encode("utf8"))
    #                 print("Acknowledged ")
    #         except:
    #             self.connect()
