# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import usbrelay_py as usbrl
import time 


class RelayControl:
    # methods
    def __init__(self):
        board_number = usbrl.board_count()
        board_detail = usbrl.board_details()
        self.count = board_number
        # print(f"Count: {self.count}")
        self.boards= board_detail
        # print(f"Boards: {self.boards}")
        
        
    def turnOnAll(self):
        "To turn on all relays"
        for board in self.boards:
            print("Board: ", board)
            relay = 1
            while(relay < board[1]+1):
                result = usbrl.board_control(board[0], relay, 1)
                print("Result: ", result)
                relay += 1
                
    def turnOffAll(self):
        "To turn off all relays"
        for board in self.boards:
            print("Board: ", board)
            relay = 1
            while(relay < board[1]+1):
                result = usbrl.board_control(board[0], relay, 0)
                print("Result: ", result)
                relay += 1
                
    def turnOn(self, relay_num=1):
        "To turn on specific relay"
        for board in self.boards:
            # print("Board: ", board)
            relay = relay_num
            result = usbrl.board_control(board[0], relay, 1)
            # print("Result: ", result)
            
    def turnOff(self, relay_num=1):
        "To turn on specific relay"
        for board in self.boards:
            # print("Board: ", board)
            relay = relay_num
            result = usbrl.board_control(board[0], relay, 0)
            # print("Result: ", result)
                
