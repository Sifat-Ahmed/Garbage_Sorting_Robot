from PyQt5 import QtCore, QtGui, QtWidgets 
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import Qt, pyqtSlot, QVariant, QUrl
from PyQt5.QtCore import QTimer,QDateTime
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QThread
import time
import sys
import redis

class ChangeCheckBox(QThread):
    def __init__(self, rs, variable, checkbox, led_width, led_height):
        super().__init__()
        self.rs = rs
        self.variable = variable
        self.checkbox = checkbox
        self.width = led_width
        self.height = led_height
        self.ret = False

    def run(self):
        while True:
            try:
                if self.rs.get(self.variable) == b'True':
                    if not self.ret:
                        self.checkbox.setStyleSheet("QCheckBox::indicator"
                                               "{"
                                               "background-color : lightgreen;"
                                               "width: " + str(self.width) + "; height: " + str(self.height) + ";"
                                               "border-radius :12px;"
                                               "}")
                        self.ret = True
                else:
                    if self.ret:
                        self.checkbox.setStyleSheet("QCheckBox::indicator"
                                               "{"
                                               "background-color : lightgrey;"
                                               "width: " + str(self.width) + "; height: " + str(self.height) + ";"
                                               "border-radius :12px;"
                                               "}")
                        self.ret = False 
            except Exception as e:
                print('Redis connection failed')
                raise 'Redis connection failed'
            time.sleep(1)

class MainWindow(QWidget):
    def __init__(self, parent):
        super(QWidget, self).__init__(parent)
        try:
            self.rs = redis.Redis(host='localhost', port=6379, db=0)
        except Exception as e:
            print('Redis connection failed')
            raise 'Redis connection failed'
        self.ui()

    def ui(self):
        font = self.font()
        font.setPointSize(18)
        QApplication.instance().setFont(font)
        layout = QHBoxLayout()
        botton_layout = QVBoxLayout()
        led_layout = QVBoxLayout()
        self.led_width = int(0.02 * height)
        self.led_height = int(0.02 * height)
        self.camera = QCheckBox('Camera OK', self)
        self.camera.setStyleSheet("QCheckBox::indicator"
                               "{"
                               "background-color : lightgray;"
                               "width: " + str(self.led_width) + "; height: " + str(self.led_height) + ";"
                               "border-radius :12px;"
                               "}")
        self.robot = QCheckBox('Robot OK', self)
        self.robot.setStyleSheet("QCheckBox::indicator"
                               "{"
                               "background-color : lightgray;"
                               "width: " + str(self.led_width) + "; height: " + str(self.led_height) + ";"
                               "border-radius :12px;"
                               "}")
        self.ai = QCheckBox('AI OK', self)
        self.ai.setStyleSheet("QCheckBox::indicator"
                               "{"
                               "background-color : lightgray;"
                               "width: " + str(self.led_width) + "; height: " + str(self.led_height) + ";"
                               "border-radius :12px;"
                               "}")
        self.start_button = QPushButton("Start")
        self.start_button.setStyleSheet("QPushButton{color:black}"
                              "QPushButton{background-color:lightgray}"
                              "QPushButton:hover{background-color:lightblue}")
        self.stop_button = QPushButton("Stop")
        self.stop_button.setStyleSheet("QPushButton{color:black}"
                              "QPushButton{background-color:darkred}"
                              "QPushButton:hover{background-color:lightblue}")
        self.start_button.setFixedHeight(int(0.075 * height))
        self.start_button.setFixedWidth(int(0.1 * height))
        self.stop_button.setFixedHeight(int(0.075 * height))
        self.stop_button.setFixedWidth(int(0.1 * height))
        botton_layout.addWidget(self.start_button)
        botton_layout.addWidget(self.stop_button)
        layout.addLayout(botton_layout)
        layout.setAlignment(Qt.AlignCenter)
        layout.setSpacing(int(0.1 * height))
        layout.addLayout(led_layout)
        led_layout.addWidget(self.camera)
        led_layout.addWidget(self.robot)
        led_layout.addWidget(self.ai)
        self.setLayout(layout)
        self.start_button.clicked.connect(self.start_button_click)
        self.stop_button.clicked.connect(self.stop_button_click)
        self.change_cam = ChangeCheckBox(self.rs, 'camera', self.camera, self.led_width, self.led_height)
        self.change_bot = ChangeCheckBox(self.rs, 'robot', self.robot, self.led_width, self.led_height)
        self.change_ai = ChangeCheckBox(self.rs, 'ai', self.ai, self.led_width, self.led_height)
        self.change_botton = QTimer()
        self.change_botton.timeout.connect(self.change_button)
        self.started = False
        self.all_ok = False
        self.change_cam.start()
        self.change_bot.start()
        self.change_ai.start()
        self.change_botton.start(1)

    def change_button(self):
        if self.change_cam.ret and self.change_bot.ret and self.change_ai.ret:
            if not self.all_ok:
                self.start_button.setStyleSheet("QPushButton{color:black}"
                                  "QPushButton{background-color:lightgreen}"
                                  "QPushButton:hover{background-color:lightgreen}")
                self.all_ok = True
        else:
            if self.all_ok:
                self.start_button.setStyleSheet("QPushButton{color:black}"
                                  "QPushButton{background-color:lightgray}"
                                  "QPushButton:hover{background-color:lightgreen}")
                self.all_ok = False


    def start_button_click(self):
        if not self.started:
            self.rs.set('start', 'True')
            self.started = True

    def stop_button_click(self):
        if self.started:
            self.rs.set('start', 'False')
            self.started = False

class App(QMainWindow):
    def __init__(self):
        super().__init__()
        self.title = 'robot'
        self.setWindowTitle(self.title)
        self.mainWidget = MainWindow(self)
        self.mainWidget.setMinimumHeight(int(0.5 * height))
        self.mainWidget.setMinimumWidth(int(0.3 * width))
        self.setCentralWidget(self.mainWidget)
        self.show()
           
    def closeEvent(self, event):
        return

def gui():
    app = QApplication(sys.argv)
    screen_resolution = app.desktop().screenGeometry()
    global width, height
    width, height = screen_resolution.width(), screen_resolution.height()
    ex = App()
    sys.exit(app.exec_())

if __name__ == '__main__':
    gui()

