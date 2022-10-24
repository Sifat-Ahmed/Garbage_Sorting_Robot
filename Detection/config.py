import os
import torch
from torchvision.transforms import transforms

class Config:

    def __init__(self) -> None:

        ## This is the path to the folder of the project

        ## this project is called 'smoke_deploy'
        ## so you put the absolute path of this project folder
        self.root_dir = '/home/eiismoke/scara_robot/src/Detection/AI/Yolo/'
        self.classification_threshold = 0.9
        self.image_size = (480, 640)
        self.neighbour_length_to_x_pixel = 25
        self.neighbour_length_to_y_pixel = 25

        self.rock_area = 100

