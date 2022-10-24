from __future__ import annotations
import numpy as np

class Point:

    ## We will send the X,Y of the object and create points
    ## From this points we will calculate all the properties of the object and
    ## then do operations on it
    def __init__(self, x, y):
        self.x = self.__convert_to_int__(x)
        self.y = self.__convert_to_int__(y)
        self.point = (self.x, self.y)

    def __str__(self):
        return f"{(self.x, self.y)}"

    def __convert_to_int__(self, value):
        return int(value)



class PointProperties:

    def __init__(self, point1: Point, point2 : Point):
        self.point1 = point1
        self.point2 = point2

        assert self.point1.x < self.point2.x, 'Start point coordinate is greater than end point'


    def __point_to_point_distance__(self):
        ## This function takes two points
        self.width = abs(self.point1.x - self.point2.x)
        self.height = abs(self.point1.y - self.point2.y)

        #self.center =

        self.distance = np.sqrt((self.width)**2 + (self.height)**2)



    def get_properties(self):
        self.__point_to_point_distance__()

        return {
            "start" : self.point1,
            "end"   : self.point2,
            "width" : self.width,
            "height": self.height,
            "center": Point(
                self.point1.x + (self.width/2),
                self.point1.y + (self.height/2)),
            "distance": self.distance,
        }

if __name__ == '__main__':
    p1 = Point(100.1, 100.2)
    p2 = Point(200.3, 200.4)

    print(PointProperties(p1, p2).get_properties())