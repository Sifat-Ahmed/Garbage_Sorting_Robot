from Detection.AI.Yolo.results.point import Point, PointProperties


class Rock:
    def __init__(self):
        pass

    def unpack_coordinates(self, array_from_yolo, image_segment = None):
        start = Point(array_from_yolo[0], array_from_yolo[1])
        end = Point(array_from_yolo[2], array_from_yolo[3])
        properties = PointProperties(start, end).get_properties()

        return properties

    def print_values(self, array_from_yolo):
        properties = self.unpack_coordinates(array_from_yolo)
        print("Start ({}, {})".format(properties["start"].x, properties["start"].y))
        print("End ({}, {})".format(properties["end"].x, properties["end"].y))
        print("Width: {}; Height: {}; Center: ({}, {})".format(properties["width"], properties["height"], properties["center"].x, properties["center"].y))
        print("Corner to Corner Distance: {} ".format(properties["distance"]))


if __name__ == '__main__':
    r = Rock()
    properties = r.unpack_coordinates(
        [449.15167236328125, 156.33737182617188, 507.04803466796875, 233.53634643554688, 0.6595696210861206, 0, 'ROCK'])

    print(properties["start"].x)
    print(properties["end"].x)