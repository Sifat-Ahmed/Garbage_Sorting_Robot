import cv2
import torch
import os
import logging
import warnings
warnings.filterwarnings('ignore')
logger = logging.getLogger("models.yolo") #.setLevel(logging.WARNING)
logger.propagate = False

class Detector:
    def __init__(self, cfg):
        self._cfg = cfg
        #try:
        self._model = torch.hub.load(self._cfg.root_dir,
                                 'custom',
                                 os.path.join(self._cfg.root_dir, r'runs/train/model/weights/best.pt'),
                                 source='local',
                                 verbose=False,
                                 )
        # except Exception as e:
        #     print('\033[101m' + 'Set the Detect/Config.py ~ self.root_path to project\'s absolute path' )


    def __unpack__bbox(self, data):
        x1 = int(data[0])
        y1 = int(data[1])
        x2 = int(data[2])
        y2 = int(data[3])
        confidence = data[4]

        return {
            'x1': x1,
            'y1': y1,
            'x2': x2,
            'y2': y2,
            'score': confidence
        }

    def get_bbox(self, image, return_box = False):
        prediction = self._model(image, size=640)
        df = prediction.pandas().xyxy[0].to_numpy()
        return df

#
# if __name__ == '__main__':
#     cfg = Config()
#     det= Detector(cfg)
#
#     image = cv2.imread('dataset/val/images/1114_25_rgb_conv.jpg')
#
#     print(det.get_bbox(image))
