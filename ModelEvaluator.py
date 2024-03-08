import torch
from torchvision import transforms
from PIL import Image
print(torch.__version__)


class Evaluator():
    def __init__(self, image:Image) -> None:
        self.image =  image

    def __resize_img(self):
        pass

    def fit(self):
        pass