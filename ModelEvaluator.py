import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from LeNet import LeNet_TryOut

class Evaluator():
    def __init__(self, image:Image) -> None:
        self.image = image
        self.preprocessed_image = self.__img_preprocess()
        self.image_tensor = self.__tensor_conversion()
        self.tensor_image_view = self.__image_conversion()


    def __img_preprocess(self):
        # Set background color (change it as per your need)
        background_color = (255, 255, 255)  # White background
        
        # Create a new RGBA image with the desired background color
        new_image = Image.new("RGB", self.image.size, background_color)
        
        # Overlay the original image on top of the new background
        new_image.paste(self.image, (0, 0), self.image)
        
        return new_image

    def __image_conversion(self):
        transform = transforms.Compose([
            transforms.ToPILImage()
        ])
        image = transform(self.image_tensor)
        return image

    def __tensor_conversion(self):
        transform = transforms.Compose([
            transforms.Resize((28,28)),
            transforms.RandomInvert(1),
            transforms.Grayscale(),
            transforms.ToTensor()]) 

        image_tensor = transform(self.preprocessed_image)
        return image_tensor

    def predict(self):
        model = LeNet_TryOut()
        model.load_state_dict(torch.load("./models/LeNet_Tryout_AVGPooling.pt"))
        model.eval()

        with torch.inference_mode():
            output = model(self.image_tensor)
        
        probabilities = torch.softmax(output, dim=1)
        _, predicted_class = torch.max(output, 1)

        return predicted_class.item(), probabilities.squeeze().tolist()