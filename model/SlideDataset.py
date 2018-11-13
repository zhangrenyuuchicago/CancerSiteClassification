import torch
import torch.utils.data as data
import numpy as np
from PIL import Image
import glob
import torchvision

class SlideDataset(data.Dataset):
    """COCO Custom Dataset compatible with torch.utils.data.DataLoader."""
    def __init__(self, dir0, dir1, transform):
        """Set the path for images, captions and vocabulary wrapper.
        
        Args:
            root: image directory.
            transform: image transformer.
        """
        self.ids0 = glob.glob(dir0 + "/*.png")
        self.ids1 = glob.glob(dir1 + "/*.png")
        print( "ids0 image number: " + str(len(self.ids0)))
        print( "ids1 image number: " + str(len(self.ids1)))
        
        self.transform = transform
        print( "Initialize end")

    def __getitem__(self, index):
        """Returns one data pair (image and caption)."""
        if index < len(self.ids0):
            img_name = self.ids0[index]
            tumer_site = torch.LongTensor([0])
        else:
            img_name = self.ids1[index - len(self.ids0)]
            tumer_site = torch.LongTensor([1])
        image = Image.open(img_name).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image, tumer_site 
                
    def __len__(self):
        return len(self.ids0) + len(self.ids1)


