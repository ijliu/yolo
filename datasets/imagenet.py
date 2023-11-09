import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as  transforms
import torchvision

class Imagenet(Dataset):

    def __init__(self, root, train=True, transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform

        self.train = train
        if self.train:
            self.root = os.path.join(root, "train")
        else:
            self.root = os.path.join(root, "val")
        self.data = []

        self.classes = [d.name for d in os.scandir(self.root) if d.is_dir()]
        self.classes.sort()
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        self.loadDatasetFromFolder(self.root)

    def __getitem__(self, index):

        img, target = self.data[index]
        img = self.loader(img)

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target
    
    def __len__(self):
        return len(self.data)
    
    def loadDatasetFromFolder(self, directory):
        for target_class in sorted(self.class_to_idx.keys()):
            class_index = self.class_to_idx[target_class]
            target_dir = os.path.join(directory, target_class)
            for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
                for fname in sorted(fnames):
                    path = os.path.join(root, fname)
                    self.data.append((path, class_index))

    def loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            img.load()
        return img.convert('RGB')

    def __repr__(self):
        repr = {
            "name" : self.__class__.__name__,
            "Length" : self.__len__(),
        }
        return str(repr)

# imgnet = Imagenet("/data/open_dataset/ILSVRC2012", train=True, transform=)
# print(imgnet)
# print(len(imgnet))
# for data in imgnet:
#     print(data[0].shape)
#     exit()
