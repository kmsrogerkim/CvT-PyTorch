import os
import torch
from PIL import Image
from torch.utils.data import Dataset
import torch.nn.functional as F
from torchvision import transforms

class CustomDataset(Dataset):
    def __init__(self, root_dir, annotation_file, transform=None, num_classes=37):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.num_classes = num_classes

        with open(annotation_file, 'r') as f:
            for line in f:
                if line.startswith('#'):
                    continue
                parts = line.strip().split()
                if not parts:
                    continue
                self.image_paths.append(os.path.join(self.root_dir, 'images', parts[0] + '.jpg'))
                self.labels.append(int(parts[1]) - 1)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        # label = F.one_hot(torch.tensor(label, dtype=torch.long), num_classes=self.num_classes).to(torch.float32)
        label = torch.tensor(label, dtype=torch.long)

        return image, label
