from PIL import Image
from datasets import load_from_disk
from torch.utils.data import Dataset
import numpy as np

class Design(Dataset):
    def __init__(self, data_dir, size, downscale_f=2.4, interpolation="bicubic"):
        self.interpolation = {"linear": Image.LINEAR,
                              "bilinear": Image.BILINEAR,
                              "bicubic": Image.BICUBIC,
                              "lanczos": Image.LANCZOS,
                              }[interpolation]
        self.size = size
        self.lr_size = int(size / downscale_f)
        self.data = load_from_disk(data_dir)
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        example = self.data[i]
        pil_img = example['image']
        h, w = pil_img.size
        pad2len = max(h, w)
        top, left = (pad2len-h)>>1, (pad2len-w)>>1
        image = Image.new(pil_img.mode, (pad2len, pad2len), (0, 0, 0))
        image.paste(pil_img, (left, top))
        
        full_image = image.resize((self.size, self.size), resample=self.interpolation)
        image = image.resize((self.lr_size, self.lr_size), resample=self.interpolation)

        image = np.array(image).astype(np.uint8)
        example['LR_image'] = (image / 127.5 - 1.0).astype(np.float32)
        full_image = np.array(full_image).astype(np.uint8)
        example['image'] = (full_image / 127.5 - 1.0).astype(np.float32)

        return example

class SuperresDesignDemoTrain(Design):
    def __init__(self, **kwargs):
        super().__init__(data_dir="data/design_demo_38", **kwargs)

class SuperresDesignDemoValidation(Design):
    def __init__(self, **kwargs):
        super().__init__(data_dir="data/design_demo_38", **kwargs)