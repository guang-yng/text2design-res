from PIL import Image
from datasets import load_from_disk
from torch.utils.data import Dataset
import numpy as np
import random
from ldm.modules.image_degradation.bsrgan_light import bicubic_degradation
from PIL import ImageFile

class Design(Dataset):
    def __init__(self, data_dir, size, downscale_f=4, degradation="bsrgan_light", min_crop_f=0.5, max_crop_f=1.0, random_crop=True):
        self.degradation = {"bsrgan_light": bicubic_degradation,
                              "bilinear": Image.BILINEAR,
                              "bicubic": Image.BICUBIC,
                              "lanczos": Image.LANCZOS,
                              }[degradation]
        self.df = downscale_f
        self.size = size
        self.lr_size = int(size / downscale_f)
        self.data = load_from_disk(data_dir)
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        example = self.data[i]
        original_img = example['image']
        if original_img.mode != 'RGB':
            original_img = original_img.convert('RGB')

        h, w = original_img.size
        assert h >= self.size and w >= self.size

        upper = random.randint(0, h-self.size)
        left = random.randint(0, w-self.size)
        img = original_img.crop((left, upper, left+self.size, upper+self.size))

        img = np.array(img).astype(np.uint8)
        img = (img/127.5 - 1.0).astype(np.float32)
        lr_img = self.degradation(img, self.df)

        example['image'] = img
        example['LR_image'] = lr_img

        return example

class SuperresDesignDemoTrain(Design):
    def __init__(self, **kwargs):
        super().__init__(data_dir="data/design_demo_38", **kwargs)

class SuperresDesignDemoValidation(Design):
    def __init__(self, **kwargs):
        super().__init__(data_dir="data/design_demo_38", **kwargs)

class SuperresDesignPresentationTrain(Design):
    def __init__(self, **kwargs):
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        super().__init__(data_dir="data/design_all_sampled_1000_filtered", **kwargs)

    def __len__(self):
        return 800

    def __getitem__(self, i):
        if i > 800:
            raise ValueError
        return super().__getitem__(i)

class SuperresDesignPresentationValidation(Design):
    def __init__(self, **kwargs):
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        super().__init__(data_dir="data/design_all_sampled_1000_filtered", **kwargs)

    def __len__(self):
        return len(self.data) - 800

    def __getitem__(self, i):
        if i > len(self.data) - 800:
            raise ValueError
        return super().__getitem__(i+800)

class SuperresDesignAll1122Train(Design):
    def __init__(self, **kwargs):
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        super().__init__(data_dir="data/design_all_1122_pad_material", **kwargs)

    def __len__(self):
        return 19136

    def __getitem__(self, i):
        if i > 19136:
            raise ValueError
        return super().__getitem__(i)

class SuperresDesignAll1122Validation(Design):
    def __init__(self, **kwargs):
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        super().__init__(data_dir="data/design_all_1122_pad_material", **kwargs)

    def __len__(self):
        return len(self.data) - 19136

    def __getitem__(self, i):
        if i > len(self.data) - 19136:
            raise ValueError
        return super().__getitem__(i+19136)