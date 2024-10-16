import torch
import random
from torchvision.transforms import functional as F
from torchvision.transforms import transforms
from PIL import ImageFilter

class RandomResizedCropWithParams(transforms.RandomResizedCrop):
    def __call__(self, img):
        # Get parameters of the crop
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        # Record the parameters
        params = {'crop': (i, j, h, w)}
        # Apply the crop
        img = F.resized_crop(img, i, j, h, w, self.size, self.interpolation)
        return img, params

class RandomHorizontalFlipWithParams(transforms.RandomHorizontalFlip):
    def __call__(self, img):
        if torch.rand(1) < self.p:
            img = F.hflip(img)
            flipped = True
        else:
            flipped = False
        params = {'flipped': flipped}
        return img, params

class ColorJitterWithParams(transforms.ColorJitter):
    def __call__(self, img):
        # Get the transformation parameters
        fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor = self.get_params(
            self.brightness, self.contrast, self.saturation, self.hue)
        
        params = {
            'brightness_factor': brightness_factor if brightness_factor is not None else 0.0,
            'contrast_factor': contrast_factor if contrast_factor is not None else 0.0,
            'saturation_factor': saturation_factor if saturation_factor is not None else 0.0,
            'hue_factor': hue_factor if hue_factor is not None else 0.0
        }

        # Apply the transformations
        for fn_id in fn_idx:
            if fn_id == 0 and brightness_factor is not None:
                img = F.adjust_brightness(img, brightness_factor)
            elif fn_id == 1 and contrast_factor is not None:
                img = F.adjust_contrast(img, contrast_factor)
            elif fn_id == 2 and saturation_factor is not None:
                img = F.adjust_saturation(img, saturation_factor)
            elif fn_id == 3 and hue_factor is not None:
                img = F.adjust_hue(img, hue_factor)
        
        return img, params


class RandomApplyWithParams(transforms.RandomApply):
    def __call__(self, img):
        if self.p < torch.rand(1):
            # The transform is not applied
            params = {
                'color_jitter_applied': False,
                'brightness_factor': 0.0,  # Default value
                'contrast_factor': 0.0,    # Default value
                'saturation_factor': 0.0,  # Default value
                'hue_factor': 0.0          # Default value
            }
            return img, params
        else:
            for t in self.transforms:
                img, params = t(img)
                params['color_jitter_applied'] = True
            return img, params


class RandomGrayscaleWithParams(transforms.RandomGrayscale):
    def __call__(self, img):
        num_output_channels = 1 if img.mode == 'L' else 3
        if torch.rand(1) < self.p:
            img = F.to_grayscale(img, num_output_channels=num_output_channels)
            grayscale = True
        else:
            grayscale = False
        params = {'grayscale': grayscale}
        return img, params

class GaussianBlurWithParams:
    def __init__(self, kernel_size, sigma=(0.1, 2.0)):
        self.kernel_size = kernel_size
        self.sigma = sigma

    def __call__(self, img):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        img = img.filter(ImageFilter.GaussianBlur(radius=sigma))
        params = {'sigma': sigma}
        return img, params
