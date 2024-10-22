from torchvision.transforms import transforms
from data_aug.gaussian_blur import GaussianBlur
from data_aug.custom_transforms import (
    RandomResizedCropWithParams, 
    ColorJitterWithParams, 
    RandomHorizontalFlipWithParams,
    RandomApplyWithParams, 
    RandomGrayscaleWithParams, 
    GaussianBlurWithParams
)
from torchvision import transforms, datasets
from data_aug.view_generator import ContrastiveLearningViewGenerator
from exceptions.exceptions import InvalidDatasetSelection
from torchvision.transforms import functional as F
import torch


class ToTensorWithParams:
    def __call__(self, img):
        img = F.to_tensor(img)
        params = {}
        return img, params

class CustomTransformPipeline:
    def __init__(self, size, s=1):
        self.transforms = [
            RandomResizedCropWithParams(size=size),
            RandomHorizontalFlipWithParams(),
            RandomApplyWithParams([ColorJitterWithParams(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)], p=0.8),
            RandomGrayscaleWithParams(p=0.2),
            GaussianBlurWithParams(kernel_size=int(0.1 * size)),
            ToTensorWithParams()
        ]

    def __call__(self, img):
        params_list = []
        for t in self.transforms:
            img, params = t(img)
            params_list.append(params)
        # Combine all parameters into a single dictionary
        transformation_params = {k: v for d in params_list for k, v in d.items()}
        #print(transformation_params)
        return img, transformation_params

class ContrastiveLearningViewGeneratorWithParams:
    """Generate multiple views of the same image with transformation parameters."""
    def __init__(self, base_transform, n_views=2):
        self.base_transform = base_transform
        self.n_views = n_views

    def __call__(self, x):
        imgs = []
        params_list = []
        for _ in range(self.n_views):
            img, params = self.base_transform(x)
            imgs.append(img)
            params_list.append(params)
        return imgs, params_list

class ContrastiveLearningDatasetWithParams:
    def __init__(self, root_folder):
        self.root_folder = root_folder

    def get_simclr_pipeline_transform(self, size, s=1):
        return CustomTransformPipeline(size=size, s=s)

    def get_dataset(self, name, n_views):
        valid_datasets = {
            'cifar10': lambda: datasets.CIFAR10(
                self.root_folder,
                train=True,
                transform=ContrastiveLearningViewGeneratorWithParams(
                    self.get_simclr_pipeline_transform(224),
                    n_views
                ),
                download=True
            ),

            'stl10': lambda: datasets.STL10(
                self.root_folder,
                split='unlabeled',
                transform=ContrastiveLearningViewGeneratorWithParams(
                    self.get_simclr_pipeline_transform(96),
                    n_views
                ),
                download=True
            )
        }

        try:
            dataset_fn = valid_datasets[name]
        except KeyError:
            raise InvalidDatasetSelection()
        else:
            return dataset_fn()

def params_collate_fn(batch):
    images_list = []
    params_list = []
    for item in batch:
        imgs_params, _ = item
        imgs, params = imgs_params
        images_list.append(imgs)    # imgs is a list of images
        params_list.append(params)  # params is a list of dictionaries

    # Transpose images_list and params_list to group by views
    images = list(zip(*images_list))   # Now images is a list of views, each containing batch_size images
    params = list(zip(*params_list))   # Now params is a list of views, each containing batch_size params

    # Stack images for each view
    images = [torch.stack(imgs, dim=0) for imgs in images]  # List of tensors with shape [batch_size, C, H, W]

    # For params, collate the dictionaries
    params_collated = []
    for view_params in params:
        # view_params is a tuple of dictionaries for the current view
        # We can collate them into a dictionary of lists
        collated_params = {}
        keys = view_params[0].keys()
        for key in keys:
            collated_params[key] = [d[key] for d in view_params]
        params_collated.append(collated_params)

    return images, params_collated

def transformation_params_to_tensor_batch(params_dict):
    """
    Converts a dictionary of transformation parameters into a tensor of shape [batch_size, 12].

    Args:
        params_dict (dict): Dictionary where each key maps to a list of parameter values for the batch.

    Returns:
        torch.Tensor: Tensor of shape [batch_size, 12], where each row contains the parameters for one sample.
    """
    # List of parameter keys in the desired order
    param_keys = ['crop_i', 'crop_j', 'crop_h', 'crop_w',
                  'flipped', 'color_jitter_applied', 'brightness_factor',
                  'contrast_factor', 'saturation_factor', 'hue_factor',
                  'grayscale', 'sigma']

    batch_size = len(next(iter(params_dict.values())))  # Get batch size from any value in the dict

    # Initialize a dictionary to hold parameter lists for each key
    param_values = {key: [0.0] * batch_size for key in param_keys}

    # Handle 'crop' separately since it contains tuples
    crop_list = params_dict.get('crop', [(0, 0, 0, 0)] * batch_size)
    for i, crop in enumerate(crop_list):
        crop_i, crop_j, crop_h, crop_w = crop
        param_values['crop_i'][i] = float(crop_i)
        param_values['crop_j'][i] = float(crop_j)
        param_values['crop_h'][i] = float(crop_h)
        param_values['crop_w'][i] = float(crop_w)

    # Process other parameters
    for key in ['flipped', 'color_jitter_applied', 'brightness_factor',
                'contrast_factor', 'saturation_factor', 'hue_factor',
                'grayscale', 'sigma']:
        value_list = params_dict.get(key, [0.0] * batch_size)
        for i, value in enumerate(value_list):
            if isinstance(value, bool):
                value = float(value)
            elif value is None:
                value = 0.0
            else:
                value = float(value)
            param_values[key][i] = value

    # Now, collect all parameter lists in the order defined by param_keys
    param_matrix = []
    for i in range(batch_size):
        params_row = [param_values[key][i] for key in param_keys]
        param_matrix.append(params_row)

    # Convert the list of lists into a tensor
    param_tensor = torch.tensor(param_matrix, dtype=torch.float32)
    return param_tensor

# class ContrastiveLearningDataset:
#     def __init__(self, root_folder):
#         self.root_folder = root_folder

#     @staticmethod
#     def get_simclr_pipeline_transform(size, s=1):
#         """Return a set of data augmentation transformations as described in the SimCLR paper."""
#         color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
#         data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=size),
#                                               transforms.RandomHorizontalFlip(),
#                                               transforms.RandomApply([color_jitter], p=0.8),
#                                               transforms.RandomGrayscale(p=0.2),
#                                               GaussianBlur(kernel_size=int(0.1 * size)),
#                                               transforms.ToTensor()])
#         return data_transforms

#     def get_dataset(self, name, n_views):
#         valid_datasets = {'cifar10': lambda: datasets.CIFAR10(self.root_folder, train=True,
#                                                               transform=ContrastiveLearningViewGenerator(
#                                                                   self.get_simclr_pipeline_transform(32),
#                                                                   n_views),
#                                                               download=True),

#                           'stl10': lambda: datasets.STL10(self.root_folder, split='unlabeled',
#                                                           transform=ContrastiveLearningViewGenerator(
#                                                               self.get_simclr_pipeline_transform(96),
#                                                               n_views),
#                                                           download=True)}

#         try:
#             dataset_fn = valid_datasets[name]
#         except KeyError:
#             raise InvalidDatasetSelection()
#         else:
#             return dataset_fn()
