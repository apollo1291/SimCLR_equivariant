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
from torchvision.datasets import ImageFolder
from data_aug.view_generator import ContrastiveLearningViewGeneratorWithParams
from exceptions.exceptions import InvalidDatasetSelection
from torchvision.transforms import functional as F
import torch


class ToTensor:
    def __call__(self, img):
        img = F.to_tensor(img)
        return img
    
class BaseTransformPipeline:
    def __init__(self, size, s=1):
        
        self.normalize = [
            transforms.Resize((size, size)),
            ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),   
        ]
    def __call__(self, img):

        for t in self.normalize:
            img = t(img)
        return img

class CustomTransformPipeline:
    def __init__(self, size, s=1):
        self.size = size
        # Normalization for imagenet
        self.transforms = [
            RandomResizedCropWithParams(size=size),
            RandomHorizontalFlipWithParams(),
            #RandomApplyWithParams([ColorJitterWithParams(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)], p=0.8),
            RandomGrayscaleWithParams(p=0.2),
            GaussianBlurWithParams(kernel_size=int(0.1 * size)),
        ]
        self.normalize = [
            ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),   
        ]

    def __call__(self, img):
        params_list = []
        for t in self.transforms:
            img, params = t(img)
            params_list.append(params)
        # Combine all parameters into a single dictionary
        transformation_params = {k: v for d in params_list for k, v in d.items()}

        for t in self.normalize:
            img = t(img)
        #print(transformation_params)
        return img, transformation_params

class ContrastiveLearningDatasetWithParams:
    def __init__(self, root_folder):
        self.root_folder = root_folder

    def get_simclr_pipeline_transform(self, size, s=1):
        return CustomTransformPipeline(size=size, s=s)
    
    def get_base_pipline_transform(self, size):
        return BaseTransformPipeline(size) 

    def get_dataset(self, name, n_views):
        valid_datasets = {
            'imagenet': lambda: ImageFolder(
                self.root_folder,
                transform=ContrastiveLearningViewGeneratorWithParams(
                    self.get_base_pipline_transform(224),
                    self.get_simclr_pipeline_transform(224),
                    n_views
                )
            ), 
            
            'cifar10': lambda: datasets.CIFAR10(
                self.root_folder,
                train=True,
                transform=ContrastiveLearningViewGeneratorWithParams(
                    self.get_base_pipline_transform(224),
                    self.get_simclr_pipeline_transform(224),
                    n_views
                ),
                download=True
            ),

            'stl10': lambda: datasets.STL10(
                self.root_folder,
                split='unlabeled',
                transform=ContrastiveLearningViewGeneratorWithParams(
                    self.get_base_pipline_transform(),
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
    class_ids = []  # List to store class IDs

    for item in batch:
        imgs_params, class_id = item  # Retain the class ID
        imgs, params = imgs_params
        images_list.append(imgs)    # imgs is a list of images
        params_list.append(params)  # params is a list of dictionaries
        class_ids.append(class_id)  # Collect class IDs

    # Transpose images_list and params_list to group by views
    images = list(zip(*images_list))   # Now images is a list of views, each containing batch_size images
    params = list(zip(*params_list))   # 

    # Stack images for each view
    images = [torch.stack(imgs, dim=0) for imgs in images]  # List of tensors with shape [batch_size, C, H, W]

    # For params, collate the dictionaries
    params_collated = []
    for view_params in params:
        collated_params = {}
        keys = view_params[0].keys()
        for key in keys:
            collated_params[key] = [d[key] for d in view_params]
        params_collated.append(collated_params)

    return images, params_collated, class_ids

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
    param_values = {key: [0.0] * batch_size for key in params_dict}


    # Process other parameters
    for key in params_dict.keys():
        value_list = params_dict.get(key)

        for i, value in enumerate(value_list):
            if isinstance(value, bool):
                value = float(value)
            elif value is None:
                value = 0.0
            else:
                value = float(value)
            param_values[key][i] = value

    param_matrix = []
    for i in range(batch_size):
        params_row = [param_values[key][i] for key in params_dict.keys()]
        param_matrix.append(params_row)

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
