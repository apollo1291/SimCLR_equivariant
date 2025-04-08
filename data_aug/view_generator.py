import numpy as np

np.random.seed(0)


class ContrastiveLearningViewGenerator(object):
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform, n_views=2):
        self.base_transform = base_transform
        self.n_views = n_views

    def __call__(self, x):
        return [self.base_transform(x) for i in range(self.n_views)]

class ContrastiveLearningViewGeneratorWithParams:
    """Generate multiple views of the same image with transformation parameters."""
    def __init__(self, base_transform, param_transform, n_views=2):
        self.base_transform = base_transform
        self.param_transform = param_transform
        self.n_views = n_views

    def __call__(self, x):

        imgs = []
        params_list = []
        for _ in range(self.n_views - 1): # x is the first view so generate n - 1 more  
            img, params = self.param_transform(x)
            x = self.base_transform(x)

            #print(params)
            
            imgs.append(x)
            imgs.append(img)
            params_list.append(params)
        return imgs, params_list

