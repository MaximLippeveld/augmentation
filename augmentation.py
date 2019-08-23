import cv2
import numpy as np
import numpy.random as rnd
import torch
import torch.nn.functional as F
from scipy.ndimage import spline_filter1d


class ToTensor(object):

    def __init__(self, cuda=True):
        """
        Transforms a numpy array into a tensor
        :param cuda: specifies whether the tensor should be transfered to the GPU or not
        """
        self.cuda = cuda

    def __call__(self, x):
        """
        Forward call
        :param x: input tensor
        :return: output tensor
        """
        if self.cuda:
            return torch.Tensor(x).cuda()
        else:
            return torch.Tensor(x)


class ToFloatTensor(object):

    def __call__(self, x):
        """
        Forward call
        :param x: input tensor
        :return: output tensor
        """
        return x.float()


class ToLongTensor(object):

    def __call__(self, x):
        """
        Forward call
        :param x: input tensor
        :return: output tensor
        """
        return x.long()


class AddChannelAxis(object):

    def __call__(self, x):
        """
        Forward call
        :param x: input tensor
        :return: output tensor
        """
        return x.unsqueeze(0)


class AddNoise(object):

    def __init__(self, prob=0.5, sigma_min=0.0, sigma_max=1.0):
        """
        Adds noise to the input
        :param prob: probability of adding noise
        :param sigma_min: minimum noise standard deviation
        :param sigma_max: maximum noise standard deviation
        """
        self.prob = prob
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def __call__(self, x):
        """
        Forward call
        :param x: input tensor
        :return: output tensor
        """

        if rnd.rand() < self.prob:
            sigma = rnd.uniform(self.sigma_min, self.sigma_max)
            noise = torch.normal(0, sigma, x.size())
            if x.is_cuda:
                noise = noise.cuda()
            return x + noise
        else:
            return x


class Normalize(object):

    def __init__(self, mu=0, std=1):
        """
        Normalizes the input
        :param mu: mean of the normalization
        :param std: standard deviation of the normalization
        """
        self.mu = mu
        self.std = std

    def __call__(self, x):
        """
        Forward call
        :param x: input tensor
        :return: output tensor
        """
        return (x - self.mu) / self.std


class FlipX(object):

    def __init__(self, shape, prob=1, cuda=True):
        """
        Perform a flip along the X axis
        :param shape: shape of the inputs
        :param prob: probability of flipping
        :param cuda: specify whether the inputs are on the GPU
        """
        self.shape = shape
        self.prob = prob
        self.cuda = cuda

        i = np.linspace(-1, 1, shape[2])
        j = np.linspace(-1, 1, shape[3])
        xv, yv = np.meshgrid(i, j)
        xv = np.fliplr(xv).copy()

        grid = torch.cat((torch.Tensor(xv).unsqueeze(-1), torch.Tensor(yv).unsqueeze(-1)), dim=-1)
        grid = grid.unsqueeze(0).repeat_interleave(shape[0], dim=0)
        if cuda:
            grid = grid.cuda()
        self.grid = grid

    def __call__(self, x):
        """
        Forward call
        :param x: input tensor
        :return: output tensor
        """

        if rnd.rand() < self.prob:
            return F.grid_sample(x, self.grid)
        else:
            return x


class FlipY(object):

    def __init__(self, shape, prob=1, cuda=True):
        """
        Perform a flip along the Y axis
        :param shape: shape of the inputs
        :param prob: probability of flipping
        :param cuda: specify whether the inputs are on the GPU
        """
        self.shape = shape
        self.prob = prob
        self.cuda = cuda

        i = np.linspace(-1, 1, shape[2])
        j = np.linspace(-1, 1, shape[3])
        xv, yv = np.meshgrid(i, j)
        yv = np.flipud(yv).copy()

        grid = torch.cat((torch.Tensor(xv).unsqueeze(-1), torch.Tensor(yv).unsqueeze(-1)), dim=-1)
        grid = grid.unsqueeze(0).repeat_interleave(shape[0], dim=0)
        if cuda:
            grid = grid.cuda()
        self.grid = grid

    def __call__(self, x):
        """
        Forward call
        :param x: input tensor
        :return: output tensor
        """

        if rnd.rand() < self.prob:
            return F.grid_sample(x, self.grid)
        else:
            return x


class Rotate90(object):

    def __init__(self, shape, prob=1, cuda=True):
        """
        Rotate the inputs by 90 degree angles
        :param prob: probability of rotating
        """
        self.shape = shape
        self.prob = prob
        self.cuda = cuda

        i = np.linspace(-1, 1, shape[2])
        j = np.linspace(-1, 1, shape[3])
        grids = []
        for m in range(4):
            xv, yv = np.meshgrid(i, j)
            xv = np.rot90(xv, m+1).copy()
            yv = np.rot90(yv, m+1).copy()

            grid = torch.cat((torch.Tensor(xv).unsqueeze(-1), torch.Tensor(yv).unsqueeze(-1)), dim=-1)
            grid = grid.unsqueeze(0).repeat_interleave(shape[0], dim=0)
            if cuda:
                grid = grid.cuda()
            grids.append(grid)
        self.grids = grids

    def __call__(self, x):
        """
        Forward call
        :param x: input tensor
        :return: output tensor
        """

        if rnd.rand() < self.prob:
            return F.grid_sample(x, self.grids[rnd.randint(0, 4)])
        else:
            return x


class RandomDeformation(object):

    def __init__(self, shape, prob=1, cuda=True, points=64, sigma=0.001):
        """
        Apply random deformation to the inputs
        :param shape: shape of the inputs
        :param prob: probability of deforming the data
        :param cuda: specifies whether the inputs are on the GPU
        :param points: seed points for deformation
        :param sigma: standard deviation for deformation
        """
        self.shape = shape
        self.prob = prob
        self.cuda = cuda
        self.points = points
        self.sigma = sigma
        self.p = 10

        i = np.linspace(-1, 1, shape[2])
        j = np.linspace(-1, 1, shape[3])
        xv, yv = np.meshgrid(i, j)

        grid = torch.cat((torch.Tensor(xv).unsqueeze(-1), torch.Tensor(yv).unsqueeze(-1)), dim=-1)
        grid = grid.unsqueeze(0).repeat_interleave(shape[0], dim=0)
        if cuda:
            grid = grid.cuda()
        self.grid = grid

    def _deformation_grid(self):
        """
        Get a new random deformation grid
        :return: deformation tensor
        """
        points = [self.points] * 2
        sigma = np.random.rand() * self.sigma
        displacement = np.random.randn(*points, 2) * sigma

        # filter the displacement
        displacement_f = np.zeros_like(displacement)
        for d in range(0, displacement.ndim - 1):
            spline_filter1d(displacement, axis=d, order=3, output=displacement_f, mode='nearest')
            displacement = displacement_f

        # resample to proper size
        displacement_f = np.zeros((self.shape[2], self.shape[3], 2))
        for d in range(0, displacement.ndim - 1):
            displacement_f[:, :, d] = cv2.resize(displacement[:, :, d], dsize=self.shape[2:],
                                                 interpolation=cv2.INTER_CUBIC)

        displacement = torch.Tensor(displacement_f).unsqueeze(0)
        if self.cuda:
            displacement = displacement.cuda()
        grid = self.grid + displacement

        return grid

    def __call__(self, x):
        """
        Forward call
        :param x: input tensor
        :return: output tensor
        """

        grid = self._deformation_grid()
        if rnd.rand() < self.prob:
            return F.grid_sample(x, grid, padding_mode="border")
        else:
            return x
