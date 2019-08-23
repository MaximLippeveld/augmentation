import imageio
import torch

from augmentation import FlipX, FlipY, Rotate90, AddNoise, RandomDeformation

x = imageio.imread('img/elaine.png').astype('float')
shape = x.shape
x = torch.Tensor(x).unsqueeze(0).unsqueeze(0).cuda()

flipx = FlipX(shape)
y = flipx(x)

flipy = FlipY(shape)
y = flipy(x)

rotate = Rotate90(shape)
y = rotate(x)

noise = AddNoise(sigma_min=20, sigma_max=20)
y = noise(x)

deform = RandomDeformation(shape, sigma=0.01)
y = deform(x)
