import imageio
import torch

from augmentation import FlipX, FlipY, Rotate90, AddNoise, RandomDeformation

x = imageio.imread('img/elaine.png').astype('float')
x = torch.Tensor(x).unsqueeze(0).unsqueeze(0).cuda()

flipx = FlipX(x.size())
y = flipx(x)

flipy = FlipY(x.size())
y = flipy(x)

rotate = Rotate90(x.size())
y = rotate(x)

noise = AddNoise(sigma_min=20, sigma_max=20)
y = noise(x)

deform = RandomDeformation(x.size(), points=x.size(2) // 64, sigma=0.01)
y = deform(x)
