import matplotlib.pyplot as plt
import imageio
import torch

from augmentation import FlipX, FlipY, Rotate90, AddNoise, RandomDeformation

x = imageio.imread('elaine.png').astype('float')
x = torch.Tensor(x).unsqueeze(0).unsqueeze(0).cuda()

plt.imshow(x[0, 0, ...].cpu().numpy(), cmap='gray')
plt.show()

flipx = FlipX(x.size())
y = flipx(x)
plt.imshow(y[0, 0, ...].cpu().numpy(), cmap='gray')
plt.show()

flipy = FlipY(x.size())
y = flipy(x)
plt.imshow(y[0, 0, ...].cpu().numpy(), cmap='gray')
plt.show()

rotate = Rotate90()
y = rotate(x)
plt.imshow(y[0, 0, ...].cpu().numpy(), cmap='gray')
plt.show()

noise = AddNoise(sigma_min=20, sigma_max=20)
y = noise(x)
plt.imshow(y[0, 0, ...].cpu().numpy(), cmap='gray')
plt.show()

deform = RandomDeformation(x.size(), points=x.size(2) // 64, sigma=0.01)
y = deform(x)
plt.imshow(y[0, 0, ...].cpu().numpy(), cmap='gray')
plt.show()
