# GPU accelerated data augmentation

This code is an extension on the [imgaug](https://github.com/aleju/imgaug) package that provides flexible use of data augmentation. In particular, we provide standard augmentation strategies on the GPU, as some of these can be intensive on the CPU. Our GPU translation is based on [PyTorch](https://pytorch.org/). The current version supports both 2D and 3D data augmentation, however the 3D augmentation is still in test phase. 

## Installation
Install the dependencies and you are ready to go! 
<pre><code>pip install -r requirements.txt</code></pre>

## Usage
Import the required modules:
<pre><code>import imageio
import torch
from augmentation_2d import FlipX, FlipY, Rotate90, AddNoise, RandomDeformation</code></pre>

Read an image into a tensor and transfer it to the GPU:
<pre><code>x = imageio.imread('img/elaine.png').astype('float')
shape = x.shape
x = torch.Tensor(x).unsqueeze(0).unsqueeze(0).cuda()</code></pre>
<img src="https://github.com/JorisRoels/augmentation/blob/master/img/elaine.png" width="256" height="256">

Flip the image along the x and y-axis: 
<pre><code>flipx = FlipX(shape)
y = flipx(x)
flipy = FlipY(shape)
y = flipy(x)</code></pre>
<img src="https://github.com/JorisRoels/augmentation/blob/master/img/elaine_flipx.png" width="256" height="256">  <img src="https://github.com/JorisRoels/augmentation/blob/master/img/elaine_flipy.png" width="256" height="256">

Rotate the image 90 degrees: 
<pre><code>rotate = Rotate90(shape)
y = rotate(x)</code></pre>
<img src="https://github.com/JorisRoels/augmentation/blob/master/img/elaine_rot.png" width="256" height="256">

Rotate the image randomly: 
<pre><code>rotate = RotateRandom(shape)
y = rotate(x)</code></pre>
<img src="https://github.com/JorisRoels/augmentation/blob/master/img/elaine_rot_random.png" width="256" height="256">

Add noise to the image:
<pre><code>noise = AddNoise(sigma_min=20, sigma_max=20)
y = noise(x)</code></pre>
<img src="https://github.com/JorisRoels/augmentation/blob/master/img/elaine_noise.png" width="256" height="256">

Apply random deformations to the image: 
<pre><code>deform = RandomDeformation(shape, sigma=0.01)
y = deform(x)</code></pre>
<img src="https://github.com/JorisRoels/augmentation/blob/master/img/elaine_deform.png" width="256" height="256">
