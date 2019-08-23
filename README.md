# GPU accelerated data augmentation

This code is an extension on the ["imgaug"](https://github.com/aleju/imgaug) package that provides flexible use of data augmentation. In particular, we provide standard augmentation strategies on the GPU, as some of these can be intensive on the CPU. 

## Installation
Install the dependencies and you are ready to go! 
`pip install -r requirements.txt`

## Usage
Import the required modules:
<pre><code>import imageio
import torch</code></pre>
