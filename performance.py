"""Script to run performance tests.
"""

import matplotlib
matplotlib.use("tkagg")
import matplotlib.pyplot as plt
from augmentation_2d import *
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy
from time import time
from tqdm import trange, tqdm

ds = datasets.FashionMNIST("img/fmnist", train=True, download=True).data
shape = tuple(ds.shape[1:])

cpu_augs = [
    Stack(cuda=False), 
    Unsqueeze(cuda=False),
    ToFloatTensor(cuda=False),
    FlipX(shape, cuda=False),
    FlipY(shape, cuda=False),
    RandomDeformation(shape, sampling_interval=7, cuda=False),
    RotateRandom(shape, cuda=False)   
]
cpu_augmenter = transforms.Compose(cpu_augs)

gpu_augs = [
    Stack(), 
    Unsqueeze(),
    ToFloatTensor(),
    FlipX(shape),
    FlipY(shape),
    RandomDeformation(shape, sampling_interval=7),
    RotateRandom(shape)   
]
gpu_augmenter = transforms.Compose(gpu_augs)

def get_loader(ds, augmenter, workers=0):
    return DataLoader(
            ds, batch_size=32, shuffle=False, 
            drop_last=False, num_workers=workers,
            collate_fn=augmenter
        )

# Difference between CPU and GPU-accelerated augmentation
N = 5
n_batches = int(len(ds)//32)
times = numpy.empty((2, N, n_batches), dtype=float)

for n, augmenter in enumerate(tqdm([cpu_augmenter, gpu_augmenter], desc="Augmenter")):
    epoch_times = numpy.empty((N,n_batches), dtype=float)
    for i in trange(N, desc="Repeats", leave=False):
        loader = get_loader(ds, augmenter) 
        batch_times = numpy.empty((n_batches,), dtype=float)
        for j in trange(len(batch_times), leave=False, desc="Batches"):
            start = time()
            next(iter(loader))
            batch_times[j] = time() - start
        epoch_times[i] = batch_times
    times[n] = epoch_times

times_mean, times_var = times.mean(axis=1), times.var(axis=1)

fig, ax = plt.subplots()

x = numpy.arange(times_mean.shape[1])
ax.errorbar(x, times_mean[0], label="CPU", yerr=times_var[0])
ax.errorbar(x, times_mean[1], label="GPU", yerr=times_var[1])
ax.legend()

plt.show()

# Changing the workers
N = 5
n_batches = int(len(ds)//32)
times = numpy.empty((2, N, n_batches), dtype=float)

for n, augmenter in enumerate(tqdm([cpu_augmenter, gpu_augmenter], desc="Augmenter")):
    epoch_times = numpy.empty((N,n_batches), dtype=float)
    for i in trange(N, desc="Repeats", leave=False):
        loader = get_loader(ds, augmenter) 
        batch_times = numpy.empty((n_batches,), dtype=float)
        for j in trange(len(batch_times), leave=False, desc="Batches"):
            start = time()
            next(iter(loader))
            batch_times[j] = time() - start
        epoch_times[i] = batch_times
    times[n] = epoch_times

times_mean, times_var = times.mean(axis=1), times.var(axis=1)

fig, ax = plt.subplots()

x = numpy.arange(times_mean.shape[1])
ax.errorbar(x, times_mean[0], label="CPU", yerr=times_var[0])
ax.errorbar(x, times_mean[1], label="GPU", yerr=times_var[1])
ax.legend()

plt.show()