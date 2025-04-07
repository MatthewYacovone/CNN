import torch, torchvision
from torchvision import transforms

# Import FashionMNIST dataset
image_path = './'
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = torchvision.datasets.FashionMNIST(root=image_path, train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.FashionMNIST(root=image_path, train=False, transform=transform, download=False)
# print(train_dataset)
# print(test_dataset)

# Load the training set into batches of 64 samples
from torch.utils.data import DataLoader
batch_size = 64
torch.manual_seed(42)
train_dl = DataLoader(train_dataset, batch_size, shuffle=True)

# Inpect the image samples and their labels from the first batch
data_iter = iter(train_dl)
images, labels = next(data_iter)
print(labels)

# Define class names
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# View the format of the image data
print(images[0].shape) # each image is 28 * 28 pixels
print(torch.max(images), torch.min(images)) # with values from [0, 1]

# Display an image
import numpy as np
import matplotlib.pyplot as plt
npimg = images[1].numpy()
plt.imshow(np.transpose(npimg, (1, 2, 0))) # tuple represents the new order of dimensions PyTorch(channels, height, width) == (0, 1, 2) --> matplotlib(height, width, channels) == (1, 2, 0)
plt.colorbar()
plt.title(class_names[labels[1]])
plt.show()

# Display the first 16 samples
for i in range(16):
    plt.subplot(4, 4, i + 1)
    plt.subplots_adjust(hspace=.3)
    plt.xticks([])
    plt.yticks([])
    npimg = images[i].numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)), cmap="Greys")
    plt.title(class_names[labels[i]])
plt.show()
