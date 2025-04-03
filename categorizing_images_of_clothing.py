import os
import torch, torchvision
from torch import nn
from torchvision import transforms

# --- Architecting the Model ---
model = nn.Sequential(
    # 1st Convolutional Layer
    nn.Conv2d(in_channels=1,
              out_channels=32,
              kernel_size=3),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2),

    # 2nd Convolutional Layer
    nn.Conv2d(in_channels=32,
              out_channels=64,
              kernel_size=3),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2),

    # 3rd Convolutional Layer
    nn.Conv2d(in_channels=64,
              out_channels=128,
              kernel_size=3),
    nn.ReLU(),

    # Flatten for the classifier backend
    nn.Flatten(),

    # Fully Connected Layers
    nn.Linear(1152, 64),
    nn.ReLU(),
    nn.Linear(64, 10), 
    # note: no softmax since CrossEntropyLoss expects raw logits
)
print(model)

# --- Setup Device, Loss Function, and Optimizer ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# --- Loading Data ---
image_path = './'
transform = transforms.Compose([transforms.ToTensor()])

# Import FashionMNIST dataset
train_dataset = torchvision.datasets.FashionMNIST(root=image_path, train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.FashionMNIST(root=image_path, train=False, transform=transform, download=False)

# Load the training set into batches of 64 samples
from torch.utils.data import DataLoader
batch_size = 64
torch.manual_seed(42)
train_dl = DataLoader(train_dataset, batch_size, shuffle=True)

# Summary of the model
from torchsummary import summary
summary(model, input_size=(1, 28, 28), batch_size=-1, device="cpu")

# --- Training Function ---
def train(model, optimizer, num_epochs, train_dl):
    for epoch in range(num_epochs):
        loss_train = 0.0
        accuracy_train = 0.0
        for x_batch, y_batch in train_dl:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            pred = model(x_batch)
            loss = loss_fn(pred, y_batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()  # Clear previous gradients
            
            loss_train += loss.item() * y_batch.size(0)
            is_correct = (torch.argmax(pred, dim=1) == y_batch).float()
            accuracy_train += is_correct.sum().cpu()
        
        loss_train /= len(train_dl.dataset)
        accuracy_train /= len(train_dl.dataset)
        print(f'Epoch {epoch+1} - loss: {loss_train:.4f} - accuracy: {accuracy_train:.4f}')

# --- Checkpointing ---
checkpoint_path = 'model_checkpoint.pth'

# Define class names
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# --- Performance on Test Set ---
test_dl = DataLoader(test_dataset, batch_size, shuffle=False)
def evaluate_model(model, test_dl):
    accuracy_test = 0.0
    with torch.no_grad():
        for x_batch, y_batch in test_dl:
            pred = model.cpu()(x_batch)
            is_correct = (torch.argmax(pred, dim=1) == y_batch).float()
            accuracy_test += is_correct.sum().item()
        print(f'Accuracy on test set: {100 * accuracy_test / 10000} %')

if __name__ == '__main__':    
    # Check if a checkpoint exists; if so, load it. Otherwise, train and save.
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print("Loaded saved model weights.")
    else:
        num_epochs = 30
        train(model, optimizer, num_epochs, train_dl)
        torch.save(model.state_dict(), checkpoint_path)
        print("Training complete and model saved.")
    
    # Evaluate the model
    evaluate_model(model, test_dl)
    print("Evaluation complete.")