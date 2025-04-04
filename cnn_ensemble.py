import os
import random
import torch, torchvision
from torch import nn
from torchvision import transforms
from torch.utils.data import DataLoader, ConcatDataset, random_split, Subset
from torchsummary import summary
from collections import Counter
from copy import deepcopy

# --- Architecting the base model ---
def create_cnn_model():
    return nn.Sequential(
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

        # Flatten for classifier backend
        nn.Flatten(),

        # Fully Connected Layers
        nn.Linear(1152, 64),
        nn.ReLU(),
        nn.Linear(64, 10)
        # no softmax because CrossEntropyLoss expects raw logits
    )

# --- Create an ensemble ---
def create_ensemble(n_models=3):
    ensemble_models = []
    for i in range(n_models):
        if i == 0:
            ensemble_models.append(create_cnn_model())
        else:
            ensemble_models.append(deepcopy(ensemble_models[0]))
        
    return ensemble_models

# --- Training Function ---
def train(model, optimizer, n_epochs, train_dl):
    for epoch in range(n_epochs):
        loss_train = 0.0
        accuracy_train = 0.0
        for x_batch, y_batch in train_dl:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            pred = model(x_batch)
            loss = loss_fn(pred, y_batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad() # clears previous gradient

            loss_train += loss.item() * y_batch.size(0)
            is_correct = (torch.argmax(pred, dim=1) == y_batch).float()
            accuracy_train += is_correct.sum().cpu()

        loss_train /= len(train_dl.dataset)
        accuracy_train /= len(train_dl.dataset)
        print(f'Epoch {epoch+1} - loss: {loss_train:.4f} - accuracy: {accuracy_train:.4f}')

# --- Checkpointing ---
checkpoint_path = 'ensemble_checkpoint_v1.pth'

# Define class names
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# --- Evaluate Ensemble Performance ---
def evaluate_ensemble(models, test_dl, device):
    total = 0
    ensemble_correct = 0
    with torch.no_grad():
        for x_batch, y_batch in test_dl:
            x_batch = x_batch.to(device)
            outputs_sum = None

            # Sum outputs for each model
            for model in models:
                outputs = model(x_batch)
                if outputs_sum is None:
                    outputs_sum = outputs
                else:
                    outputs_sum += outputs
            
            # Average the outputs
            outputs_avg = outputs_sum / len(models)
            predicted = torch.argmax(outputs_avg, dim=1).cpu()
            total += y_batch.size(0)
            ensemble_correct += (predicted == y_batch).sum().item()
    print(f'Ensemble Accuracy on test set: {100 * ensemble_correct / total:.2f}')

if __name__ == '__main__':
    # Set ensemble size
    n_models = 3

    # Create ensemble
    ensemble_models = create_ensemble(n_models=n_models)

    # Setup Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for model in ensemble_models:
        model.to(device)

    # Define Loss Function
    loss_fn = nn.CrossEntropyLoss()

    # --- Data Transformations --- 
    image_path = './'
    transform_orig = transforms.Compose([transforms.ToTensor()]) # original images

    transform_aug = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.RandomResizedCrop(size=(28,28), scale=(0.9, 1)), 
        transforms.ToTensor()
        ])

    # Load FashionMNIST training set w/o augmentation
    train_dataset_orig = torchvision.datasets.FashionMNIST(root=image_path, train=True, transform=transform_orig, download=True)

    # Create augmented copies
    augmented_datasets = [torchvision.datasets.FashionMNIST(root=image_path, train=True, transform=transform_aug, download=True)
                         for _ in range(n_models-1)]

    # Combine the original witht the augmented copies
    combined_dataset = ConcatDataset([train_dataset_orig] + augmented_datasets)
    print(f'Total training samples after combining: {len(combined_dataset)}') # n_models * 60,000

    # Randomly split the combined dataset
    subset_length = len(train_dataset_orig)
    lengths = [subset_length] * n_models
    train_subsets = random_split(combined_dataset, lengths)

    # Create a dataloader for each model
    batch_size = 64
    train_dataloaders = [DataLoader(subset, batch_size=batch_size, shuffle=True) for subset in train_subsets]

    # Testing dataset is the same for each model == the orginal training set
    test_dataset = torchvision.datasets.FashionMNIST(root=image_path, train=False, transform=transform_orig, download=False)
    test_dl = DataLoader(test_dataset, batch_size, shuffle=False)

    # Check if a checkpoint exists; if so, load it. Otherwise, train and save.
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        for idx, model in enumerate(ensemble_models):
            model.load_state_dict(checkpoint[f'model_{idx}'])
        print("Loaded saved model weights.")
    else:
        n_epochs = 30

        # Create separate optimizers for each model
        optimizers = [torch.optim.Adam(model.parameters(), lr=0.001) for model in ensemble_models]
        
        for idx, (model, optimizer, train_dl) in enumerate(zip(ensemble_models, optimizers, train_dataloaders)):
            print(f'\nTraining model {idx+1} on its own training set:')
            train(model, optimizer, n_epochs, train_dl)
        
        # Save a checkpoint for the entire ensemble
        checkpoint = {}
        for idx, model in enumerate(ensemble_models):
            checkpoint[f'model_{idx}'] = model.state_dict()
        torch.save(checkpoint, checkpoint_path)
        print("Training complete and model saved.")
    
    # Evaluate the model
    evaluate_ensemble(ensemble_models, test_dl, device)
    print("Evaluation complete.")
    