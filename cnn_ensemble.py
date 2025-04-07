import os
import random
import torch, torchvision
from torch import nn
from torchvision import transforms
from torch.utils.data import DataLoader, ConcatDataset, random_split, Subset
from copy import deepcopy
import torch.nn.functional as F
import pandas as pd

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

# --- For each image, Determine Disagreement ---
def determine_disagreement(i, model_predictions, total_disagreement, n_images_with_disagreement):
    predictions = model_predictions[:, i].tolist() # get a list of predictions of ith image for all models
    unique_predictions = set(predictions)
    disagreement = len(unique_predictions) - 1 # unanimous vote == 0; otherwise, disagreement is >0 
    total_disagreement += disagreement
    if disagreement > 0:
        n_images_with_disagreement += 1
    
    return total_disagreement, n_images_with_disagreement

# --- Measuring Polarization ---
def measure_polarization(models, x, device):
    B = x.size(0)
    n_models = len(models)
    probs_list = [] # holds each model's probability predictions (after softmax)
    for model in models:
        with torch.no_grad():
            outputs = model(x.to(device)) # (B, n_classes)
            probs = F.softmax(outputs, dim=1).cpu() # convert to probabilities
            probs_list.append(probs)

    # Compute polarization for each image in the batch
    polarization_scores = []
    for i in range(B):
        model_probs = [probs[i] for probs in probs_list] # list of (n_classes,) tensors
        avg_prob = torch.stack(model_probs, dim=0).mean(dim=0) # consensus distribution

        # Compute the KL divergence from each model's distribution to the consensus
        kl_divs = [F.kl_div(torch.log(probs), avg_prob, reduction="batchmean") for probs in model_probs]

        # Average the KL divergences as the polarization score for this image
        polarization = sum(kl_divs) / n_models
        polarization_scores.append(polarization)
    
    return torch.tensor(polarization_scores)

# --- Check Accuracy of Ensembles's Classification --
def check_accuracy_for_in_distribution_images(i, y_batch, ensemble_pred, polarization_scores, global_idx, in_distribution_count):
    results = [] # list of dictionaries for each test image
    true_label = y_batch[i].item()
    predicted_label = ensemble_pred[i].item()
    polarization = polarization_scores[i].item()
    in_distribution = 1 if global_idx < in_distribution_count else 0
    if in_distribution:
        correctness = 1 if (predicted_label == true_label) else 0
    else:
        correctness = 0 # for OOD images, there is no correct classification
    results.append({'image_idx': global_idx,
                    'polarization': polarization,
                    'in_distribution': in_distribution,
                    'correct': correctness})
    return results

# --- Evaluate Ensemble Performance and Generate Disagreement Dataset---
def evaluate_ensemble(models, test_dl, device, in_distribution_count):
    total = 0
    ensemble_correct = 0
    total_disagreement = 0
    n_images_with_disagreement = 0
    all_results = []

    with torch.no_grad():
        for batch_idx, (x_batch, y_batch) in enumerate(test_dl):
            x_batch = x_batch.to(device)
            batch_size = x_batch.size(0)

            # Collect individual model predictions
            model_predictions = []
            for model in models:
                outputs = model(x_batch)
                pred = torch.argmax(outputs, dim=1).cpu()
                model_predictions.append(pred)
            model_predictions = torch.stack(model_predictions, dim=0) # (n_models, batch_size)
            
            # Find ensemble prediction by averaging the outputs
            outputs_sum = None
            for model in models:
                outputs = model(x_batch)
                if outputs_sum is None:
                    outputs_sum = outputs
                else:
                    outputs_sum += outputs

            outputs_avg = outputs_sum / len(models)
            ensemble_pred = torch.argmax(outputs_avg, dim=1).cpu()
            total += batch_size
            ensemble_correct += (ensemble_pred == y_batch).sum().item()

            # For each image, determine disagreement
            for i in range(batch_size):
                total_disagreement, n_images_with_disagreement = determine_disagreement(i, model_predictions, total_disagreement, n_images_with_disagreement)
            
            # Compute polarization scores for this batch.
            polarization_scores = measure_polarization(models, x_batch, device)

            # Record each image's polarization and accuracy
            for i in range(batch_size):
                global_idx = batch_idx * test_dl.batch_size
                results = check_accuracy_for_in_distribution_images(i, y_batch, ensemble_pred, polarization_scores, global_idx, in_distribution_count)
                all_results.extend(results)

    in_distribution_total = sum(1 for r in all_results if r['in_distribution'] == 1)
    in_distribution_correct = sum(r['correct'] for r in all_results if r['in_distribution'] == 1)
    if in_distribution_total > 0:
        ensemble_accuracy = 100 * in_distribution_correct / in_distribution_total
    else:
        ensemble_accuracy = 0.0

    print(f'Ensemble Accuracy on test set: {ensemble_accuracy:.2f} %')
    avg_disagreement = total_disagreement / total # 0 is models always agree
    percent_disagree = 100 * n_images_with_disagreement / total
    print(f'Average disagreement per image: {avg_disagreement:.2f}')
    print(f'Percentage of images with any disagreement: {percent_disagree:.2f}')

    return pd.DataFrame(all_results)

# --- Checkpointing ---
checkpoint_path = 'ensemble_checkpoint_v1.pth'

# --- Define class names ---
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot', 'OOD']

if __name__ == '__main__':
    # --- SET UP ---
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

    transform_ood = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28,28)),
        transforms.ToTensor()
        ])
    
    # --- TRAINING ---
    # Load FashionMNIST training set w/o augmentation
    train_dataset_orig = torchvision.datasets.FashionMNIST(root=image_path, train=True, transform=transform_orig, download=True)

    # Create augmented copies
    augmented_datasets = [torchvision.datasets.FashionMNIST(root=image_path, train=True, transform=transform_aug, download=True)
                         for _ in range(n_models-1)]

    # Combine the original witht the augmented copies
    combined_train_dataset = ConcatDataset([train_dataset_orig] + augmented_datasets)
    print(f'Total training samples after combining: {len(combined_train_dataset)}') # n_models * 60,000

    # Randomly split the combined dataset
    subset_length = len(train_dataset_orig)
    lengths = [subset_length] * n_models
    train_subsets = random_split(combined_train_dataset, lengths)

    # Create a dataloader for each model
    batch_size = 64
    train_dataloaders = [DataLoader(subset, batch_size=batch_size, shuffle=True) for subset in train_subsets]

    # --- TESTING ---
    # Testing dataset is the same for each model == the orginal training set
    fmnist_test = torchvision.datasets.FashionMNIST(root=image_path, train=False, transform=transform_orig, download=False)
    in_distribution_count = len(fmnist_test)

    # Load CIFAR-10 (OOD) Testing dataset
    cifar_test = torchvision.datasets.CIFAR10(root=image_path, train=False, transform=transform_ood, download=True)
    ood_indices = random.sample(range(len(cifar_test)), 500)
    ood_subset = Subset(cifar_test, ood_indices)

    # Combine the testing datasets
    combined_test_dataset = ConcatDataset([fmnist_test, ood_subset])
    print(f'Total testing samples after combining: {len(combined_test_dataset)}')
    test_dl = DataLoader(combined_test_dataset, batch_size, shuffle=False)

    # Check if a checkpoint exists; if so, load it. Otherwise, train and save.
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        for idx, model in enumerate(ensemble_models):
            model.load_state_dict(checkpoint[f'model_{idx}'])
        print("Loaded saved model weights.")
    else:
        # Number of training dataset pass-throughs
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
    disagreement_df = evaluate_ensemble(ensemble_models, test_dl, device, in_distribution_count)
    print("Evaluation complete.")
    print(disagreement_df.head())
    disagreement_df.to_csv('testing123.csv', index=False)

    