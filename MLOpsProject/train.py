import torch
import torch.nn as nn
import torch.optim as optim
from models.model import NeuralNetwork
import wandb
from data.clean_data import *
import click

def data_loader(train_path: str,
                test_path: str,
                val_path: str,
                ):
    print("Loading data from: ", train_path)  
    train_loader = torch.load(train_path)
    test_loader = torch.load(test_path)
    val_loader = torch.load(val_path)
    return train_loader, test_loader, val_loader

# Using click to create a command line interface
@click.command()
@click.option('--train_path', default='./data/processed/train_loader.pth', help='Path to train dataloader')
@click.option('--test_path', default='./data/processed/test_loader.pth', help='Path to test dataloader')
@click.option('--val_path', default='./data/processed/val_loader.pth', help='Path to val dataloader')
@click.option('--lr', default=0.001, help='Learning rate')
@click.option('--epochs', default=20, help='Number of epochs')
@click.option('--verbose', default=True, help='Prints accuracy and device')
@click.option('--log_mode', default=False, help='Logs to wandb')
@click.option('--debug_mode', default=False, help='Prints shapes of input and output tensors')
def train(train_path: str, test_path: str,  val_path: str, lr: float, epochs: int, verbose: bool, log_mode: bool, debug_mode: bool) -> None:
    """Main training routine.

    Modes: debug, logging, normal. Change mode in config file.
        -> debug: prints out the shapes of the input and output tensors
        -> logging: logs the loss and accuracy to wandb
        -> normal: no logging or debugging. Defaults to this mode if no mode is specified.

    Args: see config file (conf/config.yaml))

    Returns: None

    """
    #Set device to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #Loading network
    net = NeuralNetwork()

    if verbose:
        print("Using device: ", device)

    #Load the data:
    train_loader,_,val_loader = data_loader(train_path=train_path, test_path=test_path, val_path=val_path)

    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = net.to(device)

    #Initiate model + wandb logging if log_mode is true
    if log_mode:
        wandb.config = {"lr": lr, "epochs": epochs}
        wandb.config.update({"lr": lr, "epochs": epochs})
        wandb.init(project="csgo")
        wandb.watch(model,log_freq=1000)
        
    #Instantiate optimizer from config file
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    #Loss function
    criterion = nn.CrossEntropyLoss()

    """Train loop of model with the given hyperparameters."""

    print(f"Training with learning rate {lr} and {epochs} epochs")
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.long().to(device)
            optimizer.zero_grad()
            outputs = model(images)
            if debug_mode:
                print("outputs: ",outputs,outputs.shape, outputs.dtype)
                print("labels: ",labels.long(),labels.long().shape, labels.long().dtype)
            loss = criterion(outputs, labels)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            accuracy = correct / total
        if log_mode:
            wandb.log({"train loss": running_loss / len(train_loader)})
            wandb.log({"train accuracy": accuracy})
        
        """Validation loop of model."""

        model.eval()
        correct_val = 0
        total_val = 0
        running_loss_val = 0.0
        with torch.no_grad():
            for images_val, labels_val in val_loader:
                images_val, labels_val = images_val.to(device), labels_val.long().to(device)
                outputs_val = model(images_val.float())  # Convert images to floats
                loss_val = criterion(outputs_val, labels_val)
                _, predicted_val = torch.max(outputs_val.data, 1)
                total_val += labels_val.size(0)
                correct_val += (predicted_val == labels_val).sum().item()
                running_loss_val += loss_val.item()
        accuracy_val = correct_val / total_val

        """Printing and logging of loss and accuracy."""
        if verbose:
            print(f'Epoch {epoch+1}/{epochs},Train Accuracy: {accuracy:.4f} Validation Accuracy: {accuracy_val:.4f}, Validation Loss: {running_loss_val / len(val_loader):.4f}')
        
        if log_mode:
            wandb.log({"train loss": running_loss / len(train_loader)})
            wandb.log({"train accuracy": accuracy})
            wandb.log({"validation accuracy": accuracy_val})
            wandb.log({"validation loss": running_loss_val / len(val_loader)})
    print('Finished Training')
    torch.save(model, f"trained_model.pt")
    print("Model saved at: ", f"trained_model.pt")  

if __name__ == "__main__":
    train() 