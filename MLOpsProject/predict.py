import torch
from data.dummy_data_loader import dummy_data
from os.path import dirname as up
from data.clean_data import *
import click


# Using click to create a command line interface
@click.command()
@click.option('--model', default="./models/trained_model.pt", help='Path to model')
@click.option('--dataloader', default='./data/processed/test_loader.pth', help='Path to dataloader')
@click.option('--verbose', default=True, help='Prints accuracy')
def predict(model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    verbose: bool = False):

    dataloader = torch.load(dataloader)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    if verbose:
        print("Using device: ", device)
        
    if device == "cuda":
        model = torch.load(model)
    else:
        model = torch.load(model, map_location=torch.device('cpu'))        

    total = 0
    correct = 0
    all_predictions = []
    model.eval()
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images.float())
            #torch max of outputs
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_predictions.append(outputs)
    test_accuracy = 100*(correct / total)
    if verbose:
        print(f'Test Accuracy: {test_accuracy:.3f}%')
    return all_predictions


if __name__ == "__main__":
    predict()