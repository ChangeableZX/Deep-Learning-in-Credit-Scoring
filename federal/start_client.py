import FlowerClient.FlowerClient
import flwr as fl

def start_client(train_loader, categories, num_continuous, criterion, num_epochs):
    client = FlowerClient(train_loader, categories, num_continuous, criterion, num_epochs)
    fl.client.start_numpy_client(server_address="localhost:8083", client=client)