import SaveModelStrategy.SaveModelStrategy
import flwr as fl

def start_server():
    strategy = SaveModelStrategy()
    fl.server.start_server(
        server_address="localhost:8083",
        config=fl.server.ServerConfig(num_rounds=1),
        strategy=strategy
    )
    print("✅ Server finished training")

    if strategy.final_parameters is not None:
        torch.save(strategy.final_parameters, "final_params.pth")