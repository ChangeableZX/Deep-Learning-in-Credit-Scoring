from collections import OrderedDict

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, train_loader, categories, num_continuous, criterion, num_epochs):
        self.train_loader = train_loader
        self.categories = categories
        self.num_continuous = num_continuous
        self.criterion = criterion
        self.num_epochs = num_epochs
        self.model = get_model().to(device)

    def get_parameters(self, config=None):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        for epoch in range(self.num_epochs):
            for X_cat, X_cont, y in self.train_loader:
                X_cat, X_cont, y = X_cat.to(device), X_cont.to(device), y.to(device)
                optimizer.zero_grad()
                outputs = self.model(X_cat, X_cont).squeeze()
                loss = self.criterion(outputs, y,X_cat)
                loss.backward()
                optimizer.step()
        return self.get_parameters(), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        return 0.0, len(self.train_loader.dataset), {}