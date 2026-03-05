from flwr.common import Parameters
from flwr.server.strategy import FedAvg

class SaveModelStrategy(FedAvg):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.final_parameters = None

    def aggregate_fit(self, server_round, results, failures):
        # 调用父类聚合方法
        aggregated_parameters = super().aggregate_fit(server_round, results, failures)
        
        if aggregated_parameters is not None:
            # 转换为正确的 Parameters 对象
            self.final_parameters = Parameters(
                tensors=aggregated_parameters[0].tensors,
                tensor_type="numpy.ndarray"
            )
        return aggregated_parameters
