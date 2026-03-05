import time
import torch
import torch.nn as nn
from multiprocessing import Process, set_start_method
import os
from multiprocessing import Process
import flwr as fl
import start_server.start_server
import start_client.start_client
import get_model.get_model
from flwr.common import Parameters, parameters_to_ndarrays

# 保证在主进程中设置 spawn
if __name__ == "__main__":
    Platform_X_res=pd.read_csv('Platform_X_res.csv')
    Platform_y_res=pd.read_csv('Platform_y_res.csv')
    nunique_counts = Platform_X_res.nunique()
    binary_columns = Platform_X_res.loc[:, nunique_counts == 2]
    other_columns = Platform_X_res.loc[:, nunique_counts > 2]
    train_loader,test_loader=loader(other_columns,binary_columns,Platform_y_res.squeeze(),batch_size=1024)
    loader(other_columns,binary_columns,y,batch_size=32)

    dataset = train_loader.dataset  # 你自己的数据集
    lengths = [int(len(dataset) / 3)] * 2 + [len(dataset) - 2 * int(len(dataset) / 3)]
    client_datasets = random_split(dataset, lengths)
    # 假设 batch_size 是你想要的批量大小
    batch_size = 64

    # 用列表推导重新包装为 dataloader
    client_loaders = [DataLoader(ds, batch_size=batch_size, shuffle=True) for ds in client_datasets]
    num_clients=3

    # 设置 start 方法为 'spawn'，防止 CUDA 报错
    categories=(2,2)
    num_continuous=37
    global_model = get_model(categories,num_continuous).to(device)

    set_start_method("spawn", force=True)

    # 启动 server
    server_process = Process(target=start_server)
    server_process.start()

    # 等待 server 启动
    time.sleep(2)

    # 启动多个客户端
    clients = []
    for i in range(num_clients):
        p = Process(
            target=start_client,
            args=(client_loaders[i], categories, num_continuous, nn.BCEWithLogitsLoss(), 60)
        )
        p.start()
        clients.append(p)

    # 等待所有客户端结束
    for p in clients:
        p.join()

    # 等待 server 结束
    server_process.join()

    # 允许加载 Flower 的 Parameters 类型
    torch.serialization.add_safe_globals([Parameters])
    if os.path.exists("final_params.pth"):
        try:
            loaded = torch.load("final_params.pth")
        
            # 处理两种可能性：元组或 Parameters 对象
            if isinstance(loaded, tuple):
                # 如果是元组，手动构造 Parameters 对象
                flower_params = Parameters(
                    tensors=[t.numpy() if torch.is_tensor(t) else t for t in loaded],
                    tensor_type="numpy.ndarray"
                )
            elif isinstance(loaded, Parameters):
                flower_params = loaded
            else:
                raise ValueError("未知的参数格式")

            # 转换为 NumPy 数组
            numpy_arrays = parameters_to_ndarrays(flower_params)
        
            # 转换为 PyTorch Tensor
            state_dict = {
                k: torch.from_numpy(v) if isinstance(v, np.ndarray) else v
                for k, v in zip(global_model.state_dict().keys(), numpy_arrays)
            }
        
            global_model.load_state_dict(state_dict, strict=False)
            torch.save(global_model.state_dict(), "global_model.pth")
            print("🌍 Global model saved to global_model.pth")
        
        except Exception as e:
            print(f"⚠️ 加载参数失败: {str(e)}")
    else:
        print("⚠️ 未找到 final_params.pth")

