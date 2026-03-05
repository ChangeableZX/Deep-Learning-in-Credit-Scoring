# 定义 TabTransformer 模型结构
def get_model(categories,num_continuous):
    model = TabTransformer(
    categories=categories,
    num_continuous = num_continuous,                # number of continuous values
    dim = 32,                           # dimension, paper set at 32
    dim_out = 1,                        # binary prediction, but could be anything
    depth = 6,                          # depth, paper recommended 6
    heads = 8,                          # heads, paper recommends 8
    attn_dropout = 0.1,                 # post-attention dropout
    ff_dropout = 0.1,                   # feed forward dropout
    mlp_hidden_mults = (8 ,4, 2),          # relative multiples of each hidden dimension of the last mlp to logits
    mlp_act = nn.ReLU(),                # activation for final mlp, defaults to relu, but could be anything else (selu etc)
    #     continuous_mean_std = cont_mean_std # (optional) - normalize the continuous values before layer norm
    )
    return model