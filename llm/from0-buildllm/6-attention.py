
# !pip install torch
# !pip install d2l
import torch
from torch import nn
import torch.nn.functional as F
try:
    from d2l import torch as d2l
except:
    print("d2l not installed")

n_train = 50  # 训练样本数
test_len = 10
x_train, _ = torch.sort(torch.rand(n_train) * test_len)   # 排序后的训练样本

def f(x):
    return 2 * torch.sin(x) + x**0.8

y_train = f(x_train) + torch.normal(0.0, 0.5, (n_train,))  # 训练样本的输出
x_test = torch.arange(0, test_len, 0.1)  # 测试样本
y_truth = f(x_test)  # 测试样本的真实输出
n_test = len(x_test)  # 测试样本数
n_test

def plot_kernel_reg(y_hat):
        d2l.plot(x_test, [y_truth, y_hat], 'query', 'key,value', legend=['Truth', 'Pred'],
                xlim=[0, test_len], ylim=[-1, test_len])
        d2l.plt.plot(x_train, y_train, 'o', alpha=0.5)

def attention_now():
    y_hat = torch.repeat_interleave(y_train.mean(), n_test)
    plot_kernel_reg(y_hat)

    # X_repeat的形状:(n_test,n_train),
    # 每一行都包含着相同的测试输入（例如：同样的查询）
    X_repeat = x_test.repeat_interleave(n_train).reshape((-1, n_train))
    # x_train包含着键。attention_weights的形状：(n_test,n_train),
    # 每一行都包含着要在给定的每个查询的值（y_train）之间分配的注意力权重
    attention_weights = nn.functional.softmax(-(X_repeat - x_train)**2 / 2, dim=1)
    # y_hat的每个元素都是值的加权平均值，其中的权重是注意力权重
    y_hat = torch.matmul(attention_weights, y_train)
    plot_kernel_reg(y_hat)

class NWKernelRegression(nn.Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.w = nn.Parameter(torch.rand((1,), requires_grad=True))

    def forward(self, queries, keys, values):
        # queries和attention_weights的形状为(查询个数，“键－值”对个数)
        queries = queries.repeat_interleave(keys.shape[1]).reshape((-1, keys.shape[1]))
        self.attention_weights = nn.functional.softmax(
            -((queries - keys) * self.w)**2 / 2, dim=1)
        # values的形状为(查询个数，“键－值”对个数)
        return torch.bmm(self.attention_weights.unsqueeze(1),
                         values.unsqueeze(-1)).reshape(-1)
    
def attention_w(epochs = 5):
    # X_tile的形状:(n_train，n_train)，每一行都包含着相同的训练输入
    X_tile = x_train.repeat((n_train, 1))
    # Y_tile的形状:(n_train，n_train)，每一行都包含着相同的训练输出
    Y_tile = y_train.repeat((n_train, 1))
    # keys的形状:('n_train'，'n_train'-1)
    keys = X_tile[(1 - torch.eye(n_train)).type(torch.bool)].reshape((n_train, -1))
    # values的形状:('n_train'，'n_train'-1)
    values = Y_tile[(1 - torch.eye(n_train)).type(torch.bool)].reshape((n_train, -1))

    net = NWKernelRegression()
    loss = nn.MSELoss(reduction='none')
    trainer = torch.optim.SGD(net.parameters(), lr=0.5)

    for epoch in range(epochs):
        trainer.zero_grad()
        l = loss(net(x_train, keys, values), y_train)
        l.sum().backward()
        trainer.step()
        print(f'epoch {epoch + 1}, loss {float(l.sum()):.6f}')

    # keys的形状:(n_test，n_train)，每一行包含着相同的训练输入（例如，相同的键）
    keys = x_train.repeat((n_test, 1))
    # value的形状:(n_test，n_train)
    values = y_train.repeat((n_test, 1))
    y_hat = net(x_test, keys, values).unsqueeze(1).detach()
    plot_kernel_reg(y_hat)


def transpose_qkv(X, num_heads):
    """为了多注意力头的并行计算而变换形状"""
    # 输入X的形状:(batch_size，查询或者“键－值”对的个数，num_hiddens)
    # 输出X的形状:(batch_size，查询或者“键－值”对的个数，num_heads，
    # num_hiddens/num_heads)
    X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)
    # 输出X的形状:(batch_size，num_heads，查询或者“键－值”对的个数,
    # num_hiddens/num_heads)
    X = X.permute(0, 2, 1, 3)
    # 最终输出的形状:(batch_size*num_heads,查询或者“键－值”对的个数,
    # num_hiddens/num_heads)
    return X.reshape(-1, X.shape[2], X.shape[3])

def transpose_output(X, num_heads):
    """逆转transpose_qkv函数的操作"""
    X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])
    X = X.permute(0, 2, 1, 3)
    return X.reshape(X.shape[0], X.shape[1], -1)

class MultiHeadAttention(nn.Module):
    """多头注意力"""
    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 num_heads, dropout, bias=False, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.attention = d2l.DotProductAttention(dropout) # 这里用到缩放点积注意力
        self.W_q = nn.Linear(query_size, num_hiddens, bias=bias)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=bias)
        self.W_v = nn.Linear(value_size, num_hiddens, bias=bias)
        self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=bias)

    def forward(self, queries, keys, values, valid_lens):
        # queries，keys，values的形状:
        # (batch_size，查询或者“键－值”对的个数，num_hiddens)
        # valid_lens　的形状:
        # (batch_size，)或(batch_size，查询的个数)
        # 经过变换后，输出的queries，keys，values　的形状:
        # (batch_size*num_heads，查询或者“键－值”对的个数，
        # num_hiddens/num_heads)
        queries = transpose_qkv(self.W_q(queries), self.num_heads)
        keys = transpose_qkv(self.W_k(keys), self.num_heads)
        values = transpose_qkv(self.W_v(values), self.num_heads)

        if valid_lens is not None:
            # 在轴0，将第一项（标量或者矢量）复制num_heads次，
            # 然后如此复制第二项，然后诸如此类。
            valid_lens = torch.repeat_interleave(
                valid_lens, repeats=self.num_heads, dim=0)

        # output的形状:(batch_size*num_heads，查询的个数，
        # num_hiddens/num_heads)
        output = self.attention(queries, keys, values, valid_lens)

        # output_concat的形状:(batch_size，查询的个数，num_hiddens)
        output_concat = transpose_output(output, self.num_heads)
        return self.W_o(output_concat)

def attention_nhead():
    num_hiddens, num_heads = 100, 5
    attention = MultiHeadAttention(num_hiddens, num_hiddens, num_hiddens,
                                num_hiddens, num_heads, 0.5)
    attention.eval()
    batch_size, num_queries = 2, 4
    num_kvpairs, valid_lens =  6, torch.tensor([3, 2])
    X = torch.ones((batch_size, num_queries, num_hiddens))
    Y = torch.ones((batch_size, num_kvpairs, num_hiddens))
    output = attention(X, Y, Y, valid_lens)
    print(output)

def attention_scaled_dot_product():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Example Usage:
    query, key, value = torch.randn(2, 3, 8, device=device), torch.randn(2, 3, 8, device=device), torch.randn(2, 3, 8, device=device)
    output = F.scaled_dot_product_attention(query, key, value)
    print(output)

attention_scaled_dot_product()