import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class TransformerConfig:
    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        d_model=512,
        n_head=8,
        num_layers=6,
        d_ff=2048,
        dropout=0.1,
        max_len=5000,
    ):
        """

        :param src_vocab_size: 源语言词表大小
        :param tgt_vocab_size: 目标语言词表大小
        :param d_model: 嵌入维度
        :param n_head: 注意力头数目
        :param num_layers: Encoder/Decoder 的层数
        :param d_ff: 前馈网络隐藏层维度
        :param dropout: 丢弃率
        :param max_len: 序列最大长度
        """
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.d_model = d_model
        self.n_head = n_head
        self.num_layers = num_layers
        self.d_ff = d_ff
        self.dropout = dropout
        self.max_len = max_len


class PositionalEncoding(nn.Module):
    def __init__(self, d_model=512, dropout=0.1, max_len=5000):
        """
        :param d_model: 嵌入维度
        :param dropout: 丢弃率
        :param max_len: 序列最大长度
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        # 创建位置编码矩阵 (max_len, d_model) 这个矩阵会存储所有可能位置的编码
        pe = torch.zeros(max_len, d_model)

        # 创建位置索引 [0, 1, 2, ..., max_len-1]
        # 为什么是浮点数? 位置编码是浮点运算;与模型其他部分类型一致;避免隐式类型转换的开销
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # 计算频率分母项
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000) / d_model)
        )

        # 计算所有位置的正弦和余弦编码
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # 为 pe 添加 batch 维度
        pe = pe.unsqueeze(0)

        # 将 pe 注册为 buffer
        # 创建属性 self.pe，值为传入的张量 pe 第一个参数 'pe' 是字符串，决定了属性的名字是 self.pe
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        前向传播

        :param x: 输入张量，shape 为 (batch_size, seq_len, d_model)
        :return 添加了位置编码后的张量，shape 不变
        """
        x = x + self.pe[:, : x.size(1), :]

        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        """
        :param d_model: 模型维度
        :param n_heads: 注意力头数
        """
        super().__init__()

        # d_model 必须能被 n_head 整除（d_k = d_model // n_head）
        assert d_model % n_heads == 0, "d_model必须能被num_heads整除"

        self.d_model = d_model
        self.n_heads = n_heads

        self.d_k = self.d_model // self.n_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(p=0.1)

    def split_heads(self, x: torch.Tensor):
        """
        把最后一维拆分成 (num_heads, d_k)
        Input:  (batch, seq_len, d_model)
        Output: (batch, num_heads, seq_len, d_k)
        """
        # 获得输入的batch size大小
        N = x.size(0)
        x = x.view(N, -1, self.n_heads, self.d_k)
        x = x.transpose(1, 2)
        return x

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor = None,
    ):
        """
        Args:
            query, key, value: (batch, seq_len, d_model)
            mask: (batch, 1, 1, seq_len) 或 (batch, 1, seq_len, seq_len)
        Returns:
            output: (batch, seq_len, d_model)
            attention_weights: (batch, num_heads, seq_len, seq_len)
        """
        N = query.size(0)

        # 线性投影 + 拆分成多头
        Q = self.split_heads(self.W_q(query))
        K = self.split_heads(self.W_k(key))
        V = self.split_heads(self.W_v(value))

        # 缩放点积注意力
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        # 应用mask
        if mask is not None:
            # float('-inf') 在Python中表示负无穷大
            scores = scores.masked_fill(mask == 0, float("-inf"))
        # Softmax + 加权求和
        attention_weights = torch.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        context = torch.matmul(attention_weights, V)
        # 合并多头
        # view() 要求tensor在内存中是连续的,而transpose/permute只改变元数据,不重排内存。
        # contiguous()会真正复制数据到新的连续内存块。
        context = context.transpose(1, 2).contiguous()
        context = context.view(N, -1, self.d_model)
        # 最终线性投影

        output = self.W_o(context)

        return output, attention_weights


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.ffn1 = nn.Linear(d_model, d_ff)
        self.ffn2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.ffn2(self.dropout(self.relu(self.ffn1(x))))

        return x


class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_head: int, d_ff=int, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model=d_model, n_heads=n_head)
        self.feed_forward = PositionwiseFeedForward(
            d_model=d_model, d_ff=d_ff, dropout=dropout
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, mask):
        # Pre-LN
        norm_x = self.norm1(x)
        attn_output, _ = self.self_attn(norm_x, norm_x, norm_x, mask)
        x = x + self.dropout(attn_output)

        norm_x = self.norm2(x)
        ff_output = self.feed_forward(norm_x)
        x = x + self.dropout(ff_output)

        return x
