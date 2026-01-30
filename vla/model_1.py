import torch
import torch.nn as nn
import torch.nn.functional as F


class Model_1_Config:

    input_size = 7
    output_size = 7

    cross_attn_input_size = 6

    d_model = 256
    nhead = 8
    num_blocks = 6
    dim_feedforward = 512
    dropout = 0.1






class MHAttentionBlock(nn.Module):
    def __init__(self, embed_dim, config: Model_1_Config):
        super(MHAttentionBlock, self).__init__()
        self.wq = nn.Linear(embed_dim, embed_dim)
        self.qk = nn.Linear(embed_dim, embed_dim)
        self.wv = nn.Linear(embed_dim, embed_dim)
        self.wo = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(config.dropout)
        self.rms = nn.RMSNorm(embed_dim)
        assert embed_dim % config.nhead == 0, "embed_dim must be divisible by nhead"
        self.d_k = embed_dim // config.nhead

        

    def get_mh_attention(self, q, k, v, mask=False):
        attn_scores: torch.Tensor = torch.matmul(q, k.transpose(-2, -1)) / (self.d_k ** 0.5)
        if mask:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        attn_scores = F.softmax(attn_scores, dim=-1)
        attn_scores = self.dropout(attn_scores)
        return torch.matmul(attn_scores, v)

    def forward(self, xq, xkv, mask=False):
        B, T, C = xq.size()
        q = self.wq(xq).reshape(B, T, -1, self.d_k).transpose(1, 2) # (B, nhead, T, d_k)
        k = self.qk(xkv).reshape(B, T, -1, self.d_k).transpose(1, 2)  # (B, nhead, T, d_k)
        v = self.wv(xkv).reshape(B, T, -1, self.d_k).transpose(1, 2)  # (B, nhead, T, d_k)
        attn_output = self.get_mh_attention(q, k, v, mask)  # (B, nhead, T, d_k)
        attn_output = attn_output.transpose(1, 2).reshape(B, T, C)  # (B, T, C)

        out = self.wo(attn_output)
        out = self.dropout(out)
        out = self.rms(out + xq)  # Residual connection

        return out


class TransformerBlock(nn.Module):
    def __init__(self, config: Model_1_Config):
        super(TransformerBlock, self).__init__()
        self.mh_attention = MHAttentionBlock(config.d_model, config)

        self.ff1 = nn.Linear(config.d_model, config.dim_feedforward)
        self.ff2 = nn.Linear(config.dim_feedforward, config.d_model)
        self.ff_rms = nn.RMSNorm(config.d_model)
        self.ff_dropout = nn.Dropout(config.dropout)

    def forward(self, xq, xkv, mask=False):
        x = self.mh_attention(xq, xkv, mask)

        ff_out = F.relu(self.ff1(x))
        ff_out = self.ff2(ff_out)
        ff_out = self.ff_dropout(ff_out)
        out = self.ff_rms(ff_out + x)  # Residual connection

        return out








class Model_1(nn.Module):
    def __init__(self,):
        super(Model_1, self).__init__()
        self.config = Model_1_Config()
        self.input_project = nn.Linear(self.config.input_size, self.config.d_model)
        self.cross_attn_project = nn.Linear(self.config.cross_attn_input_size, self.config.d_model)

        self.transformer_blocks = nn.ModuleList([TransformerBlock(self.config) for _ in range(self.config.num_blocks)])

        self.output_project = nn.Linear(self.config.d_model, self.config.output_size)

    def forward(self, x, cross_attn_x, mask=False):
        """
        x: (B, T, input_size)
        cross_attn_x: (B, T, cross_attn_input_size)
        """
        x = self.input_project(x)
        cross_attn_x = self.cross_attn_project(cross_attn_x)

        for block in self.transformer_blocks:
            x = block(x, cross_attn_x, mask)

        out = self.output_project(x)
        return out

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = Model_1().to(device)
    x = torch.randn(4, 10, model.config.input_size).to(device)
    cross_attn_x = torch.randn(4, 10, model.config.cross_attn_input_size).to(device)
    out = model(x, cross_attn_x)
    print(out.shape)  # Expected: (4, 10, output_size)
