# class MLP(nn.Module):
#     def __init__(self, dim, hidden):
#         super(MLP, self).__init__()
#         self.fc1 = nn.Linear(dim, hidden)
#         self.fc2 = nn.Linear(hidden, dim)
#         self.gelu = nn.GELU()
#         self.dropout = nn.Dropout(0.3)
#
#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.gelu(x)
#         x = self.dropout(x)
#         x = self.fc2(x)
#         x = self.gelu(x)
#         return x
#
#
# # 多头注意力机制，(TransformerEncoderLayer?)
# class Attention(nn.Module):
#     def __init__(self, dim, num_heads, dim_head=32, dropout=0.3):
#         super().__init__()
#         inner_dim = dim_head * num_heads
#         project_out = not (num_heads == 1 and dim_head == dim)
#         self.heads = num_heads
#         self.scale = dim_head ** -0.5
#         self.norm = nn.LayerNorm(dim)
#         self.attend = nn.Softmax(dim=-1)
#         self.dropout = nn.Dropout(dropout)
#         self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
#         self.to_out = nn.Sequential(
#             nn.Linear(inner_dim, dim),
#             nn.Dropout(dropout)
#         ) if project_out else nn.Identity()
#
#     def forward(self, x):
#         x = self.norm(x)
#         qkv = self.to_qkv(x).chunk(3, dim=-1)
#         q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
#         dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
#         attn = self.attend(dots)
#         attn = self.dropout(attn)
#         out = torch.matmul(attn, v)
#         out = rearrange(out, 'b h n d -> b n (h d)')
#         return self.to_out(out)
#
#
# # 单个transformer模块
# class transformer_block(nn.Module):
#     def __init__(self, dim, hidden, num_heads):
#         super(transformer_block, self).__init__()
#         self.ln1 = nn.LayerNorm(dim)
#         self.ln2 = nn.LayerNorm(dim)
#         self.MultiHeadAttention = Attention(dim, num_heads)
#         self.mlp = MLP(dim, hidden)
#
#     def forward(self, x):
#         x1 = self.ln1(x)
#         x1 = self.MultiHeadAttention(x1)
#         x1 += x
#         x2 = self.ln2(x1)
#         x2 = self.mlp(x2)
#         x2 += x1
#         return x2



# 旧版vit
# def position_embedding(h, w, dim, temperature: int = 10000, dtype=torch.float32):
#     y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
#     omega = torch.arange(dim // 4) / (dim // 4 - 1)
#     omega = 1.0 / (temperature ** omega)
#     y = y.flatten()[:, None] * omega[None, :]
#     x = x.flatten()[:, None] * omega[None, :]
#     pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)
#     return pe.type(dtype)
#
#
# class VIT(nn.Module):
#     def __init__(self, layers=6, dim=512, hidden=2048, num_heads=8, image_size=500, patch_size=100):  # 分为25块
#         super(VIT, self).__init__()
#         # patch_dim代表图像块被展平后的维度(展平：通道数x总像素)，这个维度用于将图像块映射为嵌入向量的中间步骤
#         self.patch_dim = 3 * patch_size * patch_size
#         self.patch_embedding = nn.Sequential(
#             # 将输入的图像块从二维张量重新排列成一维的块
#             Rearrange("b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=patch_size, p2=patch_size),
#             nn.LayerNorm(self.patch_dim),
#             nn.Linear(self.patch_dim, dim),  # 线性变换投影到高维度，从而生成嵌入向量
#             nn.LayerNorm(dim),
#         )
#         self.position_embedding = position_embedding(h=image_size // patch_size, w=image_size // patch_size, dim=dim)
#         self.transformer_layer = nn.TransformerEncoderLayer(dim, num_heads, hidden)
#         self.transformer = nn.TransformerEncoder(self.transformer_layer, layers)
#         self.pool = "mean"
#         self.linear_boundbox = nn.Linear(dim, 4)
#         self.linear_head = nn.Linear(dim, 20)
#
#     def forward(self, image):
#         x = self.patch_embedding(image)
#         x += self.position_embedding.to(device, dtype=x.dtype)
#         x = self.transformer(x)  # torch.Size([200batchsize, 25sequence, 512dim])
#         x = x.mean(dim=1)  # torch.Size([200, 512])
#         return self.linear_head(x), self.linear_boundbox(x).sigmoid()
