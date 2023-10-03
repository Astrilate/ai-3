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
# # 多头注意力机制，暂定(TransformerEncoderLayer?有空再说)
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


import torch
import numpy as np

# 假设你有以下两个输出张量
class_scores = torch.rand(3, 10, 21)  # 类别分数张量
bbox_coordinates = torch.rand(3, 10, 4)  # 目标框坐标张量

confidence_threshold = 0.05  # 置信度阈值

# 对类别分数进行softmax，转换为概率分布
class_probs = torch.softmax(class_scores, dim=2)

# 初始化一个字典来保存每个类别的预测结果
class_predictions = {}

# 假设你有一个包含每个目标框所属图像索引的张量
image_indices = torch.arange(3).view(-1, 1).expand(3, 10)

# 遍历每个类别
for class_idx in range(21):
    # 选择置信度高于阈值的目标框
    mask = class_probs[:, :, class_idx] > confidence_threshold

    # 提取符合条件的目标框坐标、置信度和对应的图像索引
    filtered_coordinates = bbox_coordinates[mask]
    filtered_confidences = class_probs[:, :, class_idx][mask]
    filtered_image_indices = image_indices[mask]

    # 对目标框按置信度进行排序
    sorted_indices = torch.argsort(filtered_confidences, descending=True)
    sorted_coordinates = filtered_coordinates[sorted_indices]
    sorted_image_indices = filtered_image_indices[sorted_indices]

    # 将排序后的目标框坐标、置信度和图像索引保存到字典中
    class_predictions[class_idx] = {
        'coordinates': sorted_coordinates,
        'confidences': filtered_confidences[sorted_indices],
        'image_indices': sorted_image_indices
    }


confidence = class_predictions[0]['confidences'].numpy()
BB = class_predictions[0]['coordinates'].numpy()
image_ids = class_predictions[0]['image_indices'].numpy()


# confidence = np.array([0.9, 0.8, 0.7, 0.6])  # 置信度分数
# BB = np.array([[10, 10, 50, 50], [20, 20, 60, 60], [15, 15, 45, 45], [30, 30, 70, 70]])  # 预测框坐标
# image_ids = [1, 2, 1, 3]  # 图像ID

# 示例class_recs字典，包含每个图像的信息




# 先搞定这个
class_recs = {
    0: {'bbox': np.array([[10, 10, 50, 50], [15, 15, 45, 45]]),  'det': np.array([False, False])},
    1: {'bbox': np.array([[20, 20, 60, 60]]),  'det': np.array([False])},
    2: {'bbox': np.array([[40, 40, 80, 80]]),  'det': np.array([False])}
}




ovthresh = 0.01  # 交并比阈值，判断是否检测到了
npos = 4  # 所有的正样本总数



# 按照置信度降序排序
sorted_ind = np.argsort(-confidence)
BB = BB[sorted_ind, :]   # 预测框坐标
image_ids = [image_ids[x] for x in sorted_ind] # 各个预测框的对应图片id# 便利预测框，并统计TPs和FPs
nd = len(image_ids)
tp = np.zeros(nd)
fp = np.zeros(nd)
for d in range(nd):
    R = class_recs[image_ids[d]]
    bb = BB[d, :].astype(float)
    ovmax = -np.inf
    BBGT = R['bbox'].astype(float)  # ground truth

    if BBGT.size > 0:
        # 计算IoU
        # intersection
        ixmin = np.maximum(BBGT[:, 0], bb[0])
        iymin = np.maximum(BBGT[:, 1], bb[1])
        ixmax = np.minimum(BBGT[:, 2], bb[2])
        iymax = np.minimum(BBGT[:, 3], bb[3])
        iw = np.maximum(ixmax - ixmin + 1., 0.)
        ih = np.maximum(iymax - iymin + 1., 0.)
        inters = iw * ih        # union
        uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
               (BBGT[:, 2] - BBGT[:, 0] + 1.) *
               (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)

        overlaps = inters / uni
        ovmax = np.max(overlaps)
        jmax = np.argmax(overlaps)    # 取最大的IoU

    if ovmax > ovthresh:  # 是否大于阈值
        if not R['det'][jmax]:    # 未被检测
            tp[d] = 1.
            R['det'][jmax] = 1    # 标记已被检测
        else:
            fp[d] = 1.
    else:
        fp[d] = 1.  # 计算precision recallfp = np.cumsum(fp)
tp = np.cumsum(tp)
rec = tp / float(npos)  # avoid divide by zero in case the first detection matches a difficult# ground truth
prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)


def voc_ap(rec, prec):
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], rec, [1.]))
    mpre = np.concatenate(([0.], prec, [0.]))

    # compute the precision 曲线值（也用了插值）
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


print(prec, rec)  # 列表中有（符合置信度阈值的所有目标框）个元素
print(voc_ap(rec, prec))
