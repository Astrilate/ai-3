from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
import torch
import datetime
import torchvision
from torch import nn
import numpy as np
from matplotlib import patches
import matplotlib.pyplot as plt
from PIL import Image
from scipy.optimize import linear_sum_assignment
from torchvision.ops.boxes import box_area
import torch.distributed as dist
import sys

sys.path.append(r'D:\Users\asus\Desktop\ai-cifar100')
from train import DualStreamNet

PASCAL_VOC_2012_PATH = "D:\\Users\\asus\\Desktop"

batch_size = 128
learning_rate = 0.0001
epochs = 600

train_transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize((64, 64)),
    torchvision.transforms.ToTensor(),
])


# 输入的不管是字符串还是数字都应为列表，输出也是列表，张量另行转换
def convert_labels(labels, to_text):
    class_to_label = {
        'aeroplane': 0, 'bicycle': 1, 'bird': 2, 'boat': 3, 'bottle': 4,
        'bus': 5, 'car': 6, 'cat': 7, 'chair': 8, 'cow': 9,
        'diningtable': 10, 'dog': 11, 'horse': 12, 'motorbike': 13, 'person': 14,
        'pottedplant': 15, 'sheep': 16, 'sofa': 17, 'train': 18, 'tvmonitor': 19,
        None: 20
    }
    label_to_class = {v: k for k, v in class_to_label.items()}
    if to_text:
        class_names = [label_to_class[label] for label in labels]
        return class_names
    else:
        changed_labels = [class_to_label[class_name] for class_name in labels]
        return changed_labels


def collate_fn(batch):
    images, targets = zip(*batch)  # 获取批次中的图像和标签，在这里两个都是batchsize长度，为元组，和后面循环中提取到的东西是一模一样的
    label = []
    resized_images = []
    for image, target in zip(images, targets):
        x = image.size[0]
        y = image.size[1]
        resized_image = train_transforms(image)
        resized_images.append(resized_image)
        cls, box, dic = [], [], {}
        for i in target["annotation"]["object"]:
            point = [int(i["bndbox"]["xmin"]) / x, int(i["bndbox"]["ymin"]) / y,
                     int(i["bndbox"]["xmax"]) / x, int(i["bndbox"]["ymax"]) / y]
            cls.append(i["name"])
            box.append(point)
        dic = {"cls": torch.tensor(convert_labels(cls, to_text=False)), "box": torch.tensor(box)}
        label.append(dic)
    feature = torch.stack(resized_images, dim=0)  # 生成装载图片的张量
    return feature, label


train_dataset = torchvision.datasets.VOCDetection(root=PASCAL_VOC_2012_PATH, year='2012', image_set='train')
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn,
                                           drop_last=True)

val_dataset = torchvision.datasets.VOCDetection(root=PASCAL_VOC_2012_PATH, year='2012', image_set='val')
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn,
                                         drop_last=False)


if torch.cuda.is_available():
    device = torch.device("cuda")
    print("cuda")
else:
    device = torch.device("cpu")
    print("no")


class DETR(nn.Module):
    def __init__(self, num_classes=20, hidden_dim=64, nheads=8, num_encoder_layers=3, num_decoder_layers=3):
        super().__init__()
        self.block1 = DualStreamNet().block1
        self.block2 = DualStreamNet().block2
        self.block3 = DualStreamNet().block3
        self.block4 = DualStreamNet().block4
        self.conv = nn.Conv2d(256, hidden_dim, 1)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.transformer = nn.Transformer(hidden_dim, nheads, num_encoder_layers, num_decoder_layers)
        self.linear_class = nn.Linear(hidden_dim, num_classes + 1)
        self.linear_bbox = nn.Linear(hidden_dim, 4)

        self.query_pos = nn.Parameter(torch.rand(10, batch_size, hidden_dim))  # 先输出10个框
        self.row_embed = nn.Parameter(torch.rand(16, hidden_dim // 2))  # 位置编码大了会炸显存
        self.column_embed = nn.Parameter(torch.rand(16, hidden_dim // 2))

    def forward(self, x):
        h = self.block1(x)
        h = self.block2(h)
        h = self.block3(h)
        h = self.block4(h)
        h = self.pool(self.conv(h))
        H, W = h.shape[-2:]
        pos = torch.cat([
            self.column_embed[:W].unsqueeze(0).repeat(H, 1, 1),  # 取位置编码前几列，大小为该图像需要的部分
            self.row_embed[:H].unsqueeze(1).repeat(1, W, 1),
        ], dim=-1).flatten(0, 1).unsqueeze(1)
        cat = pos + 0.1 * h.flatten(2).permute(2, 0, 1)  # 展平宽高并换位，特征张量加上位置编码
        target = self.query_pos  # 输出的目标
        h = self.transformer(cat, target).transpose(0, 1)  # torch.Size([bs, 10, 64])
        out = {'pred_logits': self.linear_class(h),  # torch.Size([bs, 10, 21])
               'pred_boxes': self.linear_bbox(h).sigmoid()}  # torch.Size([bs, 10, 4])
        return out


class detr_loss(nn.Module):
    def __init__(self):
        super(detr_loss, self).__init__()
        self.num_classes = 20
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = 0.1
        self.register_buffer('empty_weight', empty_weight)

    def loss_labels(self, outputs, targets, indices, num_boxes):
        src_logits = outputs['pred_logits']
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["cls"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes, dtype=torch.int64)
        target_classes[idx] = target_classes_o
        src_logits = src_logits.to("cpu")
        # torch.Size([32, 21, 10]) torch.Size([32, 10]) torch.Size([21])  21
        loss_ce = nn.functional.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {'loss_ce': loss_ce}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['box'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        src_boxes = src_boxes.to("cpu")
        loss_bbox = nn.functional.l1_loss(src_boxes, target_boxes, reduction='none')
        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes
        loss_giou = 1 - torch.diag(generalized_box_iou(src_boxes, target_boxes))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def _get_src_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def forward(self, outputs, targets):
        indices = match(outputs, targets)
        num_boxes = sum(len(t["cls"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()
        # 计算主网络损失
        losses = {}
        losses.update(self.loss_labels(outputs, targets, indices, num_boxes))
        losses.update(self.loss_boxes(outputs, targets, indices, num_boxes))
        Loss = losses["loss_ce"] * 1 + losses["loss_bbox"] * 5 + losses["loss_giou"] * 2
        return Loss


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def match(outputs, targets):
    bs, num_queries = outputs["pred_logits"].shape[:2]  # 2, 10
    # 把每个batch中所有图像的所有预测目标合并
    out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # [batchsize x 50预测目标数, 20]
    out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size x 50预测目标数, 4]
    # 把每个batch中所有图像的所有标签合并
    tgt_ids = torch.cat([v["cls"] for v in targets]).to(device)  # [总目标数]
    tgt_bbox = torch.cat([v["box"] for v in targets]).to(device)  # [总目标数, 4]

    cost_class = -out_prob[:, tgt_ids]
    cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
    cost_giou = - generalized_box_iou(out_bbox, tgt_bbox)
    C = cost_bbox * cost_bbox + cost_class * cost_class + cost_giou * cost_giou
    C = C.view(bs, num_queries, -1).cpu()  # [batchsize, 预测目标数10, 总目标数]
    sizes = [len(v["box"]) for v in targets]

    indices = [linear_sum_assignment(c[i].detach()) for i, c in enumerate(C.split(sizes, -1))]
    temp = [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]
    return temp


def generalized_box_iou(boxes1, boxes2):
    iou, union = box_iou(boxes1, boxes2)
    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]
    return iou - (area - union) / area


def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]
    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]
    union = area1[:, None] + area2 - inter
    iou = inter / union
    return iou, union


# def accuracy(loader):
#     ac_total = 0
#     total = 0
#     for i, (features, labels, locations) in enumerate(loader):
#         with torch.no_grad():
#             if i > 10:
#                 break
#             features = features.to(device)
#             labels = labels.to(device)
#             locations = locations.to(device)
#             predict, location = detr(features)
#             Indices = torch.argmax(predict, dim=1)
#             ac_total += torch.eq(Indices, labels).sum().item()
#             total += len(labels)
#             ls = lossF(predict, labels)
#             ls_box = lossF_box(location, locations.to(torch.float32))
#             total_loss = ls + ls_box * 50
#     return ac_total / total, total_loss


# if __name__ == '__main__':
#     detr = DETR()
#     detr = detr.to(device)
#     criterion = detr_loss()
#     optimizer = torch.optim.Adam(detr.parameters(), lr=learning_rate)
#     milestones = [450, 480]
#     scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
#     st = datetime.datetime.now()
#     for epoch in range(epochs):
#         total_loss = 0
#         count = 0
#         for i, (features, labels) in enumerate(train_loader):
#             features = features.to(device)  # 输入直接用原本的二维张量
#             output = detr(features)
#             loss = criterion(output, labels)
#             total_loss += loss
#             count += 1
#             loss.backward()
#             optimizer.step()
#             optimizer.zero_grad()
#         et = datetime.datetime.now()
#         Time = (et - st).seconds
#         scheduler.step()
#         print(f"epoch: {epoch + 1}, time:{Time}s, loss: {total_loss / count:.2f}")
#         # print(f"epoch: {epoch + 1}, time:{Time}s, train_loss: {train_loss:.2f}, "
#         #       f", train_acc: {train_acc :.2%}")
#     torch.save(detr.state_dict(), "./128.pth")


# 模型评估
detr = DETR()
detr.eval()
detr = detr.to(device)
detr.load_state_dict(torch.load("new.pth"))
for i, (features, labels) in enumerate(val_loader):
    with torch.no_grad():
        features = features.to(device)
        output = detr(features)
        probas = output['pred_logits'].softmax(-1)  # 类别维度softmax  [batchsize, 10, 20]  [batchsize, 10, 4]
        keep = probas.max(-1).values > 0.3  # 记录上面类别概率最大值  [batchsize, 10]
        for i in range(batch_size):
            Indices = torch.argmax(probas[i, keep[i]], dim=1)
            c = Indices.tolist()
            b = output['pred_boxes'][i, keep[i]].tolist()
            image_data = features[i].cpu()
            class_names = convert_labels(Indices.tolist(), to_text=True)
            plt.imshow(image_data.numpy().transpose((1, 2, 0)))
            for j in range(len(c)):
                name = class_names[j]
                if name is not None:
                    x1, y1, x2, y2 = b[j][0] * 64, b[j][1] * 64, b[j][2] * 64, b[j][3] * 64
                    bbox = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
                                             edgecolor='r', facecolor='none')  # 创建边界框
                    plt.gca().add_patch(bbox)
                    plt.text(x1, y1, name, color='r', fontsize=10)
            plt.axis('off')
            plt.show()
