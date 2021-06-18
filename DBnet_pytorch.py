import os
import cv2
import torch
import pyclipper
import numpy as np
import Polygon as plg
import torch.nn as nn
import torch.optim as optim
from shapely.geometry import Polygon
from torch.utils.data import Dataset, DataLoader


# 计算两点距离
def dist(a, b):
    return np.sqrt(np.sum((a - b) ** 2))


# 计算周长
def perimeter(bbox):
    peri = 0.0
    for i in range(bbox.shape[0]):
        peri += dist(bbox[i], bbox[(i + 1) % bbox.shape[0]])
    return peri


# 将标签轮廓向内缩小
def shrink(bbox, rate):
    area = plg.Polygon(bbox).area()
    peri = perimeter(bbox)
    pco = pyclipper.PyclipperOffset()
    pco.AddPath(bbox, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
    d = int(area * (1 - rate * rate) / peri)
    shrinked_bbox = pco.Execute(-d)
    shrinked_bbox = np.array(shrinked_bbox)[0]
    return shrinked_bbox


# 将预测结果往外扩
def shrink_out(bbox, rate):
    area = plg.Polygon(bbox).area()
    peri = perimeter(bbox)
    pco = pyclipper.PyclipperOffset()
    pco.AddPath(bbox, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
    d = int(area * rate / peri)
    shrinked_bbox = pco.Execute(d)
    shrinked_bbox = np.array(shrinked_bbox)[0]
    return shrinked_bbox


# 以矩阵方式计算点到线段的距离
def distance_matrix(xs, ys, a, b):
    x1, y1 = a[0], a[1]
    x2, y2 = b[0], b[1]
    u1 = (((xs - x1) * (x2 - x1)) + ((ys - y1) * (y2 - y1)))
    u = u1 / (np.square(x1 - x2) + np.square(y1 - y2))
    u[u <= 0] = 2
    ix = x1 + u * (x2 - x1)
    iy = y1 + u * (y2 - y1)
    distance = np.sqrt(np.square(xs - ix) + np.square(ys - iy))
    distance2 = np.sqrt(np.fmin(np.square(xs - x1) + np.square(ys - y1), np.square(xs - x2) + np.square(ys - y2)))
    distance[u >= 1] = distance2[u >= 1]
    return distance


# 计算点到各线段的最小距离
def draw_border_map(polygon, canvas, mask, shrink_ratio):
    polygon = np.array(polygon)
    assert polygon.ndim == 2
    assert polygon.shape[1] == 2
    polygon_shape = Polygon(polygon)
    if polygon_shape.area <= 0:
        return
    distance = polygon_shape.area * (1 - np.power(shrink_ratio, 2)) / polygon_shape.length
    subject = [tuple(l) for l in polygon]
    padding = pyclipper.PyclipperOffset()
    padding.AddPath(subject, pyclipper.JT_ROUND,
                    pyclipper.ET_CLOSEDPOLYGON)
    padded_polygon = np.array(padding.Execute(distance)[0])  # 往外扩
    cv2.fillPoly(mask, [padded_polygon.astype(np.int32)], 1.0)
    xmin = padded_polygon[:, 0].min()
    xmax = padded_polygon[:, 0].max()
    ymin = padded_polygon[:, 1].min()
    ymax = padded_polygon[:, 1].max()
    width = xmax - xmin + 1
    height = ymax - ymin + 1
    polygon[:, 0] = polygon[:, 0] - xmin  # 原始的polygon坐标平移到这个正的矩形框内
    polygon[:, 1] = polygon[:, 1] - ymin
    xs = np.broadcast_to(np.linspace(0, width - 1, num=width).reshape(1, width), (height, width))
    ys = np.broadcast_to(np.linspace(0, height - 1, num=height).reshape(height, 1), (height, width))
    distance_map = np.zeros(
        (polygon.shape[0], height, width), dtype=np.float32)
    for i in range(polygon.shape[0]):
        j = (i + 1) % polygon.shape[0]
        absolute_distance = distance_matrix(xs, ys, polygon[i], polygon[j])
        distance_map[i] = np.clip(absolute_distance / distance, 0, 1)
    distance_map = np.min(distance_map, axis=0)
    xmin_valid = min(max(0, xmin), canvas.shape[1] - 1)
    xmax_valid = min(max(0, xmax), canvas.shape[1] - 1)
    ymin_valid = min(max(0, ymin), canvas.shape[0] - 1)
    ymax_valid = min(max(0, ymax), canvas.shape[0] - 1)
    # 距离原始polygon越近值越接近1，超出distance的值都为0
    canvas[ymin_valid:ymax_valid + 1, xmin_valid:xmax_valid + 1] = np.fmax(
        1 - distance_map[
            ymin_valid - ymin:ymax_valid - ymax + height,
            xmin_valid - xmin:xmax_valid - xmax + width],
        canvas[ymin_valid:ymax_valid + 1, xmin_valid:xmax_valid + 1])


# 生成阈值图label
def make_threshmap(img, text_polys, shrink_ratio=0.4, thresh_min=0.3, thresh_max=0.7):
    threshmap = np.zeros(img.shape[:2], dtype=np.float32)
    mask = np.zeros(img.shape[:2], dtype=np.float32)
    for i in range(len(text_polys)):
        draw_border_map(text_polys[i], threshmap, mask=mask, shrink_ratio=shrink_ratio)
    threshmap = threshmap * (thresh_max - thresh_min) + thresh_min  # 归一化到0.3到0.7之内
    return threshmap


# 数据读取
class MyDataset(Dataset):
    def __init__(self, base_path):
        imgs = []
        labels = []
        img_dir = os.path.join(base_path, 'image')
        label_dir = os.path.join(base_path, 'label')
        img_list = os.listdir(img_dir)
        for i in range(len(img_list)):
            img_path = os.path.join(img_dir, img_list[i])
            label_path = os.path.join(label_dir, img_list[i].replace('png', 'txt'))
            imgs.append(img_path)
            labels.append(label_path)
        self.imgs_path = imgs
        self.labels_path = labels

    def __getitem__(self, index):
        img_path, label_path = self.imgs_path[index], self.labels_path[index]
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        with open(label_path) as f:
            data = f.readlines()
        gt_boxes_all = []
        height, width, _ = img.shape
        # 宽高调整为32的倍数
        new_height = int(height / 32) * 32
        scale_y = new_height / height
        new_width = int(width / 32) * 32
        scale_x = new_width / width
        img = cv2.resize(img, (new_width, new_height))
        for i in range(len(data)):
            gt_data = data[i].strip().split()
            x_list = []
            x_list.append(int(int(gt_data[0]) * scale_x))
            x_list.append(int(int(gt_data[2]) * scale_x))
            x_list.append(int(int(gt_data[4]) * scale_x))
            x_list.append(int(int(gt_data[6]) * scale_x))
            y_list = []
            y_list.append(int(int(gt_data[1]) * scale_y))
            y_list.append(int(int(gt_data[3]) * scale_y))
            y_list.append(int(int(gt_data[5]) * scale_y))
            y_list.append(int(int(gt_data[7]) * scale_y))
            gt_boxes = []
            for j in range(len(x_list)):
                gt_boxes.append([x_list[j], y_list[j]])
            gt_boxes = np.array(gt_boxes, dtype=np.int32)
            gt_boxes_all.append(gt_boxes)
        shrink_map = np.zeros(shape=(img.shape[0], img.shape[1]), dtype=np.float32)
        for gt_boxes in gt_boxes_all:
            poly = shrink(gt_boxes, 0.4)
            cv2.fillPoly(shrink_map, [poly], (1.0))
        threshold_map = make_threshmap(img=img, text_polys=gt_boxes_all)
        img = np.transpose(img, (2, 0, 1))
        img = np.array(img / 255, dtype=np.float32)
        shrink_map = np.expand_dims(shrink_map, axis=0)
        threshold_map = np.expand_dims(threshold_map, axis=0)
        return img, shrink_map, threshold_map

    def __len__(self):
        return len(self.imgs_path)


# Model
class ResNet50BasicBlock(nn.Module):
    def __init__(self, in_c, mid_c, out_c):
        super(ResNet50BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_c, out_channels=mid_c, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(mid_c)
        self.conv2 = nn.Conv2d(in_channels=mid_c, out_channels=mid_c, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(mid_c)
        self.conv3 = nn.Conv2d(in_channels=mid_c, out_channels=out_c, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out = x + out
        out = self.relu(out)
        return out


class ResNet50DownBlock(nn.Module):
    def __init__(self, in_c, mid_c, out_c, stride=1):
        super(ResNet50DownBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_c, out_channels=mid_c, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(mid_c)
        self.conv2 = nn.Conv2d(in_channels=mid_c, out_channels=mid_c, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(mid_c)
        self.conv3 = nn.Conv2d(in_channels=mid_c, out_channels=out_c, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(out_c)
        self.conv1_1 = nn.Conv2d(in_channels=in_c, out_channels=out_c, stride=stride, kernel_size=1)
        self.bn1_1 = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU()

    def forward(self, x):
        x1 = self.conv1_1(x)
        x1 = self.bn1_1(x1)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out = x1 + out
        out = self.relu(out)
        return out


class DBHead(nn.Module):
    def __init__(self, in_channels, k=50):
        super().__init__()
        self.k = k
        self.binarize = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 64, 2, 2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 1, 2, 2),
            nn.Sigmoid())

        self.thresh = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 64, 2, 2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 1, 2, 2),
            nn.Sigmoid())

    def forward(self, x):
        shrink_maps = self.binarize(x)
        threshold_maps = self.thresh(x)
        binary_maps = self.step_function(shrink_maps, threshold_maps)
        return shrink_maps, threshold_maps, binary_maps

    def step_function(self, x, y):
        return torch.reciprocal(1 + torch.exp(-self.k * (x - y)))


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3)
        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.bottleneck1_1 = ResNet50DownBlock(in_c=64, mid_c=64, out_c=256)
        self.bottleneck1_2 = ResNet50BasicBlock(in_c=256, mid_c=64, out_c=256)
        self.bottleneck1_3 = ResNet50BasicBlock(in_c=256, mid_c=64, out_c=256)
        self.bottleneck2_1 = ResNet50DownBlock(in_c=256, mid_c=128, out_c=512, stride=2)
        self.bottleneck2_2 = ResNet50BasicBlock(in_c=512, mid_c=128, out_c=512)
        self.bottleneck2_3 = ResNet50BasicBlock(in_c=512, mid_c=128, out_c=512)
        self.bottleneck2_4 = ResNet50BasicBlock(in_c=512, mid_c=128, out_c=512)
        self.bottleneck3_1 = ResNet50DownBlock(in_c=512, mid_c=256, out_c=1024, stride=2)
        self.bottleneck3_2 = ResNet50BasicBlock(in_c=1024, mid_c=256, out_c=1024)
        self.bottleneck3_3 = ResNet50BasicBlock(in_c=1024, mid_c=256, out_c=1024)
        self.bottleneck3_4 = ResNet50BasicBlock(in_c=1024, mid_c=256, out_c=1024)
        self.bottleneck3_5 = ResNet50BasicBlock(in_c=1024, mid_c=256, out_c=1024)
        self.bottleneck3_6 = ResNet50BasicBlock(in_c=1024, mid_c=256, out_c=1024)
        self.bottleneck4_1 = ResNet50DownBlock(in_c=1024, mid_c=512, out_c=2048, stride=2)
        self.bottleneck4_2 = ResNet50BasicBlock(in_c=2048, mid_c=512, out_c=2048)
        self.bottleneck4_3 = ResNet50BasicBlock(in_c=2048, mid_c=512, out_c=2048)
        # FPN
        self.conv_c5_m5 = nn.Conv2d(in_channels=2048, out_channels=64, kernel_size=1)
        self.conv_m5_p5 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.conv_c4_m4 = nn.Conv2d(in_channels=1024, out_channels=64, kernel_size=1)
        self.conv_m4_p4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.conv_c3_m3 = nn.Conv2d(in_channels=512, out_channels=64, kernel_size=1)
        self.conv_m3_p3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.conv_c2_m2 = nn.Conv2d(in_channels=256, out_channels=64, kernel_size=1)
        self.conv_m2_p2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        # concatnate
        self.conv_p2_p2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.conv_p3_p3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.conv_p4_p4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.conv_p5_p5 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        #Db_head
        self.head = DBHead(in_channels=256)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        c1 = self.relu(x)
        x = self.pool1(c1)
        x = self.bottleneck1_1(x)
        x = self.bottleneck1_2(x)
        c2 = self.bottleneck1_3(x)
        x = self.bottleneck2_1(c2)
        x = self.bottleneck2_2(x)
        x = self.bottleneck2_3(x)
        c3 = self.bottleneck2_4(x)
        x = self.bottleneck3_1(c3)
        x = self.bottleneck3_2(x)
        x = self.bottleneck3_3(x)
        x = self.bottleneck3_4(x)
        x = self.bottleneck3_5(x)
        c4 = self.bottleneck3_6(x)
        x = self.bottleneck4_1(c4)
        x = self.bottleneck4_2(x)
        c5 = self.bottleneck4_3(x)
        m5 = self.conv_c5_m5(c5)
        p5 = self.conv_m5_p5(m5)
        c4_m4 = self.conv_c4_m4(c4)
        m5_x2 = nn.functional.interpolate(m5, (m5.shape[-2] * 2, m5.shape[-1] * 2), mode='bilinear', align_corners=True)
        m4 = m5_x2 + c4_m4
        p4 = self.conv_m4_p4(m4)
        c3_m3 = self.conv_c3_m3(c3)
        m4_x2 = nn.functional.interpolate(m4, (m4.shape[-2] * 2, m4.shape[-1] * 2), mode='bilinear', align_corners=True)
        m3 = m4_x2 + c3_m3
        p3 = self.conv_m3_p3(m3)
        c2_m2 = self.conv_c2_m2(c2)
        m3_x2 = nn.functional.interpolate(m3, (m3.shape[-2] * 2, m3.shape[-1] * 2), mode='bilinear', align_corners=True)
        m2 = m3_x2 + c2_m2
        p2 = self.conv_m2_p2(m2)
        p2 = self.conv_p2_p2(p2)
        p3 = self.conv_p3_p3(p3)
        p3 = nn.functional.interpolate(p3, (p3.shape[-2] * 2, p3.shape[-1] * 2), mode='bilinear', align_corners=True)
        p4 = self.conv_p4_p4(p4)
        p4 = nn.functional.interpolate(p4, (p4.shape[-2] * 4, p4.shape[-1] * 4), mode='bilinear', align_corners=True)
        p5 = self.conv_p5_p5(p5)
        p5 = nn.functional.interpolate(p5, (p5.shape[-2] * 8, p5.shape[-1] * 8), mode='bilinear', align_corners=True)
        feature = torch.cat((p2, p3, p4, p5), dim=1)
        shrink_maps, threshold_maps, binary_maps = self.head(feature)
        return shrink_maps, threshold_maps, binary_maps


def train(model, train_loader, optimizer, epoch):
    model.train()
    all_step = 0
    for i in range(epoch):
        for step, data in enumerate(train_loader):
            all_step = all_step + 1
            img, shrink_label, threshold_label = data[0].to(torch.device('cuda')), data[1].to(torch.device('cuda')), data[2].to(torch.device('cuda'))
            shrink_pre, threshold_pre, binary_pre = model(img)
            loss_shrink_map = nn.BCELoss()(shrink_pre, shrink_label)
            loss_threshold_map = nn.L1Loss()(threshold_pre, threshold_label)
            loss_binary_map = nn.BCELoss()(binary_pre, shrink_label)
            loss = loss_shrink_map + loss_binary_map + 10 * loss_threshold_map
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (step + 1) % 100 == 0:
                print('epoch:', i + 1, 'step:', step + 1, 'loss:', loss)
    torch.save(model.state_dict(), './model/DBnet_pytorch.pth')


def inference(model, test_loader):
    model.eval()
    with torch.no_grad():
        for data in test_loader:
            img = data[0].to(torch.device('cuda'))
            shrink_pre, threshold_pre, binary_pre = model(img)
            img = img.cpu().numpy()[0]
            img = np.transpose(img, (1, 2, 0))
            img = np.array(img * 255, dtype=np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pre = binary_pre.cpu().numpy()[0][0]
            pre[pre > 0.5] = 1
            pre[pre < 1] = 0
            pre = (pre * 255).astype(np.uint8)
            _, contours, _ = cv2.findContours(pre, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            pre_boxes = []
            for i in range(len(contours)):
                contour = contours[i].squeeze(1)
                contour_perimeter = cv2.arcLength(contour, True)
                # 过小的可能是噪点，删除
                if contour_perimeter > 10:
                    bounding_box = cv2.minAreaRect(contour)
                    points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])
                    if points[1][1] > points[0][1]:
                        index_1, index_4 = 0, 1
                    else:
                        index_1, index_4 = 1, 0
                    if points[3][1] > points[2][1]:
                        index_2, index_3 = 2, 3
                    else:
                        index_2, index_3 = 3, 2
                    points = [points[index_1], points[index_2], points[index_3], points[index_4]]
                    points = np.array(points)
                    box = shrink_out(points, rate=2.0)
                    bounding_box2 = cv2.minAreaRect(box)
                    points = sorted(list(cv2.boxPoints(bounding_box2)), key=lambda x: x[0])
                    if points[1][1] > points[0][1]:
                        index_1, index_4 = 0, 1
                    else:
                        index_1, index_4 = 1, 0
                    if points[3][1] > points[2][1]:
                        index_2, index_3 = 2, 3
                    else:
                        index_2, index_3 = 3, 2
                    box = [points[index_1], points[index_2], points[index_3], points[index_4]]
                    box = np.array(box).astype(np.int32)
                    pre_boxes.append(box)
            for i in range(len(pre_boxes)):
                box = pre_boxes[i]
                for j in range(len(box)):
                    cv2.line(img, (box[j][0], box[j][1]), (box[(j + 1) % 4][0], box[(j + 1) % 4][1]), (0, 0, 255), 2)
            cv2.imwrite('./result/result.jpg', img)


if __name__ == '__main__':
    # train
    # model = Model().to(torch.device('cuda'))
    # optimizer = optim.Adam(model.parameters())
    # train_data = MyDataset(base_path='./data/train_data')
    # train_loader = DataLoader(dataset=train_data, batch_size=1, shuffle=True)
    # train(model, train_loader, optimizer, 10)

    # inference
    model = Model().to(torch.device('cuda'))
    model.load_state_dict(torch.load('./model/DBnet_pytorch.pth'))
    test_data = MyDataset(base_path='./data/test_data')
    test_loader = DataLoader(dataset=test_data, batch_size=1)
    inference(model, test_loader)
