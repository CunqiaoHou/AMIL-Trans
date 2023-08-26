# -*- coding: utf-8 -*-

import sys
import os
import numpy as np
import argparse
import random
import openslide
import PIL.Image as Image
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.transforms.functional as functional
import torchvision.models as models
from origin_lstm import ori_lstm
from transformer.transformer import visual_prompt
from torchsummary import summary
import time
from network import resnet
from sklearn.metrics import balanced_accuracy_score, recall_score
from tqdm import tqdm as tqdm
from collections import OrderedDict
from network import sa_resnet
from network import eca_resnet

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
parser = argparse.ArgumentParser(description='SoMIL-nature-medicine aggregator classifier training script')
parser.add_argument('--train_lib', type=str, default='',
                    help='path to train MIL library binary')
parser.add_argument('--output', type=str, default='', help='name of output file')
parser.add_argument('--batch_size', type=int, default=1, help='mini-batch size (default: 512)')
parser.add_argument('--nepochs', type=int, default=40, help='number of epochs')
parser.add_argument('--workers', default=0, type=int, help='number of data loading workers (default: 4)')
parser.add_argument('--weights', default=0.3799, type=float,
                    help='unbalanced positive class weight (default: 0.5, balanced classes)')
parser.add_argument('--k', default=10, type=int,
                    help='top k tiles are assumed to be of the same class as the slide (default: 1, standard MIL)')
best_acc = 0




def Parallel2Single(origin_state):
    converted = OrderedDict()

    for k, v in origin_state.items():
        name = k[7:]
        converted[name] = v

    return converted


def main():
    global args, best_acc
    args = parser.parse_args()
    # cnn
    # model = models.resnet50(num_classes=2, pretrained=False)
    model = eca_resnet.eca_resnet50()
    model.fc = nn.Linear(model.fc.in_features, 2)
    model_dict = torch.load('')
    model.cuda()
    model.load_state_dict(Parallel2Single(model_dict['state_dict']))
    get_feature_model = nn.Sequential(*list(model.children())[:-1]).cuda()
    get_feature_model.eval()
                           # 2048 or 32896
    transformer_model = visual_prompt().cuda()
    print(transformer_model)
    if args.weights == 0.5:
        criterion = nn.CrossEntropyLoss().cuda()
    else:
        w = torch.Tensor([1 - args.weights, args.weights])
        criterion = nn.CrossEntropyLoss(w).cuda()
    optimizer = optim.AdamW(transformer_model.parameters(), lr=1e-5, weight_decay=1e-4)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)
    cudnn.benchmark = True

    # normalization # 在进行根据已有的top k特征进行LSTM训练时,不再需要复杂的transforms方法,只做最基本的就好。
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    val_trans = transforms.Compose([transforms.ToTensor(), normalize])
    best_metric_probs_inf_save = torch.load('')

    # load data
    train_dset = MILdataset(args.train_lib, args.k, val_trans,
                            best_metric_probs_inf_save['train_dset_grid'],
                            best_metric_probs_inf_save['train_dset_slideIDX'])
    train_loader = torch.utils.data.DataLoader(train_dset, batch_size=args.batch_size, shuffle=False,
                                               num_workers=args.workers, pin_memory=False)

    time_mark = time.strftime('%Y_%m_%d_', time.localtime(time.time()))
    fconv = open(os.path.join(args.output, time_mark + 'Transformer_convergence.csv'), 'w')
    fconv.write(' ,Training,,,,Train_whole,,,Validation,,\n')
    fconv.write('epoch,train_acc,train_recall,train_fnr,train_loss,true_acc,true_recall,true_fnr,acc,recall,fnr')
    fconv.close()

    train_probs = best_metric_probs_inf_save['train_probs']
    topk = group_argtopk(np.array(train_dset.slideIDX), train_probs, args.k)  # 3 X train_probs[:, 1]##
    tmp_topk = group_argtopk(np.array(train_dset.slideIDX), train_probs, args.k)  # 1


    # loop through epochs
    for epoch in range(args.nepochs):
        start_time = time.time()
        # Train
        train_dset.setmode(3)
        train_dset.settopk(topk, get_feature_model)
        whole_acc, whole_recall, whole_fnr, whole_loss = train(epoch, train_loader, transformer_model, criterion, optimizer)
        print('\tTraining  Epoch: [{}/{}] Acc: {} Recall:{} Fnr:{} Loss: {}'.format(epoch + 1, \
                                                                                    args.nepochs, whole_acc,
                                                                                    whole_recall, whole_fnr,
                                                                                    whole_loss))

        # 再以top 1来计算train数据集相关指标
        train_dset.settopk(tmp_topk, get_feature_model)
        tmp_train_probs = inference(epoch, train_loader, transformer_model, args.batch_size, 'train')
        metrics_meters = calc_accuracy(np.argmax(tmp_train_probs, axis=1), train_dset.targets)
        result = '\n' + str(epoch + 1) + ',' + str(whole_acc) + ',' + str(whole_recall) + ',' + str(whole_fnr) + ',' \
                 + str(whole_loss) + ',' + str(metrics_meters['acc']) + ',' + str(metrics_meters['recall']) + ',' + str(
            metrics_meters['fnr'])

        str_logs = ['{} - {:.4}'.format(k, v) for k, v in metrics_meters.items()]
        s = ', '.join(str_logs)
        print('\tValidation  Epoch: [{}/{}]  '.format(epoch + 1, args.nepochs) + s)

        result = result + ',' + str(metrics_meters['acc']) + ',' + str(metrics_meters['recall']) + ',' \
                 + str(metrics_meters['fnr'])
        fconv = open(os.path.join(args.output, time_mark + 'LSTM_convergence.csv'), 'a')
        fconv.write(result)
        fconv.close()
        # Save best model
        tmp_acc = (metrics_meters['acc'] + metrics_meters['recall']) / 2 - metrics_meters['fnr'] * args.weights
        # if tmp_acc >= best_acc :
        if epoch >= 0:
            # if tmp_acc >= best_acc & epoch > 2:
            best_acc = tmp_acc.copy()
            obj = {
                'epoch': epoch + 1,
                'state_dict': transformer_model.state_dict(),
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict()
            }
            torch.save(obj, os.path.join(args.output, time_mark + 'transformer_checkpoint_best.pth'))
            torch.save(obj, os.path.join(args.output, time_mark + str(epoch) + '_transformer_checkpoint_best.pth'))

        print('\tEpoch %d has been finished, needed %.2f sec.' % (epoch + 1, time.time() - start_time))


def inference(run, loader, model, batch_size, phase):
    model.eval()
    probs = np.zeros((1, 2))
    whole_probably = 0.
    with torch.no_grad():
        with tqdm(loader, desc='Epoch:' + str(run + 1) + ' ' + phase + '\'s inferencing', \
                  file=sys.stdout, disable=False) as iterator:
            for i, (input, _) in enumerate(iterator):
                input = input.cuda()
                output = F.softmax(model(input), dim=1)
                prob = output.detach().clone()
                prob = prob.cpu().numpy()
                batch_proba = np.mean(prob, axis=0)
                probs = np.row_stack((probs, prob))
                whole_probably = whole_probably + batch_proba

                iterator.set_postfix_str('batch proba :' + str(batch_proba))
            whole_probably = whole_probably / (i + 1)
            iterator.set_postfix_str('Whole average probably is ' + str(whole_probably))
    probs = np.delete(probs, 0, axis=0)
    return probs.reshape(-1, 2)


def train(run, loader, model, criterion, optimizer):
    model.train()
    whole_loss = 0.
    whole_acc = 0.
    whole_recall = 0.
    whole_fnr = 0.
    logs = {}

    with tqdm(loader, desc='Epoch:' + str(run + 1) + ' is trainng', file=sys.stdout, disable=False) as iterator:
        for i, (input, target) in enumerate(iterator):
            input = input.cuda()
            target = target.cuda()
            #            features = get_feature_model(input)
            output = F.softmax(model(input), dim=1)
            loss = criterion(output, target)
            _, pred = torch.max(output, 1)
            pred = pred.data.cpu().numpy()
            metrics_meters = calc_accuracy(pred, target.cpu().numpy())
            logs.update(metrics_meters)
            logs.update({'loss': loss.item()})
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            iterator.set_postfix_str(str(metrics_meters))

            whole_acc += metrics_meters['acc']
            whole_recall += metrics_meters['recall']
            whole_fnr += metrics_meters['fnr']
            whole_loss += loss.item()
    return round(whole_acc / (i + 1), 3), round(whole_recall / (i + 1), 3), \
           round(whole_fnr / (i + 1), 3), round(whole_loss / (i + 1), 3)


def calc_accuracy(pred, real):
    if str(type(pred)) != "<class 'numpy.ndarray'>":
        pred = np.array(pred)
    if str(type(real)) != "<class 'numpy.ndarray'>":
        real = np.array(real)
    neq = np.not_equal(pred, real)
    fnr = np.logical_and(pred == 0, neq).sum() / (real == 1).sum() if (real == 1).sum() > 0 else 0.0
    # 将无法计算fnr的值从0改为0.0,保证在train和inference调用生成str_logs时不会引起Precision not allowed in integer format specifier的报错
    balanced_acc = balanced_accuracy_score(real, pred)
    recall = recall_score(real, pred, average='weighted')
    metrics_meters = {'acc': round(balanced_acc, 3), 'recall': round(recall, 3), 'fnr': round(fnr, 3)}

    return metrics_meters


def group_argtopk(groups, data, k=1):

    order = np.lexsort((data, groups))
    groups = groups[order]
    data = data[order]
    index = np.empty(len(groups), 'bool')
    index[-k:] = True
    index[:-k] = groups[k:] != groups[:-k]
    return list(order[index])


class MILdataset(data.Dataset):
    def __init__(self, libraryfile='', k=0, transform=None, load_grid=None, load_IDX=None):
        # 参数中的k用于下面是否进行重复采样的语句，仅在load_grid和load_IDX不从外部加载的时候才有意义
        lib = torch.load(libraryfile)
        slides = []
        patch_size = []
        for i, name in enumerate(lib['slides']):
            sys.stdout.write('Opening SVS headers: [{}/{}]\r'.format(i + 1, len(lib['slides'])))
            sys.stdout.flush()
            slides.append(openslide.OpenSlide(name))
            patch_size.append(int(lib['patch_size'][i]))
        print('')
        # Flatten grid
        if load_IDX is None:
            grid = []
            slideIDX = []
            # slideIDX列表存放每个WSI以及其坐标列表的标记,假设有0,1,2号三个WSI图像,分别于grid中记录4,7,3组提取的坐标,\
            # 返回为[0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2]
            for i, g in enumerate(lib['grid']):
                if len(g) < k:
                    g = g + [(g[x]) for x in np.random.choice(range(len(g)), k - len(g))]
                grid.extend(g)
                slideIDX.extend([i] * len(g))
                # 当前slide已有的grid数量在k之下时,就进行随机重复采样
        else:
            # 如果有外部传入load_grid和load_IDX时,对传参直接加载,跳过上述过程。原则上在k设置为0(即完全不需要过采样)时是不需要这样做
            # 但如果想对已有的probs数据进行最佳匹配的话,还是要进行加载,因为每次过采样的grid项都不一样,直接套用外部的probs会出现部分不匹配的情况。
            grid = load_grid
            slideIDX = load_IDX
        print('Number of tiles: {}'.format(len(grid)))
        self.slidenames = lib['slides']
        self.slides = slides
        self.targets = lib['targets']
        self.grid = grid
        self.slideIDX = slideIDX
        self.transform = transform
        self.mode = None
        self.patch_size = patch_size
        self.level = lib['level']

    def setmode(self, mode):
        self.mode = mode

    def settopk(self, top_k=None, feature_extract_model=None):
        self.top_k = top_k  # 加载top k列表用于后续提取特征用
        self.feature_extract_model = feature_extract_model

    def maketraindata(self, idxs, repeat=0):
        # repeat这个参数用于是否对采样进行复制,如果进行复制,就会在下面的_getitem_方法中对重复的样本进行不一样的颜色增强
        if abs(repeat) == 0:
            # repeat等于0的时,按用原来的方法进行生成筛选的数据,并不会进行h通道的颜色变换。
            self.t_data = [(self.slideIDX[x], self.grid[x], self.targets[self.slideIDX[x]], 0) for x in idxs]
        else:
            repeat = abs(repeat) if repeat % 2 == 1 else abs(repeat) + 1
            # 通过该操作确保非奇数的repeat传参也能变为奇数
            self.t_data = [(self.slideIDX[x], self.grid[x], self.targets[self.slideIDX[x]], 0) for x in idxs]
            for y in range(-100, int(100 + repeat / 2), int(100 * 2 / repeat)):
                # 将会在(-0,1,0.1)范围内按照repeat的数值进行区间划分(这也是要求repeat值必须为奇数的原因所在)
                # 通过上面的划分,可以确保除0外在(-0,1,0.1)都会划分为repeat-1倍,需要注意最后y的值必须控制在0.1以内
                self.t_data = self.t_data + [(self.slideIDX[x], self.grid[x], self.targets[self.slideIDX[x]], y / 1000)
                                             for x in idxs]

    def shuffletraindata(self):
        self.t_data = random.sample(self.t_data, len(self.t_data))

    def __getitem__(self, index):
        if self.mode == 1:
            # mode =1 为预测时使用，会从所有WSI文件中返回 全部region 的图像
            slideIDX = self.slideIDX[index]
            coord = self.grid[index]
            img = self.slides[slideIDX].read_region(coord, self.level, (self.patch_size[slideIDX], \
                                                                        self.patch_size[slideIDX])).convert('RGB')
            if img.size != (224, 224):
                img = img.resize((224, 224), Image.BILINEAR)
            if self.transform is not None:
                img = self.transform(img)
            return img
        elif self.mode == 2:
            # mode =2 为训练时使用，会根据index(上一轮MIL过程得出),从全部WSI文件中筛选对应的坐标列表,返回相应的训练图像和label
            slideIDX, coord, target, h_value = self.t_data[index]
            img = self.slides[slideIDX].read_region(coord, self.level, (self.patch_size[slideIDX], \
                                                                        self.patch_size[slideIDX])).convert('RGB')
            if h_value > 0:
                hue_factor = random.uniform(h_value, 0.1)
            elif h_value == 0:
                hue_factor = random.uniform(0, 0)
            elif h_value < 0:
                hue_factor = random.uniform(-0.1, h_value)
            img = functional.adjust_hue(img, hue_factor)
            # 只有在训练模式下才进行H通道变换的颜色增强方法
            # 如果在maketraindata方法设置采样复制,那么就会针对h_value的值进行不同方向的hue_factor生成,\
            #    从而达到复制的样本和原来的样本有不一样的增强的效果
            if img.size != (224, 224):
                img = img.resize((224, 224), Image.BILINEAR)
            if self.transform is not None:
                img = self.transform(img)
            return img, target
        elif self.mode == 3 and self.top_k is not None and self.feature_extract_model is not None:
            k_value = int(len(self.top_k) / len(self.targets))  # 得出实际的top k数值,省下不必要的外部传参
            for j in range(k_value):
                coord = self.grid[self.top_k[index * k_value + j]]
                img = self.slides[index].read_region(coord, self.level,
                                                     (self.patch_size[index], self.patch_size[index])).convert('RGB')
                if img.size != (224, 224):
                    img = img.resize((224, 224), Image.BILINEAR)
                img = self.transform(img).unsqueeze(0)
                if j == 0:
                    feature = self.feature_extract_model(img.cuda())  # 单个img的feature的shape是([1,512]) resnet34
                else:
                    feature = torch.cat((feature, self.feature_extract_model(img.cuda())), 0)  ######
                # 将k个feature纵向叠加在一起,变成torch.Size([k,512])
            return feature.view(-1, feature.shape[1]), self.targets[index]  # [k,512]

    def __len__(self):
        if self.mode == 1:
            return len(self.grid)
        elif self.mode == 2:
            return len(self.t_data)
        elif self.mode == 3 and self.top_k is not None and self.feature_extract_model is not None:
            return len(self.targets)  # mode为3时,len( dset )实际上就是等同于slide的数量


if __name__ == '__main__':
    main()

