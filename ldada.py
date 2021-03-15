import torch
import torch.optim as optim
import torch.nn as nn
import model_no_class as model_no
import adversarial1 as ad
import numpy as np
import os
import argparse
from data_list import ImageList
import pre_process as prep
import math
import msda
import mmd

torch.set_num_threads(1)

def test_target(loader, model, model_fc):
    with torch.no_grad():
        start_test = True
        iter_val = [iter(loader['val'+str(i)]) for i in range(10)]
        for i in range(len(loader['val0'])):
            data = [iter_val[j].next() for j in range(10)]
            inputs = [data[j][0] for j in range(10)]
            labels = data[0][1]
            for j in range(10):
                inputs[j] = inputs[j].to(device)
            labels = labels.to(device)
            outputs = []
            for j in range(10):
                features = model_fc(inputs[j])
                _, output, _ = model(features)
                outputs.append(output)
            outputs = sum(outputs)
            if start_test:
                all_output = outputs.data.float()
                all_label = labels.data.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.data.float()), 0)
                all_label = torch.cat((all_label, labels.data.float()), 0)
        _, predict = torch.max(all_output, 1)
        accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    return accuracy


def inv_lr_scheduler(param_lr, optimizer, iter_num, gamma, power, init_lr=0.001):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    lr = init_lr * (1 + gamma * iter_num) ** (-power)
    i=0
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_lr[i]
        i+=1
    return optimizer


def entropy_loss_func(input_):
    mask = input_.ge(0.000001)
    mask_out = torch.masked_select(input_, mask)
    entropy = -(torch.sum(mask_out * torch.log(mask_out)))
    return entropy / float(input_.size(0))

class predictor(nn.Module):
    def __init__(self, feature_len, cate_num):
        super(predictor, self).__init__()
        self.classifier = nn.Linear(feature_len, cate_num)
        self.classifier.weight.data.normal_(0, 0.01)
        self.classifier.bias.data.fill_(0.0)

    def forward(self, features):
        activations = self.classifier(features)
        return(activations)


class fine_net(nn.Module):
    def __init__(self, feature_len):
        super(fine_net,self).__init__()
        self.bottleneck_0 = nn.Linear(feature_len, 256)
        self.bottleneck_0.weight.data.normal_(0, 0.005)
        self.bottleneck_0.bias.data.fill_(0.1)
        self.bottleneck_layer = nn.Sequential(self.bottleneck_0, nn.ReLU(), nn.Dropout(0.5))
        self.classifier_layer = predictor(256, cate_all[0])
        self.bn = nn.BatchNorm1d(256,affine=False)

    def forward(self,features):
        #features = self.model_fc(x)
        out_bottleneck = self.bottleneck_layer(features)
        logits = self.classifier_layer(out_bottleneck)
        return(out_bottleneck, logits, self.bn(out_bottleneck))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Transfer Learning')

    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--source', type=str, nargs='?', default='c', help="source dataset")
    parser.add_argument('--target', type=str, nargs='?', default='p', help="target dataset")
    parser.add_argument('--entropy_source', type=float, nargs='?', default=0, help="target dataset")
    parser.add_argument('--entropy_target', type=float, nargs='?', default=0.01, help="target dataset")
    parser.add_argument('--mode', type=str, nargs='?', default='msda', help="msda/ldada/baseline/mmd/mmd_soft")
    parser.add_argument('--msda_wt', type=float, nargs='?', default=0.001, help="target dataset")
    parser.add_argument('--msda_raw_feat',action='store_true',default=False,help="MSDA on raw feats or bn feats")
    parser.add_argument('--lr', type=float, nargs='?', default=0.03, help="target dataset")
    parser.add_argument('--num_workers', type=int, nargs='?', default=10, help="num_workers")
    parser.add_argument('--batch_size', type=int, nargs='?', default=36, help="num_workers")
    parser.add_argument('--initial_smooth', type=float, nargs='?', default=0.9, help="target dataset")
    parser.add_argument('--final_smooth', type=float, nargs='?', default=0.1, help="target dataset")
    parser.add_argument('--max_iteration', type=float, nargs='?', default=12500, help="target dataset")
    parser.add_argument('--smooth_stratege', type=str, nargs='?', default='e', help="smooth stratege")

    parser.add_argument('--cluster_ckpt', type=str, nargs='?', default='', help="Checkpoint for cluster probabilities")
    parser.add_argument('--num_domains', type=int, nargs='?', default=3, help="num_domains")

    args = parser.parse_args()


    # device assignment
    #os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


    # file paths and domains
    file_path = {
    "i": "/vulcan-pvc1/ml_for_da_pan_base/dataset_list/bird31_ina_list_2017.txt",  # NOT AVAILABLE
    "n": "/vulcan-pvc1/ml_for_da_pan_base/dataset_list/bird31_nabirds_list.txt",
    "c": "/vulcan-pvc1/ml_for_da_pan_base/dataset_list/bird31_cub2011.txt"
    #         "pai":"/vulcan-pvc1/ml_for_da_pan_base/dataset_list/cub200_drawing_20.txt",
    #         "cub":"/vulcan-pvc1/ml_for_da_pan_base/dataset_list/cub200_2011_20.txt"
    }


    dataset_source = file_path[args.source]
    dataset_target = dataset_test = file_path[args.target]
    cate_all = [31]

    # dataset load
    batch_size = {"train": args.batch_size, "val": args.batch_size, "test": 4}
    for i in range(10):
        batch_size["val" + str(i)] = 4

    dataset_loaders = {}

    dataset_list = ImageList(open(dataset_source).readlines(), transform=prep.image_train(resize_size=256, crop_size=224))
    dataset_loaders["train"] = torch.utils.data.DataLoader(dataset_list, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    dataset_list = ImageList(open(dataset_target).readlines(), transform=prep.image_train(resize_size=256, crop_size=224))
    dataset_loaders["val"] = torch.utils.data.DataLoader(dataset_list, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    dataset_list = ImageList(open(dataset_test).readlines(), transform=prep.image_train(resize_size=256, crop_size=224))
    dataset_loaders["test"] = torch.utils.data.DataLoader(dataset_list, batch_size=4, shuffle=False, num_workers=args.num_workers)

    prep_dict_test = prep.image_test_10crop(resize_size=256, crop_size=224)
    for i in range(10):
        dataset_list = ImageList(open(dataset_test).readlines(), transform=prep_dict_test["val" + str(i)])
        dataset_loaders["val" + str(i)] = torch.utils.data.DataLoader(dataset_list, batch_size=4, shuffle=False, num_workers=6)

    # network construction
    feature_len = 2048
    # fine-grained feature extractor + fine-grained label predictor
    devices = list(range(torch.cuda.device_count()))
    model_fc = model_no.Resnet50Fc()
    model_fc = model_fc.to(device)
    model_fc = nn.DataParallel(model_fc, device_ids=devices)
    my_fine_net = fine_net(feature_len)
    my_fine_net = my_fine_net.to(device)
    my_fine_net.train(True)
    model_fc.train(True)

    # criterion and optimizer
    criterion = {
        "classifier": nn.CrossEntropyLoss(),
        "kl_loss": nn.KLDivLoss(size_average=False),
        "adversarial": nn.BCELoss()
    }

    optimizer_dict = [
        {"params": filter(lambda p: p.requires_grad, model_fc.parameters()), "lr": 0.1},
        {"params": filter(lambda p: p.requires_grad, my_fine_net.bottleneck_layer.parameters()), "lr": 1},
        {"params": filter(lambda p: p.requires_grad, my_fine_net.classifier_layer.parameters()), "lr": 1}
    ]
    optimizer = optim.SGD(optimizer_dict, lr=0.1, momentum=0.9, weight_decay=0.0005)
    param_lr = []
    for param_group in optimizer.param_groups:
        param_lr.append(param_group["lr"])
    # losses
    train_fine_cross_loss = 0.0
    train_transfer_loss = 0.0
    train_entropy_loss_source = 0.0
    train_entropy_loss_target = 0.0
    train_total_loss = 0.0

    len_source = len(dataset_loaders["train"]) - 1
    len_target = len(dataset_loaders["val"]) - 1
    iter_source = iter(dataset_loaders["train"])
    iter_target = iter(dataset_loaders["val"])
    if args.mode=="ldada" or args.mode=="mmd_soft":
        from generate_domains import generate_domains
        domain_probs = generate_domains(args.num_domains,dataset_loaders["train"], args.cluster_ckpt, device)
    for iter_num in range(1, int(args.max_iteration) + 1):
        model_fc.train(True)
        my_fine_net.train(True)
        optimizer = inv_lr_scheduler(param_lr, optimizer, iter_num, init_lr=args.lr, gamma=0.001, power=0.75)
        optimizer.zero_grad()

        if iter_num % len_source == 0:
            iter_source = iter(dataset_loaders["train"])
        if iter_num % len_target == 0:
            iter_target = iter(dataset_loaders["val"])
        data_source = iter_source.next()
        data_target = iter_target.next()
        inputs_source, labels_source, idxes_src = data_source
        inputs_target, labels_target, idxes_tgt = data_target
        inputs = torch.cat((inputs_source, inputs_target), dim=0)
        inputs = inputs.to(device)

        fine_labels_source_cpu = labels_source.view(-1, 1)
        labels_source = labels_source.to(device)
        idxes_src = idxes_src.long().to(device)
        features = model_fc(inputs)
        if args.msda_raw_feat:
            features_btnk, logits_fine, _ = my_fine_net(features)
        else:
            _, logits_fine,features_btnk = my_fine_net(features)
        features_src = features_btnk.narrow(0, 0, batch_size["train"])
        features_tgt = features_btnk.narrow(0, batch_size["train"], batch_size["train"])
        if args.mode=="msda":
            transfer_loss = msda.msda_regulizer_single(features_src, features_tgt, 5)
            transfer_loss = transfer_loss*args.msda_wt
        elif args.mode=="ldada":
            batch_domain_probs = domain_probs[idxes_src].to(device)
            transfer_loss = msda.msda_regulizer_soft(features_src, features_tgt, 5, batch_domain_probs)
            transfer_loss = transfer_loss * args.msda_wt
        elif args.mode=="mmd":
            transfer_loss = mmd.mmd(features_src, features_tgt)
            transfer_loss = transfer_loss * args.msda_wt
        elif args.mode=="mmd_soft":
            batch_domain_probs = domain_probs[idxes_src].to(device)
            transfer_loss = mmd.mmd_soft(features_src, features_tgt,batch_domain_probs)
            transfer_loss = transfer_loss * args.msda_wt
        else:
            transfer_loss = torch.zeros(1,dtype=torch.float32).to(device)
        logits_fine_source = logits_fine.narrow(0, 0, batch_size["train"])
        fine_labels_onehot = torch.zeros(logits_fine_source.size()).scatter_(1, fine_labels_source_cpu, 1)
        fine_labels_onehot = fine_labels_onehot.to(device)
        labels_onehot_smooth = fine_labels_onehot
        fine_classifier_loss = criterion["kl_loss"](nn.LogSoftmax(dim=1)(logits_fine_source), labels_onehot_smooth)
        fine_classifier_loss = fine_classifier_loss / batch_size["train"]
        classifier_loss = fine_classifier_loss
        entropy_loss_source = entropy_loss_func(nn.Softmax(dim=1)(logits_fine.narrow(0, 0, batch_size["train"])))
        entropy_loss_target = entropy_loss_func(nn.Softmax(dim=1)(logits_fine.narrow(0, batch_size["train"], batch_size["train"])))
        total_loss = classifier_loss + transfer_loss + entropy_loss_source * args.entropy_source + entropy_loss_target * args.entropy_target

        total_loss.backward()
        optimizer.step()

        train_fine_cross_loss += fine_classifier_loss.item()
        train_entropy_loss_source += entropy_loss_source.item()
        train_entropy_loss_target += entropy_loss_target.item()
        train_total_loss += total_loss.item()
        train_transfer_loss += transfer_loss.item()

        # test
        test_interval = 500
        if iter_num%10==0:
            print(iter_num)
        if iter_num % test_interval == 0:
            my_fine_net.eval()
            model_fc.eval()
            test_acc = test_target(dataset_loaders, my_fine_net, model_fc)
            print('test_acc:%.4f'%(test_acc))

            print("Iter {:05d}, Average Fine Cross Entropy Loss: {:.4f}; "
                  "Average Transfer Loss: {:.4f}; "
                  "Average Entropy Loss Source: {:.4f}; "
                  "Average Entropy Loss Target: {:.4f}; "
                  "Average Training Loss: {:.4f}".format(
                iter_num,
                train_fine_cross_loss / float(test_interval),
                train_transfer_loss / float(test_interval),
                train_entropy_loss_source / float(test_interval),
                train_entropy_loss_target / float(test_interval),
                train_total_loss / float(test_interval))
            )

            train_fine_cross_loss = 0.0
            train_coarse_cross_loss = 0.0
            train_transfer_loss = 0.0
            train_entropy_loss_source = 0.0
            train_entropy_loss_target = 0.0
            train_total_loss = 0.0

