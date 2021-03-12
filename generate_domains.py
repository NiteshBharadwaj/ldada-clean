import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import time

def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

class DomainPredictor(nn.Module):
    def __init__(self, num_domains, prob=0.5, classaware_dp=False):
        super(DomainPredictor, self).__init__()
        self.dp_model = Resnet50Fc()
        self.classaware_dp=classaware_dp
        for param in self.dp_model.conv1.parameters():
            param.requires_grad = False
        for param in self.dp_model.bn1.parameters():
            param.requires_grad = False
        for param in self.dp_model.layer1.parameters():
            param.requires_grad = True
        for param in self.dp_model.layer2.parameters():
            param.requires_grad = True
        self.inp_layer = 2048 if classaware_dp else 2048
        self.fc5 = nn.Linear(self.inp_layer, 128)
        self.bn_fc5 = nn.BatchNorm1d(128)
        self.dp_layer = nn.Linear(128, num_domains)

        self.prob = prob
        self.num_domains = num_domains

        self.relu = nn.ReLU(inplace=True)

    def set_lambda(self, lambd):
        self.lambd = lambd

    def forward(self, x, reverse=False):
        if not self.classaware_dp:
            x = self.dp_model.conv1(x)
            x = self.dp_model.bn1(x)
            x = self.dp_model.relu(x)
            x = self.dp_model.maxpool(x)

            x = self.dp_model.layer1(x)
            x = self.dp_model.layer2(x)

            x = self.dp_model.layer3(x)
            x = self.dp_model.layer4(x)
            x = self.dp_model.avgpool(x)
            #import pdb
            #pdb.set_trace()
            x = x.view(x.size(0), -1)

        x = self.relu(self.bn_fc5(self.fc5(x)))

        #x = self.avgpool(x)
        #x = x.view(x.shape[0],-1)
        #x = self.relu(self.bn_fc4(self.fc4(x)))

        dp_pred = self.dp_layer(x)

        return dp_pred

def generate_domains(num_domains, dataloader, ckpt, device):
    print("Generating domains from ckpt {}".format(ckpt))
    model_DP = DomainPredictor(num_domains)
    checkpoint = torch.load(ckpt)
    model_DP.load_state_dict(checkpoint["DP_state_dict"])
    model_DP = model_DP.to(device)
    model_DP.eval()

    dset_size = len(dataloader.dataset)
    probs_map = torch.zeros((dset_size, num_domains), dtype=torch.float32)
    total_batches = len(dataloader)
    with torch.no_grad():
        for batch_idx, data in enumerate(dataloader):
            img, label, idxes = data

            img, label,idxes = img.cuda(), label.long().cuda(), idxes.long().cuda()
            img, label = Variable(img, volatile=True), Variable(label)
            domain_logits = model_DP(img)
            domain_prob = nn.functional.softmax(domain_logits, dim=1)
            domain_prob = domain_prob.clone().detach().cpu()
            probs_map[idxes] = domain_prob
            if batch_idx%10==0:
                print("{}/{} done".format(batch_idx, total_batches))
    return probs_map



