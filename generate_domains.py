import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import time
from model_no_class import DomainPredictor

def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


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



