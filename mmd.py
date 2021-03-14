import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    n_samples = int(source.size()[0])+int(target.size()[0])
    total = torch.cat([source, target], dim=0)
    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    L2_distance = ((total0-total1)**2).sum(2)
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    return sum(kernel_val)#/len(kernel_val)


def mmd(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    batch_size = int(source.size()[0])
    kernels = guassian_kernel(source, target,
        kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)

    loss1 = 0
    for s1 in range(batch_size):
        for s2 in range(s1+1, batch_size):
            t1, t2 = s1+batch_size, s2+batch_size
            loss1 += kernels[s1, s2] + kernels[t1, t2]
    loss1 = loss1 / float(batch_size * (batch_size - 1) / 2)

    loss2 = 0
    for s1 in range(batch_size):
        for s2 in range(batch_size):
            t1, t2 = s1+batch_size, s2+batch_size
            loss2 -= kernels[s1, t2] + kernels[s2, t1]
    loss2 = loss2 / float(batch_size * batch_size)
    return loss1 + loss2

def mmd_soft_helper(kernels, batch_size, p_d1_x, p_d2_x, p_d1, p_d2):
    loss_xx = 0
    loss_yy = 0
    loss_xy = 0
    for s1 in range(batch_size):
        for s2 in range(s1, batch_size):
            if s1!=s2:
                loss_xx += kernels[s1, s2]*p_d1_x[s1]*p_d1_x[s2]
                loss_yy += kernels[s1, s2]*p_d2_x[s1]*p_d2_x[s2]
            loss_xy -= kernels[s1, s2]*p_d1_x[s1]*p_d2_x[s2] + kernels[s2, s1]*p_d1_x[s2]*p_d2_x[s1]
    loss_xx = loss_xx/p_d1/p_d1
    loss_yy = loss_yy/p_d2/p_d2
    loss_xy = loss_xy/p_d1/p_d2
    loss1 = loss_xx + loss_yy
    loss2 = loss_xy
    loss1 = loss1 / float(batch_size * (batch_size - 1) / 2)
    loss2 = loss2 / float(batch_size * batch_size)
    return loss1 + loss2


def mmd_soft_helper2(kernels, batch_size, p_d1_x, p_d1):
    loss_xx = 0
    loss_yy = 0
    loss_xy = 0
    loss1 = 0
    for s1 in range(batch_size):
        for s2 in range(s1 + 1, batch_size):
            t1, t2 = s1 + batch_size, s2 + batch_size
            if s1!=s2:
                loss_xx += kernels[s1, s2] * p_d1_x[s1] * p_d1_x[s2]
                loss_yy += kernels[t1, t2]
            loss_xy += -kernels[s1, t2] * p_d1_x[s1] + kernels[s2, t1] * p_d1_x[s2]
    loss_xx = loss_xx/p_d1/p_d1
    loss1 = loss_xx + loss_yy
    loss2 = loss_xy
    loss1 = loss1 / float(batch_size * (batch_size - 1) / 2)
    loss2 = loss2 / float(batch_size * batch_size)
    return loss1 + loss2

def mmd_soft(source, target, domain_prob, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    domain_prob_agg = domain_prob.sum(0)/domain_prob.shape[0]
    batch_size = int(source.size()[0])
    kernels = guassian_kernel(source, target,
        kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)

    num_domains = domain_prob.shape[1]
    total_mmd = 0
    for dom1 in range(num_domains):
        for dom2 in range(dom1+1,num_domains):
            total_mmd += 2*domain_prob_agg[dom1]*domain_prob_agg[dom2]* \
                         mmd_soft_helper(kernels, batch_size,
                                         domain_prob[:,dom1], domain_prob[:,dom2],
                                         domain_prob_agg[dom1], domain_prob_agg[dom2])
        total_mmd += domain_prob_agg[dom1]*\
                     mmd_soft_helper2(kernels, batch_size,
                                     domain_prob[:,dom1], domain_prob_agg[dom1])

    return total_mmd