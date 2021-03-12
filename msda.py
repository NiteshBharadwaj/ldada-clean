import torch
from itertools import combinations


def euclidean(x1, x2):
    return torch.sqrt(((x1 - x2) ** 2).sum() + 1e-8)


def k_moment(source_output, target_output, k):
    num_sources = len(source_output)
    source_output_ = []
    for i in range(num_sources):
        source_output_.append((source_output[i] ** k).mean(0))
    target_output = (target_output ** k).mean(0)

    kth_moment = 0
    for i in range(num_sources):
        kth_moment += euclidean(source_output_[i], target_output)

    comb = list(combinations(range(num_sources), 2))

    for k in range(len(comb)):
        kth_moment += euclidean(source_output_[comb[k][0]], source_output_[comb[k][1]])

    return kth_moment


def msda_regulizer(source_output, target_output, beta_moment):
    num_sources = len(source_output)
    s_mean = []
    source_output_ = []
    for i in range(num_sources):
        s_mean.append(source_output[i].mean(0))

    t_mean = target_output.mean(0)

    for i in range(num_sources):
        source_output_.append(source_output[i] - s_mean[i])

    target_output = target_output - t_mean

    # Compute first moment for nC2 combinations
    moment1 = 0
    for i in range(num_sources):
        moment1 += euclidean(source_output_[i], target_output)

    comb = list(combinations(range(num_sources), 2))

    for k in range(len(comb)):
        moment1 += euclidean(source_output_[comb[k][0]], source_output_[comb[k][1]])

    reg_info = moment1
    # print(reg_info)

    for i in range(beta_moment - 1):
        reg_info += k_moment(source_output_, target_output, i + 2)

    return reg_info / 6


def moment_soft(output_s, domain_prob, output_t):
    output_s = output_s.reshape(output_s.shape[0], output_s.shape[1],1)
    domain_prob = domain_prob.reshape(domain_prob.shape[0], 1, domain_prob.shape[1])
    output_prob = torch.matmul(output_s, domain_prob)
    output_prob_sum = domain_prob.sum(0)
    output_prob = output_prob.sum(0)/output_prob_sum.reshape(1, domain_prob.shape[2])
    loss = 0
    for i in range(output_prob.shape[1]):
        for j in range(i+1,output_prob.shape[1]):
            loss += output_prob_sum[0,i]*output_prob_sum[0,j]*euclidean(output_prob[:,i], output_prob[:,j])/output_s.shape[0]/output_s.shape[0]
        loss += output_prob_sum[0,i]*euclidean(output_prob[:,i], output_t)/output_s.shape[0]
    return loss


def k_moment_soft(output_s, output_t, k, domain_prob):
    output_s_k = (output_s**k)
    output_t = (output_t**k).mean(0)
    return moment_soft(output_s_k, domain_prob, output_t)

def msda_regulizer_soft(output_s, output_t, belta_moment, domain_prob):
    # print('s1:{}, s2:{}, s3:{}, s4:{}'.format(output_s1.shape, output_s2.shape, output_s3.shape, output_t.shape))        
    reg_info = 0
    reg_info = k_moment_soft(output_s, output_t, 1, domain_prob)
    for i in range(1,belta_moment):
        reg_info += k_moment_soft(output_s, output_t, i + 1, domain_prob)

    return reg_info / 6
# return euclidean(output_s1, output_t)

def k_moment_single(output_s, output_t, k):
	output_s_k = (output_s**k)
	output_s_mean = output_s_k.mean(0)
	output_t = (output_t**k).mean(0)
	return euclidean(output_s_mean, output_t)

def msda_regulizer_single(output_s, output_t, belta_moment):
	reg_info = 0
	reg_info += k_moment_single(output_s, output_t, 1)
# 	output_s_ = output_s -output_s.mean(0)
# 	output_t_ = output_t -output_t.mean(0)
	for i in range(1,belta_moment):
		reg_info += k_moment_single(output_s, output_t, i + 1)

	return reg_info / 6
