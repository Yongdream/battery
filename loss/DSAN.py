import torch
import torch.nn as nn

cuda_device = 1
torch.cuda.set_device(cuda_device)


def convert_to_onehot(sca_label, class_num):
    return torch.eye(class_num)[sca_label]


def gaussian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    n_samples = source[0].size(0) + target[0].size(0)
    total = torch.cat([source[0], target[0]], dim=0).cuda()
    total0 = total.unsqueeze(0).expand(total.size(0), total.size(0), total.size(1))
    total1 = total.unsqueeze(1).expand(total.size(0), total.size(0), total.size(1))
    L2_distance = ((total0 - total1) ** 2).sum(2)
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples**2 - n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    return sum(kernel_val)


def cal_weight(s_label, t_label, class_num=5):
    batch_size = s_label.shape[0]
    s_sca_label = s_label
    s_vec_label = convert_to_onehot(s_sca_label, class_num=class_num).float().cuda()
    s_sum = s_vec_label.sum(dim=0).view(1, class_num)
    s_sum[s_sum == 0] = 100
    s_vec_label = s_vec_label / s_sum

    t_sca_label = t_label.argmax(dim=1).cpu().numpy()
    t_vec_label = t_label.cpu().detach().numpy().astype('float32')
    t_sum = t_vec_label.sum(axis=0).reshape(1, class_num)
    t_sum[t_sum == 0] = 100
    t_vec_label = torch.from_numpy(t_vec_label / t_sum).cuda()

    index = list(set(s_sca_label) & set(t_sca_label))

    mask_arr = torch.zeros((batch_size, class_num)).cuda()
    mask_arr[:, index] = 1
    t_vec_label = t_vec_label * mask_arr
    s_vec_label = s_vec_label * mask_arr

    weight_ss = torch.matmul(s_vec_label, s_vec_label.t())
    weight_tt = torch.matmul(t_vec_label, t_vec_label.t())
    weight_st = torch.matmul(s_vec_label, t_vec_label.t())

    length = len(index)
    if length != 0:
        weight_ss = weight_ss / length
        weight_tt = weight_tt / length
        weight_st = weight_st / length
    else:
        weight_ss = torch.zeros(1).cuda()
        weight_tt = torch.zeros(1).cuda()
        weight_st = torch.zeros(1).cuda()
    return weight_ss, weight_tt, weight_st


def DSAN(source_list, target_list, kernel_muls=2, kernel_nums=5, fix_sigma_list=None, class_num=5):
    batch_size = source_list[0].size(0)
    s_label = source_list[1]
    t_label = target_list[1]

    s_label = s_label.argmax(dim=1).cpu().numpy()
    t_label = nn.functional.softmax(t_label, dim=1)

    weight_ss, weight_tt, weight_st = cal_weight(s_label, t_label, class_num=class_num)
    weight_ss = weight_ss.cuda()
    weight_tt = weight_tt.cuda()
    weight_st = weight_st.cuda()

    kernels = gaussian_kernel(source_list, target_list, kernel_mul=kernel_muls, kernel_num=kernel_nums, fix_sigma=fix_sigma_list)
    loss = torch.Tensor([0]).cuda()
    if torch.sum(torch.isnan(sum(kernels))):
        return loss
    SS = kernels[:batch_size, :batch_size].cuda()
    TT = kernels[batch_size:, batch_size:].cuda()
    ST = kernels[:batch_size, batch_size:].cuda()

    loss += torch.sum(weight_ss * SS + weight_tt * TT - 2 * weight_st * ST)
    return loss

# source_list = [torch.randn((10, 128)).cuda(), torch.rand((10, 5)).cuda()]
# target_list = [torch.randn((10, 128)).cuda(), torch.rand((10, 5)).cuda()]
#
# loss = DSAN(source_list, target_list)
# print(loss)
