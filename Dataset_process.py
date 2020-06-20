import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import scipy
import scipy.ndimage
import torch
import torch.utils.data as udata


def ReadData(file_path, final_num=584):
    voxel = []
    seg = []
    for i in tqdm(range(final_num), desc='reading'):
        try:
            data = np.load(file_path.format(i))
        except FileNotFoundError:
            continue
        try:
            voxel = np.append(voxel, np.expand_dims(data['voxel'], axis=0), axis=0)
            seg = np.append(seg, np.expand_dims(data['seg'], axis=0), axis=0)
        except ValueError:
            voxel = np.expand_dims(data['voxel'], axis=0)
            seg = np.expand_dims(data['seg'], axis=0)
    file_num = voxel.shape[0]
    return voxel, seg, file_num


def one_hot_label(label_file_path):
    label = pd.read_csv(label_file_path).values[:, 1].astype(int)
    label = np.eye(2)[label]
    return label


def random_split(voxel, mask, label, ratio=0.2, train_set=4):
    length = voxel.shape[1]
    label_0 = label[label == 0]
    label_1 = label[label == 1]
    voxel_0 = voxel[label == 0]
    voxel_1 = voxel[label == 1]
    mask_0 = mask[label == 0]
    mask_1 = mask[label == 1]
    num_0 = label_0.size
    num_1 = label_1.size
    rdn_idx_0 = np.random.permutation(num_0)
    label_0 = label_0[rdn_idx_0]
    voxel_0 = voxel_0[rdn_idx_0]
    mask_0 = mask_0[rdn_idx_0]
    rdn_idx_1 = np.random.permutation(num_1)
    label_1 = label_1[rdn_idx_1]
    voxel_1 = voxel_1[rdn_idx_1]
    mask_1 = mask_1[rdn_idx_1]
    num_val_0 = round(num_0 * ratio)
    num_val_1 = round(num_1 * ratio)
    num_train_0 = round((num_0 - num_val_0) / train_set)
    num_train_1 = round((num_1 - num_val_1) / train_set)
    num_train = num_train_0 + num_train_1
    train_voxel = np.zeros((train_set, num_train, length, length, length))
    train_mask = np.zeros((train_set, num_train, length, length, length))
    train_label = np.zeros((train_set, num_train))
    for i in range(train_set):
        train_voxel[i, 0 : num_train_0] = \
            voxel_0[num_train_0 * i:num_train_0 * (i + 1)]
        train_voxel[i, num_train_0 : num_train] = \
            voxel_1[num_train_1 * i:num_train_1 * (i + 1)]
        train_mask[i, 0 : num_train_0] = \
            mask_0[num_train_0 * i:num_train_0 * (i + 1)]
        train_mask[i, num_train_0 : num_train] = \
            mask_1[num_train_1 * i:num_train_1 * (i + 1)]
        train_label[i, 0 : num_train_0] = \
            label_0[num_train_0 * i:num_train_0 * (i + 1)]
        train_label[i, num_train_0 : num_train] = \
            label_1[num_train_1 * i:num_train_1 * (i + 1)]
        rdn_idx = np.random.permutation(num_train)
        train_voxel[i] = train_voxel[i][rdn_idx]
        train_mask[i] = train_mask[i][rdn_idx]
        train_label[i] = train_label[i][rdn_idx]
        print("trainset",i+1," finished !")
    val_voxel = np.append(voxel_0[num_train_0 * train_set:],voxel_1[num_train_1 * train_set:],axis=0)
    val_mask = np.append(mask_0[num_train_0 * train_set:],mask_1[num_train_1 * train_set:],axis=0)
    val_label = np.append(label_0[num_train_0 * train_set:], label_1[num_train_1 * train_set:], axis=0)
    num_val = val_voxel.shape[0]
    rdn_idx = np.random.permutation(num_val)
    val_voxel = val_voxel[rdn_idx]
    val_mask = val_mask[rdn_idx]
    val_label = val_label[rdn_idx]
    print("valset finished !")
    total_num_train = num_train * train_set
    return train_voxel, train_mask, train_label, val_voxel, val_mask, val_label, total_num_train, num_val


def random_split_2(voxel, mask, label, ratio=0.2, train_set=1):
    length = voxel.shape[1]
    indices = np.random.permutation(label.shape[0])
    label = label[indices]
    voxel = voxel[indices]
    mask = mask[indices]
    total_size = label.shape[0]
    validate_size = round(total_size * ratio)
    train_size = round((total_size - validate_size) / train_set)
    train_voxel = np.zeros((train_set, train_size, length, length, length))
    train_mask = np.zeros((train_set, train_size, length, length, length))
    train_label = np.zeros((train_set, train_size))
    for i in range(train_set):
        train_voxel[i] = voxel[train_size * i:train_size * (i + 1)]
        train_label[i] = label[train_size * i:train_size * (i + 1)]
        train_mask[i] = mask[train_size * i:train_size * (i + 1)]
    validate_voxel = voxel[train_size * train_set:]
    validate_label = label[train_size * train_set:].astype(np.float64)
    validate_mask = mask[train_size * train_set:]
    train_size = train_size * train_set
    validate_size = validate_voxel.shape[0]
    return train_voxel, train_mask, train_label, validate_voxel, validate_mask, validate_label, train_size, validate_size


def train_data_process(voxel, mask, label, batch_size):
    dataset = udata.ConcatDataset([
        data_augment(voxel, mask, label, mix_up=False, his_equalized_after_crop=True),
        data_augment(voxel, mask, label, random_move=False, mix_up=True),
        # data_augment(voxel, mask, label, masked=True,random_move=False, mix_up=False),
        # data_augment(voxel, mask, label, rotation=True, angle= [1, 1, 1], mix_up=False),
        # data_augment(voxel, mask, label, random_move=True, rotation=True, angle=[2, 2, 2], mix_up=False),
        data_augment(voxel, mask, label, masked=False, rotation=True, angle=[1, 2, 3], mix_up=False,his_equalized_after_crop=False),
        # data_augment(voxel, mask, label, random_move=True, rotation=True, angle=[2, 1, 3], mix_up=False),
        # data_augment(voxel, mask, label, masked=False, reflection=True, axis='rand', mix_up=False,his_equalized_after_crop=False),
    ])
    total_size = voxel.shape[0] * 2
    train_loader = udata.DataLoader(dataset, batch_size, shuffle=True)
    return train_loader, total_size


def val_data_process(voxel, mask, label, batch_size, crop_size=32,
                     his_equalized_before_mask=False,his_equalized_after_crop=False,
                     mix_up=False):
    data_num = voxel.shape[0]
    if his_equalized_before_mask:
        voxel = his_equalize(voxel)
    # voxel = add_mask(voxel, mask)
    center = get_center(mask)
    voxel = crop(voxel, center, crop_size)
    if his_equalized_after_crop:
        voxel = his_equalize(voxel)
    voxel = standardize(voxel)
    if mix_up:
        voxel, label = mixup(voxel, label)
    dataset = pack(voxel, label)
    val_loader = udata.DataLoader(dataset, batch_size, shuffle=True)
    return val_loader, data_num


def test_data_process(voxel, mask, batch_size, crop_size=32,
                      his_equalized_before_mask=False,his_equalized_after_crop=False,
                      mix_up=False):
    size = voxel.shape[0]
    if his_equalized_before_mask:
        voxel = his_equalize(voxel)
    # voxel = add_mask(voxel, mask)
    center = get_center(mask)
    voxel = crop(voxel, center, crop_size)
    if his_equalized_after_crop:
        voxel = his_equalize(voxel)
    voxel = standardize(voxel)
    if mix_up:
        voxel = mixup_for_test(voxel)
    voxel = torch.from_numpy(voxel).to(dtype=torch.float32)
    voxel = voxel.unsqueeze(1)
    # dataset = udata.TensorDataset(voxel)
    test_loader = udata.DataLoader(voxel, batch_size, shuffle=False)
    return test_loader, size


def data_augment(voxel, mask, label, masked=False, mix_up=True,
                 resize=False, factor_range=[0.8, 1.15],
                 random_move=False, max_move=3,
                 crop_size=32,
                 rotation=False, angle=[0, 0, 0],
                 reflection=False, axis='rand',
                 standard=True,
                 his_equalized_before_mask=False,his_equalized_after_crop=False):
    mask = mask.astype(np.float32)
    if his_equalized_before_mask:
        voxel = his_equalize(voxel)
    if masked:
        voxel = add_mask(voxel, mask)
    if resize:
        voxel, mask = random_resize(voxel, mask, factor_range)
    center = get_center(mask)
    if random_move:
        center = random_move_center(center, max_move, voxel.shape[1])
    voxel = crop(voxel, center, crop_size)
    if his_equalized_after_crop:
        voxel = his_equalize(voxel)
    if rotation:
        voxel = rotate(voxel, angle)
    if reflection:
        voxel = flip(voxel, axis)
    if standard:
        voxel = standardize(voxel)
    if mix_up:
        voxel, label = mixup(voxel, label)
    dataset = pack(voxel, label)
    return dataset


def mixup(voxel, label):
    indices = np.random.permutation(voxel.shape[0])
    voxel_mix = voxel[indices]
    label_mix = label[indices]
    for i in range(voxel.shape[0]):
        # alpha = np.random.beta(0.2, 0.2)
        alpha = np.random.random(1).item()
        voxel_mix[i] = voxel[i] * alpha + voxel_mix[i] * (1 - alpha)
        label_mix[i] = label[i] * alpha + label_mix[i] * (1 - alpha)
    return voxel_mix, label_mix


def mixup_for_test(voxel):
    indices = np.random.permutation(voxel.shape[0])
    voxel_mix = voxel[indices]
    for i in range(voxel.shape[0]):
        # alpha = np.random.beta(0.2, 0.2)
        alpha = np.random.random(1).item()
        voxel_mix[i] = voxel[i] * alpha + voxel_mix[i] * (1 - alpha)
    return voxel


def pack(voxel, label):
    voxel = torch.from_numpy(voxel).to(dtype=torch.float32)
    label = torch.from_numpy(label[0:voxel.shape[0]]).to(dtype=torch.float32)
    voxel = voxel.unsqueeze(1)
    dataset = udata.TensorDataset(voxel, label)
    return dataset


def add_mask(voxel, mask):
    voxel = voxel * (mask.astype(np.float32))
    return voxel


def random_resize(voxel, mask, factor_range=[0.8, 1.15]):
    batch_size = voxel.shape[0]
    resize_factor = np.random.rand() * (factor_range[1] - factor_range[0]) + factor_range[0]
    size = round(voxel.shape[1] * resize_factor)
    resized_data = np.zeros((batch_size, size, size, size))
    resized_mask = np.zeros_like(resized_data)
    for i in range(batch_size):
        resized_data[i] = scipy.ndimage.interpolation.zoom(voxel[i], resize_factor, order=1)
        resized_mask[i] = scipy.ndimage.interpolation.zoom(mask[i], resize_factor, order=0)
    return resized_data, resized_mask


def his_equalize(data):
    data_his = np.zeros(data.shape, dtype=float)
    for i in range(data.shape[0]):
        data_size = data[i].size
        data_part = data[i]
        v, cnt = np.unique(data_part, return_counts=True)
        sum = np.cumsum(cnt)
        v_his = np.round(sum / data_size * 255)
        data_his_part = np.zeros(data_part.shape, dtype=int)
        for j in range(v.size):
            data_his_part[data_part == v[j]] = v_his[j]
        data_his[i] = data_his_part
    return data_his


def get_center(mask):
    x = np.sum(mask, axis=(2, 3))
    y = np.sum(mask, axis=(1, 3))
    z = np.sum(mask, axis=(1, 2))
    mask_num = mask.shape[0]
    center = np.zeros((mask_num, 3), dtype=int)
    for i in range(mask_num):
        x_area = np.where(x[i] > 0)
        center[i, 0] = (np.min(x_area) + np.max(x_area)) // 2
        y_area = np.where(y[i] > 0)
        center[i, 1] = (np.min(y_area) + np.max(y_area)) // 2
        z_area = np.where(z[i] > 0)
        center[i, 2] = (np.min(z_area) + np.max(z_area)) // 2
    return center


def random_move_center(center, maxmove=3, bound=100):
    movement = np.random.randint(low=-maxmove, high=maxmove, size=center.shape)
    moved = center + movement
    moved[moved < 0] = 0
    moved[moved > bound - 1] = bound - 1
    return moved


def crop(data, center, size=32):
    bound = data.shape[1]
    data_num = data.shape[0]
    cropped = np.zeros((data_num, size, size, size))
    center[center < size // 2] = size // 2
    center[center > bound - size // 2 - size % 2] = bound - size // 2 - size % 2
    for i in range(data_num):
        low = center[i] - size // 2
        high = center[i] + size // 2 + size % 2
        cropped[i] = data[i, low[0]:high[0], low[1]:high[1], low[2]:high[2]]
    return cropped


def rotate(data, angle):
    data_num = data.shape[0]
    rotated_data = np.zeros_like(data)
    for i in range(data_num):
        X = np.rot90(data[i], angle[0], axes=(0, 1))  # rotate in X-axis
        Y = np.rot90(X, angle[1], axes=(0, 2))  # rotate in Y'-axis
        rotated_data[i] = np.rot90(Y, angle[2], axes=(1, 2))  # rotate in Z"-axis
    return rotated_data

def flip(data, axis='rand'):
    batch_size = data.shape[0]
    if axis == 'rand':
        axis_rand = np.random.randint(0, 3, batch_size)
    for i in range(batch_size):
        if axis == 'rand':
            data[i] = np.flip(data[i], axis_rand[i])
        else:
            data[i] = np.flip(data[i], int(axis))
    return data


def standardize(data):
    std = np.std(data)
    mean = np.mean(data)
    data = (data - mean) / std
    return data

def mixup_data(x, y, alpha=0.2):
    """
    Returns mixed inputs, pairs of targets, and lambda
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    mixed_x = lam * x + (1 - lam) * x.flip(dims=(0,))
    y_a, y_b = y, y.flip(dims=(0,))
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def mixup_data_for_test(x, alpha=0.2):
    """
    Returns mixed inputs, pairs of targets, and lambda
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    mixed_x = lam * x + (1 - lam) * x.flip(dims=(0,))
    return mixed_x
