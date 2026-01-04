import logging
import os
import shutil
import sys
import scipy.sparse as sparse
from medpy.metric.binary import assd, dc
import numpy as np
import torch
from logging import get_logger

def save_checkpoint(state, is_best, checkpoint_dir, logger=None, num_classes=17):
    def log_info(message):
        if logger is not None:
            logger.info(message)
    if not os.path.exists(checkpoint_dir):
        log_info(f"Checkpoint directory does not exist. Creating {checkpoint_dir}")
        os.makedirs(checkpoint_dir)
    state['num_classes'] = num_classes
    last_file_path = os.path.join(checkpoint_dir, 'last_checkpoint.pytorch')
    log_info(f"Saving last checkpoint to '{last_file_path}'")
    torch.save(state, last_file_path)
    if is_best:
        best_file_path = os.path.join(checkpoint_dir, 'best_checkpoint.pytorch')
        log_info(f"Saving best checkpoint to '{best_file_path}'")
        shutil.copyfile(last_file_path, best_file_path)

def load_checkpoint(checkpoint_path, model, optimizer=None, logger=None):
    if not os.path.exists(checkpoint_path):
        raise IOError(f"Checkpoint '{checkpoint_path}' does not exist")
    state = torch.load(checkpoint_path, map_location='cpu')
    model_dict = model.state_dict()
    pretrained_dict = state.get('model_state_dict', state)
    new_state_dict = {}
    for k, v in pretrained_dict.items():
        if k in model_dict and model_dict[k].shape == v.shape:
            new_state_dict[k] = v
        elif logger is not None:
            logger.warning(f"Skipping mismatched key: {k} (checkpoint shape: {v.shape}, model shape: {model_dict[k].shape})")
    model.load_state_dict(new_state_dict, strict=False)
    if logger is not None:
        logger.info(f"Successfully loaded checkpoint from: {checkpoint_path} with partial compatibility")
    if optimizer is not None and 'optimizer_state_dict' in state:
        optimizer.load_state_dict(state['optimizer_state_dict'])
    return state

def get_number_of_learnable_parameters(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    return sum([np.prod(p.size()) for p in model_parameters])

class RunningAverage:
    def __init__(self):
        self.count = 0
        self.sum = 0
        self.avg = 0

    def update(self, value, n=1):
        self.count += n
        self.sum += value * n
        self.avg = self.sum / self.count

def find_maximum_patch_size(model, device):
    logger = get_logger('PatchFinder')
    in_channels = model.in_channels
    patch_shapes = [(256, 256), (512, 512), (256, 512), (512, 256)]
    for shape in patch_shapes:
        H, W = shape
        patch = np.random.randn(1, in_channels, H, W).astype('float32')
        patch = torch.from_numpy(patch).to(device)
        logger.info(f"Current patch size: {shape}")
        try:
            model(patch)
        except RuntimeError as e:
            logger.warning(f"Patch size {shape} failed: {e}")
            break

def unpad(probs, index, shape, pad_width=8):
    def _new_slices(slicing, max_size):
        if slicing.start == 0:
            p_start = 0
            i_start = 0
        else:
            p_start = pad_width
            i_start = slicing.start + pad_width
        if slicing.stop == max_size:
            p_stop = None
            i_stop = max_size
        else:
            p_stop = -pad_width
            i_stop = slicing.stop - pad_width
        return slice(p_start, p_stop), slice(i_start, i_stop)
    _, H, W = shape
    i_c, i_y, i_x = index
    p_c = slice(0, probs.shape[0])
    p_y, i_y = _new_slices(i_y, H)
    p_x, i_x = _new_slices(i_x, W)
    probs_index = (p_c, p_y, p_x)
    index = (i_c, i_y, i_x)
    return probs[probs_index], index

def create_feature_maps(init_channel_number, number_of_fmaps):
    return [init_channel_number * 2 ** k for k in range(number_of_fmaps)]

def adapted_rand(seg, gt, all_stats=False):
    epsilon = 1e-6
    segA = np.ravel(gt)
    segB = np.ravel(seg)
    n = segA.size
    n_labels_A = np.amax(segA) + 1
    n_labels_B = np.amax(segB) + 1
    ones_data = np.ones(n)
    p_ij = sparse.csr_matrix((ones_data, (segA[:], segB[:])), shape=(n_labels_A, n_labels_B))
    a = p_ij[1:n_labels_A, :]
    b = p_ij[1:n_labels_A, 1:n_labels_B]
    c = p_ij[1:n_labels_A, 0].todense()
    d = b.multiply(b)
    a_i = np.array(a.sum(1))
    b_i = np.array(b.sum(0))
    sumA = np.sum(a_i * a_i)
    sumB = np.sum(b_i * b_i) + (np.sum(c) / n)
    sumAB = np.sum(d) + (np.sum(c) / n)
    precision = sumAB / max(sumB, epsilon)
    recall = sumAB / max(sumA, epsilon)
    fScore = 2.0 * precision * recall / max(precision + recall, epsilon)
    are = 1.0 - fScore
    if all_stats:
        return are, precision, recall
    else:
        return are

def dice_per_class(prediction, target, eps=1e-10):
    prediction = prediction.astype(np.float32)
    target = target.astype(np.float32)
    intersect = np.sum(prediction * target)
    return (2. * intersect / (np.sum(prediction) + np.sum(target) + eps))

def intersect_per_class(prediction, target, eps=1e-10):
    prediction = prediction.astype(np.float32)
    target = target.astype(np.float32)
    intersect = np.sum(prediction * target)
    return intersect, np.sum(prediction), np.sum(target)

def dice_whole_class(prediction, target, eps=1e-10):
    intersect_sum = np.sum(prediction * target)
    prediction_sum = np.sum(prediction)
    target_sum = np.sum(target)
    return (2. * intersect_sum / (prediction_sum + target_sum + eps))

def assds_each_class(prediction, target, voxel_size=(1, 1), connectivity=1):
    if np.sum(target) == 0 and np.sum(prediction) == 0:
        return -1.0
    target_per_class = (target > 0).astype(np.uint8)
    prediction_per_class = (prediction > 0).astype(np.uint8)
    ad = assd(prediction_per_class, target_per_class, voxelspacing=voxel_size, connectivity=connectivity)
    return ad

def evaluation_metrics_each_class(prediction, target, num_classes=2, eps=1e-10):
    dscs = np.zeros(num_classes, dtype=np.float32)
    precisions = np.zeros(num_classes, dtype=np.float32)
    recalls = np.zeros(num_classes, dtype=np.float32)
    for cls in range(num_classes):
        target_per_class = (target == cls).astype(np.float32)
        prediction_per_class = (prediction == cls).astype(np.float32)
        tp = np.sum(prediction_per_class * target_per_class)
        fp = np.sum(prediction_per_class) - tp
        fn = np.sum(target_per_class) - tp
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        dsc = 2 * tp / (2 * tp + fp + fn + eps)
        dscs[cls] = dsc
        precisions[cls] = precision
        recalls[cls] = recall
    return dscs, precisions, recalls

def evaluation_accuracy(prediction, target):
    voxel_num = np.size(target)
    voxel_num = np.float32(voxel_num)
    tp = np.sum((prediction > 0) * (target > 0))
    accuracy = tp / voxel_num
    return accuracy

def np_onehot(label, num_classes=2):
    return np.eye(num_classes)[label.astype(np.int32)]

def expand_as_one_hot(input, C, ignore_index=None):
    assert input.dim() in [3, 4]
    if input.dim() == 3:
        input = input.unsqueeze(0)  # [H, W] -> [1, H, W] or [N, H, W] -> [N, H, W]
    elif input.dim() == 4 and input.size(0) == 1:
        input = input.squeeze(0)  # [1, H, W] -> [H, W]
    N = input.size(0)
    spatial_dims = input.size()[1:]
    input_min = input.min().item()
    input_max = input.max().item()
    LOGGER = get_logger('OneHot')
    LOGGER.debug(f"Input min: {input_min}, Input max: {input_max}, Expected range: [0, {C-1}]")
    if ignore_index is not None and input_min <= ignore_index <= input_max:
        input = input.clone()
        mask = input == ignore_index
        input[mask] = 0
    if not (0 <= input_min and input_max < C):
        LOGGER.error(f"Invalid label range [{input_min}, {input_max}] for C={C}")
        return torch.zeros(N, C, *spatial_dims, device=input.device, dtype=torch.float32)
    one_hot = torch.zeros(N, C, *spatial_dims, device=input.device, dtype=torch.float32)
    return one_hot.scatter_(1, input.long().unsqueeze(1), 1)