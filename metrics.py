import importlib
import numpy as np
import torch
import torch.nn.functional as F
from skimage import measure
from losses import compute_per_channel_dice, expand_as_one_hot
from logging import get_logger
from utils import adapted_rand

LOGGER = get_logger('EvalMetric')

SUPPORTED_METRICS = ['dice', 'iou', 'boundary_ap', 'dt_ap', 'quantized_dt_ap', 'angle', 'inverse_angular']


class DiceCoefficient:
    def __init__(self, epsilon=1e-5, ignore_index=None, skip_channels=None, **kwargs):
        self.epsilon = epsilon
        self.ignore_index = ignore_index
        self.skip_channels = skip_channels if skip_channels is not None else []

    def __call__(self, input, target):
        assert input.dim() in [4, 5], f"Expected input dim 4 or 5, got {input.dim()}"
        n_classes = input.size()[1]
        batch_size = input.size()[0]
        LOGGER.debug(f"Input shape: {input.shape}, Target shape: {target.shape}, n_classes: {n_classes}")

        if input.dim() == 5 and target.dim() == 4:
            target = expand_as_one_hot(target, C=n_classes, ignore_index=self.ignore_index)
        elif input.dim() == 4 and target.dim() == 3:
            target = expand_as_one_hot(target, C=n_classes, ignore_index=self.ignore_index)

        assert input.size() == target.size(), f"Shape mismatch: input {input.shape}, target {target.shape}"
        LOGGER.debug(f"Processed Input shape: {input.shape}, Processed Target shape: {target.shape}")

        per_channel_dice = []
        for b in range(batch_size):
            binary_prediction = self._binarize_predictions(input[b], n_classes)
            target_b = target[b]

            if self.ignore_index is not None:
                mask = target_b == self.ignore_index
                binary_prediction[mask] = 0
                target_b[mask] = 0

            binary_prediction = binary_prediction.byte()
            target_b = target_b.byte()

            for c in range(n_classes):
                if c in self.skip_channels:
                    LOGGER.debug(f"Skipping channel {c} as specified in skip_channels")
                    continue
                try:
                    target_sum = torch.sum(target_b[c])
                    pred_sum = torch.sum(binary_prediction[c])
                    LOGGER.debug(f"Batch {b}, Channel {c}: Target sum: {target_sum}, Prediction sum: {pred_sum}")
                    if target_sum < self.epsilon and pred_sum < self.epsilon:
                        LOGGER.debug(f"Batch {b}, Channel {c}: Empty target and prediction, skipping")
                        continue
                    dice_score = self._dice(binary_prediction[c], target_b[c])
                    per_channel_dice.append(dice_score)
                    LOGGER.info(f"Batch {b}, Channel {c} Dice: {dice_score:.4f}")
                except (IndexError, RuntimeError) as e:
                    LOGGER.error(f"Error at Batch {b}, Channel {c}: {e}")
                    continue

        if not per_channel_dice:
            LOGGER.warning("No valid channels for Dice calculation, returning 0")
            return torch.tensor(0.0)
        mean_dice = torch.mean(torch.tensor(per_channel_dice, dtype=torch.float))
        LOGGER.info(f"Mean Dice across {len(per_channel_dice)} channels: {mean_dice:.4f}")
        return mean_dice

    def _dice(self, prediction, target, eps=1e-7):
        intersect = torch.sum(prediction & target).float()
        return (2. * intersect / ((torch.sum(prediction) + torch.sum(target)).float() + eps))

    def _binarize_predictions(self, input, n_classes):
        if n_classes == 2:
            result = input > 0.5
            return result.long()
        _, max_index = torch.max(input, dim=0, keepdim=True)
        return torch.zeros_like(input, dtype=torch.uint8).scatter_(0, max_index, 1)


class MeanIoU:
    def __init__(self, skip_channels=(), ignore_index=None, **kwargs):
        self.ignore_index = ignore_index
        self.skip_channels = skip_channels if skip_channels is not None else []

    def __call__(self, input, target):
        assert input.dim() in [4, 5]
        n_classes = input.size()[1]
        if input.dim() == 5 and target.dim() == 4:
            target = expand_as_one_hot(target, C=n_classes, ignore_index=self.ignore_index)
        elif input.dim() == 4 and target.dim() == 3:
            target = expand_as_one_hot(target, C=n_classes, ignore_index=self.ignore_index)

        input = input[0]
        target = target[0]
        assert input.size() == target.size()

        binary_prediction = self._binarize_predictions(input, n_classes)

        if self.ignore_index is not None:
            mask = target == self.ignore_index
            binary_prediction[mask] = 0
            target[mask] = 0

        binary_prediction = binary_prediction.byte()
        target = target.byte()

        per_channel_iou = []
        for c in range(n_classes):
            if c in self.skip_channels:
                LOGGER.debug(f"Skipping channel {c} as specified in skip_channels")
                continue
            if torch.sum(target[c]) == 0:
                LOGGER.debug(f"Channel {c}: Empty target, skipping")
                continue
            iou_score = self._jaccard_index(binary_prediction[c], target[c])
            per_channel_iou.append(iou_score)
            LOGGER.info(f"Channel {c} IoU: {iou_score:.4f}")

        if not per_channel_iou:
            LOGGER.warning("No valid channels for IoU calculation, returning 0")
            return torch.tensor(0.0)
        mean_iou = torch.mean(torch.tensor(per_channel_iou, dtype=torch.float))
        LOGGER.info(f"Mean IoU across {len(per_channel_iou)} channels: {mean_iou:.4f}")
        return mean_iou

    def _binarize_predictions(self, input, n_classes):
        if n_classes == 2:
            result = input > 0.5
            return result.long()
        _, max_index = torch.max(input, dim=0, keepdim=True)
        return torch.zeros_like(input, dtype=torch.uint8).scatter_(0, max_index, 1)

    def _jaccard_index(self, prediction, target, eps=1e-7):
        return torch.sum(prediction & target).float() / (torch.sum(prediction | target).float() + eps)


class AdaptedRandError:
    def __init__(self, all_stats=False, **kwargs):
        self.all_stats = all_stats

    def __call__(self, input, target):
        return adapted_rand(input, target, all_stats=self.all_stats)


class BoundaryAveragePrecision:
    def __init__(self, threshold=0.4, iou_range=(0.5, 1.0), ignore_index=-1, min_instance_size=None,
                 use_last_target=False, **kwargs):
        self.threshold = threshold
        self.iou_range = iou_range
        self.ignore_index = ignore_index
        self.min_instance_size = min_instance_size
        self.use_last_target = use_last_target

    def __call__(self, input, target):
        if isinstance(input, torch.Tensor):
            assert input.dim() == 5
            input = input[0].detach().cpu().numpy()

        if isinstance(target, torch.Tensor):
            if not self.use_last_target:
                assert target.dim() == 4
                target = target[0].detach().cpu().numpy()
            else:
                assert target.dim() == 5
                target = target[0, -1].detach().cpu().numpy()

        if isinstance(input, np.ndarray):
            assert input.ndim == 4

        if isinstance(target, np.ndarray):
            assert target.ndim == 3

        target, target_instances = self._filter_instances(target)

        per_channel_ap = []
        for c in range(input.shape[0]):
            predictions = input[c]
            predictions = predictions > self.threshold
            predictions = np.logical_not(predictions).astype(np.uint8)
            predicted = measure.label(predictions, background=0, connectivity=1)
            ap = self._calculate_average_precision(predicted, target, target_instances)
            per_channel_ap.append(ap)
            LOGGER.info(f"Channel {c} AP: {ap:.4f}")
        max_ap, c_index = np.max(per_channel_ap), np.argmax(per_channel_ap)
        LOGGER.info(f'Max average precision: {max_ap:.4f}, channel: {c_index}')
        return max_ap

    def _filter_instances(self, input):
        if self.min_instance_size is not None:
            labels, counts = np.unique(input, return_counts=True)
            for label, count in zip(labels, counts):
                if count < self.min_instance_size:
                    mask = input == label
                    input[mask] = self.ignore_index
        labels = set(np.unique(input))
        labels.discard(self.ignore_index)
        return input, labels

    def _calculate_average_precision(self, predicted, target, target_instances):
        recall, precision = self._roc_curve(predicted, target, target_instances)
        recall.insert(0, 0.0)
        recall.append(1.0)
        precision.insert(0, 0.0)
        precision.append(0.0)
        for i in range(len(precision) - 2, -1, -1):
            precision[i] = max(precision[i], precision[i + 1])
        ap = 0.0
        for i in range(1, len(recall)):
            ap += ((recall[i] - recall[i - 1]) * precision[i])
        return ap

    def _roc_curve(self, predicted, target, target_instances):
        ROC = []
        predicted, predicted_instances = self._filter_instances(predicted)
        for min_iou in np.arange(self.iou_range[0], self.iou_range[1], 0.1):
            false_negatives = set(target_instances)
            false_positives = set(predicted_instances)
            true_positives = set()
            for pred_label in predicted_instances:
                target_label = self._find_overlapping_target(pred_label, predicted, target, min_iou)
                if target_label is not None:
                    if target_label == self.ignore_index:
                        false_positives.discard(pred_label)
                    else:
                        true_positives.add(pred_label)
                        false_positives.discard(pred_label)
                        false_negatives.discard(target_label)
            tp = len(true_positives)
            fp = len(false_positives)
            fn = len(false_negatives)
            recall = tp / (tp + fn + 1e-7)
            precision = tp / (tp + fp + 1e-7)
            ROC.append((recall, precision))
        ROC = np.array(sorted(ROC, key=lambda t: t[0]))
        return list(ROC[:, 0]), list(ROC[:, 1])

    def _find_overlapping_target(self, predicted_label, predicted, target, min_iou):
        mask_predicted = predicted == predicted_label
        overlapping_labels = target[mask_predicted]
        labels, counts = np.unique(overlapping_labels, return_counts=True)
        target_label_ind = np.argmax(counts)
        target_label = labels[target_label_ind]
        mask_target = target == target_label
        if self._iou(mask_predicted, mask_target) > min_iou:
            return target_label
        return None

    @staticmethod
    def _iou(prediction, target):
        intersection = np.logical_and(prediction, target)
        union = np.logical_or(prediction, target)
        return np.sum(intersection) / (np.sum(union) + 1e-7)


def get_evaluation_metric(config):
    def _metric_class(class_name):
        m = importlib.import_module('networks.metrics')
        clazz = getattr(m, class_name)
        return clazz

    assert 'eval_metric' in config, 'Could not find evaluation metric configuration'
    metric_config = config['eval_metric']
    metric_class = _metric_class(metric_config['name'])
    return metric_class(**metric_config)