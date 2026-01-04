import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable
from torch.nn import MSELoss, SmoothL1Loss, L1Loss

def compute_per_channel_dice(input, target, epsilon=1e-5, ignore_index=None, weight=None):
    assert input.size() == target.size(), "'input' and 'target' must have the same shape"
    if ignore_index is not None:
        mask = target.clone().ne_(ignore_index)
        mask.requires_grad = False
        input = input * mask
        target = target * mask
    input = flatten(input)
    target = flatten(target)
    target = target.float()
    intersect = (input * target).sum(-1)
    if weight is not None:
        intersect = weight * intersect
    denominator = (input + target).sum(-1)
    return 2. * intersect / denominator.clamp(min=epsilon)

class DiceLoss(nn.Module):
    def __init__(self, epsilon=1e-5, weight=None, ignore_index=None, sigmoid_normalization=True,
                 skip_last_target=False):
        super(DiceLoss, self).__init__()
        self.epsilon = epsilon
        self.register_buffer('weight', weight)
        self.ignore_index = ignore_index
        self.normalization = nn.Sigmoid() if sigmoid_normalization else nn.Softmax(dim=1)
        self.skip_last_target = skip_last_target

    def forward(self, input, target):
        input = self.normalization(input)
        if self.weight is not None:
            weight = Variable(self.weight, requires_grad=False)
        else:
            weight = None
        if self.skip_last_target:
            target = target[:, :-1, ...]
        target = expand_as_one_hot(target, C=input.size()[1], ignore_index=self.ignore_index)
        per_channel_dice = compute_per_channel_dice(input, target, epsilon=self.epsilon,
                                                    ignore_index=self.ignore_index, weight=weight)
        return torch.mean(1. - per_channel_dice)

class FPFNLoss(nn.Module):
    def __init__(self, lamda=0.1):
        super(FPFNLoss, self).__init__()
        self.lamda = lamda
        self.activation = nn.Sigmoid()

    def forward(self, input, target, weights=None):
        target = expand_as_one_hot(target, C=input.size()[1])  # [N, C, H, W]
        input = self.activation(input)
        if weights is not None:
            weights = torch.unsqueeze(weights, dim=1)  # [N, 1, H, W]
            weights = Variable(weights, requires_grad=False)
        else:
            weights = 1.0
        fp = torch.sum(weights * (1 - target) * input, dim=(1, 2, 3))
        fn = torch.sum(weights * (1 - input) * target, dim=(1, 2, 3))
        loss = self.lamda * fp + fn
        return torch.mean(loss)

class LSLoss(nn.Module):
    def __init__(self, epsilon=0.05, lamda=4e-4):
        super(LSLoss, self).__init__()
        self.epsilon = epsilon
        self.lamda = lamda
        self.activation = nn.Sigmoid()
        self.ceLoss = nn.BCEWithLogitsLoss()

    def forward(self, input, target):
        eps = 1e-8
        ceLoss = self.ceLoss(input, target.unsqueeze(1).float())
        target = expand_as_one_hot(target, C=input.size()[1])  # [N, C, H, W]
        target = self.activation(target)
        sdm = target - 0.5
        h = 0.5 * (1. + torch.tanh(sdm / self.epsilon))
        h_sum = torch.sum(h, dim=(2, 3)) + eps
        h_sum_ = torch.sum(1. - h, dim=(2, 3)) + eps
        cin = torch.div(torch.sum(torch.mul(target, h), dim=(2, 3)), h_sum)
        cout = torch.div(torch.sum(torch.mul(target, 1. - h), dim=(2, 3)), h_sum_)
        cin = cin.unsqueeze(2).unsqueeze(3).expand(target.size())
        cout = cout.unsqueeze(2).unsqueeze(3).expand(target.size())
        inLoss = torch.sum(torch.mul(torch.pow(target - cin, 2), h), dim=(2, 3))
        outLoss = torch.sum(torch.mul(torch.pow(target - cout, 2), 1 - h), dim=(2, 3))
        lsLoss = torch.mean(inLoss + outLoss)
        loss = ceLoss + self.lamda * lsLoss
        return loss

class GeneralizedDiceLoss(nn.Module):
    def __init__(self, epsilon=1e-5, weight=None, ignore_index=None, sigmoid_normalization=True):
        super(GeneralizedDiceLoss, self).__init__()
        self.epsilon = epsilon
        self.register_buffer('weight', weight)
        self.ignore_index = ignore_index
        self.normalization = nn.Sigmoid() if sigmoid_normalization else nn.Softmax(dim=1)

    def forward(self, input, target):
        input = self.normalization(input)
        assert input.size() == target.size(), "'input' and 'target' must have the same shape"
        if self.ignore_index is not None:
            mask = target.clone().ne_(self.ignore_index)
            mask.requires_grad = False
            input = input * mask
            target = target * mask
        input = flatten(input)
        target = flatten(target)
        target = target.float()
        target_sum = target.sum(-1)
        class_weights = Variable(1. / (target_sum * target_sum).clamp(min=self.epsilon), requires_grad=False)
        intersect = (input * target).sum(-1) * class_weights
        if self.weight is not None:
            weight = Variable(self.weight, requires_grad=False)
            intersect = weight * intersect
        intersect = intersect.sum()
        denominator = ((input + target).sum(-1) * class_weights).sum()
        return 1. - 2. * intersect / denominator.clamp(min=self.epsilon)

class WeightedCrossEntropyLoss(nn.Module):
    def __init__(self, weight=None, ignore_index=-1):
        super(WeightedCrossEntropyLoss, self).__init__()
        self.register_buffer('weight', weight)
        self.ignore_index = ignore_index

    def forward(self, input, target):
        class_weights = self._class_weights(input)
        if self.weight is not None:
            weight = Variable(self.weight, requires_grad=False)
            class_weights = class_weights * weight
        return F.cross_entropy(input, target, weight=class_weights, ignore_index=self.ignore_index)

    @staticmethod
    def _class_weights(input):
        input = F.softmax(input, dim=1)
        flattened = flatten(input)
        nominator = (1. - flattened).sum(-1)
        denominator = flattened.sum(-1)
        class_weights = Variable(nominator / denominator, requires_grad=False)
        return class_weights

class BCELossWrapper:
    def __init__(self, loss_criterion, ignore_index=-1, skip_last_target=False):
        if hasattr(loss_criterion, 'ignore_index'):
            raise RuntimeError(f"Cannot wrap {type(loss_criterion)}. Use 'ignore_index' attribute instead")
        self.loss_criterion = loss_criterion
        self.ignore_index = ignore_index
        self.skip_last_target = skip_last_target

    def __call__(self, input, target):
        if self.skip_last_target:
            target = target[:, :-1, ...]
        assert input.size() == target.size()
        masked_input = input
        masked_target = target
        if self.ignore_index is not None:
            mask = target.clone().ne_(self.ignore_index)
            mask.requires_grad = False
            masked_input = input * mask
            masked_target = target * mask
        return self.loss_criterion(masked_input, masked_target)

class PixelWiseCrossEntropyLoss(nn.Module):
    def __init__(self, class_weights=None, ignore_index=None):
        super(PixelWiseCrossEntropyLoss, self).__init__()
        self.register_buffer('class_weights', class_weights)
        self.ignore_index = ignore_index
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, target, weights):
        assert target.size() == weights.size()
        log_probabilities = self.log_softmax(input)
        target = expand_as_one_hot(target, C=input.size()[1], ignore_index=self.ignore_index)
        weights = weights.unsqueeze(1)
        weights = weights.expand_as(input)
        if self.ignore_index is not None:
            mask = Variable(target.data.ne(self.ignore_index).float(), requires_grad=False)
            log_probabilities = log_probabilities * mask
            target = target * mask
        if self.class_weights is None:
            class_weights = torch.ones(input.size()[1]).float().to(input.device)
            self.register_buffer('class_weights', class_weights)
        result = -weights * target * log_probabilities
        return result.mean()

class MSEWithLogitsLoss(MSELoss):
    def __init__(self):
        super(MSEWithLogitsLoss, self).__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input, target):
        return super().forward(self.sigmoid(input), target)

class TagsAngularLoss(nn.Module):
    def __init__(self, tags_coefficients):
        super(TagsAngularLoss, self).__init__()
        self.tags_coefficients = tags_coefficients

    def forward(self, inputs, targets, weight):
        assert isinstance(inputs, list)
        if len(inputs) == 1:
            targets = [targets]
        assert len(inputs) == len(targets) == len(self.tags_coefficients)
        loss = 0
        for input, target, alpha in zip(inputs, targets, self.tags_coefficients):
            loss += alpha * square_angular_loss(input, target, weight)
        return loss

def square_angular_loss(input, target, weights=None):
    assert input.size() == target.size()
    stability_coeff = 0.999999
    input = input / torch.norm(input, p=2, dim=1).detach().clamp(min=1e-8) * stability_coeff
    target = target / torch.norm(target, p=2, dim=1).detach().clamp(min=1e-8) * stability_coeff
    cosines = (input * target).sum(dim=1)
    error_radians = torch.acos(cosines)
    if weights is not None:
        return (error_radians * error_radians * weights).sum()
    else:
        return (error_radians * error_radians).sum()

def flatten(tensor):
    C = tensor.size(1)
    axis_order = (1, 0) + tuple(range(2, tensor.dim()))
    transposed = tensor.permute(axis_order)
    return transposed.view(C, -1)

def expand_as_one_hot(input, C, ignore_index=None):
    assert input.dim() in [3, 4]
    shape = input.size()
    shape = list(shape)
    shape.insert(1, C)
    shape = tuple(shape)
    index = input.unsqueeze(1)
    if ignore_index is not None:
        expanded_index = index.expand(shape)
        mask = expanded_index == ignore_index
        index = index.clone()
        index[index == ignore_index] = 0
        result = torch.zeros(shape).to(input.device).scatter_(1, index, 1)
        result[mask] = ignore_index
        return result
    else:
        return torch.zeros(shape).to(input.device).scatter_(1, index, 1)

SUPPORTED_LOSSES = ['BCEWithLogitsLoss', 'CrossEntropyLoss', 'WeightedCrossEntropyLoss', 'PixelWiseCrossEntropyLoss',
                    'GeneralizedDiceLoss', 'DiceLoss', 'TagsAngularLoss', 'MSEWithLogitsLoss', 'MSELoss',
                    'SmoothL1Loss', 'L1Loss', 'FPFNLoss', 'LSLoss']

def get_loss_criterion(config):
    assert 'loss' in config, 'Could not find loss function configuration'
    loss_config = config['loss']
    name = loss_config['name']
    ignore_index = loss_config.get('ignore_index', None)
    weight = loss_config.get('weight', None)
    num_classes = config.get('num_classes', 2)
    if weight is not None:
        weight = torch.tensor(weight).to(config['device'])
    if name == 'BCEWithLogitsLoss' and num_classes == 2:
        skip_last_target = loss_config.get('skip_last_target', False)
        if ignore_index is None and not skip_last_target:
            return nn.BCEWithLogitsLoss()
        else:
            return BCELossWrapper(nn.BCEWithLogitsLoss(), ignore_index=ignore_index, skip_last_target=skip_last_target)
    elif name == 'CrossEntropyLoss' or (name == 'BCEWithLogitsLoss' and num_classes > 2):
        if ignore_index is None:
            ignore_index = -100
        return nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index)
    elif name == 'WeightedCrossEntropyLoss':
        if ignore_index is None:
            ignore_index = -100
        return WeightedCrossEntropyLoss(weight=weight, ignore_index=ignore_index)
    elif name == 'PixelWiseCrossEntropyLoss':
        return PixelWiseCrossEntropyLoss(class_weights=weight, ignore_index=ignore_index)
    elif name == 'GeneralizedDiceLoss':
        return GeneralizedDiceLoss(weight=weight, ignore_index=ignore_index, sigmoid_normalization=(num_classes == 2))
    elif name == 'DiceLoss':
        sigmoid_normalization = loss_config.get('sigmoid_normalization', num_classes == 2)
        skip_last_target = loss_config.get('skip_last_target', False)
        return DiceLoss(weight=weight, ignore_index=ignore_index, sigmoid_normalization=sigmoid_normalization,
                        skip_last_target=skip_last_target)
    elif name == 'TagsAngularLoss':
        tags_coefficients = loss_config['tags_coefficients']
        return TagsAngularLoss(tags_coefficients)
    elif name == 'MSEWithLogitsLoss':
        return MSEWithLogitsLoss()
    elif name == 'MSELoss':
        return MSELoss()
    elif name == 'SmoothL1Loss':
        return SmoothL1Loss()
    elif name == 'L1Loss':
        return L1Loss()
    elif name == 'FPFNLoss':
        return FPFNLoss(lamda=loss_config['lamda'])
    elif name == 'LSLoss':
        return LSLoss()
    else:
        raise RuntimeError(f"Unsupported loss function: '{name}'. Supported losses: {SUPPORTED_LOSSES}")