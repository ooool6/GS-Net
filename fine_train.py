import numpy as np
import h5py
import os
import shutil
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from dataset import load_data
from utils import get_logger, get_number_of_learnable_parameters
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.lr_scheduler import MultiStepLR
from losses import get_loss_criterion
from metrics import get_evaluation_metric
import importlib
from trainer_2d import Trainer
from unet_2d import UNet2D, ResidualUNet2D
from deeplab_xception_skipconnection_2d import DeepLabv3_plus_skipconnection_2d
from deeplab_xception_gcn_skipconnection_2d import DeepLabv3_plus_gcn_skipconnection_2d
import logging
import psutil
import gc

logger = get_logger('Trainer')
log_file = os.path.join('D:\\torchtestto\\multimodel\\17-coarse\\datasets\\model', 'train_fine_log_17.txt')
os.makedirs(os.path.dirname(log_file), exist_ok=True)
file_handler = logging.FileHandler(log_file, mode='a')
formatter = logging.Formatter('%(asctime)s [%(threadName)s] %(levelname)s %(name)s - %(message)s')
file_handler.setFormatter(formatter)
logger.handlers = [file_handler]
logger.setLevel(logging.INFO)

def get_parser():
    parser = ArgumentParser(description=__doc__, formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("--fold_ind", type=int, default=5, help="1 to 5")
    parser.add_argument("--data_dir", type=str, default='D:\\torchtestto\\multimodel\\17-coarse\\GCN\\data\\all-20',
                        help="the data dir")
    parser.add_argument("--coarse_identifier", type=str, default=None, help="coarse identifier")
    parser.add_argument("--model", type=str, default='DeepLabv3_plus_gcn_skipconnection_2d', help="the model name")
    parser.add_argument("--num_classes", type=int, default=17, help="number of classes")
    parser.add_argument("--nInputChannels", type=int, default=1, help="number of input channels")
    parser.add_argument("--gcn_mode", type=int, default=1, help="0, 1, 2")
    parser.add_argument("--ds_weight", type=float, default=0.3, help="the loss weight for gcn_mode=2")
    parser.add_argument("--augment", dest='augment', action='store_true', help="whether use augmentation")
    parser.add_argument("--no-augment", dest='augment', action='store_false', help="whether use augmentation")
    parser.set_defaults(augment=True)
    parser.add_argument("--unary", dest='use_unary', action='store_true', help="whether use unary")
    parser.add_argument("--no-unary", dest='use_unary', action='store_false', help="whether use unary")
    parser.set_defaults(use_unary=True)
    parser.add_argument("--pre_trained", dest='pre_trained', action='store_true', help="whether use pre_trained")
    parser.add_argument("--no-pre_trained", dest='pre_trained', action='store_false', help="whether use pre_trained")
    parser.set_defaults(pre_trained=False)
    parser.add_argument("--resume", dest='resume', action='store_true', help="whether use resume")
    parser.add_argument("--no-resume", dest='resume', action='store_false', help="whether use resume")
    parser.set_defaults(resume=False)
    parser.add_argument("--epochs", type=int, default=100, help="max number of epochs")
    parser.add_argument("--iters", type=int, default=500000, help="max number of iterations")
    parser.add_argument("--eval_score_higher_is_better", type=bool, default=True, help="model with higher eval score is better")
    parser.add_argument("--device", type=str, default='cuda:0', help="which gpu to use")
    parser.add_argument('--batch_size', type=int, default=1, help="The batch size")
    parser.add_argument('--manual_seed', type=int, default=0, help="The manual_seed")
    parser.add_argument('--loss', type=str, default='CrossEntropyLoss', help="The loss function name")
    parser.add_argument('--eval_metric', type=str, default='DiceCoefficient', help="The eval_metric name")
    parser.add_argument('--skip_channels', type=list, default=None, help="The skip_channels in eval_metric")
    parser.add_argument('--optimizer', type=str, default='Adam', help="Adam or SGD")
    parser.add_argument('--learning_rate', type=float, default=5e-4, help="The initial learning rate")
    parser.add_argument('--weight_decay', type=float, default=0.001, help="The weight decay")
    parser.add_argument('--momentum', type=float, default=0.9, help="the momentum for the optimizer")
    parser.add_argument('--nesterov', type=bool, default=False, help="the nesterov for the optimizer")
    parser.add_argument('--gamma', type=float, default=0.2, help="The gamma for the MultiStepLR scheduler")
    return parser

default_conf = {
    "pyinn": False,
    'transformer': {
        'train': {
            'raw': [
                {'name': 'RandomRotate', 'angle_spectrum': 15, 'interpolation': 'cubic'},
                {'name': 'ElasticDeformation', 'spline_order': 3},
                {'name': 'RandomContrast'},
                {'name': 'RandomFlip', 'p': 0.5},
                {'name': 'RandomBrightness', 'brightness_range': (0.8, 1.2)}
            ],
            'label': [
                {'name': 'RandomRotate', 'angle_spectrum': 15, 'interpolation': 'nearest'},
                {'name': 'ElasticDeformation', 'spline_order': 0},
                {'name': 'RandomFlip', 'p': 0.5}
            ],
            'unary': [
                {'name': 'RandomRotate', 'angle_spectrum': 15, 'interpolation': 'cubic'},
                {'name': 'ElasticDeformation', 'spline_order': 3},
                {'name': 'RandomFlip', 'p': 0.5}
            ],
            'weight': [
                {'name': 'RandomRotate', 'angle_spectrum': 15, 'interpolation': 'cubic'},
                {'name': 'ElasticDeformation', 'spline_order': 3},
                {'name': 'RandomFlip', 'p': 0.5}
            ]
        },
        'val': {'raw': None, 'label': None, 'unary': None, 'weight': None},
        'test': {'raw': None, 'label': None, 'unary': None, 'weight': None}
    }
}

def get_default_conf():
    return default_conf.copy()

def _create_optimizer(config, model):
    optimizer_config = config['optimizer']
    optimizer_name = optimizer_config['name']
    learning_rate = optimizer_config['learning_rate']
    weight_decay = optimizer_config['weight_decay']
    momentum = optimizer_config.get('momentum', 0.0)
    nesterov = optimizer_config.get('nesterov', False)
    if optimizer_name == 'Adam':
        return optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    else:
        return optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay, nesterov=nesterov)

def _create_lr_scheduler(config, optimizer):
    lr_config = config.get('lr_scheduler', None)
    if lr_config is None:
        return ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10, verbose=True)
    class_name = lr_config.pop('name')
    m = importlib.import_module('torch.optim.lr_scheduler')
    clazz = getattr(m, class_name)
    lr_config['optimizer'] = optimizer
    return clazz(**lr_config)

def _create_trainer(config, model, optimizer, lr_scheduler, loss_criterion, eval_criterion, train_loader, val_loader, logger):
    trainer_config = config['trainer']
    resume = trainer_config.get('resume', None)
    pre_trained = trainer_config.get('pre_trained', None)
    trainer_config['early_stopping_patience'] = 10
    if resume is not None:
        return Trainer.from_checkpoint(resume, model, optimizer, lr_scheduler, loss_criterion, eval_criterion, train_loader, val_loader, logger=logger, num_classes=config['num_classes'], use_unary=config.get('use_unary', False), early_stopping_patience=trainer_config['early_stopping_patience'])
    elif pre_trained is not None:
        return Trainer.from_pretrained(pre_trained, model, optimizer, lr_scheduler, loss_criterion, eval_criterion, device=config['device'], train_loader=train_loader, val_loader=val_loader, max_num_epochs=trainer_config['epochs'], batch_size=trainer_config['batch_size'], max_num_iterations=trainer_config['iters'], validate_after_iters=trainer_config['validate_after_iters'], log_after_iters=trainer_config['log_after_iters'], eval_score_higher_is_better=trainer_config['eval_score_higher_is_better'], logger=logger, num_classes=config['num_classes'], use_unary=config.get('use_unary', False), early_stopping_patience=trainer_config['early_stopping_patience'])
    return Trainer(model, optimizer, lr_scheduler, loss_criterion, eval_criterion, config['device'], train_loader, val_loader, trainer_config['checkpoint_dir'], max_num_epochs=trainer_config['epochs'], batch_size=trainer_config['batch_size'], max_num_iterations=trainer_config['iters'], validate_after_iters=trainer_config['validate_after_iters'], log_after_iters=trainer_config['log_after_iters'], eval_score_higher_is_better=trainer_config['eval_score_higher_is_better'], logger=logger, num_classes=config['num_classes'], use_unary=config.get('use_unary', False), early_stopping_patience=trainer_config['early_stopping_patience'])


def check_data_quality(data_path, logger):
    with h5py.File(data_path, 'r') as f:
        def recursive_check(group, prefix=''):
            for key, item in group.items():
                path = f"{prefix}/{key}" if prefix else key
                if isinstance(item, h5py.Dataset):
                    logger.info(f"Checking dataset: {path}, shape: {item.shape}, dtype: {item.dtype}")
                    if 'mr' in path.lower():
                        sample_data = item[0]
                        if np.any(np.isnan(sample_data)) or np.any(np.isinf(sample_data)):
                            logger.warning(f"Dataset {path} contains NaN or Inf values")
                        logger.info(f"Dataset {path} value range: [{sample_data.min():.4f}, {sample_data.max():.4f}]")
                    elif 'mask' in path.lower():
                        sample_data = item[0]
                        unique_values = np.unique(sample_data)
                        logger.info(f"Dataset {path} unique values: {unique_values}")
                        ORIGINAL_LABELS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16]

                        if not np.all(np.isin(unique_values, ORIGINAL_LABELS)):
                            logger.warning(f"Dataset {path} contains unexpected label values: {unique_values}")

                    elif 'unary' in path.lower():
                        unique_values = np.unique(sample_data)
                        logger.info(f"Dataset {path} unique values: {unique_values}")
                        if sample_data.shape[0] != 17:
                            logger.warning(f"Dataset {path} has incorrect number of channels: {sample_data.shape[0]}, expected 12")
                        if not np.all(np.isin(unique_values, [0, 1])):
                            logger.warning(f"Dataset {path} contains non-binary values, expected one-hot encoding [0, 1]: {unique_values}")
                    elif 'weight' in path.lower():
                        sample_data = item[0]
                        if np.any(np.isnan(sample_data)) or np.any(np.isinf(sample_data)):
                            logger.warning(f"Dataset {path} contains NaN or Inf values")
                        logger.info(f"Dataset {path} value range: [{sample_data.min():.4f}, {sample_data.max():.4f}]")
                elif isinstance(item, h5py.Group):
                    recursive_check(item, path)
        recursive_check(f)



def cleanup_memory():
    gc.collect()
    torch.cuda.empty_cache()
    logger.info("Memory cleanup performed")

def main():
    parser = get_parser()
    args = parser.parse_args()
    data_dir = args.data_dir

    for fold_ind in range(1, 6):
        args.fold_ind = fold_ind
        batch_size = args.batch_size
        num_classes = args.num_classes

        if args.coarse_identifier is None:
            data_path = os.path.join(data_dir, 'h5py-5-200', f'fold{fold_ind}_data.h5')
        else:
            data_path = os.path.join(data_dir, 'h5py-5-200', f'fold{fold_ind}_{args.coarse_identifier}.h5')

        fold_dir = os.path.join(data_dir, 'model', f'fold{fold_ind}')
        os.makedirs(fold_dir, exist_ok=True)

        in_channels = args.nInputChannels
        expected_channels = in_channels + (num_classes if args.use_unary else 0)
        logger.info(f"Calculated expected channels: {expected_channels} (nInputChannels: {in_channels}, use_unary: {args.use_unary}, num_classes: {num_classes})")

        if args.model == 'DeepLabv3_plus_gcn_skipconnection_2d' and args.gcn_mode == 2:
            identifier = f"{args.model}_gcn_mode_{args.gcn_mode}_ds_weight_{args.ds_weight}_{args.optimizer}_lr_{args.learning_rate}_weight_decay_{args.weight_decay}"
        else:
            identifier = f"{args.model}_gcn_mode_{args.gcn_mode}_{args.optimizer}_lr_{args.learning_rate}_weight_decay_{args.weight_decay}"
        if not args.use_unary:
            identifier += '_noUnary'
        if args.augment:
            identifier += '_augment'
        if args.loss != 'CrossEntropyLoss':
            identifier += f'_loss_{args.loss}'

        conf = get_default_conf()
        conf['device'] = args.device
        conf['manual_seed'] = args.manual_seed
        conf['num_classes'] = num_classes
        conf['use_unary'] = args.use_unary
        conf['in_channels'] = expected_channels

        conf['loss'] = {'name': args.loss, 'num_classes': num_classes}
        conf['eval_metric'] = {'name': args.eval_metric, 'skip_channels': args.skip_channels, 'num_classes': num_classes}
        conf['optimizer'] = {'name': args.optimizer, 'learning_rate': args.learning_rate, 'weight_decay': args.weight_decay, 'momentum': args.momentum, 'nesterov': args.nesterov}
        conf['lr_scheduler'] = {'name': 'MultiStepLR', 'milestones': [20, 40], 'gamma': args.gamma}
        conf['trainer'] = {'batch_size': batch_size, 'epochs': args.epochs, 'iters': args.iters, 'eval_score_higher_is_better': args.eval_score_higher_is_better, 'ds_weight': args.ds_weight, 'checkpoint_dir': os.path.join(fold_dir, identifier), 'validate_after_iters': 0, 'log_after_iters': 0, 'early_stopping_patience': 10}

        if args.loss == 'PixelWiseCrossEntropyLoss':
            return_weight = True
        else:
            return_weight = False

        if args.resume:
            identifier += '_resume'
            conf['trainer']['resume'] = os.path.join(fold_dir, identifier, 'best_checkpoint.pytorch')
            os.makedirs(os.path.join(fold_dir, identifier), exist_ok=True)
        elif args.pre_trained:
            identifier += '_pretrained'
            conf['trainer']['pre_trained'] = os.path.join(fold_dir, identifier, 'best_checkpoint.pytorch')
            os.makedirs(os.path.join(fold_dir, identifier), exist_ok=True)

        checkpoint_dir = os.path.join(fold_dir, identifier)

        mem = psutil.virtual_memory()
        logger.info(f"System memory usage: {mem.percent}% used, {mem.available / (1024 ** 3):.2f} GB available")

        logger.info(f'Starting fold {fold_ind}')
        logger.info('The configurations: ')
        for k, v in conf.items():
            logger.info(f'{k}: {v}')
        logger.info(f'Data path: {data_path}')

        check_data_quality(data_path, logger)

        if conf.get('manual_seed', None) is not None:
            logger.info(f'Seed the RNG with {conf["manual_seed"]}')
            torch.manual_seed(conf["manual_seed"])
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        logger.info(f"CUDA available: {torch.cuda.is_available()}, Device: {conf['device']}")
        torch.cuda.empty_cache()

        logger.info(f"Passing in_channels: {expected_channels} to model (adjusted for unary)")
        if args.model == 'UNet2D':
            model = UNet2D(in_channels=expected_channels, out_channels=num_classes, final_sigmoid=(num_classes==2), f_maps=32, layer_order='cbr', num_groups=8, dropout_rate=0.3)
        elif args.model == 'ResidualUNet2D':
            model = ResidualUNet2D(in_channels=expected_channels, out_channels=num_classes, final_sigmoid=(num_classes==2), f_maps=32, conv_layer_order='cbr', num_groups=8, dropout_rate=0.3)
        elif args.model == 'DeepLabv3_plus_skipconnection_2d':
            model = DeepLabv3_plus_skipconnection_2d(nInputChannels=expected_channels, n_classes=num_classes, os=16, pretrained=False, _print=False, dropout_rate=0.3)
        elif args.model == 'DeepLabv3_plus_gcn_skipconnection_2d':
            model = DeepLabv3_plus_gcn_skipconnection_2d(nInputChannels=expected_channels, n_classes=num_classes, os=16, pretrained=False, _print=False, hidden_layers=128, gcn_mode=args.gcn_mode, device=conf['device'])
            logger.info(f"Model initialized with nInputChannels: {expected_channels}, use_unary: {args.use_unary}, gcn_mode: {args.gcn_mode}, expected_channels: {expected_channels}")

        model = model.to(conf['device'])
        logger.info(f'Number of learnable params {get_number_of_learnable_parameters(model)}')

        loss_criterion = get_loss_criterion(conf)
        eval_criterion = get_evaluation_metric(conf)

        logger.info("Validating input channels with a sample batch...")
        sample_loader, _, _, sample_file_handle = load_data(data_path, batch_size=1, return_weight=False, num_classes=num_classes, use_unary=args.use_unary)
        try:
            sample_batch = next(iter(sample_loader))
            sample_img, sample_mask, sample_unary, sample_indices = sample_batch
            if sample_img.dim() == 3:
                sample_img = sample_img.unsqueeze(0)
            elif sample_img.dim() != 4:
                raise ValueError(f"Unexpected input dimension: {sample_img.dim()}")
            logger.info(f"Sample input shape: {sample_img.shape}, Sample mask shape: {sample_mask.shape if sample_mask is not None else 'None'}, "
                        f"Sample unary shape: {sample_unary.shape if sample_unary is not None else 'None'}, Sample indices shape: {sample_indices.shape}")
            if sample_unary is not None:
                logger.info(f"Sample unary unique values: {torch.unique(sample_unary)}")
                logger.info(f"Sample unary channel sum: {torch.sum(sample_unary, dim=1).mean().item():.4f}")
            mem = psutil.virtual_memory()
            logger.info(f"After sample batch: Memory usage: {mem.percent}% used, {mem.available / (1024 ** 3):.2f} GB available")

            model.eval()
            with torch.no_grad():
                sample_img = sample_img.to(conf['device'])
                sample_unary = sample_unary.to(conf['device']) if sample_unary is not None else None
                sample_output = model(sample_img, unary=sample_unary)
                if isinstance(sample_output, tuple):
                    logger.info(f"Model sample output shape: {sample_output[0].shape}, aux_output shape: {sample_output[1].shape if len(sample_output) > 1 else 'None'}")
                else:
                    logger.info(f"Model sample output shape: {sample_output.shape}")
            model.train()
        finally:
            if sample_file_handle is not None:
                sample_file_handle.close()
                logger.info("Sample batch HDF5 file handle closed")
            cleanup_memory()

        try:
            if args.augment:
                train_data_loader, val_data_loader, _, file_handle = load_data(data_path, batch_size=batch_size, transformer_config=conf['transformer'], return_weight=return_weight, num_classes=num_classes, use_unary=args.use_unary, skip_test_mask=True)
            else:
                train_data_loader, val_data_loader, _, file_handle = load_data(data_path, batch_size=batch_size, return_weight=return_weight, num_classes=num_classes, use_unary=args.use_unary, skip_test_mask=True)

            mem = psutil.virtual_memory()
            logger.info(f"After data loaders: Memory usage: {mem.percent}% used, {mem.available / (1024 ** 3):.2f} GB available")

            conf['trainer']['validate_after_iters'] = len(train_data_loader)
            conf['trainer']['log_after_iters'] = len(train_data_loader)

            optimizer = _create_optimizer(conf, model)
            lr_scheduler = _create_lr_scheduler(conf, optimizer)
            trainer = _create_trainer(conf, model=model, optimizer=optimizer, lr_scheduler=lr_scheduler,
                                      loss_criterion=loss_criterion, eval_criterion=eval_criterion,
                                      train_loader=train_data_loader, val_loader=val_data_loader,
                                      logger=logger)
            trainer.fit()
            logger.info(f'Finished fold {fold_ind}')
        except Exception as e:
            logger.error(f'Error in fold {fold_ind}: {str(e)}')
            raise
        finally:
            if 'file_handle' in locals() and isinstance(file_handle, h5py.File):
                file_handle.close()
                logger.info("HDF5 file handle closed")
            cleanup_memory()

if __name__ == '__main__':
    main()

