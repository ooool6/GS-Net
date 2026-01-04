import logging
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from utils import get_logger, RunningAverage
import utils

class Trainer:
    def __init__(self, model, optimizer, lr_scheduler, loss_criterion, eval_criterion, device,
                 train_loader, val_loader, checkpoint_dir, max_num_epochs=100, batch_size=2,
                 max_num_iterations=1e5, validate_after_iters=100, log_after_iters=100,
                 validate_iters=None, num_iterations=1, num_epoch=0, eval_score_higher_is_better=True,
                 best_eval_score=None, logger=None, num_classes=20, use_unary=False, ds_weight=0.3,
                 early_stopping_patience=10):
        if logger is None:
            self.logger = get_logger('Trainer', level=logging.DEBUG)
        else:
            self.logger = logger

        self.logger.info(model)
        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = lr_scheduler
        self.loss_criterion = loss_criterion
        self.eval_criterion = eval_criterion
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.checkpoint_dir = checkpoint_dir
        self.max_num_epochs = max_num_epochs
        self.batch_size = batch_size
        self.max_num_iterations = max_num_iterations
        self.validate_after_iters = validate_after_iters
        self.log_after_iters = log_after_iters
        self.validate_iters = validate_iters
        self.eval_score_higher_is_better = eval_score_higher_is_better
        self.num_classes = num_classes
        self.use_unary = use_unary
        self.ds_weight = ds_weight
        self.early_stopping_patience = early_stopping_patience
        self.logger.info(
            f'eval_score_higher_is_better: {eval_score_higher_is_better}, num_classes: {num_classes}, use_unary: {use_unary}, ds_weight: {ds_weight}, early_stopping_patience: {early_stopping_patience}')

        if best_eval_score is not None:
            self.best_eval_score = best_eval_score
        else:
            self.best_eval_score = float('-inf') if eval_score_higher_is_better else float('+inf')

        self.writer = SummaryWriter(log_dir=os.path.join(checkpoint_dir, 'logs'))
        self.num_iterations = num_iterations
        self.num_epoch = num_epoch

    @classmethod
    def from_checkpoint(cls, checkpoint_path, model, optimizer, lr_scheduler, loss_criterion, eval_criterion,
                        train_loader, val_loader, logger=None, num_classes=20, use_unary=False,
                        early_stopping_patience=10):
        logger.info(f"Loading checkpoint '{checkpoint_path}'...")
        state = utils.load_checkpoint(checkpoint_path, model, optimizer)
        logger.info(
            f"Checkpoint loaded. Epoch: {state['epoch']}. Best val score: {state['best_eval_score']}. Num_iterations: {state['num_iterations']}")
        checkpoint_dir = os.path.split(checkpoint_path)[0]
        return cls(model, optimizer, lr_scheduler, loss_criterion, eval_criterion, state['device'],
                   train_loader, val_loader, checkpoint_dir,
                   eval_score_higher_is_better=state['eval_score_higher_is_better'],
                   best_eval_score=state['best_eval_score'],
                   num_iterations=state['num_iterations'],
                   num_epoch=state['epoch'],
                   batch_size=state['batch_size'],
                   max_num_epochs=state['max_num_epochs'],
                   max_num_iterations=state['max_num_iterations'],
                   validate_after_iters=state['validate_after_iters'],
                   log_after_iters=state['log_after_iters'],
                   validate_iters=state['validate_iters'],
                   logger=logger, num_classes=num_classes, use_unary=use_unary, ds_weight=state.get('ds_weight', 0.3),
                   early_stopping_patience=state.get('early_stopping_patience', early_stopping_patience))

    @classmethod
    def from_pretrained(cls, pre_trained, model, optimizer, lr_scheduler, loss_criterion, eval_criterion, device,
                        train_loader, val_loader, max_num_epochs=100, batch_size=2, max_num_iterations=1e5,
                        validate_after_iters=100, log_after_iters=100, validate_iters=None,
                        num_iterations=1, num_epoch=0, eval_score_higher_is_better=True,
                        best_eval_score=None, logger=None, num_classes=20, use_unary=False, ds_weight=0.3,
                        early_stopping_patience=10):
        logger.info(f"Loading pre-trained model from '{pre_trained}'...")
        utils.load_checkpoint(pre_trained, model, None)
        checkpoint_dir = os.path.split(pre_trained)[0]
        return cls(model, optimizer, lr_scheduler, loss_criterion, eval_criterion, device,
                   train_loader, val_loader, checkpoint_dir,
                   eval_score_higher_is_better=eval_score_higher_is_better,
                   best_eval_score=best_eval_score,
                   num_iterations=num_iterations,
                   num_epoch=num_epoch,
                   max_num_epochs=max_num_epochs,
                   batch_size=batch_size,
                   max_num_iterations=max_num_iterations,
                   validate_after_iters=validate_after_iters,
                   log_after_iters=log_after_iters,
                   validate_iters=validate_iters,
                   logger=logger, num_classes=num_classes, use_unary=use_unary, ds_weight=ds_weight,
                   early_stopping_patience=early_stopping_patience)

    def fit(self):
        for _ in range(self.num_epoch, self.max_num_epochs):
            should_terminate = self.train(self.train_loader)
            if should_terminate:
                break
            self.num_epoch += 1
            self.scheduler.step()

    def train(self, train_loader):
        train_losses = RunningAverage()
        train_eval_scores = RunningAverage()
        self.model.train()

        torch.autograd.set_detect_anomaly(True)

        for i, (inputs, targets, unaries, indices) in enumerate(train_loader):
            self.logger.info(
                f'Training iteration {self.num_iterations}. Batch {i}. Epoch [{self.num_epoch}/{self.max_num_epochs - 1}]')
            inputs = inputs.to(self.device)  # [batch, channels, H, W]
            targets = targets.to(self.device)  # [batch, H, W]
            unaries = unaries.to(self.device) if unaries is not None else None

            if inputs.size(0) < self.batch_size:
                continue

            self.optimizer.zero_grad()
            outputs = self.model(inputs, unary=unaries)
            if isinstance(outputs, tuple):
                output = outputs[0]
                aux_output = outputs[1] if outputs[1] is not None else None
            else:
                output = outputs
                aux_output = None

            loss = self._forward_pass((output, aux_output), targets)
            self.logger.info(f'Loss: {loss.item():.4f}, Batch size: {inputs.size(0)}')
            train_losses.update(loss.item(), inputs.size(0))

            loss.backward()
            self.optimizer.step()

            if self.num_iterations % self.validate_after_iters == 0:
                eval_score = self.validate(self.val_loader)
                self._log_lr()
                is_best = self._is_best_eval_score(eval_score)
                self._save_checkpoint(is_best)

            if self.num_iterations % self.log_after_iters == 0:
                eval_score = self.eval_criterion(output, targets)
                train_eval_scores.update(eval_score.item(), inputs.size(0))
                self.logger.info(
                    f'Training stats. Loss: {train_losses.avg:.4f}. Eval score: {train_eval_scores.avg:.4f}')
                self._log_stats('train', train_losses.avg, train_eval_scores.avg)
                self._log_params()
                self._log_images(inputs, targets, output)

            if self.max_num_iterations < self.num_iterations:
                self.logger.info(
                    f'Maximum number of iterations {self.max_num_iterations} exceeded. Finishing training...')
                return True

            self.num_iterations += 1

        return False

    def validate(self, val_loader):
        self.logger.info('Validating...')
        val_losses = RunningAverage()
        val_scores = RunningAverage()
        self.model.eval()

        with torch.no_grad():
            for i, (inputs, targets, unaries, indices) in enumerate(val_loader):
                self.logger.info(f'Validation iteration {i}')
                inputs = inputs.to(self.device)  # [batch, channels, H, W]
                targets = targets.to(self.device)  # [batch, H, W]
                unaries = unaries.to(self.device) if unaries is not None else None

                outputs = self.model(inputs, unary=unaries)
                if isinstance(outputs, tuple):
                    output = outputs[0]
                    aux_output = outputs[1] if outputs[1] is not None else None
                    self.logger.info(
                        f"Validation output shape: {output.shape}, aux_output shape: {aux_output.shape if aux_output is not None else 'None'}")
                else:
                    output = outputs
                    aux_output = None
                    self.logger.info(f"Validation output shape: {output.shape}, aux_output shape: None")

                loss = self._forward_pass((output, aux_output), targets)
                val_losses.update(loss.item(), inputs.size(0))
                eval_score = self.eval_criterion(output, targets)
                val_scores.update(eval_score.item(), inputs.size(0))
                pred = torch.argmax(output, dim=1)
                unique_preds = torch.unique(pred).cpu().numpy()
                self.logger.info(f'Predicted classes: {unique_preds}')

                if self.validate_iters is not None and self.validate_iters <= i:
                    break

        self._log_stats('val', val_losses.avg, val_scores.avg)
        self.logger.info(f'Validation finished. Loss: {val_losses.avg:.4f}. Eval score: {val_scores.avg:.4f}')
        self.model.train()
        return val_scores.avg

    def _forward_pass(self, outputs, target):
        if isinstance(outputs, tuple):
            output = outputs[0]
            aux_output = outputs[1] if outputs[1] is not None else None
        else:
            output = outputs
            aux_output = None

        self.logger.info(f"Output shape: {output.shape}")
        self.logger.info(f"Aux output shape: {aux_output.shape if aux_output is not None else 'None'}")

        output_size = output.size()[2:]  # [H, W] of output
        target_upsampled = F.interpolate(target.unsqueeze(1).float(), size=output_size, mode='nearest').squeeze(
            1).long()

        if aux_output is not None:
            aux_output_upsampled = F.interpolate(aux_output, size=output_size, mode='bilinear', align_corners=True)
            self.logger.info(f"Aux output upsampled shape: {aux_output_upsampled.shape}")
            if self.num_classes == 2:
                output = output.squeeze(1)
                aux_output_upsampled = aux_output_upsampled.squeeze(1)
                loss = self.loss_criterion(output, target_upsampled) + self.ds_weight * self.loss_criterion(
                    aux_output_upsampled, target_upsampled)
            else:
                loss = self.loss_criterion(output, target_upsampled) + self.ds_weight * self.loss_criterion(
                    aux_output_upsampled, target_upsampled)
        else:
            if self.num_classes == 2:
                output = output.squeeze(1)
                loss = self.loss_criterion(output, target_upsampled)
            else:
                loss = self.loss_criterion(output, target_upsampled)

        return loss

    def _is_best_eval_score(self, eval_score):
        is_best = eval_score > self.best_eval_score if self.eval_score_higher_is_better else eval_score < self.best_eval_score
        if is_best:
            self.logger.info(f'Saving new best Eval score: {eval_score:.4f}')
            self.best_eval_score = eval_score
        return is_best

    def _save_checkpoint(self, is_best):
        utils.save_checkpoint({
            'epoch': self.num_epoch + 1,
            'num_iterations': self.num_iterations,
            'model_state_dict': self.model.state_dict(),
            'best_eval_score': self.best_eval_score,
            'eval_score_higher_is_better': self.eval_score_higher_is_better,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'device': str(self.device),
            'batch_size': self.batch_size,
            'max_num_epochs': self.max_num_epochs,
            'max_num_iterations': self.max_num_iterations,
            'validate_after_iters': self.validate_after_iters,
            'log_after_iters': self.log_after_iters,
            'validate_iters': self.validate_iters,
            'num_classes': self.num_classes,
            'use_unary': self.use_unary,
            'ds_weight': self.ds_weight,
            'early_stopping_patience': self.early_stopping_patience
        }, is_best, checkpoint_dir=self.checkpoint_dir, logger=self.logger)

    def _log_lr(self):
        lr = self.optimizer.param_groups[0]['lr']
        self.logger.info(f'Learning rate: {lr:.6f}')
        self.writer.add_scalar('learning_rate', lr, self.num_iterations)

    def _log_stats(self, phase, loss_avg, eval_score_avg):
        tag_value = {
            f'{phase}_loss_avg': loss_avg,
            f'{phase}_dice_score_avg': eval_score_avg
        }
        for tag, value in tag_value.items():
            self.writer.add_scalar(tag, value, self.num_iterations)

    def _log_params(self):
        self.logger.info('Logging model parameters and gradients')
        for name, value in self.model.named_parameters():
            self.writer.add_histogram(name, value.data.cpu().numpy(), self.num_iterations)
            if value.grad is not None:
                self.writer.add_histogram(f'{name}/grad', value.grad.data.cpu().numpy(), self.num_iterations)

    def _log_images(self, inputs, target, output):
        if self.num_classes == 2:
            prediction = torch.sigmoid(output)
        else:
            prediction = F.softmax(output, dim=1)[:, 1:2]
        inputs_map = {
            'inputs': inputs[:, :1],
            'targets': target.float(),
            'predictions': prediction
        }
        img_sources = {}
        for name, batch in inputs_map.items():
            img_sources[name] = batch.data.cpu().numpy()

        for name, batch in img_sources.items():
            for tag, image in self._images_from_batch(name, batch):
                self.writer.add_image(tag, image, self.num_iterations, dataformats='HW')

    def _images_from_batch(self, name, batch):
        if batch.ndim == 4:  # NCHW
            tag_template = '{}/batch_{}/channel_{}'
            tagged_images = []
            for batch_idx in range(batch.shape[0]):
                for channel_idx in range(min(batch.shape[1], 3)):
                    tag = tag_template.format(name, batch_idx, channel_idx)
                    img = batch[batch_idx, channel_idx, :, :]
                    tagged_images.append((tag, self._normalize_img(img)))
        else:  # NHW
            tag_template = '{}/batch_{}'
            tagged_images = []
            for batch_idx in range(batch.shape[0]):
                tag = tag_template.format(name, batch_idx)
                img = batch[batch_idx, :, :]
                tagged_images.append((tag, self._normalize_img(img)))
        return tagged_images

    def _normalize_img(self, img):
        return (img - np.min(img)) / (np.ptp(img) + 1e-7)

    def _batch_size(self, input):
        return input.size(0)