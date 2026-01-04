import os
import torch
import torch.nn as nn
import h5py
import numpy as np
import argparse
import psutil
import logging
from PIL import Image
from skimage import transform
import cv2
try:
    from dataset_fine import SpineParseNetDataset
    from networks.deeplab_xception_gcn_skipconnection_2d import DeepLabv3_plus_gcn_skipconnection_2d
except ImportError as e:
    print(f"Import error: {e}")
    raise


def get_utils():
    try:
        from networks import utils
        return utils
    except ImportError as e:
        print(f"Error importing networks.utils: {e}")
        raise


logger = None
def setup_logger(out_dir, fold_ind):
    global logger
    try:
        logger = get_utils().get_logger('Tester')
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        formatter = logging.Formatter('%(asctime)s %(levelname)s %(name)s - %(message)s')
        logger.setLevel(logging.INFO)
        log_file_path = os.path.join(out_dir, f'test_fine_fold_{fold_ind}_log_17.txt')
        os.makedirs(out_dir, exist_ok=True)
        file_handler = logging.FileHandler(log_file_path, mode='w')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        return file_handler, console_handler
    except Exception as e:
        print(f"Incorrect log Settings: {e}")
        raise

def dice_all_class(prediction, ground_truth, class_num):
    try:
        prediction = np.asarray(prediction, dtype=np.int32)
        ground_truth = np.asarray(ground_truth, dtype=np.int32)
        total_dice = 0.0
        valid_classes = 0
        for i in range(class_num):
            pred_i = (prediction == i).astype(np.float32)
            gt_i = (ground_truth == i).astype(np.float32)
            intersection = np.sum(pred_i * gt_i)
            union = np.sum(pred_i) + np.sum(gt_i)
            if union == 0:
                if intersection == 0:
                    continue
                else:
                    dice = 0.0
            else:
                dice = (2. * intersection) / union
            total_dice += dice
            valid_classes += 1
        return total_dice / valid_classes if valid_classes > 0 else 0.0
    except Exception as e:
        logger.error(f"Dice calculation error: {e}")
        return 0.0

def get_test_slices(test_patient_ids, mr_dir, mask_dir, unary_dir):
    try:
        slices = []
        for patient_id in test_patient_ids:
            mr_files = sorted([f for f in os.listdir(mr_dir) if f.startswith(f'Patient_{patient_id:03d}_I') and f.endswith('.png')])
            for mr_file in mr_files:
                slice_idx = mr_file.replace(f'Patient_{patient_id:03d}_I', '').replace('.png', '')
                mask_file = f'Patient_{patient_id:03d}_I{slice_idx}.png'
                unary_file = f'Patient_{patient_id:03d}_I{slice_idx}_P_L.png'
                mr_path = os.path.join(mr_dir, mr_file)
                mask_path = os.path.join(mask_dir, mask_file)
                unary_path = os.path.join(unary_dir, unary_file)
                if os.path.exists(mask_path) and os.path.exists(unary_path):
                    slices.append({
                        'patient_id': patient_id,
                        'slice_idx': slice_idx,
                        'mr_path': mr_path,
                        'mask_path': mask_path,
                        'unary_path': unary_path,
                        'slice_name': f'Patient_{patient_id:03d}_I{slice_idx}'
                    })
            else:
                    logger.warning(
                        f"Slice {mr_file} is missing mask or unary file: Mask={mask_path}, Unary={unary_path}")
        return slices
    except Exception as e:
        logger.error(f"Error retrieving test slices: {e}")
        raise

def compute_mr_stats(mr_base_dir):
    try:
        mrs = []
        for slice_file in os.listdir(mr_base_dir):
            if slice_file.endswith('.png'):
                mr_path = os.path.join(mr_base_dir, slice_file)
                try:
                    mr = np.array(Image.open(mr_path).convert('L'), dtype=np.float32)
                    mrs.append(mr.flatten())
                except Exception as e:
                    logger.warning(f"Error loading MR {mr_path}: {str(e)}")
            if mrs:
                mrs = np.concatenate(mrs)
                mean = mrs.mean()
                std = mrs.std()
                logger.info(f"Computed MR statistics: mean={mean:.4f}, std={std:.4f}")
                return mean, std
            else:
                logger.warning("No MR images found, using default mean=112.5, std=64.95")
                return 112.5, 64.95
    except Exception as e:
        logger.error(f"Error computing MR statistics: {e}")
        return 112.5, 64.95


def get_parser():
    parser = argparse.ArgumentParser(description='Spine CT Multiclass Segmentation')
    parser.add_argument("--fold_ind", type=int, default=1, help="Cross-validation fold index (1-5)")
    parser.add_argument("--data_dir", type=str, default=r'D:\torchtestto\multimodel\17-coarse\GCN\data\all-20',
                        help="Path to dataset directory")
    parser.add_argument("--model", type=str, default='DeepLabv3_plus_gcn_skipconnection_2d',
                        help="Model name")
    parser.add_argument("--num_classes", type=int, default=17, help="Number of segmentation classes")
    parser.add_argument("--nInputChannels", type=int, default=1, help="Number of input channels of MR images")
    parser.add_argument("--gcn_mode", type=int, default=1, help="GCN mode")
    parser.add_argument("--unary", dest='use_unary', action='store_true', help="Use unary term")
    parser.add_argument("--no-unary", dest='use_unary', action='store_false', help="Do not use unary term")
    parser.set_defaults(use_unary=True)
    parser.add_argument("--device", type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu',
                        help="Device to run on")
    parser.add_argument('--optimizer', type=str, default='Adam', help="Optimizer")
    parser.add_argument('--learning_rate', type=float, default=5e-4, help="Learning rate")
    parser.add_argument('--weight_decay', type=float, default=0.001, help="Weight decay")
    parser.add_argument('--loss', type=str, default='CrossEntropyLoss', help="Loss function")
    return parser


if __name__ == '__main__':
    try:
        parser = get_parser()
        args = parser.parse_args()
        num_classes = args.num_classes
        data_dir = args.data_dir
        fold_ind = args.fold_ind

        fine_h5_dir = os.path.join(data_dir, 'h5py-5-200')
        h5_path = os.path.join(fine_h5_dir, f'fold{fold_ind}_data.h5')
        mr_base_dir = os.path.join(data_dir, 'images_256')
        mask_base_dir = os.path.join(data_dir, 'masks_11_256')
        unary_base_dir = os.path.join(data_dir, '11_predictions', f'fold_{fold_ind}', 'test')

        identifier = f"{args.model}_gcn_mode_{args.gcn_mode}_{args.optimizer}_lr_{args.learning_rate}_weight_decay_{args.weight_decay}"
        if not args.use_unary:
            identifier += '_noUnary'
        identifier += '_augment'
        if args.loss != 'CrossEntropyLoss':
            identifier += f'_loss_{args.loss}'

        model_dir = os.path.join(data_dir, 'model', f'fold{fold_ind}', identifier)
        model_path = os.path.join(model_dir, 'best_checkpoint.pytorch')
        out_dir = model_dir
        prediction_images_dir = os.path.join(out_dir, 'crf_images')
        os.makedirs(prediction_images_dir, exist_ok=True)

        file_handler, console_handler = setup_logger(out_dir, fold_ind)
        logger.info(f'Start testing fold {fold_ind}')
        logger.info('Configuration parameters:')
        for k, v in vars(args).items():
            logger.info(f'{k}: {v}')

        npz_path = os.path.join(fine_h5_dir, f'split_ind_fold{fold_ind}.npz')
        logger.info(f"Loading patient split index: {npz_path}")
        try:
            foldIndData = np.load(npz_path)
            test_patient_ids = foldIndData['test_ind'].tolist()
            logger.info(f"Test set contains {len(test_patient_ids)} patient indices: {test_patient_ids}")
        except FileNotFoundError:
            logger.error(f"NPZ file not found: {npz_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading NPZ file: {e}")
            raise

        logger.info(f"Validating H5 file: {h5_path}")
        if not os.path.exists(h5_path):
            logger.error(f"H5 file not found: {h5_path}")
            raise FileNotFoundError(f"H5 file not found: {h5_path}")
        try:
            with h5py.File(h5_path, 'r') as f:
                if 'test/mr' not in f:
                    logger.error(f"H5 file missing test/mr: {h5_path}")
                    raise KeyError(f"H5 file missing test/mr")
                test_mr_shape = f['test/mr'].shape
                logger.info(f"H5 test/mr shape: {test_mr_shape}")
        except Exception as e:
            logger.error(f"Error validating H5 file: {e}")
            raise

        logger.info("Computing MR statistics...")
        mean, std = compute_mr_stats(mr_base_dir)
        logger.info("Retrieving test slice list...")
        test_slices = get_test_slices(test_patient_ids, mr_base_dir, mask_base_dir, unary_base_dir)
        logger.info(f"Total {len(test_slices)} test slices")
        for s in test_slices:
            logger.info(
                f"Slice: {s['slice_name']}, MR: {s['mr_path']}, Mask: {s['mask_path']}, Unary: {s['unary_path']}")
        model_input_channels = args.nInputChannels + (num_classes if args.use_unary else 0)
        logger.info(
            f"Model input channels: {model_input_channels} (MR: {args.nInputChannels}, Unary: {num_classes if args.use_unary else 0})")

        logger.info(f"Creating model: {args.model}")
        if args.model == 'DeepLabv3_plus_gcn_skipconnection_2d':
            model = DeepLabv3_plus_gcn_skipconnection_2d(
                nInputChannels=model_input_channels, n_classes=num_classes, os=16,
                pretrained=False, _print=True, final_sigmoid=False,
                hidden_layers=128, gcn_mode=args.gcn_mode, device=args.device
            )

        else:
            logger.error(f"Unsupported model: {args.model}")
            raise ValueError(f"Unsupported model: {args.model}")

        logger.info(f"Sending model to: {args.device}")
        model = model.to(args.device)
        if torch.cuda.device_count() > 1 and 'cuda' in args.device:
            logger.info("Using DataParallel")
            model = nn.DataParallel(model)

        logger.info(f"Loading model: {model_path}")
        if not os.path.exists(model_path):
            logger.error(f"Model file not found: {model_path}")
            raise FileNotFoundError(f"Model file not found: {model_path}")
        try:
            get_utils().load_checkpoint(model_path, model)
            logger.info(f"Successfully loaded model: {model_path}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

        model.eval()
        unary_dices = []
        crf_dices = []
        processed_slices = []
        skipped_slices = []

        height, width = 256, 256
        logger.info("Starting inference...")
        with torch.no_grad():
            total_slices = len(test_slices)
            for idx, slice_info in enumerate(test_slices):
                slice_name = slice_info['slice_name']
                logger.info(f"Processing slice {idx + 1}/{total_slices} ({slice_name})")

                try:
                    mr = np.array(Image.open(slice_info['mr_path']).convert('L'), dtype=np.float32)
                    mr = (mr - mean) / std
                    if np.isnan(mr).any() or np.isinf(mr).any():
                        logger.warning(f"MR {slice_info['mr_path']} contains NaN or Inf after normalization, skipping")
                        skipped_slices.append(slice_name)
                        continue
                    if mr.shape != (height, width):
                        mr = transform.resize(mr, (height, width), order=3, mode='constant', anti_aliasing=False)
                    mr_tensor = torch.from_numpy(mr[np.newaxis, np.newaxis, :, :]).to(args.device)
                    logger.info(f"MR tensor shape: {mr_tensor.shape}, device: {mr_tensor.device}")
                except Exception as e:
                    logger.error(f"Error loading MR image {slice_info['mr_path']}: {str(e)}")
                    skipped_slices.append(slice_name)
                    continue

                try:
                    mask = np.array(Image.open(slice_info['mask_path']).convert('L'), dtype=np.uint8)
                    if mask.shape != (height, width):
                        mask = transform.resize(mask, (height, width), order=0, anti_aliasing=False, mode='constant')
                    mask = mask.astype(np.int32)
                    mask[mask < 0] = 0
                    mask[mask > 16] = 16
                    mask_tensor = torch.from_numpy(mask).to(args.device)
                    logger.info(f"Mask tensor shape: {mask_tensor.shape}, device: {mask_tensor.device}")
                except Exception as e:
                    logger.error(f"Error loading mask {slice_info['mask_path']}: {str(e)}")
                    skipped_slices.append(slice_name)
                    continue

                if args.use_unary:
                    try:
                        unary = np.array(Image.open(slice_info['unary_path']).convert('L'), dtype=np.uint8)
                        logger.info(f"Unary {slice_info['unary_path']} raw value range: [{unary.min()}, {unary.max()}]")
                        unary = np.clip(unary // 40, 0, 16)
                        if unary.max() > 16 or unary.min() < 0:
                            logger.warning(
                                f"Unary {slice_info['unary_path']} values out of range [0, 16]: [{unary.min()}, {unary.max()}]")
                            skipped_slices.append(slice_name)
                            continue

                        unary_onehot = np.eye(num_classes)[unary.astype(np.int32)]
                        unary_onehot = unary_onehot.transpose(2, 0, 1)
                        unary_onehot = transform.resize(unary_onehot, (num_classes, height, width), order=3,
                                                        mode='constant', anti_aliasing=False)
                        unary_onehot = np.round(unary_onehot, decimals=0).astype(np.float32)
                        unary_tensor = torch.from_numpy(unary_onehot[np.newaxis, :, :, :]).to(args.device)
                        logger.info(f"Unary tensor shape: {unary_tensor.shape}, device: {unary_tensor.device}")
                    except Exception as e:
                        logger.error(f"Error loading unary {slice_info['unary_path']}: {str(e)}")
                        skipped_slices.append(slice_name)
                        continue
                else:
                    unary_tensor = None

                inputs = mr_tensor
                if args.use_unary and unary_tensor is not None:
                    inputs = torch.cat([mr_tensor, unary_tensor], dim=1)
                    logger.info(f"Input tensor shape: {inputs.shape}, device: {inputs.device}")

                try:
                    output = model(unary=unary_tensor, img=mr_tensor)
                    if isinstance(output, tuple):
                        output, _ = output
                    output_np = output.cpu().numpy().squeeze()
                    seg = np.argmax(output_np, axis=0).astype(np.uint8)
                except Exception as e:
                    logger.error(f"Model inference error ({slice_name}): {str(e)}")
                    skipped_slices.append(slice_name)
                    continue

                mask_np = mask_tensor.cpu().numpy()
                unary_label = np.argmax(unary_tensor.cpu().numpy().squeeze(),
                                        axis=0) if unary_tensor is not None else None
                unary_dice = dice_all_class(unary_label, mask_np, num_classes) if unary_label is not None else 0.0
                crf_dice = dice_all_class(seg, mask_np, num_classes)
                unary_dices.append(unary_dice)
                crf_dices.append(crf_dice)
                processed_slices.append(slice_name)
                logger.info(f"Slice {slice_name}: Unary Dice = {unary_dice:.4f}, CRF Dice = {crf_dice:.4f}")

                image_save_path = os.path.join(prediction_images_dir, f"{slice_name}.png")
                try:
                    cv2.imwrite(image_save_path, seg)
                    logger.info(f"Saved prediction image: {image_save_path}")
                except Exception as e:
                    logger.error(f"Error saving prediction image {image_save_path}: {str(e)}")

                mem = psutil.virtual_memory()
                gpu_mem = torch.cuda.memory_allocated() / (1024 ** 3) if torch.cuda.is_available() else 0.0
                logger.info(f"Memory: {mem.percent}% used, {mem.available / (1024 ** 3):.2f} GB; GPU: {gpu_mem:.2f} GB")
                torch.cuda.empty_cache()

        if crf_dices:
            mean_unary_dice = np.mean(unary_dices) if args.use_unary else 0.0
            mean_crf_dice = np.mean(crf_dices)
            logger.info(f"Mean Unary Dice = {mean_unary_dice:.4f}, Mean CRF Dice = {mean_crf_dice:.4f}")
            logger.info(f"Fold {fold_ind} Mean CRF Dice = {mean_crf_dice:.4f}")
            logger.info(f"Processed slices: {processed_slices}")
            logger.info(f"Skipped slices: {skipped_slices}")
            try:
                np.savez(os.path.join(out_dir, 'eval_scores.npz'),
                         unary_dices=unary_dices, crf_dices=crf_dices,
                         mean_unary_dice=mean_unary_dice, mean_crf_dice=mean_crf_dice,
                         processed_slices=processed_slices, skipped_slices=skipped_slices)
                logger.info("Evaluation results saved: eval_scores.npz")
            except Exception as e:
                logger.error(f"Error saving evaluation results: {e}")
        else:
            logger.warning("No slices were processed")
            logger.info(f"Skipped slices: {skipped_slices}")

    except Exception as e:
        logger.error(f"Main program error: {e}")
        raise
    finally:
        torch.cuda.empty_cache()
        logger.info(
            f"Fold {fold_ind} testing completed, Mean CRF Dice = {mean_crf_dice:.4f}" if 'mean_crf_dice' in locals() else f"Fold {fold_ind} testing completed, no CRF Dice")
        if file_handler is not None:
            logger.removeHandler(file_handler)
            file_handler.close()
        if console_handler is not None:
            logger.removeHandler(console_handler)
            console_handler.close()