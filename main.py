import os
import subprocess
import logging
import psutil
import gc
import torch
import argparse
import h5py
import numpy as np
try:
    from networks.utils import get_logger
except ImportError as e:
    print(f"Error importing networks.utils: {e}")
    raise

data_root_dir = r"D:\torchtestto\multimodel\17-coarse\GCN\data\all-20"

logger = get_logger('MainProcess')
main_log_file = os.path.join(data_root_dir, 'model', 'main_process_17_log.txt')
os.makedirs(os.path.dirname(main_log_file), exist_ok=True)
for handler in logger.handlers[:]:
    if isinstance(handler, logging.FileHandler):
        logger.removeHandler(handler)
file_handler = logging.FileHandler(main_log_file, mode='w')
formatter = logging.Formatter('%(asctime)s [%(threadName)s] %(levelname)s %(name)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.setLevel(logging.INFO)

def validate_data_dirs():
    try:
        fine_h5_dir = os.path.join(data_root_dir, 'h5py-5-200')
        if not os.path.exists(fine_h5_dir):
            logger.error(f"Data directory does not exist: {fine_h5_dir}")
            raise FileNotFoundError(f"Data directory does not exist: {fine_h5_dir}")
        return fine_h5_dir
    except Exception as e:
        logger.error(f"Error validating data directory: {e}")
        raise

def validate_data_files(fold_ind, fine_h5_dir):
    try:
        h5_path = os.path.join(fine_h5_dir, f'fold{fold_ind}_data.h5')
        npz_path = os.path.join(fine_h5_dir, f'split_ind_fold{fold_ind}.npz')
        if not os.path.exists(h5_path):
            logger.error(f"H5 file not found: {h5_path}")
            return False, None, None
        if not os.path.exists(npz_path):
            logger.error(f"NPZ file not found: {npz_path}")
            return False, None, None
        with h5py.File(h5_path, 'r') as f:
            if 'test/mr' not in f:
                logger.error(f"H5 file missing test/mr: {h5_path}")
                return False, None, None
            test_mr_shape = f['test/mr'].shape
            logger.info(f"H5 file validation passed: {h5_path}, test/mr shape: {test_mr_shape}")
        npz_data = np.load(npz_path)
        test_ind = npz_data['test_ind'].tolist()
        valid_test_ind = [i for i in test_ind if i < 200]
        if len(valid_test_ind) < len(test_ind):
            logger.warning(
                f"Fold {fold_ind} test_ind contains {len(test_ind) - len(valid_test_ind)} out-of-range indices, filtered out"
            )
        logger.info(f"NPZ file validation passed: {npz_path}, test_ind (patients): {valid_test_ind}")
        mr_dir = os.path.join(data_root_dir, 'images_256')
        mask_dir = os.path.join(data_root_dir, 'masks_11_256')
        unary_dir = os.path.join(data_root_dir, '11_predictions', f'fold_{fold_ind}', 'test')
        slice_count = 0
        for pid in valid_test_ind:
            mr_files = [f for f in os.listdir(mr_dir)
                        if f.startswith(f'Patient_{pid:03d}_I') and f.endswith('.png')]
            for mr_file in mr_files:
                slice_idx = mr_file.replace(f'Patient_{pid:03d}_I', '').replace('.png', '')
                if (os.path.exists(os.path.join(mask_dir, f'Patient_{pid:03d}_I{slice_idx}.png')) and
                    os.path.exists(os.path.join(unary_dir, f'Patient_{pid:03d}_I{slice_idx}_P_L.png'))):
                    slice_count += 1
        logger.info(f"Fold {fold_ind} test patients have {slice_count} slices in total")
        return True, h5_path, valid_test_ind
    except Exception as e:
        logger.error(f"Error validating data files: {e}")
        return False, None, None

def run_command(step_msg, command, fold_ind):
    print(step_msg.format(fold_ind=fold_ind))
    logger.info(step_msg.format(fold_ind=fold_ind))
    try:
        process = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding='utf-8',
            errors='ignore'
        )
        stdout_lines = process.stdout.splitlines()
        stderr_lines = process.stderr.splitlines()
        for line in stdout_lines:
            print(line)
            logger.info(line)
        for line in stderr_lines:
            if 'ERROR' in line:
                print(f"Subprocess error: {line}")
                logger.error(line)
            else:
                print(f"Subprocess info: {line}")
                logger.info(line)
        if process.returncode != 0:
            raise RuntimeError(f"Command failed with return code: {process.returncode}")
        return stdout_lines, stderr_lines
    except Exception as e:
        logger.error(f"Execution error: {e}, command: {' '.join(command)}")
        raise

def check_checkpoint(fold_ind, data_dir, identifier):
    try:
        checkpoint_dir = os.path.join(data_dir, 'model', f'fold{fold_ind}', identifier)
        checkpoint_path = os.path.join(checkpoint_dir, 'best_checkpoint.pytorch')
        if not os.path.exists(checkpoint_path):
            logger.warning(f"Checkpoint not found: {checkpoint_path}")
            return None
        logger.info(f"Found checkpoint: {checkpoint_path}")
        return checkpoint_path
    except Exception as e:
        logger.error(f"Error checking checkpoint: {e}")
        return None

def cleanup_memory():
    try:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gpu_mem = torch.cuda.memory_allocated() / (1024 ** 3)
            logger.info(f"GPU memory: {gpu_mem:.2f} GB")
        mem = psutil.virtual_memory()
        logger.info(f"Memory: {mem.percent}% used, {mem.available / (1024 ** 3):.2f} GB available")
    except Exception as e:
        logger.error(f"Error cleaning up memory: {e}")

def main():
    try:
        parser = argparse.ArgumentParser(description="Run SpineParseNet testing")
        parser.add_argument("--fold_ind", type=int, default=0, help="Fold index (1-5, 0 for all)")
        parser.add_argument("--device", type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu')
        parser.add_argument("--use_unary", action='store_true', default=True)
        parser.add_argument("--num_classes", type=int, default=17)
        args = parser.parse_args()

        step_msg = "Step: Testing fold {fold_ind}..."
        folds = range(1, 6) if args.fold_ind == 0 else [args.fold_ind]
        fine_h5_dir = validate_data_dirs()
        tested_folds = []

        for fold_ind in folds:
            is_valid, h5_path, test_ind = validate_data_files(fold_ind, fine_h5_dir)
            if not is_valid:
                logger.error(f"Skipping fold {fold_ind}: invalid data files")
                continue

            identifier = (
                "DeepLabv3_plus_gcn_skipconnection_2d_gcn_mode_1_Adam_lr_0.0005_weight_decay_0.001_augment"
            )
            if not args.use_unary:
                identifier += '_noUnary'

            checkpoint_path = check_checkpoint(fold_ind, data_root_dir, identifier)
            if not checkpoint_path:
                logger.error(f"Skipping fold {fold_ind}: checkpoint missing")
                continue

            command = [
                "D:\\anaconda3\\envs\\spine\\python.exe", "-u",
                r"D:\torchtestto\multimodel\17SpineParseNet-master\test_fine.py",
                f"--device={args.device}",
                f"--fold_ind={fold_ind}",
                f"--data_dir={data_root_dir}",
                "--model=DeepLabv3_plus_gcn_skipconnection_2d",
                "--gcn_mode=1",
                f"--num_classes={args.num_classes}",
                "--unary" if args.use_unary else "--no-unary",
                "--loss=CrossEntropyLoss"
            ]

            cleanup_memory()
            try:
                run_command(step_msg, command, fold_ind)
                tested_folds.append(fold_ind)
            except Exception as e:
                logger.error(f"Testing fold {fold_ind} failed: {e}")
            finally:
                cleanup_memory()

        if not tested_folds:
            logger.error("No folds were successfully tested")
            raise RuntimeError("Testing failed")
        else:
            logger.info(f"Successfully tested folds: {tested_folds}")

        logger.info("All steps completed!")
    except Exception as e:
        logger.error(f"Main program error: {e}")
        raise
    finally:
        if file_handler is not None:
            logger.removeHandler(file_handler)
            file_handler.close()

if __name__ == '__main__':
    main()
