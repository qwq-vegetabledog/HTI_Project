import logging
import os
import pprint
import random
import time
import sys
import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader

# Add current and parent directory to path
[sys.path.append(i) for i in ['.', '..']]

# --- PROJECT IMPORTS ---
# We use the _bert version of the model to avoid Embedding layer errors
from model.seq2seq_net_bert import Seq2SeqNet
from train_eval.train_seq2seq import train_iter_seq2seq
from utils.average_meter import AverageMeter
from config.parse_args import parse_args
from twh_dataset_to_lmdb import target_joints

# Using the _bert files we created for data loading
from data_loader.lmdb_data_loader_bert import TwhDataset, word_seq_collate_fn
import utils.train_utils_bert

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def init_model(args, pose_dim, _device):
    # n_words=1 and word_embed_size=3072 are passed only to satisfy the signature,
    # because the modified EncoderRNN ignores n_words and uses 3072 internally.
    n_frames = args.n_poses
    generator = Seq2SeqNet(args, 
                           pose_dim=pose_dim, 
                           n_frames=n_frames, 
                           n_words=1, 
                           word_embed_size=3072, 
                           word_embeddings=None).to(_device)
    
    # MSELoss is standard for generating continuous coordinates
    loss_fn = torch.nn.MSELoss()
    return generator, loss_fn

def evaluate_testset(test_data_loader, generator, loss_fn, args):
    generator.eval()
    losses = AverageMeter('loss')
    start = time.time()

    with torch.no_grad():
        for data in test_data_loader:
            in_text, text_lengths, target_vec, in_audio, aux_info = data
            batch_size = target_vec.size(0)

            in_text = in_text.to(device)
            target = target_vec.to(device)

            # The model generates gestures
            out_poses = generator(in_text, text_lengths, target, None)
            
            loss = loss_fn(out_poses, target)
            losses.update(loss.item(), batch_size)

    generator.train()
    elapsed_time = time.time() - start
    logging.info(f"[VAL] loss: {losses.avg:.4f} / {elapsed_time:.1f}s")
    return losses.avg

def train_epochs(args, train_data_loader, test_data_loader, pose_dim, trial_id=None):
    start = time.time()
    loss_meters = [AverageMeter('loss'), AverageMeter('var_loss')]

    # Print interval adjusted to avoid cluttering logs if dataset is small
    print_interval = max(1, int(len(train_data_loader) / 5))
    save_model_epoch_interval = 20

    # Initialize model and optimizer
    generator, loss_fn = init_model(args, pose_dim, device)
    gen_optimizer = optim.Adam(generator.parameters(), lr=args.learning_rate, betas=(0.5, 0.999))

    logging.info(f"Model initialized. Device: {device}")

    global_iter = 0
    for epoch in range(1, args.epochs + 1):
        # ----------------------
        # Validation
        # ----------------------
        val_loss = evaluate_testset(test_data_loader, generator, loss_fn, args)

        # ----------------------
        # Save Checkpoint
        # ----------------------
        if epoch % save_model_epoch_interval == 0:
            try:
                gen_state_dict = generator.module.state_dict()
            except AttributeError:
                gen_state_dict = generator.state_dict()

            save_name = os.path.join(args.model_save_path, f"{args.name}_checkpoint_{epoch:03d}.bin")
            utils.train_utils_bert.save_checkpoint({
                'args': args, 'epoch': epoch,
                'pose_dim': pose_dim, 'gen_dict': gen_state_dict
            }, save_name)

        # ----------------------
        # Training Loop
        # ----------------------
        iter_start_time = time.time()
        for iter_idx, data in enumerate(train_data_loader):
            global_iter += 1
            in_text, text_lengths, target_vec, in_audio, aux_info = data
            batch_size = target_vec.size(0)

            in_text = in_text.to(device)
            target_vec = target_vec.to(device)
            # in_audio = in_audio.to(device) # If using audio in the future

            # Call training iteration function (train_eval/train_seq2seq.py)
            loss = train_iter_seq2seq(args, epoch, in_text, text_lengths, target_vec, generator, gen_optimizer)

            # Update meters
            for loss_meter in loss_meters:
                if loss_meter.name in loss:
                    loss_meter.update(loss[loss_meter.name], batch_size)

            # Print status
            if (iter_idx + 1) % print_interval == 0:
                summary = f"EP {epoch} ({iter_idx + 1:3d}/{len(train_data_loader)}) | " \
                          f"{utils.train_utils_bert.time_since(start):>8s} | " \
                          f"{batch_size / (time.time() - iter_start_time):.0f} Hz | "
                
                for loss_meter in loss_meters:
                    if loss_meter.count > 0:
                        summary += f"{loss_meter.name}: {loss_meter.avg:.4f}, "
                        loss_meter.reset()
                logging.info(summary)

            iter_start_time = time.time()

def main(config):
    args = config['args']

    # Set Seeds for reproducibility
    if args.random_seed >= 0:
        torch.manual_seed(args.random_seed)
        torch.cuda.manual_seed_all(args.random_seed)
        np.random.seed(args.random_seed)
        random.seed(args.random_seed)
        os.environ['PYTHONHASHSEED'] = str(args.random_seed)

    # Set Logger
    os.makedirs(args.model_save_path, exist_ok=True)
    utils.train_utils_bert.set_logger(args.model_save_path, os.path.basename(__file__).replace('.py', '.log'))

    logging.info(f"PyTorch version: {torch.__version__}")
    logging.info(f"CUDA version: {torch.version.cuda}")
    logging.info(pprint.pformat(vars(args)))

    # Handle train_data_path whether list or string (depends on config)
    train_path = args.train_data_path[0] if isinstance(args.train_data_path, list) else args.train_data_path
    val_path = args.val_data_path[0] if isinstance(args.val_data_path, list) else args.val_data_path

    # Initialize Datasets
    train_dataset = TwhDataset(
        train_path,
        n_poses=args.n_poses,
        subdivision_stride=args.subdivision_stride,
        pose_resampling_fps=args.motion_resampling_framerate,
        data_mean=args.data_mean,
        data_std=args.data_std
    )
    
    val_dataset = TwhDataset(
        val_path,
        n_poses=args.n_poses,
        subdivision_stride=args.subdivision_stride,
        pose_resampling_fps=args.motion_resampling_framerate,
        data_mean=args.data_mean,
        data_std=args.data_std
    )

    # DataLoaders
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True,
        num_workers=args.loader_workers, pin_memory=True, collate_fn=word_seq_collate_fn
    )
    
    test_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True,
        num_workers=args.loader_workers, pin_memory=True, collate_fn=word_seq_collate_fn
    )

    # Start Training
    # Calculate pose dimension based on target joints
    pose_dim = len(target_joints) * 12 
    train_epochs(args, train_loader, test_loader, pose_dim=pose_dim)

if __name__ == '__main__':
    _args = parse_args()
    main({'args': _args})