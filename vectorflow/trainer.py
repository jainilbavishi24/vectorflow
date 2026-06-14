import logging
import os
import time
from typing import Dict, Optional
import argparse
import random
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import hydra
from hydra.utils import instantiate
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from vectorflow.train_utils import ddp
from vectorflow.data.utils.collect import collect_batch
from vectorflow.train_utils.ddp import reduce_and_average_losses, ddp_setup_universal
from vectorflow.train_utils.save_model import save_model, resume_model


def viz_batch(data, epoch):
    """Plot ego history, neighbor agents, and lane polylines for the first sample in the batch."""
    ego_past      = data.ego_past[0].cpu().float().numpy()       # (21, F)
    neighbor_past = data.neighbor_past[0].cpu().float().numpy()  # (32, 21, F)
    lanes         = data.lanes[0].cpu().float().numpy()          # (70, 20, 12)

    fig, ax = plt.subplots(figsize=(10, 10))

    # lanes — thin grey polylines
    for lane in lanes:
        valid = np.any(lane != 0, axis=-1)
        if valid.any():
            ax.plot(lane[valid, 0], lane[valid, 1], color='grey', linewidth=0.5, alpha=0.4)

    # neighbors — blue trails ending with a dot
    for agent in neighbor_past:
        valid = np.any(agent != 0, axis=-1)
        if valid.sum() > 1:
            ax.plot(agent[valid, 0], agent[valid, 1], color='steelblue', linewidth=0.8, alpha=0.6)
            ax.scatter(agent[valid, 0][-1], agent[valid, 1][-1], color='steelblue', s=15, zorder=3)

    # ego — red trail ending with a star
    ax.plot(ego_past[:, 0], ego_past[:, 1], color='red', linewidth=2, label='ego past')
    ax.scatter(ego_past[-1, 0], ego_past[-1, 1], color='red', s=80, marker='*', zorder=4)

    ax.set_aspect('equal')
    ax.legend(loc='upper right')
    ax.set_title(f'Epoch {epoch + 1} — batch sample 0 (normalized coords)')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def set_seed(CUR_SEED):
    random.seed(CUR_SEED)
    np.random.seed(CUR_SEED)
    torch.manual_seed(CUR_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

@hydra.main(version_base=None, config_path="script")
def trainer(cfg: DictConfig):
    
    set_seed(cfg.seed)
    
    global_rank, local_rank, world_size = ddp_setup_universal(verbose=True, cfg=cfg)

    if local_rank == 0:
        os.makedirs(cfg.save_dir, exist_ok=True)
    
    logger = logging.getLogger(__name__)
    log_path = os.path.join(cfg.save_dir, f'{cfg.job_name}.log')
    logging.basicConfig(filename=log_path, level=logging.INFO)
    
    assert cfg.train.batch_size >= world_size, f"batch size is at least world size, but got batch size {cfg.train.batch_size} running on {world_size} devices."
    
    if global_rank > 0:
        logger.setLevel(logging.WARNING)
        
    # build model
    logger.info("build model")
    model = instantiate(cfg.model)

    # load pretrain checkpoint
    if cfg.pretrained_checkpoint is not None:
        ckpt = torch.load(cfg.pretrained_checkpoint, weights_only=True)
        model.load_state_dict({n.split("module.")[1]: v for n, v in ckpt['ema_state_dict'].items()})

    model = model.to(local_rank)  # Move model to the current global_rank's GPU
    if cfg.ddp.distributed:
        model = DDP(model, device_ids=[local_rank], output_device=global_rank)
    
    # construct dataset & dataloader
    logger.info("construct dataset and dataloader")
    trainset = instantiate(cfg.data.dataset.train)
    trainsampler = DistributedSampler(trainset, num_replicas=world_size, rank=global_rank, shuffle=True)
    trainloader = DataLoader(trainset, sampler=trainsampler, batch_size=cfg.train.batch_size // world_size, num_workers=cfg.num_workers, pin_memory=cfg.pin_mem, drop_last=True, collate_fn=collect_batch)
    logger.info(f"Dataset size: {len(trainset)}, batches per epoch (rank {global_rank}): {len(trainloader)}, batch_size_per_gpu: {cfg.train.batch_size // world_size}")
    
    # build optimizer
    logger.info("build optimizer")
    optimizer = instantiate(cfg.optimizer, params=ddp.get_model(model).get_optimizer_params(), lr=cfg.optimizer.lr)

    # build scheduler
    logger.info("build scheduler")
    scheduler = instantiate(cfg.scheduler, optimizer=optimizer)
    
    # build ema
    logger.info("build ema")
    ema = instantiate(cfg.ema, model=model)
    # build core
    logger.info("build core")
    # TODO: modify augmentation for compatibility with AirFormerDataSample
    core = instantiate(cfg.core)
    
    if cfg.resume_path is not None:
        model, optimizer, scheduler, init_epoch, wandb_id, ema = resume_model(
            cfg.resume_path,
            model,
            optimizer,
            scheduler,
            ema,
            cfg.device
        )
    else:
        wandb_id = None
        init_epoch = 0
        
    # initialize recorder eg. wandb, tensorboard
    logger.info("initialize recorder")
    recorder_dict = {}
    if hasattr(cfg, "recorder"):
        for name, rec in cfg.recorder.items():
            if name == 'wandb':    
                recorder_dict[name] = instantiate(rec, wandb_id=wandb_id)
            else:
                recorder_dict[name] = instantiate(rec)

    # train model
    logger.info("Training launched")
    
    timer = time.time()

    with tqdm(total=cfg.train.epoch, initial=init_epoch, disable=(global_rank != 0)) as epoch_bar:
        for epoch in range(init_epoch, cfg.train.epoch):
            trainsampler.set_epoch(epoch)
                        
            model.train()

            # Visualize one batch before training starts for this epoch (rank 0 only)
            if global_rank == 0 and recorder_dict:
                viz_data = next(iter(trainloader)).to(cfg.device)
                fig = viz_batch(viz_data, epoch)
                for recorder in recorder_dict.values():
                    if hasattr(recorder, 'record_figure'):
                        recorder.record_figure('batch/scene', fig, epoch + 1)
                plt.close(fig)

            loss_list = []

            # Training Step
            with tqdm(total=len(trainloader), desc=f"Epoch {epoch+1} - Training", disable=(global_rank != 0), leave=False) as batch_bar:
                for k, data in enumerate(trainloader):
                    data = data.to(cfg.device)
                    loss = core.train_step(model, data)
                    
                    optimizer.zero_grad()
                    loss['total_loss'].backward()

                    nn.utils.clip_grad_norm_(ddp.get_model(model).parameters(), 5)
                    
                    optimizer.step()

                    ema.update(model)

                    loss_list.append(loss)

                    tqdm_dict = {'loss' : f"{loss['total_loss'].item():.3f}"}
                        
                    batch_bar.set_postfix(tqdm_dict)
                    batch_bar.update(1)

            scheduler.step()
            
            # record loss
            if not loss_list:
                logger.warning(f"Epoch {epoch+1}: no batches processed (batch_size may exceed dataset size). Skipping.")
                epoch_bar.update(1)
                continue

            epoch_loss = {name: ( sum([l[name] for l in loss_list]) / len(loss_list) ) for name in loss_list[-1].keys()}
            epoch_lr = {f"lr/group_{i}": param_group['lr'] for i, param_group in enumerate(optimizer.param_groups)}
            
            if cfg.ddp.distributed:
                epoch_loss = reduce_and_average_losses(epoch_loss, torch.device(cfg.device))

            logger.info(f"epoch loss : {epoch_loss['total_loss']:.3e} | epoch_lr : {epoch_lr['lr/group_0']:.3e}")

            if global_rank == 0:
                for recorder in recorder_dict.values():
                    recorder.record_loss(epoch_loss, epoch+1)
                    recorder.record_loss(epoch_lr, epoch+1)
                    
            # save model
            if global_rank == 0 and (epoch+1) % cfg.train.save_utd == 0:
                if 'wandb' in recorder_dict.keys():
                    wandb_id = recorder_dict['wandb'].id
                else:
                    wandb_id = None
                save_model(model, optimizer, scheduler, cfg.save_dir, epoch, epoch_loss['total_loss'], wandb_id, ema.ema, save_every_epoch=cfg.save_every_since)
                print(f"Model saved in {cfg.save_dir}\n")
            
            epoch_bar.update(1)
            
            if cfg.ddp.distributed:
                torch.cuda.synchronize()
            
    logger.info(f"Training finished - Time consumed: {time.strftime('%H:%M:%S', time.gmtime(time.time()-timer))}")
    
    torch.distributed.destroy_process_group()
    
if __name__ == '__main__':
    trainer()