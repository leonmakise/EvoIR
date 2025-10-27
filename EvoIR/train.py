import logging
from datetime import datetime
import subprocess

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from lightning.pytorch.loggers import WandbLogger
from timm.utils import AverageMeter
from torch.utils.data import DataLoader
import os
os.environ['NCCL_TIMEOUT_MS'] = '3600000'
os.environ['NCCL_ASYNC_ERROR_HANDLING'] = '1'

import torch.optim as optim
import torch.nn as nn
# import wandb
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from net.PerceptualLoss import PerceptualLoss
from net.model import AdaIR as AdaIRModel
from utils.dataset_utils import AdaIRTrainDataset, DenoiseTestDataset, DerainDehazeDataset
from options import options as opt
from options2 import testoptions as testopt
from utils.image_io import save_image_tensor
from utils.schedulers import LinearWarmupCosineAnnealingLR
from utils.val_utils import compute_psnr_ssim
from lightning.pytorch.callbacks import ModelCheckpoint
from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure
from datetime import timedelta
import random



def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size, timeout=timedelta(hours=1))
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def save_checkpoint(logger, model, optimizer, scheduler, epoch, task_weight, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'task_weight': task_weight
    }, path)
    logger.info(f"Model checkpoint saved to {path}")

def load_checkpoint(logger, model, optimizer, scheduler, path):
    checkpoint = torch.load(path, map_location='cpu')
    # if isinstance(model, torch.nn.parallel.DistributedDataParallel):
    #     model = model.module
    # else:
    model = model
    model.load_state_dict(checkpoint['model_state_dict'])
    
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    task_weight = checkpoint.get('task_weight', None)
    logger.info(f"Loaded checkpoint from {path}, epoch {start_epoch}")
    return start_epoch, task_weight

def init_population(size=5):
    population = []
    for _ in range(size):
        λ1 = random.uniform(0.1, 0.9)
        λ2 = 1.0 - λ1
        population.append([λ1, λ2])
    return population

def evaluate_individual_from_cache(individual, restored_list, clean_list, l1_loss, ms_ssim_loss):
    total_loss = 0
    λ1, λ2 = individual
    for restored, clean in zip(restored_list, clean_list):
        loss = λ1 * l1_loss(restored, clean) + λ2 * ms_ssim_loss(restored, clean)
        total_loss += loss.item()
    return -total_loss

def crossover(p1, p2):
    alpha = random.random()
    child1 = [alpha * p1[0] + (1 - alpha) * p2[0],
              alpha * p1[1] + (1 - alpha) * p2[1]]
    child2 = [alpha * p2[0] + (1 - alpha) * p1[0],
              alpha * p2[1] + (1 - alpha) * p1[1]]
    return child1, child2

def mutate(individual, mutation_rate=0.1):
    if random.random() < mutation_rate:
        delta = random.uniform(-0.1, 0.1)
        individual[0] = min(max(individual[0] + delta, 0.1), 0.9)
        individual[1] = 1.0 - individual[0]
    return individual

def evolve_loss_weights_from_cache(restored_list, clean_list, l1_loss, ms_ssim_loss, generations=3, pop_size=5):
    population = init_population(pop_size)
    best = None
    best_score = -float("inf")

    for _ in range(generations):
        scores = []
        for ind in population:
            score = evaluate_individual_from_cache(ind, restored_list, clean_list, l1_loss, ms_ssim_loss)
            scores.append((score, ind))

        scores.sort(reverse=True)
        best_score, best = scores[0]
        next_population = [best]

        while len(next_population) < pop_size:
            p1, p2 = random.sample(scores[:3], 2)
            child1, child2 = crossover(p1[1], p2[1])
            next_population += [mutate(child1), mutate(child2)]

        population = next_population[:pop_size]

    return best

def print_network(model):
    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1e6))

    
def setup_logger(log_dir='logs', log_file='train_log.txt'):
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_file)
    logger = logging.getLogger('train_logger')
    logger.handlers.clear()
    logger.setLevel(logging.INFO)

    rank = int(os.environ.get("RANK", "0"))
    if rank == 0:
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(logging.INFO)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger



class MS_SSIMLoss(torch.nn.Module):
    def __init__(self, data_range=1.0):
        super(MS_SSIMLoss, self).__init__()
        self.ms_ssim = MultiScaleStructuralSimilarityIndexMeasure(
            data_range=1.0,
            kernel_size=5,
            betas=(0.5, 0.5, 0.5, 0.5, 0.5)
        ).cuda()
    def forward(self, pred, target):
        return 1 - self.ms_ssim(pred, target)


def main_worker(rank, world_size):
    setup(rank, world_size)
    torch.cuda.set_device(rank)

    task_list = opt.de_type
    loss_fn = nn.L1Loss()
    λ_l1, λ_ssim = 0.8, 0.2

    ms_ssim_loss = MS_SSIMLoss(data_range=1.0)
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = f'train_log_{current_time}.txt'
    logger = setup_logger(log_dir='logger/', log_file=log_file)

    trainset = AdaIRTrainDataset(opt)
    train_sampler = DistributedSampler(trainset, num_replicas=world_size, rank=rank, shuffle=True)
    trainloader = DataLoader(trainset, batch_size=opt.batch_size, sampler=train_sampler,
                             pin_memory=True, drop_last=True, num_workers=opt.num_workers)

    model = AdaIRModel().cuda(rank)
    optimizer = optim.AdamW(model.parameters(), lr=opt.lr)
    print_network(model)

    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 
    #                                                        T_max=opt.epochs, eta_min=1e-6)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[75, 150], gamma=0.5)

    base_path = testopt.denoise_path
    derain_base_path = testopt.derain_path
    enhance_base_path = testopt.enhance_path
    deblur_base_path = testopt.gopro_path

    denoise_splits = ["bsd68"]
    # denoise_splits = ["bsd68", "urban100", "kodak"]
    derain_splits = ["Rain100L"]
    deblur_splits = ["gopro"]
    enhance_splits = ["lol"]

    denoise_tests = []


    model = DDP(model, device_ids=[rank], find_unused_parameters=True)

    start_epoch = 0
    if opt.resume:
        resume_path = opt.resume_path
        model_path = os.path.join(resume_path, 'latest.pth')
        if resume_path and os.path.exists(model_path):
            start_epoch, task_weight = load_checkpoint(logger, model, optimizer, scheduler, model_path)
            if task_weight is not None:
                λ_l1, λ_ssim = task_weight

    for epoch in tqdm(range(start_epoch, opt.epochs), desc="Training Epochs", unit="epoch"):
        train_sampler.set_epoch(epoch)
        tmp = scheduler.get_last_lr()
        if rank == 0:
            logger.info('Epoch {} learning rate: {}'.format(epoch, tmp))

        model.train()
        running_loss = 0.0
        for batch_idx, ([_, _], degrad_patch, clean_patch) in enumerate(trainloader):
            restored_list, clean_list = [], []
            degrad_patch, clean_patch = degrad_patch.cuda(rank), clean_patch.cuda(rank)
            optimizer.zero_grad()
            restored = model(degrad_patch)


            loss = λ_l1 * loss_fn(restored, clean_patch) + λ_ssim * ms_ssim_loss(restored, clean_patch)
            loss.mean().backward()
            optimizer.step()
            running_loss += loss.item()


            local_restored = restored.detach().cpu()
            local_clean = clean_patch.detach().cpu()

            if batch_idx % 500 == 499 and rank == 0:
                logger.info(f"[Epoch {epoch}/{opt.epochs}, Batch {batch_idx+1}] loss: {running_loss / 500:.4f}")
                running_loss = 0.0

            if batch_idx % 500 == 499:
                if rank == 0:
                    logger.info(f"[Epoch {epoch}] Evolving loss weights...")
  
                all_restored = [None for _ in range(world_size)]
                all_clean = [None for _ in range(world_size)]

                dist.all_gather_object(all_restored, local_restored)
                dist.all_gather_object(all_clean, local_clean)

                λ_l1, λ_ssim = evolve_loss_weights_from_cache(all_restored, all_clean, loss_fn, ms_ssim_loss)


                λ_tensor = torch.tensor([λ_l1, λ_ssim], device='cuda')
                dist.broadcast(λ_tensor, src=0)
                λ_l1, λ_ssim = λ_tensor[0].item(), λ_tensor[1].item()
                if rank == 0:
                    logger.info(f"Updated λ_l1 = {λ_l1:.3f}, λ_ssim = {λ_ssim:.3f}")

        scheduler.step()


        torch.distributed.barrier()


        denoise_tests = []
        for split in denoise_splits:
            testopt.denoise_path = os.path.join(base_path, split) + "/"
            denoise_tests.append(DenoiseTestDataset(testopt))

        for sigma in [15, 25, 50]:
            if f"denoise_{sigma}" in task_list:
                rank = dist.get_rank()
                if rank == 0:
                    logger.info(f"Start denoise sigma={sigma} testing...")
                for testset, split in zip(denoise_tests, denoise_splits):
                    test_Denoise(model.module if isinstance(model, DDP) else model, testset, sigma=sigma, logger=logger)


        if "derain" in task_list:
            for name in derain_splits:
                rank = dist.get_rank()
                if rank == 0:
                    logger.info(f'Start testing {name} rain streak removal...')
                testopt.derain_path = os.path.join(derain_base_path, name) + "/"
                derain_set = DerainDehazeDataset(testopt, addnoise=False, sigma=55, task="derain")
                test_Derain_Dehaze(model.module if isinstance(model, DDP) else model, derain_set, task="derain", logger=logger)

        if "dehaze" in task_list:
            rank = dist.get_rank()
            if rank == 0:
                logger.info('Start testing SOTS...')
            derain_set = DerainDehazeDataset(testopt, addnoise=False, sigma=55, task="dehaze")
            test_Derain_Dehaze(model.module if isinstance(model, DDP) else model, derain_set, task="dehaze", logger=logger)

        if "enhance" in task_list:
            for name in enhance_splits:
                rank = dist.get_rank()
                if rank == 0:
                    logger.info('Start testing LOL...')
                testopt.enhance_path = os.path.join(enhance_base_path) + "/"
                enhance_set = DerainDehazeDataset(testopt, addnoise=False, sigma=55, task='enhance')
                test_Derain_Dehaze(model.module if isinstance(model, DDP) else model, enhance_set, task="enhance", logger=logger)

        if "deblur" in task_list:
            rank = dist.get_rank()
            if rank == 0:
                logger.info('Start testing GOPRO...')
            testopt.gopro_path = os.path.join(deblur_base_path) + "/"
            deblur_set = DerainDehazeDataset(testopt, addnoise=False, sigma=55, task='deblur')
            test_Derain_Dehaze(model.module if isinstance(model, DDP) else model, deblur_set, task="deblur", logger=logger)


        rank = dist.get_rank()
        if rank == 0:
            task_weight = [λ_l1, λ_ssim]
            save_checkpoint(logger, model, optimizer, scheduler, epoch, task_weight,
                            f"checkpoints/model_epoch_{epoch}.pth")
            save_checkpoint(logger, model, optimizer, scheduler, epoch, task_weight,
                            f"checkpoints/latest.pth")


        torch.distributed.barrier()

    torch.distributed.barrier()
    cleanup()

        

def test_Denoise(net, dataset, sigma=15, logger=None):
    net.eval()
    dataset.set_sigma(sigma)
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    rank = dist.get_rank() if dist.is_initialized() else 0
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False) 
    loader = DataLoader(dataset, batch_size=1, sampler=sampler, pin_memory=True, num_workers=0)
    sum_psnr = sum_ssim = total_n = 0.0
    with torch.no_grad():
        for (_, degrad, clean) in loader:
            degrad = degrad.cuda(rank)
            clean = clean.cuda(rank)
            restored = net(degrad)
            p, s, n = compute_psnr_ssim(restored, clean)
            sum_psnr += p * n
            sum_ssim += s * n
            total_n += n
    if dist.is_initialized():
        stats = torch.tensor([sum_psnr, sum_ssim, total_n], device=degrad.device)
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        sum_psnr, sum_ssim, total_n = stats.tolist()
    if rank == 0 and total_n > 0:
        avg_psnr = sum_psnr / total_n
        avg_ssim = sum_ssim / total_n
        logger.info(f"Denoise sigma={sigma}: PSNR={avg_psnr:.2f}, SSIM={avg_ssim:.4f}")


def test_Derain_Dehaze(net, dataset, task="derain", logger=None):
    net.eval()
    dataset.set_dataset(task)
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    rank = dist.get_rank() if dist.is_initialized() else 0
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False)
    loader = DataLoader(dataset, batch_size=1, sampler=sampler, pin_memory=True, num_workers=0)
    sum_psnr = sum_ssim = total_n = 0.0
    with torch.no_grad():
        for (_, degrad, clean) in loader:
            degrad = degrad.cuda(rank)
            clean = clean.cuda(rank)
            restored = net(degrad)
            p, s, n = compute_psnr_ssim(restored, clean)
            sum_psnr += p * n
            sum_ssim += s * n
            total_n += n
    if dist.is_initialized():
        stats = torch.tensor([sum_psnr, sum_ssim, total_n], device=degrad.device)
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        sum_psnr, sum_ssim, total_n = stats.tolist()
    if rank == 0 and total_n > 0:
        avg_psnr = sum_psnr / total_n
        avg_ssim = sum_ssim / total_n
        logger.info(f"{task.capitalize()} PSNR={avg_psnr:.2f}, SSIM={avg_ssim:.4f}")

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    mp.spawn(main_worker, args=(world_size,), nprocs=world_size, join=True)
