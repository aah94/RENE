"""Training loops: baseline pre-training, RENE fine-tuning, and plain KD."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.common import AverageMeter
from ..utils.metrics import evaluate
from .distill import DistillationLoss, FinetuneConfig


@dataclass
class TrainConfig:
    """Standard recipe for training the uncompressed baselines."""
    epochs: int = 200
    lr: float = 0.1
    momentum: float = 0.9
    nesterov: bool = True
    weight_decay: float = 5e-4
    scheduler: str = 'cosine'
    warmup_epochs: int = 0
    label_smoothing: float = 0.0
    grad_clip: float = 0.0
    log_every: int = 100


def _make_scheduler(optimizer, kind: str, epochs: int, steps_per_epoch: int):
    if kind == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs * steps_per_epoch)
    if kind == 'step':
        return torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[int(epochs * 0.5) * steps_per_epoch,
                                   int(epochs * 0.75) * steps_per_epoch], gamma=0.1)
    return None


def train_model(model: nn.Module, train_loader, test_loader, cfg: TrainConfig,
                device, log: Callable[[str], None] = print,
                ckpt_path: Optional[str] = None) -> Dict[str, float]:
    """Train an uncompressed baseline from scratch."""
    model = model.to(device)
    opt = torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=cfg.momentum,
                          nesterov=cfg.nesterov, weight_decay=cfg.weight_decay)
    sched = _make_scheduler(opt, cfg.scheduler, cfg.epochs, len(train_loader))
    best = 0.0
    for epoch in range(cfg.epochs):
        model.train()
        meter, t0 = AverageMeter(), time.time()
        for it, (x, y) in enumerate(train_loader):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x), y, label_smoothing=cfg.label_smoothing)
            loss.backward()
            if cfg.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            if sched is not None:
                sched.step()
            meter.update(loss.item(), y.size(0))
        acc = evaluate(model, test_loader, device)['top1']
        if acc > best:
            best = acc
            if ckpt_path:
                torch.save({'model': model.state_dict(), 'top1': acc, 'epoch': epoch}, ckpt_path)
        log(f'[pretrain] epoch {epoch + 1}/{cfg.epochs} loss={meter.avg:.4f} '
            f'top1={acc:.2f} (best {best:.2f}) {time.time() - t0:.0f}s')
    return {'top1': best}


def finetune_decomposed(student: nn.Module, teacher: nn.Module, layer_names: Sequence[str],
                        train_loader, test_loader, cfg: FinetuneConfig, device,
                        log: Callable[[str], None] = print,
                        ckpt_path: Optional[str] = None) -> Dict[str, float]:
    """Fine-tune the decomposed model with ``L_f = L_ce + lambda L_diss`` (Eq. 11)."""
    student, teacher = student.to(device), teacher.to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    opt = torch.optim.SGD(student.parameters(), lr=cfg.lr, momentum=cfg.momentum,
                          nesterov=cfg.nesterov, weight_decay=cfg.weight_decay)
    sched = _make_scheduler(opt, cfg.scheduler, cfg.epochs, len(train_loader))

    best, history = 0.0, []
    criterion = DistillationLoss(teacher, student, layer_names, cfg)
    with criterion:
        for epoch in range(cfg.epochs):
            student.train()
            ce_m, diss_m, t0 = AverageMeter(), AverageMeter(), time.time()
            for it, (x, y) in enumerate(train_loader):
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                loss, parts = criterion(x, y)
                loss.backward()
                if cfg.grad_clip > 0:
                    nn.utils.clip_grad_norm_(student.parameters(), cfg.grad_clip)
                opt.step()
                if sched is not None:
                    sched.step()
                ce_m.update(parts['ce'].item(), y.size(0))
                diss_m.update(parts['diss'].item(), y.size(0))
                if cfg.log_every and (it + 1) % cfg.log_every == 0:
                    log(f'  [ft] ep{epoch + 1} it{it + 1}/{len(train_loader)} '
                        f'L_ce={ce_m.avg:.4f} L_diss={diss_m.avg:.4f}')
            acc = evaluate(student, test_loader, device)['top1']
            history.append(acc)
            if acc > best:
                best = acc
                if ckpt_path:
                    torch.save({'model': student.state_dict(), 'top1': acc,
                                'epoch': epoch}, ckpt_path)
            log(f'[finetune] epoch {epoch + 1}/{cfg.epochs} L_ce={ce_m.avg:.4f} '
                f'L_diss={diss_m.avg:.4f} top1={acc:.2f} (best {best:.2f}) '
                f'{time.time() - t0:.0f}s')
    return {'top1': best, 'history': history}


def train_kd(student: nn.Module, teacher: nn.Module, train_loader, test_loader,
             device, epochs: int = 100, lr: float = 0.05, temperature: float = 4.0,
             alpha: float = 0.9, weight_decay: float = 5e-4,
             log: Callable[[str], None] = print,
             ckpt_path: Optional[str] = None) -> Dict[str, float]:
    """Vanilla logit KD -- the ``Distillation`` row and the student of Table 3."""
    student, teacher = student.to(device), teacher.to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    opt = torch.optim.SGD(student.parameters(), lr=lr, momentum=0.9, nesterov=True,
                          weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * len(train_loader))
    best = 0.0
    for epoch in range(epochs):
        student.train()
        meter = AverageMeter()
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.no_grad():
                t_logits = teacher(x)
            s_logits = student(x)
            ce = F.cross_entropy(s_logits, y)
            kd = F.kl_div(F.log_softmax(s_logits / temperature, dim=1),
                          F.log_softmax(t_logits / temperature, dim=1),
                          reduction='batchmean', log_target=True) * temperature ** 2
            loss = (1 - alpha) * ce + alpha * kd
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            meter.update(loss.item(), y.size(0))
        acc = evaluate(student, test_loader, device)['top1']
        if acc > best:
            best = acc
            if ckpt_path:
                torch.save({'model': student.state_dict(), 'top1': acc}, ckpt_path)
        log(f'[kd] epoch {epoch + 1}/{epochs} loss={meter.avg:.4f} top1={acc:.2f} (best {best:.2f})')
    return {'top1': best}
