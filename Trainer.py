import os
import sys
import math
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from utils.util import AverageMeter, ensure_dir
from tqdm import tqdm

import torchvision
from utils.metrics import Evaluator_tensor
import shutil

import utils.mask_gen as mask_gen
from torch.cuda.amp import autocast
from torch.cuda.amp import grad_scaler

unloader = torchvision.transforms.ToPILImage()

class EMAWeightOptimizer(object):
    """
    mean teacher optimizer
    """

    def __init__(self, target_net, source_net, ema_alpha):
        self.target_net = target_net
        self.source_net = source_net
        self.ema_alpha = ema_alpha
        self.target_params0 = [p for p in target_net[0].state_dict().values() if p.dtype == torch.float]
        self.target_params1 = [p for p in target_net[1].state_dict().values() if p.dtype == torch.float]
        self.source_params = [p for p in source_net.state_dict().values() if p.dtype == torch.float]

        for tgt_p, src_p in zip(self.target_params0, self.source_params):
            tgt_p[...] = src_p[...]
        for tgt_p, src_p in zip(self.target_params1, self.source_params):
            tgt_p[...] = src_p[...]
        target_keys0 = set(target_net[0].state_dict().keys())
        target_keys1 = set(target_net[1].state_dict().keys())
        source_keys = set(source_net.state_dict().keys())
        if target_keys0 != source_keys or target_keys1 != source_keys:
            raise ValueError(
                'Source and target networks do not have the same state dict keys; do they have different architectures?')

    def step(self, step=0):
        one_minus_alpha = 1.0 - self.ema_alpha
        if step == 0:
            for tgt_p, src_p in zip(self.target_params0, self.source_params):
                tgt_p.mul_(self.ema_alpha)
                tgt_p.add_(src_p * one_minus_alpha)
        elif step == 1:
            for tgt_p, src_p in zip(self.target_params1, self.source_params):
                tgt_p.mul_(self.ema_alpha)
                tgt_p.add_(src_p * one_minus_alpha)


def cal_category_confidence(preds_student_sup, preds_student_unsup, gt, preds_teacher_unsup, num_classes):
    category_confidence = torch.zeros(num_classes).type(torch.float32)
    preds_student_sup = F.softmax(preds_student_sup, dim=1)
    for ind in range(num_classes):
        cat_mask_sup_gt = (gt == ind)
        if torch.sum(cat_mask_sup_gt) == 0:
            value = 0
        else:
            conf_map_sup = preds_student_sup[:, ind, :, :]
            value = torch.sum(conf_map_sup * cat_mask_sup_gt.float()) / (torch.sum(cat_mask_sup_gt) + 1e-12)
        category_confidence[ind] = value
    return category_confidence


class Criterion_cons(nn.Module):
    def __init__(self, gamma, sample=False, gamma2=1,
                 ignore_index=255):
        """
        consistency loss function (weighted ce loss)
        """
        super(Criterion_cons, self).__init__()
        self.gamma = gamma
        self.gamma2 = float(gamma2)
        self._ignore_index = ignore_index
        self.sample = sample
        self._criterion = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='none')

    def forward(self, preds, conf, gt, dcp_criterion=None):
        ce_loss = self._criterion(preds, gt)
        conf = torch.pow(conf, self.gamma)

        if self.sample:
            dcp_criterion = 1 - dcp_criterion
            dcp_criterion = dcp_criterion / (torch.max(dcp_criterion) + 1e-12)
            dcp_criterion = torch.pow(dcp_criterion, self.gamma2)
            pred_map = preds.max(1)[1].float()

            sample_map = torch.zeros_like(pred_map).float()
            h, w = pred_map.shape[-2], pred_map.shape[-1]

            for idx in range(len(dcp_criterion)):
                prob = 1 - dcp_criterion[idx]
                rand_map = torch.rand(h, w).cuda() * (pred_map == idx)
                rand_map = (rand_map > prob) * 1.0
                sample_map += rand_map
            conf = conf * (sample_map)
        conf = conf / (conf.sum() + 1e-12)

        loss = conf * ce_loss
        return loss.sum()


class Trainer(object):

    def __init__(self,
                 model_student,
                 model_teacher,
                 config,
                 args,
                 train_data_loader,
                 valid_data_loader,
                 train_unsup_data_loader0,
                 train_unsup_data_loader1,
                 begin_time,
                 resume_file=None):

        print("     + Training Start ... ...")
        # for general
        self.config = config
        self.args = args
        self.device = (self._device(self.args.gpu))
        self.model_student = model_student.to(self.device)
        if model_teacher is not None:
            self.model_teacher = model_teacher

        self.train_data_loader = train_data_loader
        self.valid_data_loder = valid_data_loader
        self.unsupervised_train_loader_0 = train_unsup_data_loader0
        self.unsupervised_train_loader_1 = train_unsup_data_loader1

        # for time
        self.begin_time = begin_time  # part of ckpt name
        self.save_period = self.config.save_period  # for save ckpt
        self.dis_period = self.config.dis_period  # for display

        self.model_name = self.config.model_name

        if self.config.use_seed:
            self.checkpoint_dir = os.path.join(self.args.output, self.model_name,
                                               self.begin_time + '_seed' + str(self.config.random_seed))
            self.log_dir = os.path.join(self.args.output, self.model_name,
                                        self.begin_time + '_seed' + str(self.config.random_seed), 'log')
        else:
            self.checkpoint_dir = os.path.join(self.args.output, self.model_name,
                                               self.begin_time)
            self.log_dir = os.path.join(self.args.output, self.model_name,
                                        self.begin_time, 'log')

        ensure_dir(self.checkpoint_dir)
        ensure_dir(self.log_dir)

        # log file
        log_file_path = os.path.join(self.log_dir, self.model_name + '.txt')
        self.config.write_to_file(log_file_path)

        self.history = {
            'train': {
                'epoch': [],
                'loss': [],
                'acc': [],
                'miou': [],
                'prec': [],
                'recall': [],
                'f_score': [],
            },
            'valid': {
                'epoch': [],
                'loss': [],
                'acc': [],
                'miou': [],
                'prec': [],
                'recall': [],
                'f_score': [],
            }
        }
        # for optimize
        self.weight_init_algorithm = self.config.init_algorithm
        self.current_lr = self.config.init_lr

        # for train
        self.start_epoch = 0
        self.early_stop = self.config.early_stop  # early stop steps
        self.monitor_mode = self.config.monitor.split('/')[0]
        self.monitor_metric = self.config.monitor.split('/')[1]
        self.monitor_best = 0
        self.best_epoch = -1
        self.not_improved_count = 0
        self.monitor_iou = 0

        # resume file: the confirmed ckpt file.
        self.resume_file = resume_file
        self.resume_ = True if resume_file else False
        if self.resume_file is not None:
            with open(log_file_path, 'a') as f:
                f.write('\n')
                f.write('resume_file:' + resume_file + '\n')
            self._resume_ckpt_PSMT(resume_file=resume_file)

        self.optimizer_teacher = EMAWeightOptimizer(self.model_teacher, self.model_student, 0.99)
        self.optimizer_student = self._optimizer(lr_algorithm=self.config.lr_algorithm)

        # monitor init
        if self.monitor_mode != 'off':
            assert self.monitor_mode in ['min', 'max']
            self.monitor_best = math.inf if self.monitor_mode == 'min' else -math.inf

        if self.config.use_one_cycle_lr:
            self.lr_scheduler = self._lr_scheduler_onecycle(self.optimizer_student)
        else:
            self.lr_scheduler = self._lr_scheduler_lambda(self.optimizer_student, last_epoch=self.start_epoch - 1)

        self.evaluator = Evaluator_tensor(self.config.nb_classes, self.device)
        self.evaluator_BD = Evaluator_tensor(2, self.device)
        self.choice = 0

    def _device(self, gpu):

        if gpu == -1:
            device = torch.device('cpu')
            return device
        else:
            device = torch.device('cuda:{}'.format(gpu))
            return device

    def _optimizer(self, lr_algorithm):
        assert lr_algorithm in ['adam', 'adamw', 'sgd']
        if lr_algorithm == 'adam':
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model_student.parameters()),
                                   lr=self.current_lr,
                                   betas=(0.9, 0.999),
                                   eps=1e-08,
                                   weight_decay=self.config.weight_decay,
                                   amsgrad=False
                                   )
            return optimizer
        if lr_algorithm == 'sgd':
            optimizer = optim.SGD(filter(lambda p: p.requires_grad, self.model_student.parameters()),
                                  lr=self.current_lr,
                                  momentum=self.config.momentum,
                                  dampening=0,
                                  weight_decay=self.config.weight_decay,
                                  nesterov=True)
            return optimizer
        if lr_algorithm == 'adamw':
            optimizer = optim.AdamW(filter(lambda p: p.requires_grad, self.model_student.parameters()),
                                    lr=self.current_lr,
                                    betas=(0.9, 0.999),
                                    eps=1e-08,
                                    weight_decay=self.config.weight_decay,
                                    amsgrad=False
                                    )
            return optimizer

    def _lr_scheduler_onecycle(self, optimizer):
        lr_scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=self.config.init_lr * 6,
                                                     steps_per_epoch=len(self.train_data_loader),
                                                     epochs=self.config.epochs + 1,
                                                     div_factor=6)
        return lr_scheduler

    def _lr_scheduler_lambda(self, optimizer, last_epoch):
        lambda1 = lambda epoch: pow((1 - ((epoch - 1) / self.config.epochs)), 0.9)
        lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1, last_epoch=last_epoch)

        return lr_scheduler

    def train_damage_PDMT(self):
        if self.resume_ == False:
            # ensure the same parameters between teacher and student model
            student_dict = self.model_student.state_dict()
            self.model_teacher[0].load_state_dict(student_dict, strict=True)
            self.model_teacher[1].load_state_dict(student_dict, strict=True)
            print("     + Init weight ... Done !")

        epochs = self.config.epochs
        assert self.start_epoch < epochs

        # AEL参数设置
        class_criterion = torch.rand(3, self.config.nb_classes).type(torch.float32)

        for epoch in range(self.start_epoch, epochs + 1):
            # get log information of train and evaluation phase
            train_log, class_criterion = self._train_epoch_damage_PDMT(epoch, class_criterion)
            eval_log = self._eval_epoch_damage_PDMT(epoch)

            if not self.config.use_one_cycle_lr and not self.config.learning_rate_find:
                # lr update
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step(epoch)
                    for param_group in self.optimizer_student.param_groups:
                        self.current_lr = param_group['lr']

            best = False
            diff = 0
            if self.monitor_mode != 'off':
                improved = (self.monitor_mode == 'min' and eval_log[
                    'val_' + self.monitor_metric] < self.monitor_best) or \
                           (self.monitor_mode == 'max' and eval_log['val_' + self.monitor_metric] > self.monitor_best)
                if improved:
                    self.monitor_best = eval_log['val_' + self.monitor_metric]
                    self.monitor_iou = eval_log['val_MIoU']
                    best = True
                    self.best_epoch = eval_log['epoch']
                    self.not_improved_count = 0
                else:
                    self.not_improved_count += 1

                if self.not_improved_count > self.early_stop:
                    print("     + Validation Performance didn\'t improve for {} epochs."
                          "     + Training stop :/"
                          .format(self.not_improved_count))
                    break
            if epoch % self.save_period == 0 or best == True:
                self._save_ckpt(epoch, best=best)

        # save history file
        print("     + Saving History ... ... ")
        hist_path = os.path.join(self.log_dir, 'history1.txt')
        with open(hist_path, 'w') as f:
            f.write(str(self.history))

    def _train_epoch_damage_PDMT(self, epoch, class_criterion):
        """
        train one epoch
        """
        ave_total_loss = AverageMeter()
        ave_total_sup_loss = AverageMeter()
        ave_total_unsup_loss = AverageMeter()

        scaler = grad_scaler.GradScaler()

        self.evaluator.reset()
        self.evaluator_BD.reset()
        bce_loss=nn.BCEWithLogitsLoss()
        ce_loss=nn.CrossEntropyLoss()

        cons_loss = Criterion_cons(gamma=self.config.gamma, sample=False, gamma2=1, ignore_index=255)
        class_momentum = 0.999

        display = False
        # only for display
        if display:
            conf = 1 - class_criterion[0]
            conf = (conf ** 0.5).numpy()
            conf_print = np.exp(conf) / np.sum(np.exp(conf))
            print('epoch [', epoch, ': ]', 'sample_rate_target_class_conf', conf_print)  # sample rate
            print('epoch [', epoch, ': ]', 'criterion_per_class', class_criterion[0])
            print('epoch [', epoch, ': ]', 'sample_rate_per_class_conf',
                  (1 - class_criterion[0]) / (torch.max(1 - class_criterion[0]) + 1e-12))

        # set model mode
        self.model_student.train()

        train_dataloader = iter(self.train_data_loader)
        unsupervised_dataloader_0 = iter(self.unsupervised_train_loader_0)
        unsupervised_dataloader_1 = iter(self.unsupervised_train_loader_1)

        mask_generator = mask_gen.BoxMaskGenerator(prop_range=(0.4, 0.4), n_boxes=3,
                                                   random_aspect_ratio=True,
                                                   prop_by_area=True, within_bounds=True,
                                                   invert=True)

        max_samples = max(len(self.train_data_loader), len(self.unsupervised_train_loader_0)) * self.config.batch_size
        niters_per_epoch = max_samples // self.config.batch_size
        bar_format = '{desc}[{elapsed}<{remaining},{rate_fmt}]'
        pbar = tqdm(range(niters_per_epoch), file=sys.stdout, bar_format=bar_format)
        for idx in pbar:
            train_minibatch = train_dataloader.next()
            imgs = train_minibatch[0].to(self.device, non_blocking=True)
            gts = train_minibatch[1].to(self.device, non_blocking=True)

            target_BD = gts.clone().detach()
            target_BD[target_BD != 0] = 1

            self.optimizer_student.zero_grad()

            # sup
            with autocast():
                sup_logits = self.model_student(imgs)
                gts = gts.long()
                sup_D_loss = ce_loss(sup_logits['damage'], gts)
                sup_BD_loss = bce_loss(torch.squeeze(sup_logits['building'], dim=1), target_BD)
                sup_loss = sup_D_loss + sup_BD_loss

            # update class_criterion
            with torch.no_grad():
                # cal category-wise confidence
                category_entropy = cal_category_confidence(sup_logits['damage'].detach(),
                                                                   None, gts,
                                                                   None, self.config.nb_classes)
                # update category-wise confidence by EMA
                class_criterion = class_criterion * class_momentum + category_entropy * (1 - class_momentum)

            # unsup
            unsup_loss = 0
            if epoch > self.config.warmup_period:
                unsup_minibatch_0 = unsupervised_dataloader_0.next()
                unsup_minibatch_1 = unsupervised_dataloader_1.next()
                unsup_imgs_0 = unsup_minibatch_0[0].to(self.device, non_blocking=True)
                unsup_imgs_1 = unsup_minibatch_1[0].to(self.device, non_blocking=True)

                if self.config.use_cutmix:
                    batch_mix_masks = torch.from_numpy(mask_generator.generate_params(unsup_imgs_0.shape[0], (
                        self.config.input_size, self.config.input_size)).astype(dtype=np.float32)).to(self.device,
                                                                                                      non_blocking=True)
                    unsup_imgs_mixed = unsup_imgs_0 * (1 - batch_mix_masks) + unsup_imgs_1 * batch_mix_masks

                else:
                    unsup_imgs_mixed = unsup_imgs_0

                with torch.no_grad():
                    logits_u0_tea_1 = self.model_teacher[0](unsup_imgs_0)
                    prob_u0_tea_BD_1 = torch.sigmoid(logits_u0_tea_1['building']).detach()
                    prob_u0_tea_D_1 = torch.sigmoid(logits_u0_tea_1['damage']).detach()

                    if self.config.use_mix:
                        logits_u1_tea_1 = self.model_teacher[1](unsup_imgs_1)
                        prob_u1_tea_BD_1 = torch.sigmoid(logits_u1_tea_1['building']).detach()
                        prob_u1_tea_D_1 = torch.sigmoid(logits_u1_tea_1['damage']).detach()
                        prob_cons_tea_BD_1 = prob_u0_tea_BD_1 * (
                                    1 - batch_mix_masks) + prob_u1_tea_BD_1 * batch_mix_masks
                        prob_cons_tea_D_1 = prob_u0_tea_D_1 * (1 - batch_mix_masks) + prob_u1_tea_D_1 * batch_mix_masks
                    else:
                        prob_cons_tea_BD_1 = prob_u0_tea_BD_1
                        prob_cons_tea_D_1 = prob_u0_tea_D_1

                    conf_unsup1 = F.softmax(prob_cons_tea_D_1, dim=1).max(1)[0]
                    ps_label_D_1 = torch.argmax(prob_cons_tea_D_1, dim=1)

                    prob_cons_tea_BD_1[prob_cons_tea_BD_1 < 0.5] = 0
                    prob_cons_tea_BD_1[prob_cons_tea_BD_1 >= 0.5] = 1
                    ps_label_BD_1 = torch.squeeze(prob_cons_tea_BD_1, dim=1)

                with autocast():
                    logits_cons_stu_1 = self.model_student(unsup_imgs_mixed)

                    unsup_loss_BD = bce_loss(torch.squeeze(logits_cons_stu_1['building'], dim=1), ps_label_BD_1)
                    unsup_loss_D = cons_loss(logits_cons_stu_1['damage'], conf_unsup1, ps_label_D_1,
                                         class_criterion[0])

                    unsup_loss=unsup_loss_D + unsup_loss_BD

            loss = sup_loss + unsup_loss * 1.0

            scaler.scale(loss).backward()
            scaler.step(self.optimizer_student)
            scaler.update()
            # update the parameters of two teacher models iteratively
            self.optimizer_teacher.step(step=self.choice)
            if self.choice == 0:
                self.choice == 1
            elif self.choice == 1:
                self.choice == 0

            pred_D = torch.argmax(sup_logits['damage'], dim=1)
            pred_D = pred_D.view(-1).long()
            label_D = gts.view(-1).long()

            pred_BD = 0.5 < sup_logits['building']
            pred_BD = pred_BD.view(-1).long()

            # Add batch sample into evaluator
            self.evaluator.add_batch(label_D, pred_D)
            self.evaluator_BD.add_batch(target_BD.view(-1).long(), pred_BD)
            ave_total_loss.update(loss.item())
            ave_total_sup_loss.update(sup_loss.item())
            ave_total_unsup_loss.update(unsup_loss)

            if self.config.use_one_cycle_lr:
                # lr update
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()
                    for param_group in self.optimizer_student.param_groups:
                        self.current_lr = param_group['lr']

        # Building acc
        acc0 = self.evaluator_BD.Pixel_Accuracy().cpu().detach().numpy()
        acc_class0 = self.evaluator_BD.Pixel_Accuracy_Class().cpu().detach().numpy()
        miou0 = self.evaluator_BD.Mean_Intersection_over_Union().cpu().detach().numpy()
        fwiou0 = self.evaluator_BD.Frequency_Weighted_Intersection_over_Union().cpu().detach().numpy()
        confusion_matrix0 = self.evaluator_BD.get_confusion_matrix().cpu().detach().numpy()
        TP0, FP0, FN0, TN0 = self.evaluator_BD.get_base_value()
        iou0 = self.evaluator_BD.get_iou().cpu().detach().numpy()
        prec0 = self.evaluator_BD.Pixel_Precision_Class().cpu().detach().numpy()
        recall0 = self.evaluator_BD.Pixel_Recall_Class().cpu().detach().numpy()
        f1_score0 = self.evaluator_BD.Pixel_F1_score_Class().cpu().detach().numpy()
        kappa_coe0 = self.evaluator_BD.Kapaa_coefficient().cpu().detach().numpy()

        acc = self.evaluator.Pixel_Accuracy().cpu().detach().numpy()
        acc_class = self.evaluator.Pixel_Accuracy_Class().cpu().detach().numpy()
        miou = self.evaluator.Mean_Intersection_over_Union().cpu().detach().numpy()
        confusion_matrix1 = self.evaluator.get_confusion_matrix().cpu().detach().numpy()
        TP, FP, FN, TN = self.evaluator.get_base_value()
        iou = self.evaluator.get_iou().cpu().detach().numpy()
        prec = self.evaluator.Pixel_Precision_Class().cpu().detach().numpy()
        recall = self.evaluator.Pixel_Recall_Class().cpu().detach().numpy()
        f1_score = self.evaluator.Pixel_F1_score_Class().cpu().detach().numpy()

        #  train log and return
        self.history['train']['epoch'].append(epoch)
        self.history['train']['loss'].append(ave_total_sup_loss.average())
        self.history['train']['acc'].append(acc.tolist())
        self.history['train']['miou'].append(miou.tolist())

        self.history['train']['prec'].append(prec[1])
        self.history['train']['recall'].append(recall[1])
        self.history['train']['f_score'].append(f1_score[1])
        result = {
            'epoch': epoch,
            'loss': ave_total_loss.average(),
            'acc': acc,
            'miou': miou,
            'mprec': np.mean(prec),
            'mrecall': np.mean(recall),
            'mf_score': np.mean(f1_score),
            'acc_BD': acc0,
            'iou_BD': iou0[1],
            'prec_BD': prec0[1],
            'recall_BD': recall0[1],
            'f_score_BD': f1_score0[1],
        }
        return result, class_criterion

    def _eval_epoch_damage_PDMT(self, epoch):
        ave_total_loss = AverageMeter()
        ce_loss = nn.CrossEntropyLoss()
        bce_loss = nn.BCEWithLogitsLoss()
        self.evaluator.reset()
        self.evaluator_BD.reset()
        # set model mode
        self.model_teacher[0].eval()
        self.model_teacher[1].eval()

        with torch.no_grad():
            for steps, (imgs, gts, filename) in enumerate(self.valid_data_loder, start=1):
                imgs = imgs.to(self.device, non_blocking=True)
                gts = gts.to(self.device, non_blocking=True)

                target_BD = gts.clone().detach()
                target_BD[target_BD != 0] = 1

                # supervised loss on both models
                sup_logits0 = self.model_teacher[0](imgs)
                sup_logits1 = self.model_teacher[1](imgs)

                sup_logits={}
                sup_logits['damage']=(sup_logits0['damage']+sup_logits1['damage'])/2
                sup_logits['building']=(sup_logits0['building']+sup_logits1['building'])/2

                gts = gts.long()
                loss_D = ce_loss(sup_logits['damage'], gts)
                loss_BD = bce_loss(torch.squeeze(sup_logits['building'], dim=1), target_BD)
                loss = loss_BD + loss_D

                pred_D = torch.argmax(sup_logits['damage'], dim=1)
                pred_D = pred_D.view(-1).long()
                label = gts.view(-1).long()

                pred_BD = 0.5 < sup_logits['building']
                pred_BD = pred_BD.view(-1).long()

                # Add batch sample into evaluator
                self.evaluator.add_batch(label, pred_D)
                self.evaluator_BD.add_batch(target_BD.view(-1).long(), pred_BD)

                # update ave metrics
                ave_total_loss.update(loss.item())

            # Building acc
            acc0 = self.evaluator_BD.Pixel_Accuracy().cpu().detach().numpy()
            acc_class0 = self.evaluator_BD.Pixel_Accuracy_Class().cpu().detach().numpy()
            miou0 = self.evaluator_BD.Mean_Intersection_over_Union().cpu().detach().numpy()
            fwiou0 = self.evaluator_BD.Frequency_Weighted_Intersection_over_Union().cpu().detach().numpy()
            confusion_matrix0 = self.evaluator_BD.get_confusion_matrix().cpu().detach().numpy()
            TP0, FP0, FN0, TN0 = self.evaluator_BD.get_base_value()
            iou0 = self.evaluator_BD.get_iou().cpu().detach().numpy()
            prec0 = self.evaluator_BD.Pixel_Precision_Class().cpu().detach().numpy()
            recall0 = self.evaluator_BD.Pixel_Recall_Class().cpu().detach().numpy()
            f1_score0 = self.evaluator_BD.Pixel_F1_score_Class().cpu().detach().numpy()
            kappa_coe0 = self.evaluator_BD.Kapaa_coefficient().cpu().detach().numpy()

            # calculate metrics
            acc = self.evaluator.Pixel_Accuracy().cpu().detach().numpy()
            acc_class = self.evaluator.Pixel_Accuracy_Class().cpu().detach().numpy()
            miou = self.evaluator.Mean_Intersection_over_Union().cpu().detach().numpy()
            fwiou = self.evaluator.Frequency_Weighted_Intersection_over_Union().cpu().detach().numpy()
            confusion_matrix1 = self.evaluator.get_confusion_matrix().cpu().detach().numpy()
            TP, FP, FN, TN = self.evaluator.get_base_value()
            iou = self.evaluator.get_iou().cpu().detach().numpy()
            prec = self.evaluator.Pixel_Precision_Class().cpu().detach().numpy()
            recall = self.evaluator.Pixel_Recall_Class().cpu().detach().numpy()
            f1_score = self.evaluator.Pixel_F1_score_Class().cpu().detach().numpy()
            kappa_coe = self.evaluator.Kapaa_coefficient().cpu().detach().numpy()


            print('Epoch {} validation done !'.format(epoch))
            print('lr: {:.8f}\n'
                  'MIoU: {:6.4f},       Accuracy: {:6.4f},    Loss: {:.6f},\n'
                  'Precision: {:6.4f},  Recall: {:6.4f},      F_Score: {:6.4f}'
                  .format(self.current_lr,
                          miou, acc, ave_total_loss.average(),
                          prec[1], recall[1], f1_score[1]))

        self.history['valid']['epoch'].append(epoch)
        self.history['valid']['loss'].append(ave_total_loss.average())
        self.history['valid']['acc'].append(acc.tolist())
        self.history['valid']['miou'].append(miou.tolist())
        self.history['valid']['prec'].append(prec[1])
        self.history['valid']['recall'].append(recall[1])
        self.history['valid']['f_score'].append(f1_score[1])

        #  validation log and return
        return {
            'epoch': epoch,
            'val_Loss': ave_total_loss.average(),
            'val_Acc': acc,
            'val_MIoU': miou,
            'val_mprec': np.mean(prec),
            'val_mrecall': np.mean(recall),
            'val_mf_score': np.mean(f1_score),
            'val_acc_BD': acc0,
            'val_iou_BD': iou0[1],
            'val_prec_BD': prec0[1],
            'val_recall_BD': recall0[1],
            'val_f_score_BD': f1_score0[1],
        }

    def _save_ckpt(self, epoch, best):
        # save model ckpt
        state = {
            'epoch': epoch,
            'arch': str(self.model_teacher[0]),
            'history': self.history,
            'state_dict1': self.model_teacher[0].state_dict(),
            'state_dict2': self.model_teacher[1].state_dict(),
            'monitor_best': self.monitor_best,
        }
        filename = os.path.join(self.checkpoint_dir, 'checkpoint-ep{}.pth'.format(epoch))
        best_filename = os.path.join(self.checkpoint_dir, 'checkpoint-best.pth')
        last_best_filename = os.path.join(self.checkpoint_dir,
                                          'checkpoint-ep{}-iou{:.4f}.pth'.format(epoch, self.monitor_iou))
        if best:
            if os.path.exists(best_filename):
                shutil.copyfile(best_filename, last_best_filename)

            print("     + Saving Best Checkpoint : Epoch {}  path: {} ...  ".format(epoch, best_filename))
            torch.save(state, best_filename)
        else:
            start_save_epochs = math.ceil(self.config.epochs * 0.5)
            if epoch > start_save_epochs:
                print("     + After {} epochs, saving Checkpoint per {} epochs, path: {} ... ".format(start_save_epochs,
                                                                                                      self.save_period,
                                                                                                      filename))
                torch.save(state, filename)

    def _resume_ckpt_PSMT(self, resume_file):
        print("     + Loading ckpt path : {} ...".format(resume_file))
        checkpoint = torch.load(resume_file)
        self.model_student.load_state_dict(checkpoint['state_dict1'], strict=True)
        self.model_teacher[0].load_state_dict(checkpoint['state_dict1'], strict=True)
        self.model_teacher[1].load_state_dict(checkpoint['state_dict2'], strict=True)
        print("     + Model State Loaded ! :D ")
        print("     + Checkpoint file: '{}' , Loaded ! \n"
              "     + Prepare to test ! ! !"
              .format(resume_file))

    def state_cuda(self, msg):
        print("--", msg)
        print("allocated: %dM, max allocated: %dM, cached: %dM, max cached: %dM" % (
            torch.cuda.memory_allocated(self.device) / 1024 / 1024,
            torch.cuda.max_memory_allocated(self.device) / 1024 / 1024,
            torch.cuda.memory_cached(self.device) / 1024 / 1024,
            torch.cuda.max_memory_cached(self.device) / 1024 / 1024,
        ))
