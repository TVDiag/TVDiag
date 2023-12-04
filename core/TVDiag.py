import os
import time

import dgl
import torch
import torch.nn.functional as F
# from torch.cuda import amp
from torch.utils.tensorboard import SummaryWriter
import shutil
from core.ita import cal_task_affinity
from core.loss.UnsupervisedContrastiveLoss import UspConLoss
from core.loss.SupervisedContrastiveLoss import SupConLoss
from core.loss.AutomaticWeightedLoss import AutomaticWeightedLoss
from core.model.MainModel import MainModel
from core.aug import *
from helper.eval import *

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class TVDiag(object):

    def __init__(self, args, logger, device):
        self.args = args
        self.device = device
        self.logger = logger

        self.lr = args.lr
        self.weight_decay = args.weight_decay
        self.epochs = args.epochs
        self.eval_period = args.eval_period
        self.log_every_n_steps = args.log_step
        self.tau = args.temperature
        log_dir = f"logs/{args.dataset}"
        os.makedirs(log_dir, exist_ok=True)

        # alias tensorboard='python3 -m tensorboard.main'
        # tensorboard --logdir=logs --host=192.168.31.201
        self.writer = SummaryWriter(log_dir)
        self.printParams()

    def printParams(self):
        self.logger.info(f"Training with: {self.device}")
        self.logger.info(f"Status of task-oriented learning: {self.args.TO}")
        self.logger.info(f"Status of cross-modal association: {self.args.CM}")
        self.logger.info(f"Status of dynamic_weight: {self.args.dynamic_weight}")
        if self.args.aug:
            self.logger.info(f"Augmente the data: {self.args.aug_method}")
        self.logger.info(f"lr: {self.args.lr}, weight_decay: {self.args.weight_decay}")

    def train(self, train_dl, test_data):

        model = MainModel(self.args).to(self.device)
        opt = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        
        awl = AutomaticWeightedLoss(3)
        supConLoss = SupConLoss(self.tau)
        uspConLoss = UspConLoss(self.tau)

        self.logger.info(model)
        self.logger.info(f"Start training for {self.epochs} epochs.")
        
        n_test=0
        best_avg, best_f1 = 0, 0
        best_data = {}

        # Overhead
        fit_times = []
        test_times = []

        # Inter-Task Affinity (RCL -> FTI, FTI -> RCL)
        Z_r2fs, Z_f2rs = [], []
        
        for epoch in range(self.epochs):
            n_iter = 0
            start_time = time.time()
            model.train()
            epoch_loss = 0

            for batch_graphs, batch_labels in train_dl:
                instance_labels = batch_labels[:, 0].to(self.device)
                type_labels = batch_labels[:, 1].to(self.device)

                opt.zero_grad()
                
                # aug
                if self.args.aug:
                    p = self.args.aug_percent
                    graph_list = dgl.unbatch(batch_graphs)
                    if self.args.aug_method == 'node_drop':
                        aug_graph_list = aug_drop_node_list(graph_list, instance_labels, p)
                    elif self.args.aug_method == 'random_walk':
                        aug_graph_list = aug_random_walk_list(graph_list, instance_labels, p)
                    batch_graphs = dgl.batch(aug_graph_list + graph_list)
                    instance_labels = torch.cat((instance_labels, instance_labels), dim=0)
                    type_labels = torch.cat((type_labels, type_labels), dim=0)

                (f_m, f_t, f_l), root_logit, type_logit = model(batch_graphs)

                # Task-oriented learning
                l_to, l_cm = 0, 0
                if self.args.TO:
                    l_to = supConLoss(f_m, instance_labels) + \
                        supConLoss(f_t, instance_labels) + \
                                   supConLoss(f_l, type_labels)

                # Cross-modal association
                if self.args.CM:
                    l_cm = uspConLoss(f_m, f_t) + \
                        uspConLoss(f_m, f_l)
                
                # Failure Diagnosis
                gamma = self.args.guide_weight
                # if epoch < 300:
                #     l_con = gamma * (l_to + l_cm) 
                # else:
                #     l_con = gamma* (l_to)
                
                l_con = gamma * (l_to + l_cm)

                l_rcl = F.cross_entropy(root_logit, instance_labels)
                l_fti = F.cross_entropy(type_logit, type_labels)
                if self.args.dynamic_weight:
                    total_loss = awl(l_rcl, l_fti, l_con)
                else:
                    total_loss = l_con + l_rcl + l_fti

                self.logger.debug("con_loss: {:.3f}, RCA_loss: {:.3f}, TC_loss: {:.3f}"
                      .format(l_con, l_rcl, l_fti))

                
                # Calculate Inter-Task Affinity
                if epoch == 0:
                    Z_r2f, Z_f2r = cal_task_affinity(model=model, 
                                    optimizer=opt, 
                                    batch_graphs=batch_graphs,
                                    instance_labels=instance_labels,
                                    type_labels=type_labels)
                    Z_r2fs.append(Z_r2f)
                    Z_f2rs.append(Z_f2r)

                total_loss.backward()
                opt.step()
                epoch_loss += total_loss.detach().item()
                n_iter += 1
                
            mean_epoch_loss = epoch_loss / n_iter
            end_time = time.time()
            time_per_epoch = (end_time - start_time)
            fit_times.append(time_per_epoch)
            self.logger.info("Epoch {} done. Loss: {:.3f}, Time per epoch: {:.3f}[s]"
                         .format(epoch, mean_epoch_loss, time_per_epoch))

            top1, top3, top5 = accuracy(root_logit, instance_labels, topk=(1, 3, 5))
            pre = precision(type_logit, type_labels, k=5)
            rec = recall(type_logit, type_labels, k=5)
            f1 = f1score(type_logit, type_labels, k=5)
            self.writer.add_scalar('loss', epoch_loss / n_iter, global_step=epoch)
            self.writer.add_scalar('train/top1', top1, global_step=epoch)
            self.writer.add_scalar('train/top3', top3, global_step=epoch)
            self.writer.add_scalar('train/top5', top5, global_step=epoch)
            self.writer.add_scalar('train/precision', pre, global_step=epoch)
            self.writer.add_scalar('train/recall', rec, global_step=epoch)
            self.writer.add_scalar('train/f1-score', f1, global_step=epoch)

            # evaluate
            if epoch % self.eval_period == 0:
                n_test += 1
                graphs, instance_labels, type_labels = [],[],[]
                
                for data in test_data:
                    graphs.append(data[0])
                    instance_labels.append(data[1][0])
                    type_labels.append(data[1][1])
                batch_graphs = dgl.batch(graphs)
                instance_labels = torch.tensor(instance_labels)
                type_labels = torch.tensor(type_labels)
                
                model.eval()
                start_time = time.time()
                with torch.no_grad():
                    _, root_logit, type_logit = model(batch_graphs)

                end_time = time.time()
                time_test = (end_time - start_time)
                test_times.append(time_test)
                top1, top2, top3, top4, top5 = accuracy(root_logit, instance_labels, topk=(1, 2, 3, 4, 5))
                avg_5 = np.mean([top1, top2, top3, top4, top5])
                pre = precision(type_logit, type_labels, k=5)
                rec = recall(type_logit, type_labels, k=5)
                f1 = f1score(type_logit, type_labels, k=5)

                self.logger.info("Validation Results - Epoch: {}".format(epoch))
                self.logger.info("[Root localization] top1: {:.3%}, top2: {:.3%}, top3: {:.3%}, top4: {:.3%}, top5: {:.3%}, avg@5: {:.3f}".format(top1, top2, top3, top4, top5, avg_5))
                self.logger.info("[Failure type classification] precision: {:.3%}, recall: {:.3%}, f1-score: {:.3%}".format(pre, rec, f1))

                self.writer.add_scalar('test/top1', top1, global_step=n_test)
                self.writer.add_scalar('test/top3', top3, global_step=n_test)
                self.writer.add_scalar('test/top5', top5, global_step=n_test)
                self.writer.add_scalar('test/precision', pre, global_step=n_test)
                self.writer.add_scalar('test/recall', rec, global_step=n_test)
                self.writer.add_scalar('test/f1-score', f1, global_step=n_test)
                self.logger.info("Time of test: {:.3f}[s]"
                         .format(time_test))
                
                if (avg_5 + f1) > (best_avg + best_f1):
                    best_avg = avg_5
                    best_f1 = f1
                    best_data['top1'] = top1
                    best_data['top2'] = top2
                    best_data['top3'] = top3
                    best_data['top4'] = top4
                    best_data['top5'] = top5
                    best_data['avg_5'] = avg_5
                    best_data['precision'] = pre
                    best_data['recall'] = rec
                    best_data['f1-score'] = f1
                    state = {
                        'epoch': self.epochs,
                        'model': model.state_dict(),
                        'opt': opt.state_dict(),
                    }
                    torch.save(state, os.path.join(self.writer.log_dir, 'model_best.pth.tar'))   

                 
        self.logger.info("Training has finished.")
        # calculate the training time for raw data
        self.logger.debug(f"The average training time per epoch is {np.mean(fit_times)/2}")
        self.logger.debug(f"The average predict time is {np.mean(test_times)}")
        self.logger.debug(f"The affinity of RCL -> FTI is {np.mean(Z_r2f)}")
        self.logger.debug(f"The affinity of FTI -> RCL is {np.mean(Z_f2r)}")
        for key in best_data.keys():
            self.logger.debug(f'{key}: {best_data[key]}')
            print(f'{key}: {best_data[key]}')
        self.logger.info(f"Model checkpoint and metadata has been saved at {self.writer.log_dir}.")
