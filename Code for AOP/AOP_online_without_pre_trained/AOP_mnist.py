import sys, time
import numpy as np
import torch
import torch.nn as nn

dtype = torch.cuda.FloatTensor  # run on GPU
import utils
''' 
from CSL import tao as TL
from CSL import classifier as C
from CSL.utils import normalize
from CSL.contrastive_learning import get_similarity_matrix, NT_xent
import torch.optim.lr_scheduler as lr_scheduler
from CSL.shedular import GradualWarmupScheduler
'''
import torch

########################################################################################################################

class Appr(object):

    def __init__(self, model, nepochs=0, sbatch=10, lr=0, clipgrad=10, args=None):
        self.model = model
        self.nepochs = nepochs
        self.sbatch = sbatch
        self.lr = lr
        self.clipgrad = clipgrad

        self.ce = torch.nn.CrossEntropyLoss()
        self.optimizer = self._get_optimizer()
        with torch.no_grad():
            #self.Pc1 = torch.autograd.Variable(torch.eye(3 * 2 * 2).type(dtype))
            #self.Pc2 = torch.autograd.Variable(torch.eye(64 * 2 * 2).type(dtype))
           # self.Pc3 = torch.autograd.Variable(torch.eye(128 * 2 * 2).type(dtype))
            self.P1 = torch.autograd.Variable(torch.eye(28*28).type(dtype))
            self.P2 = torch.autograd.Variable(torch.eye(1000).type(dtype))
            self.P3 = torch.autograd.Variable(torch.eye(1000).type(dtype))

        self.test_max = 0

        return

    def _get_optimizer(self, t=0, lr=None):
        # if lr is None:
        #     lr = self.lr
        lr = self.lr+0.000*t
        lr_owm = self.lr+0.000*t
        fc1_params = list(map(id, self.model.fc1.parameters()))
        fc2_params = list(map(id, self.model.fc2.parameters()))
        fc3_params = list(map(id, self.model.fc3.parameters()))
        base_params = filter(lambda p: id(p) not in fc1_params + fc2_params+fc3_params,
                             self.model.parameters())
        optimizer = torch.optim.SGD([{'params': base_params},
                                     {'params': self.model.fc1.parameters(), 'lr': lr_owm,'momentum':0.9},
                                     {'params': self.model.fc2.parameters(), 'lr': lr_owm,'momentum':0.9},
                                     {'params': self.model.fc3.parameters(), 'lr': lr_owm, 'momentum': 0.9},
                                     ], lr=lr)

        return optimizer

    def train(self, t, xtrain, ytrain, xvalid, yvalid, data,lr=0.03):
        best_loss = np.inf
        best_acc = 0
        best_model = utils.get_model(self.model)
        lr = self.lr
        # patience = self.lr_patience
        self.optimizer = self._get_optimizer(t, lr)
        #self.old_model.eval()
        #scheduler = lr_scheduler.CosineAnnealingLR(self.optimizer, self.nepochs)
        #scheduler_warmup = GradualWarmupScheduler(self.optimizer, multiplier=10.0, total_epoch=10,
                                               #  after_scheduler=scheduler)
        #scheduler=scheduler_warmup
        nepochs = self.nepochs
        TASK_NUM=4
        # Loop epochs
        try:
            for e in range(nepochs):
                # Tra in

               # print(ytrain)
                l_sim=0
                l_shift = 0
                #if self.uu==2500:
                 #   continue
                self.train_epoch(xtrain, ytrain, cur_epoch=e, nepoch=nepochs)
                #self.old_model= copy.deepcopy(self.model)
                #self.old_model = self.old_model.eval()
                print(e)
                #if (e+1) % 5  !=0:
                 #   if t<TASK_NUM:
                  #      continue
                #print("e",e)
                train_loss, train_acc = self.eval(xtrain, ytrain)
                #print("t",train_loss)
                print('| [{:d}/10], Epoch {:d}/{:d}, | Train: loss={:.3f}, acc={:2.2f}% |Train: sim_loss={:.3f}|Train: shift_loss={:.3f} |'.format(t + 1, e + 1,
                                                                                   nepochs, train_loss,100 * train_acc,l_sim,l_shift),
                      end='')
                # # Valid
                valid_loss, valid_acc= self.eval(xvalid, yvalid)
                print(' Valid: loss={:.3f}, acc={:5.2f}% |'.format(valid_loss, 100 * valid_acc), end='')
                print()

                xtest = data[TASK_NUM+1]['test']['x'].cuda()
                ytest = data[TASK_NUM+1]['test']['y'].cuda()
                #print(ytest)

                _, test_acc = self.eval(xtest, ytest)

                # # Adapt lr
                # if valid_loss < best_loss:
                #     best_loss = min(best_loss,valid_loss)

                # if valid_acc > best_acc:
                #     best_acc = max(best_acc, valid_acc)
                if test_acc >= self.test_max:
                    self.test_max = max(self.test_max, test_acc)
                    best_model = utils.get_model(self.model)

                print('>>> Test on All Task:->>> Max_acc : {:2.2f}%  Curr_acc : {:2.2f}%<<<'.format(100 * self.test_max,
                                                                                                    100 * test_acc))

        except KeyboardInterrupt:
            print()

        # Restore best validation model
        utils.set_model_(self.model, best_model)
      #  print("THIS task",self.u,self
       #       .uu)
        #a=self.u[0]/self.uu
        #b=self.u[1]/self.uu
        #ab=[a,b]
        #self.U.append(ab)
        #print("U",self.U)
       # utils.set_model_(self.old_model,best_model)
        return

    def train_epoch(self, x, y, cur_epoch=0, nepoch=0,scheduler=None):
        self.model.train()

        r_len = np.arange(x.size(0))
        np.random.shuffle(r_len)
        r_len = torch.LongTensor(r_len).cuda()

        # Loop batches
        for i_batch in range(0, len(r_len), self.sbatch):
          #  if i_batch >= 120*self.sbatch:
           #     continue
            b = r_len[i_batch:min(i_batch + self.sbatch, len(r_len))]
           # if len(b)//12 != 0:
            b1 = b[:len(b)]
            with torch.no_grad():
                images = torch.autograd.Variable(x[b])
                targets = torch.autograd.Variable(y[b])
            #hflip = TL.HorizontalFlipLayer().cuda()
            #rotation = TL.Rotation().cuda()
            #batch_size = images.data.shape[0]
            # print("i",images.shape)
            #positive_data2 = images.data
            # positive_data1=positive_data.reshape(-1,)
            # images = images.to(device)
            #labels = targets.data
            #images1, images2 = hflip(positive_data2.repeat(2, 1, 1, 1)).chunk(2)
            #images1 = torch.cat(
             #   [rotation(images1, k) for k in range(1)])  # P.K是4，这里对image1生成4个旋转混原数据数据集
            #images2 = torch.cat([rotation(images2, k) for k in range(1)])  # 也就是4个数据迁移
            # shift_labels = torch.cat([torch.ones_like(labels) * k for k in range(self.K_shift)],
            #                        0)  # 生成batch大小的只有0，1，2，3的label，并在一起B -> 4B
            # print("A",images2.shape)
            # shift_labels = shift_labels.repeat(2).long().to(self.opt.gpu)
            #images_pair = torch.cat([images1, images2], dim=0).cuda()  # 8B
            # 产生pair，估计是堆在一起0维
            # print("pair", images_pair.shape)  # 256
            #with torch.no_grad():
             #   resize_scale = (0.08, 1.0)  # resize scaling factor,default [0.08,1]
                # if P.resize_fix: # if resize_fix is True, use same scale
                #    resize_scale = (P.resize_factor, P.resize_factor)

                # Align augmentation
              #  color_jitter = TL.ColorJitterLayer(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p=0.8).cuda()
               # color_gray = TL.RandomColorGrayLayer(p=0.2).cuda()
                #resize_crop = TL.RandomResizedCropLayer(scale=resize_scale, size=[32, 32, 3]).cuda()
                #simclr_aug = transform = torch.nn.Sequential(
                 #   color_jitter,  # 这个不会变换大小，但是会变化通道值，新旧混杂
                  #  color_gray,  # 这个也不会，混搭
                   # resize_crop, )
                # shift_labels = shift_labels.repeat(2).long().to(self.opt.gpu)
                #images_pair = simclr_aug(images_pair)  # transform
            # images_pair = images_pair.reshape(images_pair.shape[0], 3, -1)
            #output = self.model.forward(images_pair, is_simclr=True)
            # print(outputs_aux['simclr'].shape,outputs_aux['shift'].shape,outputs_aux['penultimate'].shape)
            # [256,128],[256,4],[256,512]
            # print(batch_size)

            #simclr = normalize(output)
            # print("s",simclr.shape)# normalize 按范数逐一正则化.sim的分母部分完成
            #sim_matrix = get_similarity_matrix(simclr)
            # print(sim_matrix.shape)# 得到的是【256，256】，sim乘积部分
            #loss_sim1 = NT_xent(sim_matrix, 0.5)  # p.sim是1？为对角矩阵

            #  images_cl = torch.autograd.Variable(x[b1])
               # targets_cl = torch.autograd.Variable(y[b1])
            #hflip = TL.HorizontalFlipLayer().to(self.opt.gpu)
            #cutperm = TL.CutPerm().to(self.opt.gpu)
            #batch_size = images_cl.data.shape[0]
            #print("i",images.shape)
            '''  
            positive_data2 = images_cl.data
            # positive_data1=positive_data.reshape(-1,)
            # images = images.to(device)
            labels = targets_cl.data
            images1,images2 = hflip(positive_data2.repeat(2, 1, 1, 1)).chunk(2)
            images1 = torch.cat([self.shift_trans(images1, k) for k in range(self.K_shift)])
           # images12 = torch.cat([cutperm(images1, k) for k in range(self.K_shift)])
            #images1 = torch.cat([images11,images12],dim=0)# P.K是4，这里对image1生成4个旋转混原数据数据集
            images2 = torch.cat([self.shift_trans(images2, k) for k in range(self.K_shift)])
            #images22 = torch.cat([cutperm(images2, k) for k in range(self.K_shift)])
            #images2 = torch.cat([images21, images22],dim=0)  # P.K是4，这里对image1生成4个旋转混原数据数据集
          #  images2 = torch.cat([self.shift_trans(images2, k) for k in range(self.K_shift)])  # 也就是4个数据迁移
            shift_labels = torch.cat([torch.ones_like(labels) * k for k in range(self.K_shift)],
                                     0)  # 生成batch大小的只有0，1，2，3的label，并在一起B -> 4B
            #print("A",images2.shape)
           # shift_labels = shift_labels.repeat(2).long().to(self.opt.gpu)
            images_pair = torch.cat([images1, images2], dim=0)  # 8B
            # 产生pair，估计是堆在一起0维
            #print("pair", images_pair.shape)  # 256
            with torch.no_grad():
                shift_labels = shift_labels.repeat(2).long().to(self.opt.gpu)
                images_pair = self.opt.simclr_aug(images_pair)  # transform
            # images_pair = images_pair.reshape(images_pair.shape[0], 3, -1)
            outputs_aux = self.model.forward(images_pair, simclr=True, shift=True)
            # print(outputs_aux['simclr'].shape,outputs_aux['shift'].shape,outputs_aux['penultimate'].shape)
            # [256,128],[256,4],[256,512]
            #print(batch_size)

            simclr = normalize(outputs_aux['simclr'])
            #print("s",simclr.shape)# normalize 按范数逐一正则化.sim的分母部分完成
            sim_matrix = get_similarity_matrix(simclr)
            #print(sim_matrix.shape)# 得到的是【256，256】，sim乘积部分
            loss_sim = NT_xent(sim_matrix, self.opt.temperature)  # p.sim是1？为对角矩阵
            criterion2 = nn.CrossEntropyLoss().to(self.opt.gpu)
            loss_shift = criterion2(outputs_aux['shift'], shift_labels)
            #print("A",shift_labels)

            # Forward
            '''
            output, h_list = self.model.forward(images)

           # print("目标",targets)
            loss1 = self.ce(output, targets)#+0*loss_sim1
            #print("损失",loss1)
            loss_dist3=0
            loss_dist2 = 0
            #print("y",y[0])
            ''' 
            #output = self.model(images_pair, path, -1, is_simclr=True)
            outputs_aux_old = self.old_model.forward(images_pair, simclr=True, shift=True)
            output_old =outputs_aux_old['simclr']
           # output = outputs_aux['simclr']
            simclr = normalize(outputs_aux['simclr'])
            # print("s",simclr.shape)# normalize 按范数逐一正则化.sim的分母部分完成
            sim_matrix = get_similarity_matrix(simclr)
            simclr_old = normalize(outputs_aux_old['simclr'])
            # print("s",simclr.shape)# normalize 按范数逐一正则化.sim的分母部分完成
            sim_matrix_old = get_similarity_matrix(simclr_old)
           # print("sim",sim_matrix.shape,sim_matrix_old.shape)
                #= self.old_model(images_pair, path, -1,is_simclr=True)  # 老模型也这样跑一次

            #t_one_hot = targets_one_hot.clone()
            #t_one_hot[:, 0:self.args.class_per_task * self.args.sess] = outputs_old[:,
             #                                                           0:self.args.class_per_task * self.args.sess]  # 老模型那一代的预测

            #if (self.args.sess in range(1 + self.args.jump)):  # 0，1，2
             #   cx = 1
            #else:
            cx = 1
             #   cx = self.args.rigidness_coff * (self.args.sess - self.args.jump)  # 2.5乘以一个差距，jump是1，越后面越大
            #  a=F.log_softmax(outputs / 2.0, dim=1)
            # b=F.softmax(t_one_hot / 2.0, dim=1)
            # print("奇怪的10",a.shape,b.shape)
           # print("345",output.shape,output_old.shape)
            #print("???",cx,torch.sum(F.kl_div(output / 2.0, output_old / 2.0,
             ##  F.kl_div(output / 2.0, output_old / 2.0,
               #          reduce=False)),F.kl_div(F.softmax(output / 2.0,dim=1), F.softmax(output_old / 2.0,dim=1),
                #         reduce=False))
            #print("567", F.softmax(output / 2.0,dim=1).shape, F.softmax(output_old / 2.0,dim=1).shape )
            #loss_dist2 = (cx / (64 * 1.0)) * torch.sum(
             #   F.kl_div(output / 0.5, output_old / 2.0,
             #            reduce=False))
          #  loss_dist3 = (cx / (64 * 1.0)) * torch.sum(
           #     F.kl_div(F.softmax(output / 0.5,dim=1), F.softmax(output_old / 0.5,dim=1),
            #             reduce=False))
            loss_dist4 = torch.mean(torch.sum(F.softmax(sim_matrix / 0.5, dim=1) * F.softmax(sim_matrix_old / 0.5, dim=1),dim=1))#)torch.sum(
           #     F.kl_div(F.softmax(sim_matrix / 0.5, dim=1), F.softmax(sim_matrix_old / 0.5, dim=1),
            #             reduce=False))
           # print("LOOOOOs,损失",loss_dist2,-loss_dist3,loss_dist4)
            #print("a")
           # if y[0]>=10:
           '''
         ##      if n == 'fc1.weight':
           #         weight1=w.detach()
                    #pro_weight(self.P1, h_list[0], w, alpha=alpha_array[0], cnn=False)

            #    if n == 'fc2.weight':
             #       weight2=w.detach()#pro_weight(self.P2, h_list[1], w, alpha=alpha_array[1], cnn=False)
            #lambda_loss = 1e-3
            loss= 1.0*loss1#+lambda_loss*(torch.norm(weight1)+torch.norm(weight2))#+0.0*loss_shift+0.0*loss_sim+0*loss_dist4
           # print("loss",loss)
            #print("los",loss_sim1)
            #else:
             #   loss=0*loss1
            #print("sim", loss_ sim)
            # Backward
            self.optimizer.zero_grad()
            #for n, w in self.model.named_parameters():
             #   if n == 'shift_cls_layer.weight':

              #      print(w.data)
            loss.backward()
            #cur_epoch1=cu r_ep och%5
            lamda = i_batch / len(r_len) / nepoch + cur_epoch / nepoch

            alpha_array = [ 0.9 * 0.001**lamda,1.0*0.1**lamda,0.6 ]
            #h_list=h_list.detach()

            mean_h_1 = torch.mean(h_list[0])#torch.matmul(h_list[0].t(),h_list[0]))
            mean_h_2 = torch.mean(h_list[1])#torch.matmul(h_list[1].t(),h_list[1]))#h_list[1])torch.matmul(h_list[1].t(),h_list[1]))
            mean_h_3 = torch.mean(h_list[2])#torch.matmul(h_list[2].t(), h_list[2]))#h_list[2])#torch.matmul(h_list[2].t(), h_list[2]))
            #mean_h_3 = 0#torch.mean(h_list[2])
            std_v_1 = torch.var(h_list[0])
            std_v_2 = torch.var(h_list[1])
            std_v_3 = torch.var(h_list[2])

            alpha_array[0] = (mean_h_1.data.detach()**2+std_v_1.detach())**lamda
            alpha_array[1] = (mean_h_2.data.detach()**2+std_v_2.detach())**lamda
            alpha_array[2] = (mean_h_3.data.detach() ** 2 + std_v_3.detach())**lamda


            def pro_weight(p, x, w, alpha=1.0, cnn=True, stride=1):
                if cnn:
                    _, _, H, W = x.shape
                    F, _, HH, WW = w.shape
                    S = stride  # stride
                    Ho = int(1 + (H - HH) / S)
                    Wo = int(1 + (W - WW) / S)
                    for i in range(Ho):
                        for j in range(Wo):
                            # N*C*HH*WW, C*HH*WW = N*C*HH*WW, sum -> N*1
                            r = x[:, :, i * S: i * S + HH, j * S: j * S + WW].contiguous().view(1, -1).data
                            # r = r[:, range(r.shape[1] - 1, -1, -1)]
                            k = torch.mm(p.data, torch.t(r))
                            p.sub_(torch.mm(k, torch.t(k)) / (alpha + torch.mm(r, k)))
                    w.grad.data = torch.mm(w.grad.data.view(F, -1), torch.t(p.data)).view_as(w)
                else:
                    r = x.data
                    k = torch.mm(p.data, torch.t(r))
                    p.sub_(torch.mm(k, torch.t(k)) / (alpha + torch.mm(r, k)))
                    w.grad.data = torch.mm(w.grad.data, torch.t(p.data))
            # Compensate embedding gradients
            for n, w in self.model.named_parameters():
                if n == 'fc1.weight':
                    pro_weight(self.P1, h_list[0], w, alpha=alpha_array[0], cnn=False)

                if n == 'fc2.weight':
                    pro_weight(self.P2, h_list[1], w, alpha=alpha_array[1], cnn=False)
                if n == 'fc3.weight':
                    pro_weight(self.P3, h_list[2], w, alpha=alpha_array[2], cnn=False)

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clipgrad)
            self.optimizer.step()
        print("alpha", alpha_array)

        return #loss_sim,loss_shift

    def eval(self, x, y):
        total_loss = 0
        total_acc = 0
        total_num = 0
        a=0
        c=0
        self.model.eval()

        r = np.arange(x.size(0))
        r = torch.LongTensor(r).cuda()

        # Loop batches
        for i in range(0, len(r), self.sbatch):
            b = r[i:min(i + self.sbatch, len(r))]
            with torch.no_grad():
                images = torch.autograd.Variable(x[b])
                targets = torch.autograd.Variable(y[b])

            # print("pair", images_pair.shape)  # 256
        #    print("target",targets)
            # Forward
            output, _ = self.model.forward(images)
            loss = self.ce(output, targets)
            #print(output.shape,output.max(1))
            _, pred = output.max(1)
           # print("pred",pred)
            hits = (pred % 10 == targets).float()

            # Log
            total_loss += loss.data.cpu().numpy().item() * len(b)
            total_acc += hits.sum().data.cpu().numpy().item()
            total_num += len(b)
            #a+=loss_shift*len(b)
            #c+=loss_sim*len(b)

        return total_loss / total_num, total_acc / total_num
