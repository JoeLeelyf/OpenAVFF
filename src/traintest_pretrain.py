import sys
import os
import datetime
import time
from utilities import *
import torch
import torch.nn as nn
import numpy as np
from torch.cuda.amp import autocast, GradScaler

def train(model, train_loader, test_loader, args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_grad_enabled(True)
    
    batch_time, per_sample_time, data_time, per_sample_data_time, per_sample_dnn_time = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
    loss_av_meter, loss_a_meter, loss_v_meter, loss_c_meter = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
    
    best_epoch, best_loss = 0, np.inf
    global_step, epoch = 0, 0
    exp_dir = args.save_dir
    
    if not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)
    
    model.to(device)
    trainables = [p for p in model.parameters() if p.requires_grad]
    print('Total parameter number is : {:.3f} million'.format(sum(p.numel() for p in model.parameters()) / 1e6))
    print('Total trainable parameter number is : {:.3f} million'.format(sum(p.numel() for p in trainables) / 1e6))
    optimizer = torch.optim.Adam(trainables, lr=args.lr, weight_decay=5e-7, betas=(0.95, 0.999))
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, list(range(args.lrscheduler_start, 1000, args.lrscheduler_step)),gamma=args.lrscheduler_decay)
    
    epoch += 1
    scaler = GradScaler()
    
    print("current #steps=%s, #epochs=%s" % (global_step, epoch))
    print("start training...")
    result = np.zeros([args.n_epochs, 10])  # for each epoch, 10 metrics to record
    model.train()
    while epoch < args.n_epochs + 1:
        begin_time = time.time()
        end_time = time.time()
        model.train()
        print('---------------')
        print(datetime.datetime.now())
        print("current #epochs=%s, #steps=%s" % (epoch, global_step))
        
        for i, (a_input, v_input, _) in enumerate(train_loader):
            assert a_input.shape[0] == v_input.shape[0]
            B = a_input.shape[0]
            a_input = a_input.to(device, non_blocking=True)
            v_input = v_input.to(device, non_blocking=True)
            
            data_time.update(time.time() - end_time)
            per_sample_data_time.update((time.time() - end_time) / B)
            dnn_start_time = time.time()
            
            with autocast():
                loss, loss_c, c_acc, loss_mae_a, loss_mae_v, _, _ = model(a_input, v_input, mae_loss_weight=args.mae_loss_weight, contrast_loss_weight=args.contrast_loss_weight)
                # this is due to for torch.nn.DataParallel, the output loss of 4 gpus won't be automatically averaged, need to be done manually
                loss, loss_c, loss_mae_a, loss_mae_v, c_acc = loss.sum(), loss_c.sum(), loss_mae_a.sum(), loss_mae_v.sum(), c_acc.mean()
            
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            # loss_av is the main loss
            loss_av_meter.update(loss.item(), B)
            loss_a_meter.update(loss_mae_a.item(), B)
            loss_v_meter.update(loss_mae_v.item(), B)
            loss_c_meter.update(loss_c.item(), B)
            batch_time.update(time.time() - end_time)
            per_sample_time.update((time.time() - end_time)/a_input.shape[0])
            per_sample_dnn_time.update((time.time() - dnn_start_time)/a_input.shape[0])

            print_step = global_step % args.n_print_steps == 0
            early_print_step = epoch == 0 and global_step % (args.n_print_steps/10) == 0
            print_step = print_step or early_print_step

            if print_step and global_step != 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                  'Per Sample Total Time {per_sample_time.avg:.5f}\t'
                  'Per Sample Data Time {per_sample_data_time.avg:.5f}\t'
                  'Per Sample DNN Time {per_sample_dnn_time.avg:.5f}\t'
                  'Train Total Loss {loss_av_meter.val:.4f}\t'
                  'Train MAE Loss Audio {loss_a_meter.val:.4f}\t'
                  'Train MAE Loss Visual {loss_v_meter.val:.4f}\t'
                  'Train Contrastive Loss {loss_c_meter.val:.4f}\t'
                  'Train Contrastive Acc {c_acc:.3f}\t'.format(
                   epoch, i, len(train_loader), per_sample_time=per_sample_time, per_sample_data_time=per_sample_data_time,
                      per_sample_dnn_time=per_sample_dnn_time, loss_av_meter=loss_av_meter, loss_a_meter=loss_a_meter, loss_v_meter=loss_v_meter, loss_c_meter=loss_c_meter, c_acc=c_acc), flush=True)
                if np.isnan(loss_av_meter.avg):
                    print("training diverged...")
                    return

            end_time = time.time()
            global_step += 1
        
        print('start validation')
        eval_loss_av, eval_loss_c, eval_c_acc, eval_loss_mae_a, eval_loss_mae_v = validate(model, test_loader, args)

        print("Eval Audio MAE Loss: {:.6f}".format(eval_loss_mae_a))
        print("Eval Visual MAE Loss: {:.6f}".format(eval_loss_mae_v))
        print("Eval Contrastive Loss: {:.6f}".format(eval_loss_c))
        print("Eval Total Loss: {:.6f}".format(eval_loss_av))
        print("Eval Contrastive Accuracy: {:.6f}".format(eval_c_acc))

        print("Train Audio MAE Loss: {:.6f}".format(loss_a_meter.avg))
        print("Train Visual MAE Loss: {:.6f}".format(loss_v_meter.avg))
        print("Train Contrastive Loss: {:.6f}".format(loss_c_meter.avg))
        print("Train Total Loss: {:.6f}".format(loss_av_meter.avg))
        
        result[epoch-1, :] = [loss_a_meter.avg, loss_v_meter.avg, loss_c_meter.avg, loss_av_meter.avg, eval_loss_mae_a, eval_loss_mae_v, eval_loss_c, eval_loss_av, eval_c_acc, optimizer.param_groups[0]['lr']]
        
        if eval_loss_av < best_loss:
            best_loss = eval_loss_av
            best_epoch = epoch
            
        if best_epoch == epoch:
            torch.save(model.state_dict(), "%s/models/best_model.pth" % (exp_dir))
            torch.save(optimizer.state_dict(), "%s/models/best_optim_state.pth" % (exp_dir))
            
        if args.save_model == True:
            torch.save(model.state_dict(), "%s/models/model.%d.pth" % (exp_dir, epoch))
            
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(-eval_loss_av)
        else:
            scheduler.step()

        print('Epoch-{0} lr: {1}'.format(epoch, optimizer.param_groups[0]['lr']))
        
        finish_time = time.time()
        print('epoch {:d} training time: {:.3f}'.format(epoch, finish_time-begin_time))

        epoch += 1

        batch_time.reset()
        per_sample_time.reset()
        data_time.reset()
        per_sample_data_time.reset()
        per_sample_dnn_time.reset()
        loss_av_meter.reset()
        loss_a_meter.reset()
        loss_v_meter.reset()
        loss_c_meter.reset()
        
def validate(model, val_loader, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_time = AverageMeter()
    if not isinstance(model, nn.DataParallel):
        model = nn.DataParallel(model)
    model = model.to(device)
    model.eval()

    end = time.time()
    A_loss, A_loss_c, A_c_acc, A_loss_mae_a, A_loss_mae_v = [], [], [], [], []
    with torch.no_grad():
        for i, (a_input, v_input, _) in enumerate(val_loader):
            a_input = a_input.to(device)
            v_input = v_input.to(device)
            with autocast():
                loss, loss_c, c_acc, loss_mae_a, loss_mae_v, _, _ = model(a_input, v_input, mae_loss_weight=args.mae_loss_weight, contrast_loss_weight=args.contrast_loss_weight)
                # this is due to for torch.nn.DataParallel, the output loss of 4 gpus won't be automatically averaged, need to be done manually
                loss, loss_c, loss_mae_a, loss_mae_v, c_acc = loss.sum(), loss_c.sum(), loss_mae_a.sum(), loss_mae_v.sum(), c_acc.mean()
            A_loss.append(loss.to('cpu').detach())
            A_loss_mae_a.append(loss_mae_a.to('cpu').detach())
            A_loss_mae_v.append(loss_mae_v.to('cpu').detach())
            A_loss_c.append(loss_c.to('cpu').detach())
            A_c_acc.append(c_acc.to('cpu').detach())
            batch_time.update(time.time() - end)
            end = time.time()

        loss = np.mean(A_loss)
        loss_mae_a = np.mean(A_loss_mae_a)
        loss_mae_v = np.mean(A_loss_mae_v)
        loss_c = np.mean(A_loss_c)
        c_acc = np.mean(A_c_acc)

    return loss, loss_c, c_acc, loss_mae_a, loss_mae_v