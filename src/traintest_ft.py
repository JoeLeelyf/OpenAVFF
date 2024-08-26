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
    loss_meter = AverageMeter()
    
    best_epoch, best_mAP, best_acc = 0, -np.inf, -np.inf
    global_step, epoch = 0, 0
    start_time = time.time()
    exp_dir = args.save_dir
    
    if not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)
    
    model.to(device)
    
    # possible mlp layer name list, mlp layers are newly initialized layers in the finetuning stage (i.e., not pretrained) and should use a larger lr during finetuning
    mlp_list = [
        'a2v.mlp.linear.weight',
        'a2v.mlp.linear.bias',
        'v2a.mlp.linear.weight',
        'v2a.mlp.linear.bias',
        'mlp_vision.weight',
        'mlp_vision.bias',
        'mlp_audio.weight',
        'mlp_audio.bias',
        'mlp_head.fc1.weight',
        'mlp_head.fc1.bias',
        'mlp_head.fc2.weight',
        'mlp_head.fc2.bias'
    ]
    mlp_params = list(filter(lambda kv: kv[0] in mlp_list, model.module.named_parameters()))
    base_params = list(filter(lambda kv: kv[0] not in mlp_list, model.module.named_parameters()))
    mlp_params = [i[1] for i in mlp_params]
    base_params = [i[1] for i in base_params]
    
    trainables = [p for p in model.parameters() if p.requires_grad]
    print('Total parameter number is : {:.3f} million'.format(sum(p.numel() for p in model.parameters()) / 1e6))
    print('Total trainable parameter number is : {:.3f} million'.format(sum(p.numel() for p in trainables) / 1e6))
    optimizer = torch.optim.Adam([{'params': base_params, 'lr': args.lr}, {'params': mlp_params, 'lr': args.lr * args.head_lr}], weight_decay=5e-7, betas=(0.95, 0.999))
    base_lr = optimizer.param_groups[0]['lr']
    mlp_lr = optimizer.param_groups[1]['lr']
    lr_list = [args.lr, mlp_lr]
    print('base lr, mlp lr : ', base_lr, mlp_lr)
    
    print('Total newly initialized MLP parameter number is : {:.3f} million'.format(sum(p.numel() for p in mlp_params) / 1e6))
    print('Total pretrained backbone parameter number is : {:.3f} million'.format(sum(p.numel() for p in base_params) / 1e6))
    
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, list(range(args.lrscheduler_start, 1000, args.lrscheduler_step)),gamma=args.lrscheduler_decay)
    main_metrics = args.metrics
    
    if args.loss == 'BCE':
        loss_fn = nn.BCEWithLogitsLoss()
    elif args.loss == 'CE':
        loss_fn = nn.CrossEntropyLoss()
    args.loss_fn = loss_fn
    
    epoch += 1
    scaler = GradScaler()
    
    print("current #steps=%s, #epochs=%s" % (global_step, epoch))
    print("start training...")
    result = np.zeros([args.n_epochs, 4])  # for each epoch, 10 metrics to record
    model.train()
    
    while epoch < args.n_epochs + 1:
        begin_time = time.time()
        end_time = time.time()
        model.train()
        print('---------------')
        print(datetime.datetime.now())
        print("current #epochs=%s, #steps=%s" % (epoch, global_step))
        
        for i, (a_input, v_input, labels) in enumerate(train_loader):
            assert a_input.shape[0] == v_input.shape[0]
            B = a_input.shape[0]
            a_input = a_input.to(device, non_blocking=True)
            v_input = v_input.to(device, non_blocking=True)
            labels = labels.to(device)
            
            data_time.update(time.time() - end_time)
            per_sample_data_time.update((time.time() - end_time) / B)
            dnn_start_time = time.time()
            
            with autocast():
                output = model(a_input, v_input)
                loss = loss_fn(output, labels)
            
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            # loss_av is the main loss
            loss_meter.update(loss.item(), B)
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
                  'Train Loss {loss_meter.val:.4f}\t'.format(
                   epoch, i, len(train_loader), per_sample_time=per_sample_time, per_sample_data_time=per_sample_data_time,
                      per_sample_dnn_time=per_sample_dnn_time, loss_meter=loss_meter), flush=True)
                if np.isnan(loss_meter.avg):
                    print("training diverged...")
                    return

            end_time = time.time()
            global_step += 1
        
        print('start validation')
        stats, valid_loss = validate(model, test_loader, args)

        mAP = np.mean([stat['AP'] for stat in stats])
        mAUC = np.mean([stat['auc'] for stat in stats])
        acc = stats[0]['acc'] # this is just a trick, acc of each class entry is the same, which is the accuracy of all classes, not class-wise accuracy

        if main_metrics == 'mAP':
            print("mAP: {:.6f}".format(mAP))
        else:
            print("acc: {:.6f}".format(acc))
        print("AUC: {:.6f}".format(mAUC))
        print("d_prime: {:.6f}".format(d_prime(mAUC)))
        print("train_loss: {:.6f}".format(loss_meter.avg))
        print("valid_loss: {:.6f}".format(valid_loss))

        result[epoch-1, :] = [acc, mAP, mAUC, optimizer.param_groups[0]['lr']]
        np.savetxt(exp_dir + '/result.csv', result, delimiter=',')
        print('validation finished')
        
        if mAP > best_mAP:
            best_mAP = mAP
            if main_metrics == 'mAP':
                best_epoch = epoch

        if acc > best_acc:
            best_acc = acc
            if main_metrics == 'acc':
                best_epoch = epoch

        if best_epoch == epoch:
            torch.save(model.state_dict(), "%s/models/best_audio_model.pth" % (exp_dir))
            torch.save(optimizer.state_dict(), "%s/models/best_optim_state.pth" % (exp_dir))
        if args.save_model == True:
            torch.save(model.state_dict(), "%s/models/audio_model.%d.pth" % (exp_dir, epoch))
        
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            if main_metrics == 'mAP':
                scheduler.step(mAP)
            elif main_metrics == 'acc':
                scheduler.step(acc)
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
        loss_meter.reset()
        
def validate(model, val_loader, args, output_pred=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_time = AverageMeter()
    if not isinstance(model, nn.DataParallel):
        model = nn.DataParallel(model)
    model = model.to(device)
    model.eval()

    end = time.time()
    A_predictions, A_targets, A_loss = [], [], []
    with torch.no_grad():
        for i, (a_input, v_input, labels) in enumerate(val_loader):
            a_input = a_input.to(device)
            v_input = v_input.to(device)
            labels = labels.to(device)

            with autocast():
                audio_output = model(a_input, v_input)

            predictions = audio_output.to('cpu').detach()

            A_predictions.append(predictions)
            A_targets.append(labels)

            labels = labels.to(device)
            loss = args.loss_fn(audio_output, labels)
            A_loss.append(loss.to('cpu').detach())

            batch_time.update(time.time() - end)
            end = time.time()

        audio_output = torch.cat(A_predictions)
        target = torch.cat(A_targets)
        loss = np.mean(A_loss)

        stats = calculate_stats(audio_output.cpu(), target.cpu())

    if output_pred == False:
        return stats, loss
    else:
        # used for multi-frame evaluation (i.e., ensemble over frames), so return prediction and target
        return stats, audio_output, target