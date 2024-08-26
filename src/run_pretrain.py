import argparse
import os
import torch
from torch.utils.data import DataLoader
from dataloader import VideoAudioDataset
from models.video_cav_mae import VideoCAVMAE
from traintest_pretrain import train

parser = argparse.ArgumentParser(description='Video CAV-MAE')
parser.add_argument('--data-train', type=str, help='path to train data csv')
parser.add_argument('--data-val', type=str, help='path to val data csv')
parser.add_argument('--target_length', default=1024, type=int, help='audio target length')
parser.add_argument("--dataset_mean", default=-5.081, type=float, help="the dataset audio spec mean, used for input normalization")
parser.add_argument("--dataset_std", default=4.4849, type=float, help="the dataset audio spec std, used for input normalization")
parser.add_argument("--noise", default=False, type=bool, help="add noise to the input")

parser.add_argument('--batch-size', default=32, type=int, help='batch size')
parser.add_argument('--num_workers', default=32, type=int, help='number of workers')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--n-epochs', default=25, type=int, help='number of epochs')
parser.add_argument('--save-dir', default='checkpoints', type=str, help='directory to save checkpoints')
parser.add_argument('--pretrain_path', default=None, type=str, help='path to pretrain model')
parser.add_argument("--contrast_loss_weight", type=float, default=0.01, help="weight for contrastive loss")
parser.add_argument("--mae_loss_weight", type=float, default=1.0, help="weight for mae loss")
parser.add_argument('--save_model', default=True)
parser.add_argument("--lrscheduler_start", default=10, type=int, help="when to start decay in finetuning")
parser.add_argument("--lrscheduler_step", default=5, type=int, help="the number of step to decrease the learning rate in finetuning")
parser.add_argument("--lrscheduler_decay", default=0.5, type=float, help="the learning rate decay ratio in finetuning")
parser.add_argument("--lr_adapt", help='if use adaptive learning rate')
parser.add_argument('--norm_pix_loss', help='if use norm_pix_loss', default=None)
parser.add_argument("--n_print_steps", default=100, type=int)

args = parser.parse_args()

im_res = 224
audio_conf = {'num_mel_bins': 128, 'target_length': args.target_length, 'freqm': 0, 'timem': 0, 'mode':'train', 
            'mean':args.dataset_mean, 'std':args.dataset_std, 'noise':args.noise, 'im_res': im_res}
val_audio_conf = {'num_mel_bins': 128, 'target_length': args.target_length, 'freqm': 0, 'timem': 0, 'mixup': 0,'mode':'eval', 
            'mean': args.dataset_mean, 'std': args.dataset_std, 'noise': False, 'im_res': im_res}

print('current mae loss {:.3f}, and contrastive loss {:.3f}'.format(args.mae_loss_weight, args.contrast_loss_weight))

# Construct dataloader
train_loader = DataLoader(VideoAudioDataset(args.data_train, audio_conf, stage=1), batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
val_loader = DataLoader(VideoAudioDataset(args.data_val, val_audio_conf, stage=1), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=True)

# Construct model
cavmae = VideoCAVMAE()

# init model
if args.pretrain_path is not None:
    mdl_weight = torch.load(args.pretrain_path, map_location='cpu')
    if not isinstance(cavmae, torch.nn.DataParallel):
        cavmae = torch.nn.DataParallel(cavmae)
    miss, unexpected = cavmae.load_state_dict(mdl_weight, strict=False)
    print('Missing: ', miss)
    print('Unexpected:', unexpected)
    print('now load pretrain model from {:s}, missing keys: {:d}, unexpected keys: {:d}'.format(args.pretrain_path, len(miss), len(unexpected)))
    
print("\n Creating experiment directory: %s"%args.save_dir)
if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

# Train model
print("Now start training for %d epochs"%args.n_epochs)
train(cavmae, train_loader, val_loader, args)
