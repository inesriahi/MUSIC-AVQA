from __future__ import print_function
import ast
import json
import os
import sys
sys.path.append('.')
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
import time
from datetime import timedelta

import dataloader_avst as dataloader_avst
from net_avst import AVQA_Fusion_Net
import yaml

TIMESTAMP = "{0:%Y-%m-%d-%H-%M-%S/}".format(datetime.now()) 
writer = SummaryWriter('runs/net_avst/'+TIMESTAMP)
dist.init_process_group("nccl")
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)
global_rank = int(os.environ["RANK"])

print("\n--------------- Audio-Visual Spatial-Temporal Model --------------- \n")

def batch_organize(out_match_posi,out_match_nega):
    # audio B 512
    # posi B 512
    # nega B 512

    # print("audio data: ", audio_data.shape)
    out_match = torch.zeros(out_match_posi.shape[0] * 2, out_match_posi.shape[1])
    batch_labels = torch.zeros(out_match_posi.shape[0] * 2)
    for i in range(out_match_posi.shape[0]):
        out_match[i * 2, :] = out_match_posi[i, :]
        out_match[i * 2 + 1, :] = out_match_nega[i, :]
        batch_labels[i * 2] = 1
        batch_labels[i * 2 + 1] = 0
    
    return out_match, batch_labels


def train(config, model, train_loader, optimizer, criterion, epoch):
    model.train()
    total_qa = 0
    correct_qa = 0
    for batch_idx, sample in enumerate(train_loader):
        audio,visual_posi,visual_nega, target, question = sample['audio'].to('cuda'), sample['visual_posi'].to('cuda'),sample['visual_nega'].to('cuda'), sample['label'].to('cuda'), sample['question'].to('cuda')

        optimizer.zero_grad()
        # model = model.cuda()
        # model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
        out_qa, out_match_posi,out_match_nega = model(audio, visual_posi,visual_nega, question)  
        out_match,match_label=batch_organize(out_match_posi,out_match_nega)  
        out_match,match_label = out_match.type(torch.FloatTensor).cuda(), match_label.type(torch.LongTensor).cuda()
    
        # output.clamp_(min=1e-7, max=1 - 1e-7)
        loss_match=criterion(out_match,match_label)
        loss_qa = criterion(out_qa, target)
        loss = loss_qa + 0.5*loss_match

        writer.add_scalar('run/match',loss_match.item(), epoch * len(train_loader) + batch_idx)
        writer.add_scalar('run/qa_test',loss_qa.item(), epoch * len(train_loader) + batch_idx)
        writer.add_scalar('run/both',loss.item(), epoch * len(train_loader) + batch_idx)
        loss.backward()
        optimizer.step()
        if batch_idx % config["training"]["log_interval"] == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(audio), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


def eval(model, val_loader,epoch):
    model.eval()
    total_qa = 0
    total_match=0
    correct_qa = 0
    correct_match=0
    with torch.no_grad():
        for batch_idx, sample in enumerate(val_loader):
            audio,visual_posi,visual_nega, target, question = sample['audio'].to('cuda'), sample['visual_posi'].to('cuda'),sample['visual_nega'].to('cuda'), sample['label'].to('cuda'), sample['question'].to('cuda')

            preds_qa, out_match_posi,out_match_nega = model(audio, visual_posi,visual_nega, question)

            _, predicted = torch.max(preds_qa.data, 1)
            total_qa += preds_qa.size(0)
            correct_qa += (predicted == target).sum().item()

    print('Accuracy qa: %.2f %%' % (100 * correct_qa / total_qa))
    writer.add_scalar('metric/acc_qa',100 * correct_qa / total_qa, epoch)

    return 100 * correct_qa / total_qa


def test(model, val_loader):
    model.eval()
    total = 0
    correct = 0
    samples = json.load(open('./data/json/avqa-test.json', 'r'))
    A_count = []
    A_cmp = []
    V_count = []
    V_loc = []
    AV_ext = []
    AV_count = []
    AV_loc = []
    AV_cmp = []
    AV_temp = []
    with torch.no_grad():
        for batch_idx, sample in enumerate(val_loader):
            audio,visual_posi,visual_nega, target, question = sample['audio'].to('cuda'), sample['visual_posi'].to('cuda'),sample['visual_nega'].to('cuda'), sample['label'].to('cuda'), sample['question'].to('cuda')

            preds_qa,out_match_posi,out_match_nega = model(audio, visual_posi,visual_nega, question)
            preds = preds_qa
            _, predicted = torch.max(preds.data, 1)

            total += preds.size(0)
            correct += (predicted == target).sum().item()

            x = samples[batch_idx]
            type =ast.literal_eval(x['type'])
            if type[0] == 'Audio':
                if type[1] == 'Counting':
                    A_count.append((predicted == target).sum().item())
                elif type[1] == 'Comparative':
                    A_cmp.append((predicted == target).sum().item())
            elif type[0] == 'Visual':
                if type[1] == 'Counting':
                    V_count.append((predicted == target).sum().item())
                elif type[1] == 'Location':
                    V_loc.append((predicted == target).sum().item())
            elif type[0] == 'Audio-Visual':
                if type[1] == 'Existential':
                    AV_ext.append((predicted == target).sum().item())
                elif type[1] == 'Counting':
                    AV_count.append((predicted == target).sum().item())
                elif type[1] == 'Location':
                    AV_loc.append((predicted == target).sum().item())
                elif type[1] == 'Comparative':
                    AV_cmp.append((predicted == target).sum().item())
                elif type[1] == 'Temporal':
                    AV_temp.append((predicted == target).sum().item())

    print('Audio Counting Accuracy: %.2f %%' % (
            100 * sum(A_count)/len(A_count)))
    print('Audio Cmp Accuracy: %.2f %%' % (
            100 * sum(A_cmp) / len(A_cmp)))
    print('Audio Accuracy: %.2f %%' % (
            100 * (sum(A_count) + sum(A_cmp)) / (len(A_count) + len(A_cmp))))
    print('Visual Counting Accuracy: %.2f %%' % (
            100 * sum(V_count) / len(V_count)))
    print('Visual Loc Accuracy: %.2f %%' % (
            100 * sum(V_loc) / len(V_loc)))
    print('Visual Accuracy: %.2f %%' % (
            100 * (sum(V_count) + sum(V_loc)) / (len(V_count) + len(V_loc))))
    print('AV Ext Accuracy: %.2f %%' % (
            100 * sum(AV_ext) / len(AV_ext)))
    print('AV counting Accuracy: %.2f %%' % (
            100 * sum(AV_count) / len(AV_count)))
    print('AV Loc Accuracy: %.2f %%' % (
            100 * sum(AV_loc) / len(AV_loc)))
    print('AV Cmp Accuracy: %.2f %%' % (
            100 * sum(AV_cmp) / len(AV_cmp)))
    print('AV Temporal Accuracy: %.2f %%' % (
            100 * sum(AV_temp) / len(AV_temp)))

    print('AV Accuracy: %.2f %%' % (
            100 * (sum(AV_count) + sum(AV_loc)+sum(AV_ext)+sum(AV_temp)
                   +sum(AV_cmp)) / (len(AV_count) + len(AV_loc)+len(AV_ext)+len(AV_temp)+len(AV_cmp))))

    print('Overall Accuracy: %.2f %%' % (
            100 * correct / total))

    return 100 * correct / total

def main():
    with open('/scratch/project_462000189/ines/MUSIC-AVQA/net_grd_avst/config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    print(yaml.dump(config, default_flow_style=False, sort_keys=False))
    # os.environ['CUDA_VISIBLE_DEVICES'] = config["global"]["gpu"]

    torch.manual_seed(config["global"]["seed"])

    if config["model"]["name"] == 'AVQA_Fusion_Net':
        model = AVQA_Fusion_Net(config)
        model = model.cuda()
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    else:
        raise ('not recognized')

    if config["model"]["mode"] == 'train':
        train_dataset = dataloader_avst.AVQA_dataset(label=config["directories"]["label_train"], audio_dir=config["directories"]["audio_dir"], video_res14x14_dir=config["directories"]["video_res14x14_dir"],
                                    transform=transforms.Compose([dataloader_avst.ToTensor()]), mode_flag='train')
        train_sampler = DistributedSampler(train_dataset)
        train_loader = DataLoader(train_dataset, batch_size=config["training"]["batch_size"], num_workers=config["training"]["num_workers"], pin_memory=True, sampler=train_sampler)
        val_dataset = dataloader_avst.AVQA_dataset(label=config["directories"]["label_val"], audio_dir=config["directories"]["audio_dir"], video_res14x14_dir=config["directories"]["video_res14x14_dir"],
                                    transform=transforms.Compose([dataloader_avst.ToTensor()]), mode_flag='val')
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=config["training"]["num_workers"], pin_memory=True)


        # ===================================== load pretrained model ===============================================
        ####### concat model
        pretrained_file = "grounding_gen/models_grounding_gen/main_grounding_gen_best.pt"
        checkpoint = torch.load(pretrained_file)
        print("\n-------------- loading pretrained models --------------")
        model_dict = model.state_dict()
        tmp = ['module.fc_a1.weight', 'module.fc_a1.bias','module.fc_a2.weight','module.fc_a2.bias','module.fc_gl.weight','module.fc_gl.bias','module.fc1.weight', 'module.fc1.bias','module.fc2.weight', 'module.fc2.bias','module.fc3.weight', 'module.fc3.bias','module.fc4.weight', 'module.fc4.bias']
        tmp2 = ['module.fc_a1.weight', 'module.fc_a1.bias','module.fc_a2.weight','module.fc_a2.bias']
        pretrained_dict1 = {k: v for k, v in checkpoint.items() if k in tmp}
        pretrained_dict2 = {str(k).split('.')[0]+'.'+str(k).split('.')[1]+'_pure.'+str(k).split('.')[-1]: v for k, v in checkpoint.items() if k in tmp2}

        model_dict.update(pretrained_dict1) #利用预训练模型的参数，更新模型
        model_dict.update(pretrained_dict2) #利用预训练模型的参数，更新模型
        model.load_state_dict(model_dict)

        print("\n-------------- load pretrained models --------------")

        # ===================================== load pretrained model ===============================================

        optimizer = optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config["training"]["step_lr"]["step_size"], gamma=config["training"]["step_lr"]["gamma"])
        criterion = nn.CrossEntropyLoss()
        best_F = 0
        total_start_time = time.time()  # Record the start time of the training

        for epoch in range(1, config["training"]["epochs"] + 1):
            epoch_start_time = time.time()  # Record the start time of the epoch
            
            # Your training and evaluation code here
            train(config, model, train_loader, optimizer, criterion, epoch=epoch)
            scheduler.step(epoch)
            F = eval(model, val_loader, epoch)
            
            # Calculate and print the time each epoch takes
            epoch_end_time = time.time()
            epoch_duration = timedelta(seconds=(epoch_end_time - epoch_start_time))
            print(f"Epoch {epoch} completed in {epoch_duration}")
            
            if F >= best_F:
                best_F = F
                torch.save(model.state_dict(), config["directories"]["model_save_dir"] + config["model"]["checkpoint"] + ".pt")

        total_end_time = time.time()
        total_duration = timedelta(seconds=(total_end_time - total_start_time))
        print(f"Total training time: {total_duration}")


    else:
        test_dataset = dataloader_avst.AVQA_dataset(label=config["directories"]["label_test"], audio_dir=config["directories"]["audio_dir"], video_res14x14_dir=config["directories"]["video_res14x14_dir"],
                                   transform=transforms.Compose([dataloader_avst.ToTensor()]), mode_flag='test')
        print(test_dataset.__len__())
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
        model.load_state_dict(torch.load(config["directories"]["model_save_dir"] + config["model"]["checkpoint"] + ".pt"))
        test(model, test_loader)


if __name__ == '__main__':
    main()