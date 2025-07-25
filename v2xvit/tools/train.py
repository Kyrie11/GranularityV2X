import argparse
import os,sys,random
import statistics

import torch
torch.multiprocessing.set_sharing_strategy('file_system')
import os,time
torch.autograd.set_detect_anomaly(True)
import tqdm
from torch.utils.data import DataLoader, DistributedSampler
from tensorboardX import SummaryWriter
import yaml
from datetime import datetime
import v2xvit.hypes_yaml.yaml_utils as yaml_utils
from v2xvit.tools import train_utils,infrence_utils
from v2xvit.data_utils.datasets import build_dataset
from v2xvit.tools import multi_gpu_utils
import gc
gc.collect()
torch.cuda.empty_cache()

def train_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument("--hypes_yaml", type=str, required=True,
                        help='data generation yaml file needed ')
    parser.add_argument('--model_dir', default='',
                        help='Continued training path')
    parser.add_argument("--half", action='store_true', help="whether train with half precision")
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    opt = parser.parse_args()
    return opt

def main():

    opt = train_parser()
    hypes = yaml_utils.load_yaml(opt.hypes_yaml, opt)
    n = hypes['train_params']['lsh']['n']
    m = hypes['train_params']['lsh']['m']
    p = hypes['train_params']['lsh']['p']

    multi_gpu_utils.init_distributed_mode(opt)

    print('-----------------Dataset Building------------------')
    opencood_train_dataset = build_dataset(hypes, visualize=False, train=True)
    opencood_validate_dataset = build_dataset(hypes,
                                              visualize=False,
                                              train=False)

    if opt.distributed:
        sampler_train = DistributedSampler(opencood_train_dataset)
        sampler_val = DistributedSampler(opencood_validate_dataset,
                                         shuffle=False)

        batch_sampler_train = torch.utils.data.BatchSampler(
            sampler_train, hypes['train_params']['batch_size'], drop_last=True)

        train_loader = DataLoader(opencood_train_dataset,
                                  batch_sampler=batch_sampler_train,
                                  num_workers=32,
                                  collate_fn=opencood_train_dataset.collate_batch_train)
        val_loader = DataLoader(opencood_validate_dataset,
                                sampler=sampler_val,
                                num_workers=32,
                                collate_fn=opencood_train_dataset.collate_batch_train,
                                drop_last=False)
    else:
        train_loader = DataLoader(opencood_train_dataset,
                                batch_size=hypes['train_params']['batch_size'],
                                num_workers=8,
                                collate_fn=opencood_train_dataset.collate_batch_train,
                                shuffle=True,
                                pin_memory=False,
                                drop_last=True)
        val_loader = DataLoader(opencood_validate_dataset,
                                batch_size=hypes['train_params']['batch_size'],
                                num_workers=8,
                                collate_fn=opencood_train_dataset.collate_batch_train,
                                shuffle=False,
                                pin_memory=False,
                                drop_last=True)

    print('---------------Creating Model------------------')
    model = train_utils.create_model(hypes)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # if we want to train from last checkpoint.
    if opt.model_dir:
        saved_path = opt.model_dir
        init_epoch, model = train_utils.load_saved_model(saved_path, model)

    else:
        init_epoch = 0
        # if we train the model from scratch, we need to create a folder
        # to save the model,
        # saved_path = train_utils.setup_train(hypes)
        saved_path = train_utils.setup_train(hypes)

    # we assume gpu is necessary
    if torch.cuda.is_available():
        model.to(device)
    model_without_ddp = model

    if opt.distributed:
        model = \
            torch.nn.parallel.DistributedDataParallel(model,
                                                      device_ids=[opt.gpu],
                                                      find_unused_parameters=True)
        model_without_ddp = model.module

    # define the loss
    criterion = train_utils.create_loss(hypes)

    # optimizer setup
    # optimizer = train_utils.setup_optimizer(hypes, model)
    optimizer = train_utils.setup_optimizer(hypes, model_without_ddp)
    # lr scheduler setup
    num_steps = len(train_loader)
    scheduler = train_utils.setup_lr_schedular(hypes, optimizer, num_steps)

    # record training
    writer = SummaryWriter(saved_path)

    # half precision training
    if opt.half:
        scaler = torch.cuda.amp.GradScaler()

    print('Training start')
    epoches = hypes['train_params']['epoches']
    # used to help schedule learning rate
    for epoch in range(init_epoch, max(epoches, init_epoch)):
        model.fusion_net.reset_hidden_state()
        if hypes['lr_scheduler']['core_method'] != 'cosineannealwarm':
            scheduler.step(epoch)
        if hypes['lr_scheduler']['core_method'] == 'cosineannealwarm':
            scheduler.step_update(epoch * num_steps + 0)
        for param_group in optimizer.param_groups:
            print('learning rate %.7f' % param_group["lr"])

        if opt.distributed:
            sampler_train.set_epoch(epoch)

        pbar2 = tqdm.tqdm(total=len(train_loader), leave=True)
        record_len_list = []
        for i, (batch_data_list, ego_indices_batch) in enumerate(train_loader):
            if batch_data_list is None:
                continue

            # batch_data_list = train_utils.to_device(batch_data_list, device)

            print(f"一共有{len(batch_data_list)}帧")
            historical_data = batch_data_list[1:]
            short_his_data = historical_data[:n]
            long_his_data = []
            record_len = batch_data_list[0]['ego']['record_len']
            print("第0帧的时间戳：", batch_data_list[0]['ego']['agent_timestamps'])

            historical_ego_indices = ego_indices_batch[0]
            for j, frame_data in enumerate(historical_data):
                # The ego timestamp for this frame is the j-th element in the historical index list
                ego_ts_for_frame = historical_ego_indices[j].item()
                agent_ts_list = frame_data['ego']['agent_timestamps']
                print(f"每个车的延迟分别为{frame_data['ego']['time_delay']}")
                print(f"\n[Frame {j + 1} - Historical (Ego-time: {ego_ts_for_frame})]:")
                print(f"  > Agent Timestamps: {agent_ts_list}")
            print(f"historical_ego_indices={historical_ego_indices}")
            if historical_ego_indices.nelement() > 0:
                # The timeline starts from the most recent historical frame (e.g., t-1)
                start_index = historical_ego_indices[0].item()
                target_long_indices = [start_index - j * p for j in range(m)]

                for target_idx in target_long_indices:
                    # Find the position of target_idx in the historical timeline
                    match_pos = (historical_ego_indices == target_idx).nonzero(as_tuple=True)[0]
                    if match_pos.nelement() > 0:
                        # We found it, now grab the corresponding data snapshot
                        # The index `frame_index` corresponds to the `historical_data` list
                        frame_index = match_pos.item()
                        long_his_data.append(historical_data[frame_index])

            current_data = batch_data_list[0]
            # the model will be evaluation mode during validation
            model.train()
            model.zero_grad()
            optimizer.zero_grad()

            # case1 : late fusion train --> only ego needed
            # case2 : early fusion train --> all data projected to ego
            # case3 : intermediate fusion --> ['ego']['processed_lidar']
            # becomes a list, which containing all data from other cavs
            # as well
            if not opt.half:
                ouput_dict = model(current_data, short_his_data, long_his_data)
                final_loss = criterion(ouput_dict,
                                       current_data['ego']['label_dict'])
                final_loss += ouput_dict["offset_loss"] + ouput_dict["commu_loss"]
            else:
                with torch.cuda.amp.autocast():
                    ouput_dict = model(short_his_data, long_his_data)
                    # first argument is always your output dictionary,
                    # second argument is always your label dictionary.
                    final_loss = criterion(ouput_dict,
                                        current_data['ego']['label_dict'])
                    final_loss += ouput_dict["offset_loss"]+ ouput_dict["commu_loss"]
            criterion.logging(epoch, i, len(train_loader), writer)
            pbar2.update(1)
            time.sleep(0.001)
            # back-propagation
            if not opt.half:
                final_loss.backward()
                optimizer.step()
            else:
                scaler.scale(final_loss).backward()
                scaler.step(optimizer)
                scaler.update()
        if epoch % hypes['train_params']['save_freq'] == 0:
            torch.save(model.state_dict(),
                       os.path.join(saved_path,
                                    'net_epoch%d.pth' % (epoch + 1)))

    print('Training Finished, checkpoints saved to %s' % saved_path)
    torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
