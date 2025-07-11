import argparse
import os
import time
from collections import OrderedDict

import torch
import os
import open3d as o3d
from torch.utils.data import DataLoader

from v2xvit.data_utils.datasets.intermediate_fusion_dataset import IntermediateFusionDataset
import v2xvit.hypes_yaml.yaml_utils as yaml_utils
from v2xvit.tools import train_utils, infrence_utils
from v2xvit.data_utils.datasets import build_dataset
from v2xvit.visualization import vis_utils
from v2xvit.utils import eval_utils
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


def test_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Continued training path')
    parser.add_argument('--fusion_method', required=False, type=str,
                        default='intermediate_with_comm',
                        help='late, early or intermediate')
    parser.add_argument('--show_vis', action='store_true',
                        help='whether to show image visualization result')
    parser.add_argument('--show_sequence', action='store_true',
                        help='whether to show video visualization result.'
                             'it can note be set true with show_vis together ')
    parser.add_argument('--eval_epoch', type=str, default=14,
                        help='Set the checkpoint')
    parser.add_argument('--save_vis', action='store_true',
                        help='whether to save visualization result')
    parser.add_argument('--save_npy', action='store_true',
                        help='whether to save prediction and gt result'
                             'in npy file')
    parser.add_argument('--comm_thre', type=float, default=None,
                        help='Communication confidence threshold')
    parser.add_argument('--score_thre', type=float, default=None,
                    help='Confidence score threshold')
    parser.add_argument('--xyz_std', type=float, default=None,
                    help='position error')
    parser.add_argument('--ryp_std', type=float, default=None,
                help='rotation error')
    opt = parser.parse_args()
    return opt


def main():
    opt = test_parser()
    assert opt.fusion_method in ['late', 'early', 'intermediate',"intermediate_with_comm"]
    assert not (opt.show_vis and opt.show_sequence), \
        'you can only visualize ' \
        'the results in single ' \
        'image mode or video mode'

    hypes = yaml_utils.load_yaml(None, opt)
    if opt.comm_thre is not None:
        hypes['model']['args']['fusion_args']['communication']['thre'] = opt.comm_thre
    if opt.score_thre is not None:
        hypes['postprocess']['target_args']['score_threshold'] = opt.score_thre
    score_threshold = hypes['postprocess']['target_args']['score_threshold']
    if opt.xyz_std is not None:
        hypes['wild_setting']['xyz_std'] = opt.xyz_std
    if opt.ryp_std is not None:
        hypes['wild_setting']['ryp_std'] = opt.ryp_std

    print('Dataset Building')
    # opencood_dataset = build_dataset(hypes, visualize=True, train=False)
    opencood_dataset = IntermediateFusionDataset(hypes, visualize=False, train=False)
    data_loader = DataLoader(opencood_dataset,
                             batch_size=1,
                             num_workers=10,
                             collate_fn=opencood_dataset.collate_batch_test,
                             shuffle=False,
                             pin_memory=False,
                             drop_last=False)

    print('Creating Model')
    model = train_utils.create_model(hypes)
    # we assume gpu is necessary
    if torch.cuda.is_available():
        model.cuda()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('Loading Model from checkpoint')
    saved_path = opt.model_dir
    last_epoch = train_utils.findLastCheckpoint(saved_path)
    if opt.eval_epoch is None:
        epoch_id_list = [last_epoch]
    else:
        epoch_id_list = [opt.eval_epoch]
    
    for epoch_id in epoch_id_list:
        epoch_id, model = train_utils.load_saved_model(saved_path, model, epoch_id)
        model.eval()

        # Create the dictionary for evaluation
        result_stat = {0.3: {'tp': [], 'fp': [], 'gt': 0},
                    0.5: {'tp': [], 'fp': [], 'gt': 0},
                    0.7: {'tp': [], 'fp': [], 'gt': 0}}

        total_comm_rates = []
        for i, batch_data_list in enumerate(data_loader):
            print("{}".format(i))
            with torch.no_grad():
                torch.cuda.synchronize()
                batch_data = batch_data_list[0]
                batch_data = train_utils.to_device(batch_data, device)  
                batch_data_list = train_utils.to_device(batch_data_list, device)
                output_dict = OrderedDict()
                batch_data = batch_data_list[0]

                output_dict['ego'] = model(batch_data_list)
                print("output_dict:", output_dict)
                print("batch_data:", batch_data)
                pred_box_tensor, pred_score, gt_box_tensor = \
                    opencood_dataset.post_process(batch_data,
                                         output_dict)

                eval_utils.caluclate_tp_fp(pred_box_tensor,
                                        pred_score,
                                        gt_box_tensor,
                                        result_stat,
                                        0.3)
                eval_utils.caluclate_tp_fp(pred_box_tensor,
                                        pred_score,
                                        gt_box_tensor,
                                        result_stat,
                                        0.5)
                eval_utils.caluclate_tp_fp(pred_box_tensor,
                                        pred_score,
                                        gt_box_tensor,
                                        result_stat,
                                        0.7)
                if opt.save_npy:
                    npy_save_path = os.path.join(opt.model_dir, 'npy')
                    if not os.path.exists(npy_save_path):
                        os.makedirs(npy_save_path)
                    infrence_utils.save_prediction_gt(pred_box_tensor,
                                                    gt_box_tensor,
                                                    batch_data['ego'][
                                                        'origin_lidar'][0],
                                                    i,
                                                    npy_save_path)

                if opt.show_vis or opt.save_vis:
                    vis_save_path = ''
                    if opt.save_vis:
                        vis_save_path = os.path.join(opt.model_dir, 'vis')
                        if not os.path.exists(vis_save_path):
                            os.makedirs(vis_save_path)
                        vis_save_path = os.path.join(vis_save_path, '%05d.png' % i)

                    opencood_dataset.visualize_result(pred_box_tensor,
                                                    gt_box_tensor,
                                                    batch_data['ego'][
                                                        'origin_lidar'][0],
                                                    opt.show_vis,
                                                    vis_save_path,
                                                    dataset=opencood_dataset)
        if len(total_comm_rates) > 0:
            comm_rates = (sum(total_comm_rates)/len(total_comm_rates)).item()
        else:
            comm_rates = 0
        ap_30, ap_50, ap_70 = eval_utils.eval_final_results(result_stat, opt.model_dir)
        current_time = time.ctime()
        
        with open(os.path.join(saved_path, 'result.txt'), 'a+') as f:
            msg = 'Epoch: {} | AP @0.3: {:.04f} | AP @0.5: {:.04f} | AP @0.7: {:.04f} | comm_rate: {:.06f}\n'.format(epoch_id, ap_30, ap_50, ap_70, comm_rates)
            if opt.comm_thre is not None:
                msg = 'Epoch: {} | AP @0.3: {:.04f} | AP @0.5: {:.04f} | AP @0.7: {:.04f} | comm_rate: {:.06f} | comm_thre: {:.04f} | score_threshold: {:.02f} | xyz_std: {:.01f} | ryp_std: {:.01f} | time: {}\n'.format(epoch_id, ap_30, ap_50, ap_70, comm_rates, opt.comm_thre,score_threshold,opt.xyz_std,opt.ryp_std,current_time)
            f.write(msg)
            print(msg)

if __name__ == '__main__':
    main()