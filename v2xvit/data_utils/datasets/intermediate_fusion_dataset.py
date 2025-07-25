"""
Dataset class for early fusion
"""
import math
from collections import OrderedDict

import numpy as np
import torch

import v2xvit
import v2xvit.data_utils.post_processor as post_processor
from v2xvit.utils import box_utils
from v2xvit.data_utils.datasets import basedataset
from v2xvit.data_utils.pre_processor import build_preprocessor
from v2xvit.utils.pcd_utils import \
    mask_points_by_range, mask_ego_points, shuffle_points, \
    downsample_lidar_minimum


class IntermediateFusionDataset(basedataset.BaseDataset):
    def __init__(self, params, visualize, train=True):
        super(IntermediateFusionDataset, self). \
            __init__(params, visualize, train) 
        self.cur_ego_pose_flag = params['fusion']['args']['cur_ego_pose_flag']
        self.frame = params['train_params']['frame'] 
        self.pre_processor = build_preprocessor(params['preprocess'],
                                                train) 
        self.post_processor = post_processor.build_postprocessor(
            params['postprocess'],
            train)  # 3D Anchor Generator for Voxel

    # In IntermediateFusionDataset class

    # In IntermediateFusionDataset class

    def __getitem__(self, idx):
        # 1. Get LSH parameters
        m = self.params['train_params']['lsh']['m']
        n = self.params['train_params']['lsh']['n']
        p = self.params['train_params']['lsh']['p']

        # 2. Generate the "Ground Truth" frame for t=0 (no delay)
        gt_base_data_dict, _, current_timestamp = self.retrieve_base_data(idx, cur_ego_pose_flag=True)

        # 3. Generate the realistic, delayed historical frames
        historical_base_data_list, agent_delays, ego_historical_indices = \
            self.retrieve_lsh_data(idx, m, n, p, cur_ego_pose_flag=False)

        # 4. Combine them: GT frame at index 0, history starts at index 1
        base_data_list = [gt_base_data_dict] + historical_base_data_list

        processed_data_list = []
        ego_id = -1

        # 5. Establish 'Ground Truth' ego pose and communication range
        ego_lidar_pose = []
        cav_id_list = []

        for cav_id, cav_content in gt_base_data_dict.items():
            if cav_content['ego']:
                ego_id = cav_id
                ego_lidar_pose = cav_content['params']['lidar_pose']
                break
        assert ego_id != -1, "Ego vehicle not found"

        for cav_id, selected_cav_base in gt_base_data_dict.items():
            distance = math.sqrt((selected_cav_base['params']['lidar_pose'][0] - ego_lidar_pose[0]) ** 2 +
                                 (selected_cav_base['params']['lidar_pose'][1] - ego_lidar_pose[1]) ** 2)
            if distance <= v2xvit.data_utils.datasets.COM_RANGE:
                cav_id_list.append(cav_id)

        # 6. Process EACH frame in the combined list (GT + History)
        for i, base_data_dict in enumerate(base_data_list):
            processed_data_dict = OrderedDict({'ego': {}})
            pairwise_t_matrix = self.get_pairwise_transformation(base_data_dict, self.max_cav)

            processed_features, object_stack, object_id_stack = [], [], []
            velocity, time_delay, infra, spatial_correction_matrix = [], [], [], []

            agent_timestamps = []

            for cav_id in cav_id_list:
                if cav_id not in base_data_dict:
                    continue

                selected_cav_base = base_data_dict[cav_id]
                selected_cav_processed, void_lidar = self.get_item_single_car(selected_cav_base, ego_lidar_pose)

                if void_lidar:
                    continue

                # Append standard data
                object_stack.append(selected_cav_processed['object_bbx_center'])
                object_id_stack += selected_cav_processed['object_ids']
                processed_features.append(selected_cav_processed['processed_features'])
                velocity.append(selected_cav_processed['velocity'])
                spatial_correction_matrix.append(selected_cav_base['params']['spatial_correction_matrix'])
                infra.append(1 if int(cav_id) < 0 else 0)

                # Calculate and append delay and the absolute timestamp
                if i == 0:  # This is the Ground Truth frame
                    time_delay.append(0.0)
                    agent_timestamps.append(float(current_timestamp))
                else:  # This is a historical frame
                    frame_delay = float(agent_delays[cav_id])
                    time_delay.append(frame_delay)

                    # ego_historical_indices is a list of ints, not a tensor. No .item() needed.
                    ego_timestamp_for_this_frame = ego_historical_indices[i - 1]
                    agent_absolute_timestamp = ego_timestamp_for_this_frame - frame_delay
                    agent_timestamps.append(agent_absolute_timestamp)

            if not processed_features:
                continue

            # ... (Post-processing logic remains unchanged) ...
            unique_indices = [object_id_stack.index(x) for x in set(object_id_stack)]
            object_stack = np.vstack(object_stack)[unique_indices]
            object_bbx_center = np.zeros((self.params['postprocess']['max_num'], 7))
            mask = np.zeros(self.params['postprocess']['max_num'])
            object_bbx_center[:object_stack.shape[0], :] = object_stack
            mask[:object_stack.shape[0]] = 1
            cav_num = len(processed_features)
            merged_feature_dict = self.merge_features_to_dict(processed_features)
            anchor_box = self.post_processor.generate_anchor_box()
            label_dict = self.post_processor.generate_label(
                gt_box_center=object_bbx_center, anchors=anchor_box, mask=mask)
            velocity += (self.max_cav - len(velocity)) * [0.]
            time_delay += (self.max_cav - len(time_delay)) * [0.]
            infra += (self.max_cav - len(infra)) * [0.]
            agent_timestamps += (self.max_cav - len(agent_timestamps)) * [0.]
            spatial_correction_matrix = np.stack(spatial_correction_matrix)
            padding_eye = np.tile(np.eye(4)[None], (self.max_cav - len(spatial_correction_matrix), 1, 1))
            spatial_correction_matrix = np.concatenate([spatial_correction_matrix, padding_eye], axis=0)

            processed_data_dict['ego'].update({
                'object_bbx_center': object_bbx_center,
                'object_bbx_mask': mask,
                'object_ids': [object_id_stack[i] for i in unique_indices],
                'anchor_box': anchor_box,
                'processed_lidar': merged_feature_dict,
                'label_dict': label_dict,
                'cav_num': cav_num,
                'velocity': velocity,
                'time_delay': time_delay,
                'infra': infra,
                'spatial_correction_matrix': spatial_correction_matrix,
                'pairwise_t_matrix': pairwise_t_matrix,
                'agent_timestamps': agent_timestamps
            })
            processed_data_list.append(processed_data_dict)

        return processed_data_list, ego_historical_indices

    @staticmethod
    def get_pairwise_transformation(base_data_dict, max_cav):
        """
        Get pair-wise transformation matrix across different agents.
        This is only used for v2vnet and disconet. Currently we set
        this as identity matrix as the pointcloud is projected to
        ego vehicle first.

        Parameters
        ----------
        base_data_dict : dict
            Key : cav id, item: transformation matrix to ego, lidar points.

        max_cav : int
            The maximum number of cav, default 5

        Return
        ------
        pairwise_t_matrix : np.array
            The pairwise transformation matrix across each cav.
            shape: (L, L, 4, 4)
        """
        pairwise_t_matrix = np.zeros((max_cav, max_cav, 4, 4))
        # default are identity matrix
        pairwise_t_matrix[:, :] = np.identity(4)

        return pairwise_t_matrix

    def get_item_single_car(self, selected_cav_base, ego_pose):
        """
        Project the lidar and bbx to ego space first, and then do clipping.

        Parameters
        ----------
        selected_cav_base : dict
            The dictionary contains a single CAV's raw information.
        ego_pose : list
            The ego vehicle lidar pose under world coordinate.

        Returns
        -------
        selected_cav_processed : dict
            The dictionary contains the cav's processed information.
        """
        selected_cav_processed = {}

        # calculate the transformation matrix
        transformation_matrix = \
            selected_cav_base['params']['transformation_matrix']

        # retrieve objects under ego coordinates
        object_bbx_center, object_bbx_mask, object_ids = \
            self.post_processor.generate_object_center([selected_cav_base],
                                                       ego_pose)

        # filter lidar
        lidar_np = selected_cav_base['lidar_np']
        lidar_np = shuffle_points(lidar_np)
        # remove points that hit itself
        lidar_np = mask_ego_points(lidar_np)
        # project the lidar to ego space
        lidar_np[:, :3] = \
            box_utils.project_points_by_matrix_torch(lidar_np[:, :3],
                                                     transformation_matrix)
        lidar_np = mask_points_by_range(lidar_np,
                                        self.params['preprocess'][
                                            'cav_lidar_range'])
        # Check if filtered LiDAR points are not void
        void_lidar = True if lidar_np.shape[0] < 1 else False

        processed_lidar = self.pre_processor.preprocess(lidar_np)

        # velocity
        velocity = selected_cav_base['params']['ego_speed']
        # normalize veloccity by average speed 30 km/h
        velocity = velocity / 30

        selected_cav_processed.update(
            {'object_bbx_center': object_bbx_center[object_bbx_mask == 1],
             'object_ids': object_ids,
             'projected_lidar': lidar_np,
             'processed_features': processed_lidar,
             'velocity': velocity})

        return selected_cav_processed, void_lidar

    @staticmethod
    def merge_features_to_dict(processed_feature_list):
        """
        Merge the preprocessed features from different cavs to the same
        dictionary.

        Parameters
        ----------
        processed_feature_list : list
            A list of dictionary containing all processed features from
            different cavs.

        Returns
        -------
        merged_feature_dict: dict
            key: feature names, value: list of features.
        """

        merged_feature_dict = OrderedDict()

        for i in range(len(processed_feature_list)):
            for feature_name, feature in processed_feature_list[i].items():
                if feature_name not in merged_feature_dict:
                    merged_feature_dict[feature_name] = []
                if isinstance(feature, list):
                    merged_feature_dict[feature_name] += feature
                else:
                    merged_feature_dict[feature_name].append(feature)

        return merged_feature_dict

    @staticmethod
    def merge_features_to_dict(processed_feature_list):
        """
        Merge the preprocessed features from different cavs to the same
        dictionary.

        Parameters
        ----------
        processed_feature_list : list
            A list of dictionary containing all processed features from
            different cavs.

        Returns
        -------
        merged_feature_dict: dict
            key: feature names, value: list of features.
        """

        merged_feature_dict = OrderedDict()

        for i in range(len(processed_feature_list)):
            for feature_name, feature in processed_feature_list[i].items():
                if feature_name not in merged_feature_dict:
                    merged_feature_dict[feature_name] = []
                if isinstance(feature, list):
                    merged_feature_dict[feature_name] += feature
                else:
                    merged_feature_dict[feature_name].append(feature)

        return merged_feature_dict

    def collate_batch_train(self, batch):
        """
        Custom collate function for training.

        Parameters
        ----------
        batch : list
            List of tuples, where each tuple contains (processed_data_list, ego_historical_indices).

        Returns
        -------
        collated_data : tuple
            A tuple containing the collated processed_data_list and ego_historical_indices.
        """
        # Unzip the batch into separate lists
        processed_data_list_batch = [item[0] for item in batch]
        ego_indices_batch = [item[1] for item in batch]

        # The number of frames (GT + history) is determined by the first sample in the batch
        num_frames = len(processed_data_list_batch[0])

        # This will be our final list of batched frame dictionaries
        output_data_list = []

        # Iterate through each frame index (e.g., frame 0 is GT, frame 1 is t-1, etc.)
        for frame_idx in range(num_frames):
            # Create a temporary list of data for the current frame from all samples in the batch
            frame_batch = [sample_frames[frame_idx] for sample_frames in processed_data_list_batch]

            # Initialize lists to gather data from all samples for this frame
            object_bbx_center_list = []
            object_bbx_mask_list = []
            object_ids_list = []

            processed_lidar_list = []
            record_len = []

            velocity_list = []
            time_delay_list = []
            infra_list = []
            pairwise_t_matrix_list = []
            spatial_correction_matrix_list = []

            ## NEW ##: Initialize the list for our new timestamp data
            agent_timestamps_list = []

            # Now, loop through the data for each sample *within this specific frame*
            for data in frame_batch:
                # Data that is batched at the "sample" level
                object_bbx_center_list.append(data['ego']['object_bbx_center'])
                object_bbx_mask_list.append(data['ego']['object_bbx_mask'])
                object_ids_list.append(data['ego']['object_ids'])

                # Data that is batched at the "agent" level
                processed_lidar_list.append(data['ego']['processed_lidar'])
                record_len.append(data['ego']['cav_num'])

                # Extend the lists with per-agent information
                velocity_list.extend(data['ego']['velocity'])
                time_delay_list.extend(data['ego']['time_delay'])
                infra_list.extend(data['ego']['infra'])

                ## NEW ##: Extend the list with the agent timestamps
                agent_timestamps_list.extend(data['ego']['agent_timestamps'])

                pairwise_t_matrix_list.append(data['ego']['pairwise_t_matrix'])
                spatial_correction_matrix_list.append(data['ego']['spatial_correction_matrix'])

            # Collate the stacked agent data into tensors
            merged_feature_dict = merge_features_to_dict(processed_lidar_list)
            velocity_list = torch.from_numpy(np.array(velocity_list)).float()
            time_delay_list = torch.from_numpy(np.array(time_delay_list)).float()
            infra_list = torch.from_numpy(np.array(infra_list)).float()

            ## NEW ##: Convert the gathered timestamps into a tensor
            agent_timestamps_list = torch.from_numpy(np.array(agent_timestamps_list)).float()

            # Create the final batched dictionary for this frame
            final_frame_dict = {
                'ego': {
                    'object_bbx_center': torch.from_numpy(np.array(object_bbx_center_list)).float(),
                    'object_bbx_mask': torch.from_numpy(np.array(object_bbx_mask_list)).float(),
                    'processed_lidar': merged_feature_dict,
                    'record_len': torch.from_numpy(np.array(record_len)),
                    'velocity': velocity_list,
                    'time_delay': time_delay_list,
                    'infra': infra_list,
                    'pairwise_t_matrix': torch.from_numpy(np.array(pairwise_t_matrix_list)).float(),
                    'spatial_correction_matrix': torch.from_numpy(np.array(spatial_correction_matrix_list)).float(),

                    ## NEW ##: Add the final batched tensor to the dictionary
                    'agent_timestamps': agent_timestamps_list
                }
            }
            output_data_list.append(final_frame_dict)

        # Collate the ego historical indices (this part is simpler)
        # Note: This might need adjustment if history lengths are variable and require padding
        collated_ego_indices = torch.stack([torch.from_numpy(np.array(indices)) for indices in ego_indices_batch])

        return output_data_list, collated_ego_indices
    
    def collate_batch_test(self, batch):
            # output_dict_list = []
        # for batch in batch_lit:
        assert len(batch) <= 1, "Batch size 1 is required during testing!"
        output_dict_list = self.collate_batch_train(batch)

        # check if anchor box in the batch
        for i in range(len(batch[0])):
            if batch[0][i]['ego']['anchor_box'] is not None:
                output_dict_list[i]['ego'].update({'anchor_box':
                    torch.from_numpy(np.array(
                        batch[0][i]['ego'][
                            'anchor_box']))})

            # save the transformation matrix (4, 4) to ego vehicle
            transformation_matrix_torch = \
                torch.from_numpy(np.identity(4)).float()
            output_dict_list[i]['ego'].update({'transformation_matrix':
                                        transformation_matrix_torch})
        # output_dict_list.append(output_dict)

        return output_dict_list

    def post_process(self, data_dict, output_dict):
        """
        Process the outputs of the model to 2D/3D bounding box.

        Parameters
        ----------
        data_dict : dict
            The dictionary containing the origin input data of model.

        output_dict :dict
            The dictionary containing the output of the model.

        Returns
        -------
        pred_box_tensor : torch.Tensor
            The tensor of prediction bounding box after NMS.
        gt_box_tensor : torch.Tensor
            The tensor of gt bounding box.
        """
        pred_box_tensor, pred_score = \
            self.post_processor.post_process(data_dict, output_dict)
        gt_box_tensor = self.post_processor.generate_gt_bbx(data_dict)

        return pred_box_tensor, pred_score, gt_box_tensor