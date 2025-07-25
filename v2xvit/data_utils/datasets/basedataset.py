"""
Basedataset class for lidar data pre-processing
"""

import os
import math
from collections import OrderedDict

import torch
import numpy as np
from torch.utils.data import Dataset

import v2xvit.utils.pcd_utils as pcd_utils
from v2xvit.data_utils.augmentor.data_augmentor import DataAugmentor
from v2xvit.hypes_yaml.yaml_utils import load_yaml
from v2xvit.utils.pcd_utils import downsample_lidar_minimum
from v2xvit.utils.transformation_utils import x1_to_x2


class BaseDataset(Dataset):
    """
    Base dataset for all kinds of fusion. Mainly used to assign correct
    index and add noise.

    Parameters
    __________
    params : dict
        The dictionary contains all parameters for training/testing.

    visualize : false
        If set to true, the dataset is used for visualization.

    Attributes
    ----------
    scenario_database : OrderedDict
        A structured dictionary contains all file information.

    len_record : list
        The list to record each scenario's data length. This is used to
        retrieve the correct index during training.

    """

    def __init__(self, params, visualize, train=True):
        self.params = params
        self.visualize = visualize
        self.train = train

        self.pre_processor = None
        self.post_processor = None
        self.data_augmentor = DataAugmentor(params['data_augment'],
                                            train)

        if 'wild_setting' in params:
            self.seed = params['wild_setting']['seed']
            # whether to add time delay
            self.async_flag = params['wild_setting']['async']
            self.async_mode = \
                'sim' if 'async_mode' not in params['wild_setting'] \
                    else params['wild_setting']['async_mode']
            self.async_overhead = params['wild_setting']['async_overhead']

            # localization error
            self.loc_err_flag = params['wild_setting']['loc_err']
            self.xyz_noise_std = params['wild_setting']['xyz_std']
            self.ryp_noise_std = params['wild_setting']['ryp_std']

            # transmission data size
            self.data_size = \
                params['wild_setting']['data_size'] \
                    if 'data_size' in params['wild_setting'] else 0
            self.transmission_speed = \
                params['wild_setting']['transmission_speed'] \
                    if 'transmission_speed' in params['wild_setting'] else 27
            self.backbone_delay = \
                params['wild_setting']['backbone_delay'] \
                    if 'backbone_delay' in params['wild_setting'] else 0

        else:
            self.async_flag = False
            self.async_overhead = 0
            self.async_mode = 'sim'
            self.loc_err_flag = False
            self.xyz_noise_std = 0
            self.ryp_noise_std = 0
            self.data_size = 0
            self.transmission_speed = 27
            self.backbone_delay = 0
        if self.train:
            root_dir = params['root_dir']
        else:
            root_dir = params['validate_dir']

        if 'max_cav' not in params['train_params']:
            self.max_cav = 7
        else:
            self.max_cav = params['train_params']['max_cav']

        # first load all paths of different scenarios
        scenario_folders = sorted([os.path.join(root_dir, x)
                                   for x in os.listdir(root_dir) if
                                   os.path.isdir(os.path.join(root_dir, x))])
        # Structure: {scenario_id : {cav_1 : {timestamp1 : {yaml: path,
        # lidar: path, cameras:list of path}}}}
        self.scenario_database = OrderedDict()
        self.len_record = []

        # loop over all scenarios
        for (i, scenario_folder) in enumerate(scenario_folders):
            self.scenario_database.update({i: OrderedDict()})

            # at least 1 cav should show up
            cav_list = sorted([x for x in os.listdir(scenario_folder)
                               if os.path.isdir(
                    os.path.join(scenario_folder, x))])
            assert len(cav_list) > 0

            # roadside unit data's id is always negative, so here we want to
            # make sure they will be in the end of the list as they shouldn't
            # be ego vehicle.
            if int(cav_list[0]) < 0:
                cav_list = cav_list[1:] + [cav_list[0]]

            # loop over all CAV data
            for (j, cav_id) in enumerate(cav_list):
                if j > self.max_cav - 1:
                    print('too many cavs')
                    break
                self.scenario_database[i][cav_id] = OrderedDict()

                # save all yaml files to the dictionary
                cav_path = os.path.join(scenario_folder, cav_id)

                # use the frame number as key, the full path as the values
                yaml_files = \
                    sorted([os.path.join(cav_path, x)
                            for x in os.listdir(cav_path) if
                            x.endswith('.yaml') and 'additional' not in x])
                timestamps = self.extract_timestamps(yaml_files)

                for timestamp in timestamps: 
                    self.scenario_database[i][cav_id][timestamp] = \
                        OrderedDict()

                    yaml_file = os.path.join(cav_path,
                                             timestamp + '.yaml')
                    lidar_file = os.path.join(cav_path,
                                              timestamp + '.pcd')
                    camera_files = self.load_camera_files(cav_path, timestamp)

                    self.scenario_database[i][cav_id][timestamp]['yaml'] = \
                        yaml_file
                    self.scenario_database[i][cav_id][timestamp]['lidar'] = \
                        lidar_file
                    self.scenario_database[i][cav_id][timestamp]['camera0'] = \
                        camera_files
                # Assume all cavs will have the same timestamps length. Thus
                # we only need to calculate for the first vehicle in the
                # scene.
                if j == 0:
                    self.scenario_database[i][cav_id]['ego'] = True
                    if not self.len_record:
                        self.len_record.append(len(timestamps))
                    else:
                        prev_last = self.len_record[-1]
                        self.len_record.append(prev_last + len(timestamps))
                else:
                    self.scenario_database[i][cav_id]['ego'] = False

    def __len__(self):
        return self.len_record[-1]

    def __getitem__(self, idx):
        """
        Abstract method, needs to be definecollate_batch_train by the children class.
        """
        pass
    
    def retrieve_multi_data(self, idx, select_num, cur_ego_pose_flag=True):
        scenario_index = 0
        for i, ele in enumerate(self.len_record):
            if idx < ele:
                scenario_index = i
                break

        # check the timestamp index
        timestamp_index = idx if scenario_index == 0 else \
            idx - self.len_record[scenario_index - 1]
            
        if timestamp_index < select_num:
            idx += select_num
        
        select_dict = OrderedDict()
        # [index-3,index-2,index-1,index]
        scenario_index_list = []
        index_list = []
        scenario_index = 0
        timestamp_key = 0
        for j in range(idx,idx-select_num-1,-1):
            if j == idx:
                base_data_dict,cur_scenario,cur_timestamp = self.retrieve_base_data(j,cur_ego_pose_flag)
                scenario_index = cur_scenario
                timestamp_key = cur_timestamp
            else:
                base_data_dict = self.retrieve_base_data_before(scenario_index,j,timestamp_key,cur_ego_pose_flag)
            scenario_index_list.append(scenario_index)
            index_list.append(j)
            select_dict[j] = base_data_dict
        return select_dict,scenario_index_list,index_list,timestamp_index



    def retrieve_base_data(self, idx, cur_ego_pose_flag=True):
        """
        Given the index, return the corresponding data.

        Parameters
        ----------
        idx : int
            Index given by dataloader.

        cur_ego_pose_flag : bool
            Indicate whether to use current timestamp ego pose to calculate
            transformation matrix.
        Returns
        -------
        data : dict
            The dictionary contains loaded yaml params and lidar data for
            each cav.
        """
        # we loop the accumulated length list to see get the scenario index
        scenario_index = 0
        for i, ele in enumerate(self.len_record):
            if idx < ele:
                scenario_index = i
                break
        scenario_database = self.scenario_database[scenario_index]

        # check the timestamp index
        timestamp_index = idx if scenario_index == 0 else \
            idx - self.len_record[scenario_index - 1]
        timestamp_key = self.return_timestamp_key(scenario_database,
                                                  timestamp_index)
        # calculate distance to ego for each cav for time delay estimation
        ego_cav_content = \
            self.calc_dist_to_ego(scenario_database, timestamp_key)

        data = OrderedDict()
        # load files for all CAVs self.scenario_database[i][cav_id]['ego'] = True
        for cav_id, cav_content in scenario_database.items():
            data[cav_id] = OrderedDict()
            data[cav_id]['ego'] = cav_content['ego']

            # calculate delay for this vehicle
            timestamp_delay = \
                self.time_delay_calculation(cav_content['ego'])

            if timestamp_index - timestamp_delay <= 0:
                timestamp_delay = timestamp_index
            timestamp_index_delay = max(0, timestamp_index - timestamp_delay)
            timestamp_key_delay = self.return_timestamp_key(scenario_database,
                                                            timestamp_index_delay)
            # add time delay to vehicle parameters
            data[cav_id]['time_delay'] = timestamp_delay
            # load the corresponding data into the dictionary
            data[cav_id]['params'] = self.reform_param(cav_content,
                                                    ego_cav_content,
                                                    timestamp_key,
                                                    timestamp_key_delay,
                                                    cur_ego_pose_flag)
            data[cav_id]['lidar_np'] = \
                pcd_utils.pcd_to_np(cav_content[timestamp_key_delay]['lidar'])
        return data,scenario_index,timestamp_key
    
    def retrieve_base_data_before(self, scenario_index, idx, cur_timestamp_key, cur_ego_pose_flag=True):
        
        scenario_database = self.scenario_database[scenario_index]

        # check the timestamp index
        timestamp_index = idx if scenario_index == 0 else \
            idx - self.len_record[scenario_index - 1]
        # retrieve the corresponding timestamp key
        timestamp_key = self.return_timestamp_key(scenario_database,
                                                  timestamp_index)
        # calculate distance to ego for each cav for time delay estimation
        ego_cav_content = \
            self.calc_dist_to_ego(scenario_database, timestamp_key)

        data = OrderedDict()
        # load files for all CAVs
        for cav_id, cav_content in scenario_database.items():
            if True:
                data[cav_id] = OrderedDict()
                data[cav_id]['ego'] = cav_content['ego']

                # calculate delay for this vehicle
                timestamp_delay = \
                    self.time_delay_calculation(cav_content['ego'])

                if timestamp_index - timestamp_delay <= 0:
                    timestamp_delay = timestamp_index
                timestamp_index_delay = max(0, timestamp_index - timestamp_delay)
                timestamp_key_delay = self.return_timestamp_key(scenario_database,
                                                                timestamp_index_delay)
                # add time delay to vehicle parameters
                data[cav_id]['time_delay'] = timestamp_delay
                # load the corresponding data into the dictionary
                data[cav_id]['params'] = self.reform_param(cav_content,
                                                        ego_cav_content,
                                                        cur_timestamp_key,
                                                        timestamp_key_delay,
                                                        cur_ego_pose_flag)
                data[cav_id]['lidar_np'] = \
                    pcd_utils.pcd_to_np(cav_content[timestamp_key_delay]['lidar'])
        return data

    """
    获取长短期历史帧
    Parameters
        ----------
    idx: int
            the current index given by the dataloader
    m : int
            The number of long-term history frames (m). 
    n : int
            The stride for sampling long-term history frames (n). E.g., n=3 means every 3rd frame.
    p : int
            The number of recent frames for short-term history (p).
    cur_ego_pose_flag : bool
            Passed to the underlying data retrieval functions.
            
    Returns
        -------
        select_dict : OrderedDict
            A dictionary where keys are the frame indices and values are the loaded data dictionaries.
        
        final_indices : list
            A sorted list of unique global indices that were loaded.
    """
    def retrieve_long_short_his(self, idx, p, m, n, cur_ego_pose_flag=True):
        # 1. Find the scenario and timestamp index for the current frame `idx`
        scenario_index = 0
        for i, ele in enumerate(self.len_record):
            if idx < ele:
                scenario_index = i
                break

        scenario_start_idx = 0 if scenario_index == 0 else self.len_record[scenario_index - 1]

        # 2. Generate the desired indices for short and long-term history
        short_term_target_indices = []
        # Short-term history: p consecutive frames ending at idx
        if p>0:
            for i in range(p):
                short_term_target_indices.append(idx - i)

        # Long-term history: m frames with interval n, starting from idx
        long_term_target_indices = []
        if m>0 and n>0:
            for i in range(m):
                long_term_target_indices.append(idx - (i*n))
        # 3. Get the combined set of unique indices needed for data fetching
        all_target_indices = set(short_term_target_indices) | set(long_term_target_indices)

        # 4. Filter ALL index lists to ensure they are within the current scenario's bounds
        valid_unique_indices = sorted([i for i in all_target_indices if i >= scenario_start_idx], reverse=True)

        valid_short_indices = [i for i in short_term_target_indices if i in valid_unique_indices]
        valid_long_indices = [i for i in long_term_target_indices if i in valid_unique_indices]
        if not valid_unique_indices:
            return OrderedDict(), [], [], []

        # 5. Retrieve data ONLY for the unique, valid indices
        select_dict = OrderedDict()
        # First frame (current) is retrieved with its own pose as reference
        current_idx = valid_unique_indices[0]
        base_data_dict, _, cur_timestamp_key = self.retrieve_base_data(current_idx, cur_ego_pose_flag)
        select_dict[current_idx] = base_data_dict

        assert current_idx == idx, "The first valid index must be the current index"
        # Retrieve all other past frames
        for past_idx in valid_unique_indices[1:]:
            past_data_dict = self.retrieve_base_data_before(scenario_index, past_idx, cur_timestamp_key,
                                                            cur_ego_pose_flag)
            select_dict[past_idx] = past_data_dict

        return select_dict, valid_unique_indices, valid_short_indices, valid_long_indices



    @staticmethod
    def extract_timestamps(yaml_files):
        """
        Given the list of the yaml files, extract the mocked timestamps.

        Parameters
        ----------
        yaml_files : list
            The full path of all yaml files of ego vehicle

        Returns
        -------
        timestamps : list
            The list containing timestamps only.
        """
        timestamps = []

        for file in yaml_files:
            res = file.split('/')[-1]

            timestamp = res.replace('.yaml', '')
            timestamps.append(timestamp)

        return timestamps

    @staticmethod
    def return_timestamp_key(scenario_database, timestamp_index):
        """
        Given the timestamp index, return the correct timestamp key, e.g.
        2 --> '000078'.

        Parameters
        ----------
        scenario_database : OrderedDict
            The dictionary contains all contents in the current scenario.

        timestamp_index : int
            The index for timestamp.

        Returns
        -------
        timestamp_key : str
            The timestamp key saved in the cav dictionary.
        """
        # get all timestamp keys
        timestamp_keys = list(scenario_database.items())[0][1]
        # retrieve the correct index
        timestamp_key = list(timestamp_keys.items())[timestamp_index][0]

        return timestamp_key

    def calc_dist_to_ego(self, scenario_database, timestamp_key):
        """
        Calculate the distance to ego for each cav.
        """
        ego_lidar_pose = None
        ego_cav_content = None
        # Find ego pose first
        for cav_id, cav_content in scenario_database.items():
            if cav_content['ego']:
                ego_cav_content = cav_content
                ego_lidar_pose = \
                    load_yaml(cav_content[timestamp_key]['yaml'])['lidar_pose']
                break

        assert ego_lidar_pose is not None

        # calculate the distance
        for cav_id, cav_content in scenario_database.items():
            cur_lidar_pose = \
                load_yaml(cav_content[timestamp_key]['yaml'])['lidar_pose']
            distance = \
                math.sqrt((cur_lidar_pose[0] -
                           ego_lidar_pose[0]) ** 2 +
                          (cur_lidar_pose[1] - ego_lidar_pose[1]) ** 2)
            cav_content['distance_to_ego'] = distance
            scenario_database.update({cav_id: cav_content})

        return ego_cav_content

    def time_delay_calculation(self, ego_flag):
        """
        Calculate the time delay for a certain vehicle.

        Parameters
        ----------
        ego_flag : boolean
            Whether the current cav is ego.

        Return
        ------
        time_delay : int
            The time delay quantization.
        """
        # there is not time delay for ego vehicle
        if ego_flag:
            return 0
        # time delay real mode
        if self.async_mode == 'real':
            # noise/time is in ms unit
            overhead_noise = np.random.uniform(0, self.async_overhead)
            tc = self.data_size / self.transmission_speed * 1000
            time_delay = int(overhead_noise + tc + self.backbone_delay)
        elif self.async_mode == 'sim':
            time_delay = np.abs(self.async_overhead)

        time_delay = time_delay // 100
        return time_delay if self.async_flag else 0

    def add_loc_noise(self, pose, xyz_std, ryp_std):
        """
        Add localization noise to the pose.

        Parameters
        ----------
        pose : list
            x,y,z,roll,yaw,pitch

        xyz_std : float
            std of the gaussian noise on xyz

        ryp_std : float
            std of the gaussian noise
        """
        np.random.seed(self.seed)
        xyz_noise = np.random.normal(0, xyz_std, 3)
        ryp_std = np.random.normal(0, ryp_std, 3)
        noise_pose = [pose[0] + xyz_noise[0],
                      pose[1] + xyz_noise[1],
                      pose[2] + xyz_noise[2],
                      pose[3],
                      pose[4] + ryp_std[1],
                      pose[5]]
        return noise_pose

    def reform_param(self, cav_content, ego_content, timestamp_cur,
                     timestamp_delay, cur_ego_pose_flag):
        """
        Reform the data params with current timestamp object groundtruth and
        delay timestamp LiDAR pose.

        Parameters
        ----------
        cav_content : dict
            Dictionary that contains all file paths in the current cav/rsu.

        ego_content : dict
            Ego vehicle content.

        timestamp_cur : str
            The current timestamp.

        timestamp_delay : str
            The delayed timestamp.

        cur_ego_pose_flag : bool
            Whether use current ego pose to calculate transformation matrix.

        Return
        ------
        The merged parameters.
        """
        cur_params = load_yaml(cav_content[timestamp_cur]['yaml'])
        delay_params = load_yaml(cav_content[timestamp_delay]['yaml'])

        cur_ego_params = load_yaml(ego_content[timestamp_cur]['yaml'])
        delay_ego_params = load_yaml(ego_content[timestamp_delay]['yaml'])

        # we need to calculate the transformation matrix from cav to ego
        # at the delayed timestamp
        delay_cav_lidar_pose = delay_params['lidar_pose']
        delay_ego_lidar_pose = delay_ego_params["lidar_pose"]

        cur_ego_lidar_pose = cur_ego_params['lidar_pose']
        cur_cav_lidar_pose = cur_params['lidar_pose']

        if not cav_content['ego'] and self.loc_err_flag:
            delay_cav_lidar_pose = self.add_loc_noise(delay_cav_lidar_pose,
                                                      self.xyz_noise_std,
                                                      self.ryp_noise_std)
            cur_cav_lidar_pose = self.add_loc_noise(cur_cav_lidar_pose,
                                                    self.xyz_noise_std,
                                                    self.ryp_noise_std)

        if cur_ego_pose_flag:
            transformation_matrix = x1_to_x2(delay_cav_lidar_pose,
                                             cur_ego_lidar_pose)
            spatial_correction_matrix = np.eye(4)
        else:
            transformation_matrix = x1_to_x2(delay_cav_lidar_pose,
                                             delay_ego_lidar_pose)
            spatial_correction_matrix = x1_to_x2(delay_ego_lidar_pose,
                                                 cur_ego_lidar_pose)
        # This is only used for late fusion, as it did the transformation
        # in the postprocess, so we want the gt object transformation use
        # the correct one
        gt_transformation_matrix = x1_to_x2(cur_cav_lidar_pose,
                                            cur_ego_lidar_pose)

        # we always use current timestamp's gt bbx to gain a fair evaluation
        delay_params['vehicles'] = cur_params['vehicles']
        delay_params['transformation_matrix'] = transformation_matrix
        delay_params['gt_transformation_matrix'] = \
            gt_transformation_matrix
        delay_params['spatial_correction_matrix'] = spatial_correction_matrix

        return delay_params

    @staticmethod
    def load_camera_files(cav_path, timestamp):
        """
        Retrieve the paths to all camera files.

        Parameters
        ----------
        cav_path : str
            The full file path of current cav.

        timestamp : str
            Current timestamp

        Returns
        -------
        camera_files : list
            The list containing all camera png file paths.
        """
        camera0_file = os.path.join(cav_path,
                                    timestamp + '_camera0.png')
        camera1_file = os.path.join(cav_path,
                                    timestamp + '_camera1.png')
        camera2_file = os.path.join(cav_path,
                                    timestamp + '_camera2.png')
        camera3_file = os.path.join(cav_path,
                                    timestamp + '_camera3.png')
        return [camera0_file, camera1_file, camera2_file, camera3_file]

    def project_points_to_bev_map(self, points, ratio=0.1):
        """
        Project points to BEV occupancy map with default ratio=0.1.

        Parameters
        ----------
        points : np.ndarray
            (N, 3) / (N, 4)

        ratio : float
            Discretization parameters. Default is 0.1.

        Returns
        -------
        bev_map : np.ndarray
            BEV occupancy map including projected points
            with shape (img_row, img_col).

        """
        return self.pre_processor.project_points_to_bev_map(points, ratio)

    def augment(self, lidar_np, object_bbx_center, object_bbx_mask):
        """
        Data augmentation operation.
        """
        tmp_dict = {'lidar_np': lidar_np,
                    'object_bbx_center': object_bbx_center,
                    'object_bbx_mask': object_bbx_mask}
        tmp_dict = self.data_augmentor.forward(tmp_dict)

        lidar_np = tmp_dict['lidar_np']
        object_bbx_center = tmp_dict['object_bbx_center']
        object_bbx_mask = tmp_dict['object_bbx_mask']

        return lidar_np, object_bbx_center, object_bbx_mask

    def collate_batch_train(self, batch):
        """
        Customized collate function for pytorch dataloader during training
        for late fusion dataset.

        Parameters
        ----------
        batch : dict

        Returns
        -------
        batch : dict
            Reformatted batch.
        """
        # during training, we only care about ego.
        output_dict = {'ego': {}}

        object_bbx_center = []
        object_bbx_mask = []
        processed_lidar_list = []
        label_dict_list = []

        if self.visualize:
            origin_lidar = []

        for i in range(len(batch)):
            ego_dict = batch[i]['ego']
            object_bbx_center.append(ego_dict['object_bbx_center'])
            object_bbx_mask.append(ego_dict['object_bbx_mask'])
            processed_lidar_list.append(ego_dict['processed_lidar'])
            label_dict_list.append(ego_dict['label_dict'])

            if self.visualize:
                origin_lidar.append(ego_dict['origin_lidar'])

        # convert to numpy, (B, max_num, 7)
        object_bbx_center = torch.from_numpy(np.array(object_bbx_center))
        object_bbx_mask = torch.from_numpy(np.array(object_bbx_mask))

        processed_lidar_torch_dict = \
            self.pre_processor.collate_batch(processed_lidar_list)
        label_torch_dict = \
            self.post_processor.collate_batch(label_dict_list)
        output_dict['ego'].update({'object_bbx_center': object_bbx_center,
                                   'object_bbx_mask': object_bbx_mask,
                                   'processed_lidar': processed_lidar_torch_dict,
                                   'label_dict': label_torch_dict})
        if self.visualize:
            origin_lidar = \
                np.array(downsample_lidar_minimum(pcd_np_list=origin_lidar))
            origin_lidar = torch.from_numpy(origin_lidar)
            output_dict['ego'].update({'origin_lidar': origin_lidar})

        return output_dict

    def visualize_result(self, pred_box_tensor,
                         gt_tensor,
                         pcd,
                         show_vis,
                         save_path,
                         dataset=None):
        self.post_processor.visualize(pred_box_tensor,
                                      gt_tensor,
                                      pcd,
                                      show_vis,
                                      save_path,
                                      dataset=dataset)

    # Add this method inside the BaseDataset class. It's a more advanced replacement for retrieve_multi_data
    def retrieve_lsh_data(self, idx, m, n, p, cur_ego_pose_flag=True):
        """
        Retrieves Long-Short History (LSH) data for all agents.

        For each required historical frame of the ego-vehicle, this function
        assembles a corresponding "snapshot" of data from all other agents,
        respecting their individual, calculated delays.

        Parameters
        ----------
        idx : int
            The dataset index, corresponding to the ego-vehicle's current timestamp.
        m : int, n : int, p : int
            Parameters for Long-Short History.
        cur_ego_pose_flag : bool
            Flag to use the current ego pose for all transformations.

        Returns
        -------
        output_data_dicts : list
            A list of OrderedDicts. Each OrderedDict is a "snapshot" containing
            the base data for all agents at a specific historical time.

        timestamp_index: int
            The timestamp index of the current frame for the ego vehicle.
        """
        # 1. Find scenario and ego's current timestamp index (relative to scenario)
        scenario_index = 0
        for i, ele in enumerate(self.len_record):
            if idx < ele:
                scenario_index = i
                break

        timestamp_index = idx if scenario_index == 0 else \
            idx - self.len_record[scenario_index - 1]

        scenario_database = self.scenario_database[scenario_index]
        max_len = len(list(list(scenario_database.values())[0].items())) - 2  # -2 to be safe

        # 2. Get the full data for the CURRENT timestamp to calculate delays for all agents
        current_data_dict, _, current_timestamp_key = self.retrieve_base_data(idx, cur_ego_pose_flag)

        # Store the calculated delay (in frames) for each agent
        agent_delays = {cav_id: content['time_delay'] for cav_id, content in current_data_dict.items()}

        # 3. Generate the target historical indices FOR THE EGO VEHICLE
        ego_target_indices = self._generate_lsh_indices(timestamp_index, m, n, p, max_len)

        # Find ego content, which is needed for transformations
        ego_content = None
        for cav_content in scenario_database.values():
            if cav_content['ego']:
                ego_content = cav_content
                break

        # 4. Loop through each of the ego's target frames and build a snapshot
        output_data_dicts = []
        for target_ego_idx in ego_target_indices:
            snapshot_data = OrderedDict()
            # The offset of this historical frame from the present
            offset = timestamp_index - target_ego_idx

            # Loop through every agent in the scenario to get its corresponding data
            for cav_id, cav_content in scenario_database.items():
                delay = agent_delays[cav_id]

                # Calculate the agent's target index for this snapshot
                # Start from its delayed time, and apply the same offset
                agent_delayed_start_idx = timestamp_index - delay
                target_agent_idx = agent_delayed_start_idx - offset

                # Boundary check: clamp to the first frame if index is too small
                target_agent_idx = max(0, target_agent_idx)

                # Get the actual data for the agent at its calculated historical index
                timestamp_key_delay = self.return_timestamp_key(scenario_database, target_agent_idx)

                reformed_params = self.reform_param(cav_content,
                                                    ego_content,
                                                    current_timestamp_key,  # Ego pose is ALWAYS current
                                                    timestamp_key_delay,  # Agent data is from its past
                                                    cur_ego_pose_flag)

                lidar_np = pcd_utils.pcd_to_np(cav_content[timestamp_key_delay]['lidar'])

                # Assemble the single agent's data for this snapshot
                snapshot_data[cav_id] = OrderedDict()
                snapshot_data[cav_id]['ego'] = cav_content['ego']
                snapshot_data[cav_id]['params'] = reformed_params
                snapshot_data[cav_id]['lidar_np'] = lidar_np
                # We don't need to add time_delay here as it's already been applied

            output_data_dicts.append(snapshot_data)

        return output_data_dicts, agent_delays, ego_target_indices

    @staticmethod
    def __generate_lsh_inference(start_index, m, n, p, max_len):
        """
            Generates a unique, sorted list of indices for long and short term history.

            Parameters
            ----------
            start_index : int
                The starting timestamp index (e.g., current time or delayed time).
            m : int
                Number of long-term history frames.
            n : int
                Number of short-term history frames.
            p : int
                Period/interval for long-term frames.
            max_len : int
                The maximum possible length of the scenario to prevent out-of-bounds.
                (Although we clamp at 0, this could be used for further checks).

            Returns
            -------
            list
                A list of unique, valid timestamp indices, sorted in descending order.
        """
        # Generate short-term indices: [t, t-1, t-2, ..., t-(n-1)]
        short_indices = [start_index - i for i in range(n)]

        # Generate long-term indices: [t, t-p, t-2p, ..., t-(m-1)p]
        long_indices = [start_index - i * p for i in range(m)]

        # Combine, get unique indices, and filter out any negative indices
        combined_indices = list(set(short_indices + long_indices))
        valid_indices = [i for i in combined_indices if i >= 0]

        # Sort in descending order to have the latest timestamp first
        valid_indices.sort(reverse=True)

        return valid_indices

    @staticmethod
    def _generate_lsh_indices(start_index, m, n, p, max_len):
        """
        Generates a unique, sorted list of indices for long and short term history.

        Parameters
        ----------
        start_index : int
            The starting timestamp index (e.g., current time or delayed time).
        m : int
            Number of long-term history frames.
        n : int
            Number of short-term history frames.
        p : int
            Period/interval for long-term frames.
        max_len : int
            The maximum possible length of the scenario to prevent out-of-bounds.
            (Although we clamp at 0, this could be used for further checks).

        Returns
        -------
        list
            A list of unique, valid timestamp indices, sorted in descending order.
        """
        # Generate short-term indices: [t, t-1, t-2, ..., t-(n-1)]
        short_indices = [start_index - i for i in range(n)]

        # Generate long-term indices: [t, t-p, t-2p, ..., t-(m-1)p]
        long_indices = [start_index - i * p for i in range(m)]

        # Combine, get unique indices, and filter out any negative indices
        combined_indices = list(set(short_indices + long_indices))
        valid_indices = [i for i in combined_indices if i >= 0]

        # Sort in descending order to have the latest timestamp first
        valid_indices.sort(reverse=True)

        return valid_indices