import os
import os.path
import numpy as np
import torch
import pandas
from collections import OrderedDict
from .base_video_dataset import BaseVideoDataset
from lib.train.data import jpeg4py_loader
from lib.train.admin import env_settings


class UAV123(BaseVideoDataset):
    """UAV123 dataset for training.

    Publication:
        A Benchmark and Simulator for UAV Tracking
        Matthias Mueller, Neil Smith and Bernard Ghanem
        ECCV, 2016

    Structure expected under root:
        {root}/data_seq/UAV123/{seq_folder}/{frame:06d}.jpg
        {root}/anno/UAV123/{seq_name}.txt   (comma-separated x,y,w,h per line)

    Note: UAV123 splits some long videos into sub-sequences (e.g. car1_1, car1_2).
    Each sub-sequence entry stores its startFrame so frame indexing is correct.
    """

    def __init__(self, root=None, image_loader=jpeg4py_loader):
        root = env_settings().uav_dir if root is None else root
        super().__init__('UAV123', root, image_loader)
        self.sequence_list = self._build_sequence_list()

    def get_name(self):
        return 'uav123'

    def _build_sequence_list(self):
        # Each entry: (seq_name, folder_name, start_frame, anno_file)
        # start_frame is the physical frame number (1-indexed) that maps to index 0.
        # anno files contain only the frames for that sub-segment.
        return [
            ('bike1',       'bike1',    1,    'anno/UAV123/bike1.txt'),
            ('bike2',       'bike2',    1,    'anno/UAV123/bike2.txt'),
            ('bike3',       'bike3',    1,    'anno/UAV123/bike3.txt'),
            ('bird1_1',     'bird1',    1,    'anno/UAV123/bird1_1.txt'),
            ('bird1_2',     'bird1',    775,  'anno/UAV123/bird1_2.txt'),
            ('bird1_3',     'bird1',    1573, 'anno/UAV123/bird1_3.txt'),
            ('boat1',       'boat1',    1,    'anno/UAV123/boat1.txt'),
            ('boat2',       'boat2',    1,    'anno/UAV123/boat2.txt'),
            ('boat3',       'boat3',    1,    'anno/UAV123/boat3.txt'),
            ('boat4',       'boat4',    1,    'anno/UAV123/boat4.txt'),
            ('boat5',       'boat5',    1,    'anno/UAV123/boat5.txt'),
            ('boat6',       'boat6',    1,    'anno/UAV123/boat6.txt'),
            ('boat7',       'boat7',    1,    'anno/UAV123/boat7.txt'),
            ('boat8',       'boat8',    1,    'anno/UAV123/boat8.txt'),
            ('boat9',       'boat9',    1,    'anno/UAV123/boat9.txt'),
            ('building1',   'building1',1,    'anno/UAV123/building1.txt'),
            ('building2',   'building2',1,    'anno/UAV123/building2.txt'),
            ('building3',   'building3',1,    'anno/UAV123/building3.txt'),
            ('building4',   'building4',1,    'anno/UAV123/building4.txt'),
            ('building5',   'building5',1,    'anno/UAV123/building5.txt'),
            ('car1_1',      'car1',     1,    'anno/UAV123/car1_1.txt'),
            ('car1_2',      'car1',     751,  'anno/UAV123/car1_2.txt'),
            ('car1_3',      'car1',     1627, 'anno/UAV123/car1_3.txt'),
            ('car2',        'car2',     1,    'anno/UAV123/car2.txt'),
            ('car3',        'car3',     1,    'anno/UAV123/car3.txt'),
            ('car4',        'car4',     1,    'anno/UAV123/car4.txt'),
            ('car5',        'car5',     1,    'anno/UAV123/car5.txt'),
            ('car6_1',      'car6',     1,    'anno/UAV123/car6_1.txt'),
            ('car6_2',      'car6',     487,  'anno/UAV123/car6_2.txt'),
            ('car6_3',      'car6',     1807, 'anno/UAV123/car6_3.txt'),
            ('car6_4',      'car6',     2953, 'anno/UAV123/car6_4.txt'),
            ('car6_5',      'car6',     3925, 'anno/UAV123/car6_5.txt'),
            ('car7',        'car7',     1,    'anno/UAV123/car7.txt'),
            ('car8_1',      'car8',     1,    'anno/UAV123/car8_1.txt'),
            ('car8_2',      'car8',     1357, 'anno/UAV123/car8_2.txt'),
            ('car9',        'car9',     1,    'anno/UAV123/car9.txt'),
            ('car10',       'car10',    1,    'anno/UAV123/car10.txt'),
            ('car11',       'car11',    1,    'anno/UAV123/car11.txt'),
            ('car12',       'car12',    1,    'anno/UAV123/car12.txt'),
            ('car13',       'car13',    1,    'anno/UAV123/car13.txt'),
            ('car14',       'car14',    1,    'anno/UAV123/car14.txt'),
            ('car15',       'car15',    1,    'anno/UAV123/car15.txt'),
            ('car16_1',     'car16',    1,    'anno/UAV123/car16_1.txt'),
            ('car16_2',     'car16',    415,  'anno/UAV123/car16_2.txt'),
            ('car17',       'car17',    1,    'anno/UAV123/car17.txt'),
            ('car18',       'car18',    1,    'anno/UAV123/car18.txt'),
            ('car1_s',      'car1_s',   1,    'anno/UAV123/car1_s.txt'),
            ('car2_s',      'car2_s',   1,    'anno/UAV123/car2_s.txt'),
            ('car3_s',      'car3_s',   1,    'anno/UAV123/car3_s.txt'),
            ('car4_s',      'car4_s',   1,    'anno/UAV123/car4_s.txt'),
            ('group1_1',    'group1',   1,    'anno/UAV123/group1_1.txt'),
            ('group1_2',    'group1',   1333, 'anno/UAV123/group1_2.txt'),
            ('group1_3',    'group1',   2515, 'anno/UAV123/group1_3.txt'),
            ('group1_4',    'group1',   3925, 'anno/UAV123/group1_4.txt'),
            ('group2_1',    'group2',   1,    'anno/UAV123/group2_1.txt'),
            ('group2_2',    'group2',   907,  'anno/UAV123/group2_2.txt'),
            ('group2_3',    'group2',   1771, 'anno/UAV123/group2_3.txt'),
            ('group3_1',    'group3',   1,    'anno/UAV123/group3_1.txt'),
            ('group3_2',    'group3',   1567, 'anno/UAV123/group3_2.txt'),
            ('group3_3',    'group3',   2827, 'anno/UAV123/group3_3.txt'),
            ('group3_4',    'group3',   4369, 'anno/UAV123/group3_4.txt'),
            ('person1',     'person1',  1,    'anno/UAV123/person1.txt'),
            ('person2_1',   'person2',  1,    'anno/UAV123/person2_1.txt'),
            ('person2_2',   'person2',  1189, 'anno/UAV123/person2_2.txt'),
            ('person3',     'person3',  1,    'anno/UAV123/person3.txt'),
            ('person4_1',   'person4',  1,    'anno/UAV123/person4_1.txt'),
            ('person4_2',   'person4',  1501, 'anno/UAV123/person4_2.txt'),
            ('person5_1',   'person5',  1,    'anno/UAV123/person5_1.txt'),
            ('person5_2',   'person5',  877,  'anno/UAV123/person5_2.txt'),
            ('person6',     'person6',  1,    'anno/UAV123/person6.txt'),
            ('person7_1',   'person7',  1,    'anno/UAV123/person7_1.txt'),
            ('person7_2',   'person7',  1249, 'anno/UAV123/person7_2.txt'),
            ('person8_1',   'person8',  1,    'anno/UAV123/person8_1.txt'),
            ('person8_2',   'person8',  1075, 'anno/UAV123/person8_2.txt'),
            ('person9',     'person9',  1,    'anno/UAV123/person9.txt'),
            ('person10',    'person10', 1,    'anno/UAV123/person10.txt'),
            ('person11',    'person11', 1,    'anno/UAV123/person11.txt'),
            ('person12_1',  'person12', 1,    'anno/UAV123/person12_1.txt'),
            ('person12_2',  'person12', 601,  'anno/UAV123/person12_2.txt'),
            ('person13',    'person13', 1,    'anno/UAV123/person13.txt'),
            ('person14_1',  'person14', 1,    'anno/UAV123/person14_1.txt'),
            ('person14_2',  'person14', 847,  'anno/UAV123/person14_2.txt'),
            ('person14_3',  'person14', 1813, 'anno/UAV123/person14_3.txt'),
            ('person15',    'person15', 1,    'anno/UAV123/person15.txt'),
            ('person16',    'person16', 1,    'anno/UAV123/person16.txt'),
            ('person17_1',  'person17', 1,    'anno/UAV123/person17_1.txt'),
            ('person17_2',  'person17', 1501, 'anno/UAV123/person17_2.txt'),
            ('person18',    'person18', 1,    'anno/UAV123/person18.txt'),
            ('person19_1',  'person19', 1,    'anno/UAV123/person19_1.txt'),
            ('person19_2',  'person19', 1243, 'anno/UAV123/person19_2.txt'),
            ('person19_3',  'person19', 2791, 'anno/UAV123/person19_3.txt'),
            ('person20',    'person20', 1,    'anno/UAV123/person20.txt'),
            ('person21',    'person21', 1,    'anno/UAV123/person21.txt'),
            ('person22',    'person22', 1,    'anno/UAV123/person22.txt'),
            ('person23',    'person23', 1,    'anno/UAV123/person23.txt'),
            ('person1_s',   'person1_s',1,    'anno/UAV123/person1_s.txt'),
            ('person2_s',   'person2_s',1,    'anno/UAV123/person2_s.txt'),
            ('person3_s',   'person3_s',1,    'anno/UAV123/person3_s.txt'),
            ('truck1',      'truck1',   1,    'anno/UAV123/truck1.txt'),
            ('truck2',      'truck2',   1,    'anno/UAV123/truck2.txt'),
            ('truck3',      'truck3',   1,    'anno/UAV123/truck3.txt'),
            ('truck4_1',    'truck4',   1,    'anno/UAV123/truck4_1.txt'),
            ('truck4_2',    'truck4',   577,  'anno/UAV123/truck4_2.txt'),
            ('uav1_1',      'uav1',     1,    'anno/UAV123/uav1_1.txt'),
            ('uav1_2',      'uav1',     1555, 'anno/UAV123/uav1_2.txt'),
            ('uav1_3',      'uav1',     2473, 'anno/UAV123/uav1_3.txt'),
            ('uav2',        'uav2',     1,    'anno/UAV123/uav2.txt'),
            ('uav3',        'uav3',     1,    'anno/UAV123/uav3.txt'),
            ('uav4',        'uav4',     1,    'anno/UAV123/uav4.txt'),
            ('uav5',        'uav5',     1,    'anno/UAV123/uav5.txt'),
            ('uav6',        'uav6',     1,    'anno/UAV123/uav6.txt'),
            ('uav7',        'uav7',     1,    'anno/UAV123/uav7.txt'),
            ('uav8',        'uav8',     1,    'anno/UAV123/uav8.txt'),
            ('wakeboard1',  'wakeboard1',1,   'anno/UAV123/wakeboard1.txt'),
            ('wakeboard2',  'wakeboard2',1,   'anno/UAV123/wakeboard2.txt'),
            ('wakeboard3',  'wakeboard3',1,   'anno/UAV123/wakeboard3.txt'),
            ('wakeboard4',  'wakeboard4',1,   'anno/UAV123/wakeboard4.txt'),
            ('wakeboard5',  'wakeboard5',1,   'anno/UAV123/wakeboard5.txt'),
            ('wakeboard6',  'wakeboard6',1,   'anno/UAV123/wakeboard6.txt'),
            ('wakeboard7',  'wakeboard7',1,   'anno/UAV123/wakeboard7.txt'),
            ('wakeboard8',  'wakeboard8',1,   'anno/UAV123/wakeboard8.txt'),
            ('wakeboard9',  'wakeboard9',1,   'anno/UAV123/wakeboard9.txt'),
            ('wakeboard10', 'wakeboard10',1,  'anno/UAV123/wakeboard10.txt'),
        ]

    def get_num_sequences(self):
        return len(self.sequence_list)

    def _read_bb_anno(self, anno_path):
        gt = pandas.read_csv(anno_path, delimiter=',', header=None,
                             dtype=np.float32, na_filter=False, low_memory=False).values
        return torch.tensor(gt)

    def get_sequence_info(self, seq_id):
        _, _, _, anno_rel = self.sequence_list[seq_id]
        anno_path = os.path.join(self.root, anno_rel)
        bbox = self._read_bb_anno(anno_path)
        valid = (bbox[:, 2] > 0) & (bbox[:, 3] > 0)
        visible = valid.byte()
        return {'bbox': bbox, 'valid': valid, 'visible': visible}

    def _get_frame_path(self, folder_name, start_frame, frame_id):
        # frame_id is 0-indexed relative to sub-sequence start
        physical_frame = start_frame + frame_id
        return os.path.join(self.root, 'data_seq', 'UAV123', folder_name,
                            '{:06d}.jpg'.format(physical_frame))

    def _get_frame(self, folder_name, start_frame, frame_id):
        return self.image_loader(self._get_frame_path(folder_name, start_frame, frame_id))

    def get_frames(self, seq_id, frame_ids, anno=None):
        _, folder_name, start_frame, _ = self.sequence_list[seq_id]
        frame_list = [self._get_frame(folder_name, start_frame, f_id) for f_id in frame_ids]

        if anno is None:
            anno = self.get_sequence_info(seq_id)

        anno_frames = {key: [value[f_id, ...].clone() for f_id in frame_ids]
                       for key, value in anno.items()}

        object_meta = OrderedDict({'object_class_name': None,
                                   'motion_class': None,
                                   'major_class': None,
                                   'root_class': None,
                                   'motion_adverb': None})
        return frame_list, anno_frames, object_meta
