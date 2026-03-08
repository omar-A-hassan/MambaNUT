import os
import os.path
import numpy as np
import torch
import pandas
from collections import OrderedDict
from .base_video_dataset import BaseVideoDataset
from lib.train.data import jpeg4py_loader
from lib.train.admin import env_settings


class UAV20L(BaseVideoDataset):
    """UAV20L long-term tracking dataset for training.

    UAV20L is a subset of 20 long-term sequences from UAV123.
    It shares video frames with UAV123 but provides its own annotation files.

    Structure expected under root (same root as UAV123):
        {root}/data_seq/UAV123/{seq_name}/{frame:06d}.jpg  (shared with UAV123)
        {root}/anno/UAV20L/{seq_name}.txt                  (comma-separated x,y,w,h)
    """

    def __init__(self, root=None, image_loader=jpeg4py_loader):
        root = env_settings().uav_dir if root is None else root
        super().__init__('UAV20L', root, image_loader)
        self.sequence_list = self._build_sequence_list()

    def get_name(self):
        return 'uav20l'

    def _build_sequence_list(self):
        anno_dir = os.path.join(self.root, 'anno', 'UAV20L')
        seqs = sorted([f[:-4] for f in os.listdir(anno_dir) if f.endswith('.txt')])
        return seqs

    def get_num_sequences(self):
        return len(self.sequence_list)

    def _read_bb_anno(self, seq_name):
        anno_path = os.path.join(self.root, 'anno', 'UAV20L', seq_name + '.txt')
        gt = pandas.read_csv(anno_path, delimiter=',', header=None,
                             dtype=np.float32, na_filter=False, low_memory=False).values
        return torch.tensor(gt)

    def get_sequence_info(self, seq_id):
        seq_name = self.sequence_list[seq_id]
        bbox = self._read_bb_anno(seq_name)
        valid = (bbox[:, 2] > 0) & (bbox[:, 3] > 0)
        visible = valid.byte()
        return {'bbox': bbox, 'valid': valid, 'visible': visible}

    def _get_frame_path(self, seq_name, frame_id):
        # UAV20L uses the same video folders as UAV123, frames are 1-indexed
        return os.path.join(self.root, 'data_seq', 'UAV123', seq_name,
                            '{:06d}.jpg'.format(frame_id + 1))

    def _get_frame(self, seq_name, frame_id):
        return self.image_loader(self._get_frame_path(seq_name, frame_id))

    def get_frames(self, seq_id, frame_ids, anno=None):
        seq_name = self.sequence_list[seq_id]
        frame_list = [self._get_frame(seq_name, f_id) for f_id in frame_ids]

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
