from __future__ import absolute_import

import torch
import time
import cv2
import mc
import numpy as np
import os.path as osp

def get_time():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

def read_image_mc(file_path,
    server_list_config_file = '/mnt/lustre/share/memcached_client/server_list.conf',
    client_config_file = '/mnt/lustre/share/memcached_client/client.conf'):
    mclient = mc.MemcachedClient.GetInstance(server_list_config_file, client_config_file)
    value = mc.pyvector()
    mclient.Get(file_path, value)
    value_str = mc.ConvertString(value)
    img_array = np.fromstring(value_str, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    return img

def load_model(model, path):
    pretrained_dict = torch.load(path)
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
