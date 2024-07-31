import pickle
import torch
from pasco.visual.helper_kitti import *


def convert_panoptic_to_semantic_output(panoptic_seg, segments_info):
    """
    panoptic_seg: [256, 256, 32]
    segments_info: [{'id': 1, 'isthing': False, 'category_id': 9}, 
                    {'id': 2, 'isthing': True, 'category_id': 6}, ...]
    """
    # pdb.set_trace()
    semseg = torch.zeros_like(panoptic_seg)
    for segment in segments_info:
        semseg[panoptic_seg == segment['id']] = segment['category_id']
    return semseg

def convert_panoptic_to_instance_output(panoptic_seg, segments_info):
    """
    panoptic_seg: [256, 256, 32]
    segments_info: [{'id': 1, 'isthing': False, 'category_id': 9}, 
                    {'id': 2, 'isthing': True, 'category_id': 6}, ...]
    """
    # pdb.set_trace()
    instance_seg = torch.zeros_like(panoptic_seg)
    for segment in segments_info:
        if segment['isthing']:
            instance_seg[panoptic_seg == segment['id']] = segment['id']
    return instance_seg


if __name__ == "__main__":
    with open("/home/acao/jeanzay/scratch/uncertainty/000875.pkl", 'rb') as handle:
        data = pickle.load(handle)
    
    panoptic_seg = data['pred_panoptic_seg']
    segments_info = data['pred_segments_info']
    valid_mask = data['valid_mask']
    
    semseg = convert_panoptic_to_semantic_output(panoptic_seg, segments_info)
    semseg[~valid_mask] = 0
    semseg = semseg.detach().cpu().numpy()
    # semseg[semseg!=6] = 0
    # draw_semantic(semseg)

    semantic_label = data['semantic_label']
    semantic_label[~valid_mask] = 0
    semantic_label = semantic_label.detach().cpu().numpy()
    # semantic_label[semantic_label!=6] = 0
    # draw_semantic(semantic_label)

    instance_seg = convert_panoptic_to_instance_output(panoptic_seg, segments_info)
    instance_seg[~valid_mask] = 0
    instance_seg = instance_seg.detach().cpu().numpy()
    # draw_instance(instance_seg)

    instance_label = data['instance_label']
    instance_label[~valid_mask] = 0
    instance_label = instance_label.detach().cpu().numpy()
    draw_instance(instance_label)