import torch
import MinkowskiEngine as ME
from uncertainty.models.augmenter import Augmenter
from uncertainty.models.misc import compute_scene_size


def collate_fn_simple(batch):
    return batch[0]


def collate_fn(batch, complete_scale):
    in_feats = []
    in_coords = []
    min_Cs = []
    max_Cs = []
    semantic_labels = []
    instance_labels = []
    semantic_label_origins = []
    instance_label_origins = []
    mask_labels = []
    mask_label_origins = []
    Ts = []
    geo_labels = {}
    sem_labels = {}
    frame_ids = []
    sequences = []
    input_pcd_instance_labels = []
    xyzs = []

    scales = [1, 2, 4]
    for scale in scales:       
        geo_labels['1_{}'.format(scale)] = []
        sem_labels['1_{}'.format(scale)] = []
        
    for idx, input_dict in enumerate(batch):
        frame_id = input_dict['frame_id']
        sequence = input_dict['sequence']
        in_feat = input_dict['in_feat']
        in_coord = input_dict['in_coord']
        # print("collate", in_coord.max(0))
        min_C = input_dict['min_C']
        max_C = input_dict['max_C']
        T = input_dict['T']
        semantic_label = input_dict['semantic_label']
        instance_label = input_dict['instance_label']
        mask_label = input_dict['mask_label']
        semantic_label_origin = input_dict['semantic_label_origin']
        instance_label_origin = input_dict['instance_label_origin']
        mask_label_origin = input_dict['mask_label_origin']
        input_pcd_instance_label = input_dict['input_pcd_instance_label']
        
        for scale in scales:       
            geo_labels['1_{}'.format(scale)].append(input_dict['geo_labels']["1_{}".format(scale)])
            sem_labels['1_{}'.format(scale)].append(input_dict['sem_labels']["1_{}".format(scale)])
        
        in_feats.append(in_feat)
        in_coords.append(in_coord)
        Ts.append(T)
        sequences.append(sequence)
        frame_ids.append(frame_id)
        semantic_labels.append(semantic_label)
        instance_labels.append(instance_label)
        semantic_label_origins.append(semantic_label_origin)
        instance_label_origins.append(instance_label_origin)
        mask_labels.append(mask_label)
        mask_label_origins.append(mask_label_origin)
        min_Cs.append(min_C)
        max_Cs.append(max_C)
        input_pcd_instance_labels.append(input_pcd_instance_label)
        xyzs.append(input_dict['xyz'])
        
 
    # The smaller scale is 8 so the dimension should divide 8
    global_min_Cs = torch.min(torch.stack(min_Cs), dim=0)[0]
    global_max_Cs = torch.max(torch.stack(max_Cs), dim=0)[0]
    combine_scene_size = compute_scene_size(global_min_Cs, global_max_Cs, scale=complete_scale)
    global_max_Cs = global_min_Cs + combine_scene_size - 1 # inclusive coords
    semantic_label_origin = torch.stack(semantic_label_origins)
    instance_label_origin = torch.stack(instance_label_origins)
    
    
    ret_data = {
        "frame_id": frame_ids,
        "xyz": xyzs,
        
        "geo_labels": geo_labels,
        "sem_labels": sem_labels,

        "sequence": sequences,
        
        "semantic_label": semantic_labels,
        "instance_label": instance_labels,
        "semantic_label_origin": semantic_label_origin,
        "instance_label_origin": instance_label_origin,
        
        "mask_label": mask_labels,
        "mask_label_origin": mask_label_origins,
        
        "in_coords": in_coords,
        "in_feats": in_feats,
        "input_pcd_instance_label": input_pcd_instance_labels,

        "Ts": Ts,
        "global_min_Cs": global_min_Cs,
        "global_max_Cs": global_max_Cs,
        "min_Cs": min_Cs,
        "max_Cs": max_Cs,
    }
    
    return ret_data


