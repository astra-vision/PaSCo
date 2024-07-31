from pasco.utils.helper_kitti_mayavi import draw_instance, draw_semantic, draw_uncertainty, draw_panoptic, draw_pcd
import pickle
import numpy as np
import pickle
import torch
from pasco.utils.transform_utils import *
from tqdm import tqdm
import os
from mayavi import mlab
try:
    engine = mayavi.engine
except NameError:
    from mayavi.api import Engine
    engine = Engine()
    engine.start()
from numba import jit 
import click


@jit(nopython=True)
def median_filter_3d_core(input_array, kernel_size, ops):
    output_array = np.zeros_like(input_array)
    pad_size = kernel_size // 2

    for z in range(input_array.shape[2]):
        for y in range(input_array.shape[1]):
            for x in range(input_array.shape[0]):
                if kernel_size % 2 == 0:
                    start_x = max(x, 0)
                    end_x = min(x + pad_size + 1, input_array.shape[0])
                    start_y = max(y, 0)
                    end_y = min(y + pad_size + 1, input_array.shape[1])
                    start_z = max(z, 0)
                    end_z = min(z + pad_size + 1, input_array.shape[2])
                else:
                    start_x = max(x - pad_size, 0)
                    end_x = min(x + pad_size + 1, input_array.shape[0])
                    start_y = max(y - pad_size, 0)
                    end_y = min(y + pad_size + 1, input_array.shape[1])
                    start_z = max(z - pad_size, 0)
                    end_z = min(z + pad_size + 1, input_array.shape[2])

                window = input_array[start_x:end_x, start_y:end_y, start_z:end_z]
                # print(window.shape)
                flat_window = window.flatten()
                flat_window = flat_window[flat_window != 255]
                if len(flat_window) == 0:
                    output_array[x, y, z] = 255
                else:
                    if ops == "avg":
                        output_array[x, y, z] = np.mean(flat_window)
                    elif ops == "max":
                        output_array[x, y, z] = np.max(flat_window)
                    elif ops == "median":
                        output_array[x, y, z] = np.median(flat_window)
    return output_array

def median_filter_3d(input_tensor, kernel_size=3, ops="avg"):
    input_array = input_tensor.numpy()
    output_array = median_filter_3d_core(input_array, kernel_size, ops)
    output_tensor = torch.from_numpy(output_array)
    return output_tensor


def visualize_mask(pred_panoptic_seg, pred_segments_info, min_C, T, 
                       figure=None,
                       filename=None, 
                       ssc=None,
                       set_stuff_color=False,
                       highlight_instance=False):
    cnt = 0
    thing_id = 0
    pred_panoptic_seg_n = pred_panoptic_seg.copy()

  
    pred_panoptic_seg_n = torch.from_numpy(pred_panoptic_seg_n)
    from_feature, from_coords = sample_scene(min_C, T, pred_panoptic_seg_n.unsqueeze(0), (256, 256, 32))
    thing_features =[]
    thing_coords = []
   
    for seg in pred_segments_info:
        id = seg["id"]
        mask = (from_feature == id).squeeze()
        thing_id += 1
        thing_features.append(torch.full((from_feature[mask, :].shape[0],1), thing_id))
        
        thing_coords.append(from_coords[mask, :])

    if len(thing_features) > 0:
        thing_features = torch.cat(thing_features, dim=0).long().squeeze().numpy()
        thing_coords = torch.cat(thing_coords, dim=0).long().numpy()

    
    
    draw_instance(thing_coords, thing_features, filename=filename, figure=figure)
   



def visualize_panoptic(pred_panoptic_seg, pred_segments_info, min_C, T, 
                       figure=None,
                       filename=None, 
                       ssc=None,
                       highlight_instance=False):
    cnt = 0
    thing_id = 0
    pred_panoptic_seg_n = pred_panoptic_seg.copy()
  
    pred_panoptic_seg_n = torch.from_numpy(pred_panoptic_seg_n)
    from_feature, from_coords = sample_scene(min_C, T, pred_panoptic_seg_n.unsqueeze(0), (256, 256, 32))
    thing_features =[]
    thing_coords = []
    stuff_features = []
    stuff_coords = []
   
    for seg in pred_segments_info:
        id = seg["id"]
        mask = (from_feature == id).squeeze()
        if seg['isthing']:
            thing_id += 1
            thing_features.append(torch.full((from_feature[mask, :].shape[0],1), thing_id))
            
            thing_coords.append(from_coords[mask, :])
        else:
            stuff_features.append(torch.full((from_feature[mask, :].shape[0],1), seg['category_id']))
            stuff_coords.append(from_coords[mask, :])
    if len(thing_features) > 0:
        thing_features = torch.cat(thing_features, dim=0).long().squeeze().numpy()
        thing_coords = torch.cat(thing_coords, dim=0).long().numpy()

        
    if ssc is not None:
        ssc = ssc.squeeze().long()
        stuff_coords = torch.nonzero((ssc > 8) & (ssc < 20) & (ssc != 0)).long().numpy()
        stuff_features = ssc[stuff_coords[:, 0], stuff_coords[:, 1], stuff_coords[:, 2]]
    else:
        stuff_features = torch.cat(stuff_features, dim=0).long().squeeze().numpy()
        stuff_coords = torch.cat(stuff_coords, dim=0).long().numpy()
    
    if highlight_instance:
        draw_instance(thing_coords, thing_features, stuff_coords, stuff_features)
    else:
        draw_panoptic(thing_coords, thing_features, stuff_coords, stuff_features, filename=filename, figure=figure)
    

def visualize_semantic(ssc_pred, min_C, T, filename=None, scale=1, figure=None):
    from_coords =np.argwhere(ssc_pred != 0)
    from_feature = ssc_pred[from_coords[:, 0], from_coords[:, 1], from_coords[:, 2]]
    draw_semantic(from_coords, from_feature, filename=filename, voxel_size=0.2 * scale, figure=figure)


def visualize_uncertainty(confidence, min_C, T, filename=None, figure=None, vrange=(0, 1)):
    from_feature, from_coords = sample_scene(min_C, T, confidence, (256, 256, 32))
    draw_uncertainty(from_coords.numpy(), from_feature.squeeze().numpy(), 
                     vrange=vrange,
                     figure=figure, filename=filename)
    

def majority_pooling(grid, k_size=2):
  result = np.zeros((grid.shape[0] // k_size, grid.shape[1] // k_size, grid.shape[2] // k_size))
  for xx in range(0, int(np.floor(grid.shape[0]/k_size))):
    for yy in range(0, int(np.floor(grid.shape[1]/k_size))):
      for zz in range(0, int(np.floor(grid.shape[2]/k_size))):

        sub_m = grid[(xx*k_size):(xx*k_size)+k_size, (yy*k_size):(yy*k_size)+k_size, (zz*k_size):(zz*k_size)+k_size]
        unique, counts = np.unique(sub_m, return_counts=True)
        if True in ((unique != 0) & (unique != 255)):
          # Remove counts with 0 and 255
          counts = counts[((unique != 0) & (unique != 255))]
          unique = unique[((unique != 0) & (unique != 255))]
        else:
          if True in (unique == 0):
            counts = counts[(unique != 255)]
            unique = unique[(unique != 255)]
        value = unique[np.argmax(counts)]
        result[xx, yy, zz] = value
  return result


@click.command()
@click.option('--start_frame_id', default=0)
@click.option('--end_frame_id', default=4080)
@click.option('--save_folder', default="output/images")
@click.option('--draw_conf', default=True)
@click.option('--draw_mask', default=True)
@click.option('--is_draw_panoptic', default=True)
@click.option('--is_draw_multi_scale_sem', default=True)
@click.option('--is_draw_all_subnets', default=False)
@click.option('--is_draw_input', default=True)
def main(start_frame_id, end_frame_id, save_folder, 
         draw_conf, draw_mask, is_draw_panoptic, is_draw_multi_scale_sem, is_draw_all_subnets, is_draw_input):   
     
    model_names = ["pasco_single"]
    frame_ids = list(range(start_frame_id, end_frame_id, 5))
    
    
    figure = mlab.figure(size=(1400, 1400), bgcolor=(1, 1, 1), engine=engine)
    

    frame_ids = [ '{:06d}'.format(int(number)) for number in frame_ids]

    for conf_post_process in ["median"]:
            
        os.makedirs(save_folder, exist_ok=True)
        
        for i_subnet in [1]: 
            for frame_id in tqdm(frame_ids):
                predictions = []
                vmins = []
                vmaxs = []
                ins_vmins, ins_vmaxs = [], []
                instance_confidence_denses = []
                vox_confidence_denses = []
                xyz = []
                pred_panoptic_segs, pred_segments_infos = [], []
                for model_name in model_names:
                    filepath = "output/{}/{}_{}.pkl".format(model_name, frame_id, i_subnet)
                    
                    with open(filepath, 'rb') as handle:
                        prediction = pickle.load(handle)
                        predictions.append(prediction)
                    ssc_pred = torch.from_numpy(prediction["ssc_pred"].astype(int)).reshape(1, 256, 256, 32)
                    vox_confidence_dense = torch.from_numpy(prediction["vox_confidence_denses"])
                    vmins.append(torch.min(vox_confidence_dense[ssc_pred == 0]))
                    vmaxs.append(torch.max(vox_confidence_dense[ssc_pred == 0]))
                    
                    pred_panoptic_seg = prediction["pred_panoptic_seg"].squeeze()
                    pred_segments_info = prediction["pred_segments_info"][0]    
                    pred_panoptic_segs.append(pred_panoptic_seg)
                    pred_segments_infos.append(pred_segments_info)
                    
                    instance_confidence_dense = torch.zeros(1, 256, 256, 32).float()
                    pred_panoptic_seg = torch.from_numpy(pred_panoptic_seg).unsqueeze(0)
                    for segment_info in pred_segments_info:
                        if segment_info["isthing"]:
                            instance_confidence_dense[pred_panoptic_seg == segment_info["id"]] = segment_info["confidence"]
                            ins_vmins.append(segment_info["confidence"])
                            ins_vmaxs.append(segment_info["confidence"])
                    instance_confidence_denses.append(instance_confidence_dense)
                    
                    xyz.append(prediction["xyz"])
                    
                
                for i in range(len(predictions)):                                                    
                    prediction = predictions[i]
                    instance_confidence_dense = instance_confidence_denses[i]
                    method = model_names[i]
                    pred_panoptic_seg = pred_panoptic_segs[i]
                    pred_segments_info = pred_segments_infos[i]
                    
                    if is_draw_input:
                        input_filename = "{}/input_{}.png".format(save_folder, frame_id)
                        draw_pcd(xyz[i], filename=input_filename, figure=figure)
                    
                    ssc_pred = prediction["ssc_pred"].astype(int).reshape(256, 256, 32)
                    semantic_label_origin = prediction["semantic_label_origin"].astype(int)
                    
                    min_C = torch.zeros(3)
                    T = torch.eye(4)
                    
                    scales = [1]
                    if is_draw_multi_scale_sem:
                        scales = [1, 2, 4]
                    ssc_pred_t = ssc_pred.copy()
                    for scale in scales:
                        if scale != 1:
                            ssc_pred_t = majority_pooling(ssc_pred_t, k_size=scale)
                            semantic_label_origin = majority_pooling(semantic_label_origin, k_size=scale)
                        
                        sem_filename = "{}/{}_sem_{}_{}_{}.png".format(save_folder, method, frame_id, scale, i_subnet)
                        sem_gt_filename = "{}/{}_sem_gt_{}_{}_{}.png".format(save_folder, method, frame_id, scale, i_subnet)     
                        visualize_semantic(ssc_pred_t, min_C, T, filename=sem_filename, scale=scale, figure=figure)
                        
                        semantic_label_origin[semantic_label_origin == 255] = 0
                        visualize_semantic(semantic_label_origin, min_C, T, filename=sem_gt_filename, scale=scale, figure=figure)
                   
                    if is_draw_panoptic:
                        panop_filename = "{}/{}_panop_pred_{}_{}_{}.png".format(save_folder, method, frame_id, scale, i_subnet)
                        ssc_pred = torch.from_numpy(ssc_pred)
                        visualize_panoptic(pred_panoptic_seg, pred_segments_info, min_C, T, filename=panop_filename, figure=figure, ssc=ssc_pred)
                    
                    if draw_mask:
                        mask_filename = "{}/{}_mask_pred_{}_{}_{}.png".format(save_folder, method, frame_id, scale, i_subnet)
                        visualize_mask(pred_panoptic_seg, pred_segments_info, min_C, T, filename=mask_filename, figure=figure)

                    if draw_conf:
                        if "vox_confidence_denses" in prediction:
                            vox_conf_filename = "{}/{}_vox_conf_{}_{}_{}.png".format(save_folder, method, frame_id, scale, i_subnet)
                            vox_confidence_denses = torch.from_numpy(prediction["vox_confidence_denses"])
                            # ssc_pred = torch.from_numpy(ssc_pred).unsqueeze(0)
                            ssc_pred = ssc_pred.unsqueeze(0)
                            vox_confidence_denses[ssc_pred == 0] = 255
                            
                            if conf_post_process != "raw":
                                vox_confidence_denses = median_filter_3d(vox_confidence_denses, kernel_size=3, ops=conf_post_process)
                            vox_confidence_denses[ssc_pred == 0] = 255
                            
                            visualize_uncertainty(vox_confidence_denses, min_C, T,
                                            filename=vox_conf_filename,
                                            figure=figure)
                        
                            ins_conf_filename = "{}/{}_ins_conf_{}_{}_{}.png".format(save_folder, method, frame_id, scale, i_subnet)
                            instance_confidence_denses = torch.zeros(1, 256, 256, 32).float()
                            pred_panoptic_seg = torch.from_numpy(pred_panoptic_seg).unsqueeze(0)
                            for segment_info in pred_segments_info:
                                if segment_info["isthing"]:
                                    instance_confidence_denses[pred_panoptic_seg == segment_info["id"]] = segment_info["confidence"]
                            visualize_uncertainty(instance_confidence_denses, min_C, T,
                                            filename=ins_conf_filename,
                                            figure=figure)
                        
                    
if __name__ == "__main__":
    main()