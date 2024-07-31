import pytorch_lightning as pl
import torch
from collections import defaultdict
from pasco.models.metrics import SSCMetrics, UncertaintyMetrics

from pasco.models.transformer.transformer_predictor_v2 import TransformerPredictorV2 as TransformerPredictor
from pasco.loss.matcher_sparse import HungarianMatcher
from pasco.loss.criterion_sparse import SetCriterion
import torch.nn.functional as F
import numpy as np
from pasco.loss.panoptic_quality import pq_compute_single_core, PQStat, convert_mask_label_to_panoptic_output, find_matched_segment
from pasco.data.kitti360.params import thing_ids

from pasco.models.unet3d_sparse_v2 import UNet3DV2, CylinderFeat

import MinkowskiEngine as ME
from pasco.models.helper import  panoptic_inference
from pasco.loss.losses import compute_sem_compl_loss_kitti360
from pasco.utils.torch_util import WarmupCosine
from pasco.models.ensembler import Ensembler
from pasco.models.utils import print_metrics_table_panop_ssc, print_metrics_table_uncertainty, print_metrics_table_panop_per_class
from pasco.models.augmenter import Augmenter
from pasco.models.misc import compute_scene_size, prune_outside_coords
import time


class Net_kitti360(pl.LightningModule):
    def __init__(
        self,
        n_classes,
        class_names,
        class_weights,
        encoder_dropouts,
        decoder_dropouts,
        dense3d_dropout,
        n_infers,
        class_frequencies,
        in_channels=27 + 256,
        transformer_dropout=0.0,
        lr=1e-4,
        scale=1,
        overlap_threshold=0.4,
        object_mask_threshold=0.7,
        weight_decay=1e-4,
        dec_layers=1,
        enc_layers=0,
        num_queries=100,
        aux_loss=False,
        mask_weight=20.0,
        use_se_layer=False,
        alpha=0.1,
        no_object_weight = 0.1,
        query_sample_ratio=1.0,
        occ_weight=1.0,
        f=64,
        compl_labelweights=None,
        use_voxel_query_loss=True,
        iou_threshold=0.2,
        heavy_decoder=True
    ):
        super().__init__()
        self.n_infers = n_infers
        self.lr = lr
        self.weight_decay = weight_decay
        self.n_classes = n_classes
        self.class_names = class_names
        self.object_mask_threshold = object_mask_threshold
        self.overlap_threshold = overlap_threshold
        self.scene_size = (256//scale, 256//scale, 32//scale)
        self.class_weights = class_weights
        self.compl_labelweights = compl_labelweights
        self.class_frequencies = class_frequencies
        self.iou_threshold = iou_threshold
        self.augmenter = Augmenter()
        self.save_hyperparameters()
            
        self.scale = scale
        
        
        self.sigmoid = ME.MinkowskiSigmoid()
        self.softmax = ME.MinkowskiSoftmax(dim=1)
        
        
        self.num_queries = num_queries
        self.num_queries_sampled = int(num_queries * query_sample_ratio)
     

        self.transformer_predictor = TransformerPredictor(
            dropout=transformer_dropout,
            num_classes=n_classes,
            nheads=8,
            hidden_dim=384,
            enc_layers=enc_layers,
            num_queries=self.num_queries,
            dim_feedforward=1024,
            dec_layers=dec_layers,
            aux_loss=aux_loss,
            mask_dim=f,
            n_infers=n_infers,
            query_sample_ratio=query_sample_ratio,
            in_channels=[f*4, f*2, f])
        
        
        self.unet3d = UNet3DV2(
            heavy_decoder=heavy_decoder,
            drop_path_rate=0.0,
            n_classes=n_classes,
            in_channels=f * n_infers,
            transformer_predictor=self.transformer_predictor,
            f_maps=[f, f * 2, f * 4, f * 4],
            dense3d_dropout=dense3d_dropout, 
            n_infers=n_infers,
            decoder_dropouts=decoder_dropouts, 
            num_queries=num_queries,
            query_sample_ratio=query_sample_ratio,
            encoder_dropouts=encoder_dropouts,
            use_se_layer=use_se_layer)

        
        self.ensembler = Ensembler()
        
        
        self.panop_weight = 1.0
        self.ce_weight = 2.0
        self.occ_weight = occ_weight

        
        self.sem_weight_dict = {"loss_ce": 0.3, "loss_lovasz": 1.0}
        self.dice_weight = 1.0
        self.mask_weight = mask_weight
        
        self.weight_dict = {
            "ssc_ce": 0.3, "ssc_lovasz": 1.0,
            "loss_ce": self.ce_weight, 
            "loss_mask": self.mask_weight, 
            "loss_dice": self.dice_weight}
        
        if not use_voxel_query_loss:
            self.weight_dict["ssc_ce"] = 0.0
            self.weight_dict["ssc_lovasz"] = 0.0
        
        # building criterion
        matcher = HungarianMatcher(
            cost_class=1.0,
            cost_mask=self.mask_weight,
            cost_dice=self.dice_weight,
        )

        self.feat = CylinderFeat(fea_dim=in_channels, out_pt_fea_dim=f)

        self.criterion = SetCriterion(
            n_classes,
            alpha=alpha,
            matcher=matcher,
            weight_dict=self.weight_dict,
            eos_coef=no_object_weight,
            class_weights=class_weights,
            compl_labelweights=compl_labelweights
        )

        self.thing_ids = thing_ids

        self.sync_dist = True
        self.pq_stat = {}
        self.ssc_metrics = {}
        self.uncertainty_metrics = {}
        self.steps = ["train", "val", "test"] 
        
        for step in self.steps:
            self.pq_stat[step] = {}
            self.ssc_metrics[step] = {}
            self.uncertainty_metrics[step] = {}
            for i_infer in range(self.n_infers + 1): # +1 for ensemble
                self.pq_stat[step][i_infer] = PQStat()
                self.ssc_metrics[step][i_infer] = SSCMetrics(self.n_classes)
                self.uncertainty_metrics[step][i_infer] = UncertaintyMetrics()
 
        self.uncertainty_metrics_by_thresholds = {}
        self.uncertainty_thresholds = [0.5]
        for threshold in self.uncertainty_thresholds:
                self.uncertainty_metrics_by_thresholds[threshold] = {} 
                for i_infer in range(self.n_infers + 1):
                    self.uncertainty_metrics_by_thresholds[threshold][i_infer] = UncertaintyMetrics()
        
        self.inference_times = []
        self.ensemble_times = []
        self.memories = []
        
        

    def forward(self, in_feat, 
                batch_size: int, sem_labels=None, 
                global_min_coords=None, global_max_coords=None,
                min_Cs=None, max_Cs=None,
                Ts=None, is_predict_panop=True,
                ensemble_confidence_type="max_prob", # max_prob
                return_ensemble=True, test=False, measure_time=False)-> torch.Tensor:
        if measure_time:
            inference_time = 0.0
            torch.cuda.synchronize()
            time_start = time.time() 
        ret = self.unet3d(in_feat, batch_size, 
                            global_min_coords=global_min_coords, global_max_coords=global_max_coords,
                            min_Cs=min_Cs, max_Cs=max_Cs,
                            class_frequencies=self.class_frequencies,
                            Ts=Ts, is_predict_panop=is_predict_panop,
                            sem_labels=sem_labels,
                            test=test)
        
        if measure_time:
            torch.cuda.synchronize()
            inference_time += time.time() - time_start
            self.inference_times.append(inference_time)    
            
        if return_ensemble:
            ensemble_sem_prob_denses = self.ensembler.ensemble_sem_compl(ret["sem_logits_at_scales"], Ts)
            ssc_confidences = []
            for i in range(len(ensemble_sem_prob_denses)):
                ensemble_sem_prob_dense = ensemble_sem_prob_denses[i]
                if i == len(ensemble_sem_prob_denses) - 1: # ensemble output
                    if ensemble_confidence_type == "var":
                        var = torch.stack(ensemble_sem_prob_denses[:-1]).var(dim=0)
                        ssc_confidence = (1 - var).mean(dim=0)
                        ssc_confidence = (ssc_confidence - ssc_confidence.min()) / (ssc_confidence.max() - ssc_confidence.min())
                    elif ensemble_confidence_type == "max_prob":
                        ssc_confidence = ensemble_sem_prob_dense.max(dim=0)[0]
                    else:
                        raise NotImplementedError
                else:
                    ssc_confidence = ensemble_sem_prob_dense.max(dim=0)[0]
                ssc_confidences.append(ssc_confidence)
           
            panop_prob_predictions = self.ensembler.ensemble_panop(ret["panop_predictions"], 
                                                                    ensemble_sem_prob_denses, 
                                                                    self.scene_size, Ts, 
                                                                    iou_threshold=self.iou_threshold,
                                                                    measure_time=measure_time)
            if measure_time:
                self.ensemble_times.append(panop_prob_predictions[0]["ensemble_time"])
                
                free, total = torch.cuda.mem_get_info()
                allocated = total - free
                self.memories.append(allocated)
                torch.cuda.empty_cache()
                if len(self.memories) % 10 == 0:
                    print("allocated", len(self.memories), np.mean(self.memories)/1024/1024)
            return ssc_confidences, ensemble_sem_prob_denses, panop_prob_predictions

        return ret

    

    def step(self, batch, step_type):
       
        geo_labels = batch['geo_labels']
        sem_labels = batch['sem_labels']

        semantic_labels = batch['semantic_label']  # [bs, 256, 256, 32]
        mask_labels = batch['mask_label']
        
        in_coords, in_feats = self.feat(batch['in_feats'], batch['in_coords'])
        in_feat = ME.SparseTensor(in_feats, in_coords.int())
        in_feat = self.augmenter.merge(in_feat)
        
        Ts = batch['Ts']
        
        batch_size = 1
        global_min_coords, global_max_coords = batch['global_min_Cs'], batch['global_max_Cs']

        min_Cs = batch['min_Cs']
        max_Cs = batch['max_Cs']
        
        
        is_predict_panop = True
        out = self(in_feat, batch_size, sem_labels, 
                   global_min_coords=global_min_coords, global_max_coords=global_max_coords,
                   min_Cs=min_Cs, max_Cs=max_Cs,
                   is_predict_panop=is_predict_panop,
                   Ts=Ts, return_ensemble=False)
        
        sem_logits_at_scales = out["sem_logits_at_scales"]
        
        
        try:
            compl_ce_loss, compl_lovasz_loss = compute_sem_compl_loss_kitti360(sem_labels, sem_logits_at_scales, min_Cs, max_Cs, self.class_frequencies)
        except:
            import pdb; pdb.set_trace()
        compl_ce_loss *= self.occ_weight
        compl_lovasz_loss *= self.occ_weight
        total_loss = (compl_ce_loss + compl_lovasz_loss) 
        self.log(step_type + "/compl_ce_loss", compl_ce_loss, on_epoch=True, sync_dist=self.sync_dist, batch_size=batch_size)
        self.log(step_type + "/compl_lovasz_loss", compl_lovasz_loss, on_epoch=True, sync_dist=self.sync_dist, batch_size=batch_size)
        if not is_predict_panop:
            return {
                "loss": total_loss,
            }
        panop_predictions = out["panop_predictions"]
        sem_logits_pruneds = out["sem_logits_pruneds"]
        
        panop_out_subnets = []
        indices = []
        loss_ce = 0
        loss_mask = 0
        loss_dice = 0
        ssc_ce_loss = 0.0
        ssc_lovasz_loss = 0.0
        loss_aux = defaultdict(float)
        for i_infer in range(self.n_infers):
            panop_prediction = panop_predictions[i_infer]
            
            min_C = min_Cs[i_infer]
            max_C = max_Cs[i_infer]
            sem_logits_pruned_1_1 = sem_logits_pruneds[i_infer]

            
            
            semantic_label = semantic_labels[i_infer].unsqueeze(0)
            voxel_probs = self.sigmoid(panop_prediction['voxel_logits'])
            
            query_logits = panop_prediction['query_logits'] # [1, 100, 21]
            
            assert len(mask_labels) == self.n_infers, "batch size must equal n_infers"
            mask_label = [mask_labels[i_infer]]
            geo_label_1_1 = geo_labels['1_1'][i_infer].unsqueeze(0)
            unknown_mask_dense = (geo_label_1_1 == 255)
            losses = self.criterion(sem_logits_pruned_1_1, panop_prediction, mask_label, semantic_label, 
                                    unknown_mask_dense, i_infer, indices, min_C)
            loss_ce += losses['loss_ce'] / self.n_infers
            loss_mask += losses['loss_mask'] / self.n_infers
            loss_dice += losses['loss_dice'] / self.n_infers
            for key in losses['loss_aux'] :
                loss_aux[key] += losses['loss_aux'][key]  / self.n_infers
           

            # ========= Evaluate panoptic
            if i_infer == 0 or i_infer == self.n_infers - 1: # evaluate the 1st subnet and the ensemble to save time
                with torch.no_grad():
                
                    query_probs = F.softmax(query_logits, dim=-1)
                    scene_size = compute_scene_size(min_C, max_C, scale=8).int()
                    panop_out = panoptic_inference(voxel_probs, query_probs, 
                                                overlap_threshold=self.overlap_threshold,
                                                object_mask_threshold=self.object_mask_threshold,
                                                thing_ids=self.thing_ids,
                                                scene_size=scene_size,
                                                min_C=min_C,
                                                input_query_logit=False,
                                                input_voxel_logit=False)
                    panop_out_subnets.append(panop_out)
                    
                    sem_logits = sem_logits_at_scales[1][i_infer]
                    sem_logits = prune_outside_coords(sem_logits, min_C, max_C)
                    shape = torch.Size([1, self.n_classes, scene_size[0], scene_size[1], scene_size[2]])
                    sem_logits = sem_logits.dense(shape, min_coordinate=torch.IntTensor([*min_C]))[0]
                    
                    sem_logits = sem_logits.squeeze()
                    sem_prob = F.softmax(sem_logits, dim=0)
                    ssc_confidence = sem_prob.max(dim=0)[0]
                    self.evaluate_all(panop_out=panop_out,
                                    sem_prob=sem_prob,
                                    i_infer=i_infer,
                                    ssc_confidence=ssc_confidence,
                                    compute_uncertainty=False,
                                    semantic_label=semantic_label, mask_labels=mask_label,
                                    ssc_metrics=self.ssc_metrics[step_type][i_infer],
                                    pq_stat=self.pq_stat[step_type][i_infer],   
                                    uncertainty_metrics=self.uncertainty_metrics[step_type][i_infer])
            # ========= Evaluate panoptic
       
        try:
            total_loss += (loss_dice +  loss_ce + loss_mask) * self.panop_weight + (ssc_ce_loss + ssc_lovasz_loss)
            for k in loss_aux:
                total_loss += loss_aux[k]
                self.log(step_type + "/" + k, loss_aux[k], on_epoch=True, sync_dist=self.sync_dist, batch_size=batch_size)
        except:
            import pdb; pdb.set_trace()
    
        self.log(step_type + "/ssc_ce_loss", ssc_ce_loss, on_epoch=True, sync_dist=self.sync_dist, batch_size=batch_size)
        self.log(step_type + "/ssc_lovasz_loss", ssc_lovasz_loss, on_epoch=True, sync_dist=self.sync_dist, batch_size=batch_size)
        self.log(step_type + "/total_loss", total_loss, on_epoch=True, sync_dist=self.sync_dist, batch_size=batch_size)
        self.log(step_type + "/loss_ce", loss_ce, on_epoch=True, sync_dist=self.sync_dist, batch_size=batch_size)
        self.log(step_type + "/loss_mask", loss_mask, on_epoch=True, sync_dist=self.sync_dist, batch_size=batch_size)
        self.log(step_type + "/loss_dice", loss_dice, on_epoch=True, sync_dist=self.sync_dist, batch_size=batch_size)

        return {
            "loss": total_loss,
            "panop_out_subnets": panop_out_subnets,
        }
    

    def step_inference(self, batch, step_type, eval=False, return_ensemble=True,  draw=False, measure_time=False):
        torch.cuda.empty_cache()
        in_coords, in_feats = self.feat(batch['in_feats'], batch['in_coords'])
        in_feat = ME.SparseTensor(in_feats, in_coords.int())
        in_feat = self.augmenter.merge(in_feat)
        Ts = batch['Ts']
        min_Cs = batch['min_Cs']
        max_Cs = batch['max_Cs']
        geo_labels = batch['geo_labels']
        
        global_min_coords, global_max_coords = batch['global_min_Cs'], batch['global_max_Cs']
        batch_size = 1
        
        ssc_confidences, sem_prob_denses, panop_prob_predictions = self(in_feat, batch_size, geo_labels,
                   global_min_coords=global_min_coords, 
                   global_max_coords=global_max_coords,
                   min_Cs=min_Cs, max_Cs=max_Cs,
                   Ts=Ts,
                   measure_time=measure_time,
                   return_ensemble=return_ensemble,
                   test=step_type == "test")
        
        panop_outs = []
        if step_type == "val":
            eval_list = [0, len(panop_prob_predictions) - 1]
        elif step_type == "test":
            eval_list = range(len(panop_prob_predictions))
        if draw:
            eval_list =  [len(panop_prob_predictions)-1]
            
        gt_panoptic_segs = []
        gt_segments_infos = []
        ssc_preds = []
        for i_infer in eval_list: # evaluate the 1st subnet and the ensemble to save time
            
            query_probs = panop_prob_predictions[i_infer]['query_probs']
            voxel_probs = panop_prob_predictions[i_infer]['voxel_probs']
            sem_probs = panop_prob_predictions[i_infer]['sem_probs']
            
            scene_size = (256, 256, 32)
            min_C = torch.tensor([0, 0, 0], dtype=torch.int32)
            
            panop_out = panoptic_inference(voxel_probs, query_probs,
                                            overlap_threshold=self.overlap_threshold,
                                            object_mask_threshold=self.object_mask_threshold,
                                            thing_ids=self.thing_ids,
                                            scene_size=scene_size,
                                            min_C=min_C,
                                            input_query_logit=False,
                                            input_voxel_logit=False)
            
            
            ssc_confidence = ssc_confidences[i_infer]
            panop_out['ssc_confidence'] = ssc_confidence
            panop_outs.append(panop_out)
                
            if eval or draw:
                item_id = 0 # same items for all subnets
                geo_labels = batch['geo_labels']

                semantic_labels = batch['semantic_label']  # [bs, 256, 256, 32]
                semantic_label = semantic_labels[item_id].unsqueeze(0)
                semantic_label_origins = batch['semantic_label_origin']
                semantic_label_origin = semantic_label_origins[item_id].unsqueeze(0)
                
                mask_labels = batch['mask_label'][item_id]
                mask_label = [mask_labels]
                mask_label_origin = batch['mask_label_origin'][item_id]
                mask_label_origin = [mask_label_origin]


                sem_prob_dense = sem_prob_denses[i_infer]
        
                gt_panoptic_seg, gt_segments_info, ssc_pred = self.evaluate_all(panop_out=panop_out,
                                  ssc_confidence=ssc_confidence,
                                  i_infer=i_infer,
                                  sem_prob=sem_prob_dense,
                                  semantic_label=semantic_label_origin, mask_labels=mask_label_origin,
                                  ssc_metrics=self.ssc_metrics[step_type][i_infer],
                                  pq_stat=self.pq_stat[step_type][i_infer],
                                  uncertainty_metrics=self.uncertainty_metrics[step_type][i_infer], 
                                  compute_uncertainty=step_type == "test")
                gt_panoptic_segs.append(gt_panoptic_seg)
                gt_segments_infos.append(gt_segments_info)
                ssc_preds.append(ssc_pred)
        if measure_time and len(self.inference_times) % 20 == 0:
            print(self.inference_times[:6])
            print("inference time: ", np.mean(self.inference_times[1:]))
            print(self.ensemble_times[:6])
            print("ensemble time: ", np.mean(self.ensemble_times[1:]))

        return panop_outs, gt_panoptic_segs, gt_segments_infos, ssc_preds


    def evaluate_all(self, 
                     ssc_confidence, i_infer, sem_prob,
                     panop_out,
                     semantic_label,
                     mask_labels,
                     ssc_metrics, pq_stat, uncertainty_metrics, compute_uncertainty=False):
        with torch.no_grad():
       
            if not compute_uncertainty:
                uncertainty_metrics = None
                
            invalid_masks = semantic_label == 255
            gt_panoptic_seg, gt_segments_info = self.evaluate_panoptic(pq_stat,
                                   i_infer,
                            invalid_masks,
                            mask_labels, 
                            panop_out["vox_all_mask_probs_denses"],
                            panop_out["panoptic_seg_denses"], 
                            panop_out["segments_infos"], 
                            panop_out["vox_confidence_denses"],
                            uncertainty_metrics=uncertainty_metrics)
            
            ssc_pred = torch.argmax(sem_prob, dim=0).squeeze()
            if compute_uncertainty:
                try:
                    ssc_metrics.add_batch_ece(ssc_confidence, ssc_pred, sem_prob, semantic_label, inference_time=0.0)
                except:
                    print("ssc_confidence", ssc_confidence)
                    print("ssc_pred", ssc_pred)
            ssc_pred = ssc_pred.unsqueeze(0).detach().cpu().numpy()
            ssc_gt = semantic_label.cpu().numpy()
            ssc_metrics.add_batch(ssc_pred, ssc_gt)
            
            return gt_panoptic_seg, gt_segments_info, ssc_pred
            
            

    
    def evaluate_panoptic(self, 
                            pq_metric, i_infer,
                            invald_masks, 
                            mask_labels, 
                            vox_all_mask_probs_denses,
                            panoptic_seg_denses, 
                            segments_infos,
                            vox_confidence_denses=None,
                            uncertainty_metrics=None):
        for i in range(len(panoptic_seg_denses)):
            vox_all_mask_probs_dense = vox_all_mask_probs_denses[i]
            invald_mask = invald_masks[i]
            mask_label = mask_labels[i]
            pred_panoptic_seg = panoptic_seg_denses[i].detach().cpu().numpy()
            pred_segments_info = segments_infos[i]
            vox_confidence_dense = vox_confidence_denses[i]
            
            gt_panoptic_seg, gt_segments_info = convert_mask_label_to_panoptic_output(
                mask_label["labels"], mask_label['masks'], self.thing_ids)
            gt_panoptic_seg = gt_panoptic_seg.detach().cpu().numpy()

            unknown_mask = invald_mask.detach().cpu().numpy()
            pred_panoptic_seg[unknown_mask] = 0
            gt_panoptic_seg[unknown_mask] = 0
            pred_ids = np.unique(pred_panoptic_seg)
            gt_ids = np.unique(gt_panoptic_seg)
            pred_segments_info = [el for el in pred_segments_info if el['id'] in pred_ids]
            gt_segments_info = [el for el in gt_segments_info if el['id'] in gt_ids]
            

            _ = pq_compute_single_core(
                pq_metric,
                gt_segments_info, 
                pred_segments_info, 
                gt_panoptic_seg,
                pred_panoptic_seg,
                thing_ids=self.thing_ids)
     
            
            
            if uncertainty_metrics is not None:
                for threshold in self.uncertainty_thresholds:
                    pred_gt_matched_segms = find_matched_segment(gt_segments_info, 
                        pred_segments_info, 
                        gt_panoptic_seg,
                        pred_panoptic_seg, 
                        threshold=threshold)
                    
                    self.uncertainty_metrics_by_thresholds[threshold][i_infer].compute_ece_panop(
                        pred_panoptic_seg, pred_segments_info, 
                        vox_confidence_dense, vox_all_mask_probs_dense,
                        pred_gt_matched_segms, 
                        gt_panoptic_seg, gt_segments_info,
                        n_classes=self.n_classes)
            
            return gt_panoptic_seg, gt_segments_info
        
        

    def training_step(self, batch, batch_idx):
        sch = self.lr_schedulers()
        sch.step(self.global_step)
        return self.step(batch, "train")['loss']


    def validation_step(self, batch, batch_idx):
        self.step_inference(batch, "val", eval=True)

    def test_step(self, batch, batch_idx):
        self.step_inference(batch, "test", eval=True, measure_time=True)

    
    def validation_epoch_end(self, outputs):
        print("validation_epoch_end")
     
        
        for i_infer in range(self.n_infers + 1):
            for step_type in ['train', 'val']:
                prefix = "{}_subnet{}".format(step_type, i_infer)
                pq_stat = self.pq_stat[step_type][i_infer]
                ssc_metrics = self.ssc_metrics[step_type][i_infer]
                uncertainty_metrics = self.uncertainty_metrics[step_type][i_infer]

                ssc_stat = ssc_metrics.get_stats()
                for i, class_name in enumerate(self.class_names):
                    self.log(
                        "{}_SemIoU/{}".format(prefix, class_name),
                        ssc_stat["iou_ssc"][i],
                        sync_dist=self.sync_dist
                    )
                    self.log("{}/mIoU".format(prefix),
                                ssc_stat["iou_ssc_mean"] * 100, sync_dist=self.sync_dist)
                    self.log("{}/IoU".format(prefix), ssc_stat["iou"]* 100, sync_dist=self.sync_dist)
                    self.log("{}/Precision".format(prefix),
                                ssc_stat["precision"]* 100, sync_dist=self.sync_dist)
                    self.log("{}/Recall".format(prefix),
                                ssc_stat["recall"]* 100, sync_dist=self.sync_dist)
                   
                ssc_metrics.reset()
                self.panoptic_metrics(pq_stat, prefix)
                pq_stat.reset()



    def test_epoch_end(self, outputs):
        print("test_epoch_end")
        step_type = "test"
        n_preds = len(self.pq_stat[step_type])
        pq_stats = [self.pq_stat[step_type][i] for i in range(n_preds)] 
        ssc_metrics = [self.ssc_metrics[step_type][i] for i in range(n_preds)] 
        print_metrics_table_panop_ssc(pq_stats, ssc_metrics, self)
        print_metrics_table_panop_per_class(pq_stats, self)
        
        print(self.inference_times[:6])
        print("inference time: ", np.mean(self.inference_times[1:]))
        print(self.ensemble_times[:6])
        print("ensemble time: ", np.mean(self.ensemble_times[1:]))
        
        for threshold in self.uncertainty_thresholds:
            print("Uncertainty threshold: ", threshold)
            uncertainty_metrics = [self.uncertainty_metrics_by_thresholds[threshold][i] for i in range(n_preds)] 
            print_metrics_table_uncertainty(uncertainty_metrics, ssc_metrics, self)
        
        print("allocated", np.mean(self.memories)/1024/1024)
    
    def panoptic_metrics(self, pq_stat, prefix="", is_logging=True):
        
        metrics = [("All", None), ("Things", True), ("Stuff", False)]
        results = {}
        for name, isthing in metrics:
            results[name], per_class_results = pq_stat.pq_average(isthing=isthing, ignore_cat_id=0, thing_ids=self.thing_ids)
            if name == 'All':
                results['per_class'] = per_class_results     
        
        self.log("{}/pq_dagger_all".format(prefix), 100 *results["All"]['pq_dagger'], sync_dist=self.sync_dist)
        for name, _isthing in metrics:
            if is_logging:
                self.log("{}/{}_pq".format(prefix, name), 100 *results[name]['pq'], sync_dist=self.sync_dist)
                self.log("{}/{}_sq".format(prefix, name), 100 *results[name]['sq'], sync_dist=self.sync_dist)
                self.log("{}/{}_rq".format(prefix, name), 100 *results[name]['rq'], sync_dist=self.sync_dist)
                self.log("{}/{}_n".format(prefix, name),  results[name]['n'], sync_dist=self.sync_dist)
        return results


    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            WarmupCosine(
                0,
                50000, # config["scheduler"]["max_epoch"] * len_train_loader,
                0.01 # config["scheduler"]["min_lr"] / config["optim"]["lr"],
            ),
        )

        scheduler = {
            'scheduler': scheduler,
            'interval': 'epoch'
        }

        return [optimizer], [scheduler]

