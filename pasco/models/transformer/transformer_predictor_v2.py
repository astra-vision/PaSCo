# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from: https://github.com/facebookresearch/detr/blob/master/models/detr.py
import torch
from torch import nn

from pasco.models.transformer.position_encoding import PositionEmbeddingSineSparse
import MinkowskiEngine as ME
import pasco.models.transformer.blocks as blocks


class TransformerPredictorV2(nn.Module):
    def __init__(
        self,
        in_channels,
        query_sample_ratio=1.0,
        num_classes=20,
        hidden_dim=384,
        num_queries=100,
        nheads=8,
        dropout=0.0,
        dim_feedforward=2048,
        enc_layers=0,
        dec_layers=6,
        pre_norm=False,
        aux_loss=True,
        mask_dim=256,
        enforce_input_project=True,
        mask_classification=True,
        sample_query_class=False,
        transformer_input_dims=[256],
        n_infers=2,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            in_channels: channels of the input features
            num_classes: number of classes
            hidden_dim: Transformer feature dimension
            num_queries: number of queries
            nheads: number of heads
            dropout: dropout in Transformer
            dim_feedforward: feature dimension in feedforward network
            enc_layers: number of Transformer encoder layers
            dec_layers: number of Transformer decoder layers
            pre_norm: whether to use pre-LayerNorm or not
            deep_supervision: whether to add supervision to every decoder layers
            mask_dim: mask feature dimension
            enforce_input_project: add input project 1x1 conv even if input
                channels and hidden dim is identical
            mask_classification: whether to add mask classifier or not
        """
        super().__init__()
        # print("Transformer dropout: ", dropout)
        print("sample_query_class: ", sample_query_class)
        self.transformer_input_dims = transformer_input_dims
        self.mask_classification = mask_classification
        self.sample_query_class = sample_query_class

        # positional encoding
        self.nheads = nheads
        self.n_infers = n_infers
        self.hidden_dim = hidden_dim
        self.query_dim = hidden_dim
        N_steps = hidden_dim // 3
        self.decoder_norm = nn.LayerNorm(hidden_dim)
        self.pe_layer = PositionEmbeddingSineSparse(N_steps, normalize=True)

        self.src_scales = [4, 2, 1]
        self.num_layers = len(self.src_scales)
        self.transformer_self_attention_layers = nn.ModuleList()
        self.transformer_cross_attention_layers = nn.ModuleList()
        self.transformer_ffn_layers = nn.ModuleList()
        for _ in range(self.num_layers):
            self.transformer_self_attention_layers.append(
                blocks.SelfAttentionLayer(
                    d_model=hidden_dim, nhead=self.nheads, dropout=0.0
                )
            )
            self.transformer_cross_attention_layers.append(
                blocks.CrossAttentionLayer(
                    d_model=hidden_dim, nhead=self.nheads, dropout=0.0
                )
            )
            self.transformer_ffn_layers.append(
                blocks.FFNLayer(
                    d_model=hidden_dim, dim_feedforward=dim_feedforward, dropout=0.0
                )
            )

        self.num_queries = num_queries

        self.query_sample_ratio = query_sample_ratio
        self.num_queries_sampled = int(self.num_queries * self.query_sample_ratio)

        # learnable query features
        self.query_feat = nn.Embedding(self.num_queries * n_infers, hidden_dim)
        self.query_embed = nn.Embedding(self.num_queries * n_infers, hidden_dim)

        self.input_projs = nn.ModuleList()
        self.max_pools = nn.ModuleDict()
        for i in range(self.num_layers):
            scale = self.src_scales[i]
            self.input_projs.append(nn.Linear(in_channels[i], hidden_dim))
            self.max_pools[str(scale)] = ME.MinkowskiMaxPooling(
                kernel_size=scale, stride=scale, dimension=3
            )

        self.aux_loss = aux_loss

        # output FFNs
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.mask_embed = blocks.MLP(hidden_dim, hidden_dim, hidden_dim, 3)
        self.mask_feat_proj = nn.Linear(mask_dim, hidden_dim)

    def forward(self, xs, sem_logits, min_Cs, max_Cs, keep_pad):
        """
        xs[scale] = (voxel_F, voxel_C)
        """
        sem_logits_F, sem_logits_C = sem_logits
        bs = sem_logits_F.shape[0]

        assert self.n_infers == bs, "batch size should be equal to number of inference"

        output = self.query_feat.weight.reshape(bs, -1, self.hidden_dim)
        query_embed = self.query_embed.weight.reshape(bs, -1, self.hidden_dim)

        srcs = []
        src_Cs = []
        pos = []
        for i in range(self.num_layers):
            scale = self.src_scales[i]
            # src = xs[scale]
            batch_F, batch_C = xs[scale]
            srcs.append(batch_F)
            src_Cs.append(batch_C)

            pe = batch_C.reshape(-1, 4)
            pe = self.pe_layer(pe[:, 1:])
            pe = pe.reshape(bs, -1, self.hidden_dim)
            pos.append(pe)  # (bs, N, hidden_dim)

        predictions_class = []
        predictions_mask = []

        # predictions on learnable query features, first attn_mask
        voxel_coord = xs[1][1]
        voxel_feat = self.mask_feat_proj(xs[1][0]) + pos[-1]

        outputs_class, outputs_mask = self.pred_heads(output, voxel_feat)
        predictions_class.append(outputs_class)
        predictions_mask.append(outputs_mask)
        
        for i in range(self.num_layers):
            src_F = self.input_projs[i](srcs[i])
            src_C = src_Cs[i]

            attn_mask = self.compute_attn_mask(
                outputs_mask,
                voxel_coord,
                srcs[i],
                src_C,
                self.src_scales[i],
                min_Cs,
                max_Cs,
            )

            if attn_mask is not None:
                attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False

            # cross-attention first
            output = self.transformer_cross_attention_layers[i](
                output,
                src_F,
                attn_mask=attn_mask,
                pos=pos[i],
                query_pos=query_embed,
            )

            output = self.transformer_self_attention_layers[i](
                output, attn_mask=None, padding_mask=None, query_pos=query_embed
            )
            # FFN
            output = self.transformer_ffn_layers[i](output)

            # get predictions and attn mask for next feature level
            outputs_class, outputs_mask = self.pred_heads(
                output,
                voxel_feat,
            )

            predictions_class.append(outputs_class)
            predictions_mask.append(outputs_mask)

        panop_predictions = []
        for i in range(bs):
            # sem_logits_F_i = sem_logits_F[i]
            sem_logits_C_i = sem_logits_C[i]
            keep = keep_pad[i]
            # sem_logits_F_i = sem_logits_F_i[keep]
            sem_logits_C_i = sem_logits_C_i[keep]

            predictions_class_per_item = [t[i].unsqueeze(0) for t in predictions_class]
            predictions_masks_per_item = []
            for j in range(len(predictions_mask)):
                predictions_mask_F = predictions_mask[j][i]
                predictions_mask_C = xs[1][1][i]  # xs[scale][coord/feature][batch]
                predictions_mask_j = ME.SparseTensor(
                    predictions_mask_F[keep], predictions_mask_C[keep]
                )
                predictions_masks_per_item.append(predictions_mask_j)

            query_logits = predictions_class_per_item[-1]
            out = {
                "query_logits": query_logits,
                "voxel_logits": predictions_masks_per_item[-1],
            }
            out["aux_outputs"] = self.set_aux(
                predictions_class_per_item, predictions_masks_per_item
            )
            panop_predictions.append(out)

        return panop_predictions

    def compute_attn_mask(
        self, outputs_mask, voxel_coord, src, src_C, src_scale, min_Cs, max_Cs
    ):

        # src_scale = src.tensor_stride[0]

        keep_mask = (outputs_mask.sigmoid() > 0.5).detach().float()

        keep_mask_F = torch.cat([t for t in keep_mask], dim=0)
        keep_mask_C = [t[:, 1:] for t in voxel_coord]
        keep_mask_C = ME.utils.batched_coordinates(keep_mask_C).to(keep_mask_F.device)
        keep_mask_sparse = ME.SparseTensor(keep_mask_F, keep_mask_C)

        if src_scale != 1:
            keep_mask_sparse_downscaled = self.max_pools[str(src_scale)](
                keep_mask_sparse
            )
        else:
            keep_mask_sparse_downscaled = keep_mask_sparse

        attn_mask = []
        for i in range(len(src_C)):
            src_C_i = (src_C[i][:, 1:] - min_Cs[i]) / src_scale
            scene_size = ((max_Cs[i] - min_Cs[i]) / src_scale + 1).int()

            batch_indices = torch.nonzero(
                keep_mask_sparse_downscaled.C[:, 0] == i, as_tuple=True
            )[0]
            keep_mask_sparse_downscaled_F_i = keep_mask_sparse_downscaled.F[
                batch_indices
            ]
            keep_mask_sparse_downscaled_C_i = keep_mask_sparse_downscaled.C[
                batch_indices
            ]
            keep_mask_sparse_downscaled_C_i = ME.utils.batched_coordinates(
                [keep_mask_sparse_downscaled_C_i[:, 1:]]
            ).to(keep_mask_sparse_downscaled_F_i.device)

            keep_mask_sparse_downscaled_i = ME.SparseTensor(
                keep_mask_sparse_downscaled_F_i,
                keep_mask_sparse_downscaled_C_i,
                tensor_stride=keep_mask_sparse_downscaled.tensor_stride,
            )
            keep_mask_dense_downscaled_i = keep_mask_sparse_downscaled_i.dense(
                shape=torch.Size(
                    [
                        1,
                        keep_mask_sparse_downscaled.shape[1],
                        scene_size[0],
                        scene_size[1],
                        scene_size[2],
                    ]
                ),
                min_coordinate=torch.IntTensor([*min_Cs[i]]),
            )[0]
            src_C_i = src_C_i.long()
            keep_mask_i = keep_mask_dense_downscaled_i[
                0, :, src_C_i[:, 0], src_C_i[:, 1], src_C_i[:, 2]
            ].T
            attn_mask_i = ~(keep_mask_i.bool())
            attn_mask.append(attn_mask_i)
        attn_mask = torch.stack(attn_mask, dim=0)
        attn_mask = (
            attn_mask.unsqueeze(1)
            .repeat(1, self.nheads, 1, 1)
            .flatten(0, 1)
            .permute(0, 2, 1)
        )

        return attn_mask

    def pred_heads(
        self,
        output,
        mask_features,
    ):
        decoder_output = self.decoder_norm(output)
        outputs_class = self.class_embed(decoder_output)
        mask_embed = self.mask_embed(decoder_output)
        n_queries = mask_embed.shape[1]

        outputs_mask = torch.einsum("bqc,bpc->bpq", mask_embed, mask_features)

        return outputs_class, outputs_mask

    @torch.jit.unused
    def set_aux(self, outputs_class, outputs_seg_masks):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [
            {"query_logits": a, "voxel_logits": b}
            for a, b in zip(outputs_class[:-1], outputs_seg_masks[:-1])
        ]

    def sample_mask_embed(self, query_features):
        mask_embed, query_means, log_std = self.mask_embed(query_features)
        return (
            mask_embed,
            query_means[-1],
            log_std[-1],
        )  # NOTE use only the last layer output of the transformer for now

    def forward_prediction_head(
        self, mask_embed, query_features, mask_features, scale=1
    ):

        outputs_class = self.class_embed(query_features)

        out = {"pred_logits": outputs_class[-1]}

        bs = mask_embed.shape[1]
        mask_embed = mask_embed[
            -1
        ]  # NOTE: use only the last layer output of the transformer for now

        outputs_seg_masks = []
        mask_coords = []

        outputs_seg_masks = torch.zeros(
            (mask_features.C.shape[0], mask_embed.shape[1]),
            dtype=mask_embed.dtype,
            device=mask_embed.device,
        )
        mask_coords_cat = torch.zeros_like(mask_features.C)

        for i in range(bs):
            mask_embed_i = mask_embed[i, :, :]  # [num_queries, dim]

            mask_indices_i = torch.nonzero(mask_features.C[:, 0] == i, as_tuple=True)[
                0
            ]  # [N]
            mask_features_i = mask_features.F[mask_indices_i]  # [N, dim]
            mask_coords_i = mask_features.C[mask_indices_i]  # [N, dim]

            outputs_seg_mask = mask_features_i @ mask_embed_i.T  # [N, queries]

            outputs_seg_masks[mask_indices_i] = outputs_seg_mask
            mask_coords_cat[mask_indices_i] = mask_coords_i

        # Check if mask_coords_cat and mask_features.C are equal
        equality_check = torch.all(torch.eq(mask_coords_cat, mask_features.C))
        assert equality_check, "Are mask_coords_cat and mask_features.C equal?"

        output_seg_masks = ME.SparseTensor(
            features=outputs_seg_masks,
            coordinates=mask_coords_cat,
            tensor_stride=mask_features.tensor_stride,
        )

        out["pred_masks"] = output_seg_masks  # SparseTensor(N * bs, queries)

        return out

    def _set_aux_output(self, outputs_class, outputs_seg_masks):

        return [
            {"pred_logits": a, "pred_masks": b}
            for a, b in zip(outputs_class[:-1], outputs_seg_masks[:-1])
        ]
