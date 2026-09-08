import os
import json
import copy
import torch
import random
import numpy as np
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
from utils.logger import LOGGER
from .general_module import TokenMasker, MMGeneralModule, Contra_head, Match_head
from utils.distributed import all_gather_with_grad, concat_all_gather, all_gather_list
from torch.nn import LayerNorm as LayerNorm
from easydict import EasyDict as edict
from utils.volume import volume_computation2,volume_computation4,volume_computation3, volume_computation5
from utils.pmrl_loss import PMRLLoss


class GRAM(MMGeneralModule):
    """ VLP pretraining """
    def __init__(self, config):
        super().__init__()
    
        self.config = config
        self.construct_vision_encoder()
        self.construct_audio_encoder()
        self.construct_multimodal_encoder()

        contra_dim = self.config.contra_dim
        self.contra_head_t = Contra_head(self.multimodal_dim, contra_dim)
        self.contra_head_s = Contra_head(self.multimodal_dim, contra_dim)
        self.contra_head_v = Contra_head(self.vision_dim, contra_dim)
        self.contra_head_a = Contra_head(self.audio_dim, contra_dim)
        self.contra_head_d = Contra_head(self.vision_dim, contra_dim)
        self.contra_head_va = nn.Linear(self.vision_dim + self.audio_dim, contra_dim)
        self.contra_head_vs = nn.Linear(self.vision_dim + self.multimodal_dim, contra_dim)
        self.contra_head_vas = nn.Linear(self.vision_dim + self.audio_dim + self.multimodal_dim, contra_dim)
        self.contra_temp = nn.Parameter(torch.tensor(0.07))
        # ---- hypergraph (GatedHGNN) knobs; defaults => stage A == plain HyperGRAM (byte-identical) ----
        self.stage = config.get('stage', 'A')
        self.w_doc  = float(config.get('w_doc', 0.0))
        self.w_reg  = float(config.get('w_reg', 0.0))
        self.semantic_edges = bool(config.get('semantic_edges', False))
        self.knn_k = int(config.get('knn_k', 4))
        self.edge_dropout = float(config.get('edge_dropout', 0.3))
        _ss = config.get('sem_sim_std', None)
        self.sem_sim_std = float(_ss) if _ss is not None else None
        if self.stage == 'B':
            from .hypergraph import GatedHGNN
            self.hgnn = GatedHGNN(contra_dim, n_layers=int(config.get('hgnn_layers', 2)))
        self.itm_head = Match_head(self.multimodal_dim)
        self.vision_frame_embedding = nn.Parameter(0.02 * torch.randn(1, self.config.max_vision_sample_num, self.multimodal_dim))
        self.audio_frame_embedding = nn.Parameter(0.02 * torch.randn(1, self.config.max_audio_sample_num, self.multimodal_dim))
        self.hidden_trans_vision_multimodal = nn.Sequential(nn.Linear(self.vision_dim, self.multimodal_dim),LayerNorm(self.multimodal_dim, eps=1e-12))
        self.hidden_trans_audio_multimodal = nn.Sequential(nn.Linear(self.audio_dim, self.multimodal_dim),LayerNorm(self.multimodal_dim, eps=1e-12))
        self.hidden_trans_subtitle_multimodal = nn.Sequential(nn.Linear(self.multimodal_dim, self.multimodal_dim),LayerNorm(self.multimodal_dim, eps=1e-12))
        self.vision_type_embeddings = nn.Parameter(0.02 * torch.randn(1, 1, self.multimodal_dim)) 
        self.audio_type_embeddings = nn.Parameter(0.02 * torch.randn(1, 1, self.multimodal_dim)) 
        self.subtitle_type_embeddings = nn.Parameter(0.02 * torch.randn(1, 1, self.multimodal_dim)) 
        self.beam_size  = config.beam_size
        self.itm_ratio = config.itm_ratio
        self.max_omni_caption_len = config.max_omni_caption_len
        self.max_caption_len = config.max_caption_len
        self.max_subtitle_len = config.max_subtitle_len

        # HyperGRAM: Geometry mode configuration
        self.geometry_mode = config.get('geometry_mode', 'euclidean')  # 'euclidean', 'hyperbolic', or 'hybrid'

        if self.geometry_mode in ['hyperbolic', 'hybrid', 'hybrid_pmrl']:
            # Learnable curvature parameters for hyperbolic space
            curv_init = config.get('curvature_init', 1.0)
            self.learn_curvature = config.get('learn_curvature', True)

            if isinstance(curv_init, list):
                # Different curvature for each modality
                # Order: [language, video, audio, subtitle, depth]
                self.curvatures = nn.Parameter(
                    torch.tensor(curv_init[:5], dtype=torch.float32),
                    requires_grad=self.learn_curvature
                )
            else:
                # Single shared curvature for all modalities
                self.curvature = nn.Parameter(
                    torch.tensor(curv_init, dtype=torch.float32),
                    requires_grad=self.learn_curvature
                )

        # Hybrid mode weights
        if self.geometry_mode in ['hybrid', 'hybrid_pmrl']:
            # Initialize weights for euclidean and hyperbolic components
            init_euclidean_weight = config.get('initial_euclidean_weight', 0.5)
            init_hyperbolic_weight = config.get('initial_hyperbolic_weight', 0.5)
            self.learn_hybrid_weights = config.get('learn_hybrid_weights', True)

            self.euclidean_weight = nn.Parameter(
                torch.tensor(init_euclidean_weight, dtype=torch.float32),
                requires_grad=self.learn_hybrid_weights
            )
            self.hyperbolic_weight = nn.Parameter(
                torch.tensor(init_hyperbolic_weight, dtype=torch.float32),
                requires_grad=self.learn_hybrid_weights
            )

        # PMRL configuration
        if self.geometry_mode in ['pmrl', 'pmrl_volume', 'hybrid_pmrl']:
            pmrl_lambda1 = config.get('pmrl_lambda1', 1.0)
            pmrl_lambda2 = config.get('pmrl_lambda2', 0.1)
            pmrl_temperature = config.get('pmrl_temperature', 0.07)

            # Enable hybrid space SVD for hybrid_pmrl mode
            use_hybrid_space = (self.geometry_mode == 'hybrid_pmrl')

            self.pmrl_loss = PMRLLoss(
                lambda1=pmrl_lambda1,
                lambda2=pmrl_lambda2,
                temperature=pmrl_temperature,
                use_hybrid_space=use_hybrid_space
            )

            LOGGER.info(f"PMRL mode initialized: {self.geometry_mode}")
            LOGGER.info(f"  lambda1={pmrl_lambda1}, lambda2={pmrl_lambda2}, tau={pmrl_temperature}")
            if use_hybrid_space:
                LOGGER.info(f"  Hybrid space SVD enabled (Strategy 5)")


    def construct_multimodal_encoder(self):    
        
        from .text_encoders.bert.bert import BertForMaskedLM, BertConfig
     
        bertconfig = BertConfig.from_pretrained("./pretrained_weights/bert/bert-base-uncased")
        bertconfig.add_cross_attention = True
        bertconfig.is_decoder = True
        self.multimodal_encoder = BertForMaskedLM.from_pretrained("./pretrained_weights/bert/bert-base-uncased",config = bertconfig )
        self.multimodal_dim = 768

        if self.config.checkpointing:
            self.multimodal_encoder._set_gradient_checkpointing(self.multimodal_encoder.bert.encoder, True)

        from transformers import BertTokenizer


        self.multimodal_encoder.tokenizer = BertTokenizer.from_pretrained('./pretrained_weights/bert/bert-base-uncased')
        self.multimodal_encoder.tokenizer.bos_token_id = self.multimodal_encoder.tokenizer.convert_tokens_to_ids(['[CLS]'])[0]
        self.multimodal_encoder.tokenizer.eos_token_id = self.multimodal_encoder.tokenizer.convert_tokens_to_ids(['[SEP]'])[0]
        self.multimodal_encoder.tokenizer.pad_token_id = self.multimodal_encoder.tokenizer.convert_tokens_to_ids(['[PAD]'])[0]
        self.multimodal_encoder.tokenizer.mask_token_id = self.multimodal_encoder.tokenizer.convert_tokens_to_ids(['[MASK]'])[0]

        self.text_masker = TokenMasker(mask_token = self.multimodal_encoder.tokenizer.mask_token_id, range_start=106, range_end = 30522)

        

    def batch_get(self, batch, key):
        if key in batch:
            return batch[key]


        elif key == 'caption_tokens':

            caption_tokens = self.multimodal_encoder.tokenizer(batch.raw_captions,
                                                    padding="max_length",
                                                    truncation=True,
                                                    max_length=self.max_caption_len,
                                                    return_tensors="pt").to(torch.device('cuda'))
         
            batch[key] = caption_tokens
        
        elif key == 'subtitle_tokens':
         
            subtitle_tokens = self.multimodal_encoder.tokenizer(batch.raw_subtitles,
                                                    padding="max_length",
                                                    truncation=True,
                                                    max_length=self.max_subtitle_len,
                                                    return_tensors="pt")
            subtitle_tokens = subtitle_tokens.to(torch.device('cuda'))
            batch[key] = subtitle_tokens
                                        

        elif key == 'vision_caption_tokens':
            caption_tokens = self.multimodal_encoder.tokenizer(batch.vision_captions,
                                                    padding="max_length",
                                                    truncation=True,
                                                    max_length=self.max_caption_len,
                                                    return_tensors="pt")

            caption_tokens = caption_tokens.to(torch.device('cuda'))
            batch[key] = caption_tokens


        
        elif key == 'audio_caption_tokens':
            caption_tokens = self.multimodal_encoder.tokenizer(batch.audio_captions,
                                                    padding="max_length",
                                                    truncation=True,
                                                    max_length=self.max_caption_len,
                                                    return_tensors="pt")

            caption_tokens = caption_tokens.to(torch.device('cuda'))
            batch[key] = caption_tokens

        elif key == 'omni_caption_tokens':
            caption_tokens = self.multimodal_encoder.tokenizer(batch.omni_captions,
                                                    padding="max_length",
                                                    truncation=True,
                                                    max_length=self.max_omni_caption_len,
                                                    return_tensors="pt")

            caption_tokens = caption_tokens.to(torch.device('cuda'))
            batch[key] = caption_tokens


        elif key == 'caption_output':
            caption_tokens = self.batch_get(batch, 'caption_tokens')
            input_ids = caption_tokens.input_ids
            attention_mask = caption_tokens.attention_mask
            caption_output = self.multimodal_encoder.bert(input_ids = input_ids,
                                            attention_mask = attention_mask).last_hidden_state
            batch[key] = caption_output



        elif key == 'vision_caption_output':
            caption_tokens = self.batch_get(batch, 'vision_caption_tokens')
            input_ids = caption_tokens.input_ids
            attention_mask = caption_tokens.attention_mask
            caption_output = self.multimodal_encoder.bert(input_ids = input_ids,
                                            attention_mask = attention_mask).last_hidden_state
            batch[key] = caption_output


        elif key == 'audio_caption_output':
            caption_tokens = self.batch_get(batch, 'audio_caption_tokens')
            input_ids = caption_tokens.input_ids
            attention_mask = caption_tokens.attention_mask
            caption_output = self.multimodal_encoder.bert(input_ids = input_ids,
                                            attention_mask = attention_mask).last_hidden_state
            batch[key] = caption_output

       
        elif key == 'subtitle_output':
            subtitle_tokens = self.batch_get(batch, 'subtitle_tokens')
            input_ids = subtitle_tokens.input_ids
            attention_mask = subtitle_tokens.attention_mask
            subtitle_output = self.multimodal_encoder.bert(input_ids = input_ids,
                                            attention_mask = attention_mask).last_hidden_state
            batch[key] = subtitle_output
  

        elif key == 'vision_output':
            vision_output = self.forward_vision_encoder(batch.vision_pixels)
            batch[key] = vision_output
            
        elif key == 'depth_output':
            depth_output = self.forward_vision_encoder(batch.depth_pixels)
            batch[key] = depth_output
        
        elif key == 'audio_output':
            audio_output = self.forward_audio_encoder(batch.audio_spectrograms) 
            batch[key] = audio_output


        elif key == 'condition_feats_v':
            vision_output = self.batch_get(batch, 'vision_output')
            condition_feats_v = self.get_multimodal_forward_input_vision(vision_output)
            batch[key] = condition_feats_v
            
        elif key == 'condition_feats_d':
            vision_output = self.batch_get(batch, 'depth_output')
            condition_feats_d = self.get_multimodal_forward_input_vision(vision_output)
            batch[key] = condition_feats_d
            
        elif key == 'condition_feats_a':
            audio_output = self.batch_get(batch, 'audio_output')
            condition_feats_a = self.get_multimodal_forward_input_audio(audio_output)
            batch[key] = condition_feats_a

        elif key == 'condition_feats_s':
            subtitle_output = self.batch_get(batch, 'subtitle_output')
            condition_feats_s = self.get_multimodal_forward_input_subtitle(subtitle_output)
            batch[key] = condition_feats_s

        elif key == 'condition_feats_va':
            condition_feats_v = self.batch_get(batch, 'condition_feats_v')
            condition_feats_a = self.batch_get(batch, 'condition_feats_a')
            condition_feats_va = torch.cat((condition_feats_v, condition_feats_a),dim=1)
            batch[key] = condition_feats_va

        elif key == 'condition_feats_vs':
            condition_feats_v = self.batch_get(batch, 'condition_feats_v')
            condition_feats_s = self.batch_get(batch, 'condition_feats_s')
            condition_feats_vs = torch.cat((condition_feats_v, condition_feats_s),dim=1)
            batch[key] = condition_feats_vs

        elif key == 'condition_feats_vas':
            condition_feats_v = self.batch_get(batch, 'condition_feats_v')
            condition_feats_a = self.batch_get(batch, 'condition_feats_a')
            condition_feats_s = self.batch_get(batch, 'condition_feats_s')
            condition_feats_vas = torch.cat((condition_feats_v, condition_feats_a, condition_feats_s),dim=1)
            batch[key] = condition_feats_vas
            
        elif key == 'condition_feats_vasd':
            condition_feats_v = self.batch_get(batch, 'condition_feats_v')
            condition_feats_a = self.batch_get(batch, 'condition_feats_a')
            condition_feats_s = self.batch_get(batch, 'condition_feats_s')
            condition_feats_d = self.batch_get(batch, 'condition_feats_d')
            condition_feats_vas = torch.cat((condition_feats_v, condition_feats_a, condition_feats_s, condition_feats_d),dim=1)
            batch[key] = condition_feats_vas



        elif key == 'feat_v':
            vision_output = self.batch_get(batch, 'vision_output')
            vision_output_pooled = self.pool_vision_for_contra(vision_output)
            feat_v = self.contra_head_v(vision_output_pooled)
            batch['u_v'] = feat_v   # pre-norm proj for hypergraph loss_reg (stage B)
            feat_v = F.normalize(feat_v,dim=-1)
            batch[key] = feat_v
        
        elif key == 'feat_d':
            depth_output = self.batch_get(batch, 'depth_output')
            depth_output_pooled = self.pool_vision_for_contra(depth_output)
            feat_d = self.contra_head_d(depth_output_pooled) #TODO do we have to use a contra head depth?
            batch['u_d'] = feat_d   # pre-norm proj for hypergraph loss_reg (stage B)
            feat_d = F.normalize(feat_d,dim=-1)
            batch[key] = feat_d
        
        elif key == 'feat_a':
            audio_output = self.batch_get(batch, 'audio_output')
            audio_output_pooled = self.pool_audio_for_contra(audio_output)
            feat_a = self.contra_head_a(audio_output_pooled)
            batch['u_a'] = feat_a   # pre-norm proj for hypergraph loss_reg (stage B)
            feat_a = F.normalize(feat_a,dim=-1)
            batch[key] = feat_a       

        elif key == 'feat_s':
            subtitle_output = self.batch_get(batch, 'subtitle_output')
            subtitle_output_pooled = self.pool_text_for_contra(subtitle_output)
            feat_s = self.contra_head_s(subtitle_output_pooled)
            batch['u_s'] = feat_s   # pre-norm proj for hypergraph loss_reg (stage B)
            feat_s = F.normalize(feat_s,dim=-1)
            batch[key] = feat_s

        elif key == 'feat_t':
            caption_output = self.batch_get(batch, 'caption_output')
            caption_output_pooled = self.pool_text_for_contra(caption_output)
            feat_t = self.contra_head_t(caption_output_pooled)
            batch['u_t'] = feat_t   # pre-norm proj for hypergraph loss_reg (stage B)
            feat_t = F.normalize(feat_t,dim=-1)
            batch[key] = feat_t

        elif key == 'feat_va':
            vision_output = self.batch_get(batch, 'vision_output')
            vision_output_pooled = self.pool_vision_for_contra(vision_output)
            audio_output = self.batch_get(batch, 'audio_output')
            audio_output_pooled = self.pool_audio_for_contra(audio_output)
            feat_va = torch.cat((vision_output_pooled, audio_output_pooled), dim=1)
            feat_va = self.contra_head_va(feat_va)
            feat_va = F.normalize(feat_va,dim=-1)
            batch[key] = feat_va

        elif key == 'feat_vs': 
            vision_output = self.batch_get(batch, 'vision_output')
            vision_output_pooled = self.pool_vision_for_contra(vision_output)
            subtitle_output = self.batch_get(batch, 'subtitle_output')
            subtitle_output_pooled = self.pool_text_for_contra(subtitle_output)
            feat_vs = torch.cat((vision_output_pooled, subtitle_output_pooled), dim=1)
            feat_vs = self.contra_head_vs(feat_vs)
            feat_vs = F.normalize(feat_vs,dim=-1) 
            batch[key] = feat_vs

        elif key == 'feat_vas':     
            vision_output = self.batch_get(batch, 'vision_output')
            vision_output_pooled = self.pool_vision_for_contra(vision_output)
            audio_output = self.batch_get(batch, 'audio_output')
            audio_output_pooled = self.pool_audio_for_contra(audio_output)
            subtitle_output = self.batch_get(batch, 'subtitle_output')
            subtitle_output_pooled = self.pool_text_for_contra(subtitle_output)
            feat_vas = torch.cat((vision_output_pooled, audio_output_pooled, subtitle_output_pooled), dim=1)
            feat_vas = self.contra_head_vas(feat_vas)
            feat_vas = F.normalize(feat_vas,dim=-1)
            batch[key] = feat_vas  
            
            

        elif key == 'feat_t_omni_caption':
            caption_tokens = self.batch_get(batch, 'omni_caption_tokens')
            input_ids = caption_tokens.input_ids
            attention_mask = caption_tokens.attention_mask
            caption_tokens = self.multimodal_encoder.bert(input_ids = input_ids,
                                            attention_mask = attention_mask).last_hidden_state
            caption_tokens_pooled = self.pool_text_for_contra(caption_tokens)
            feat_t = self.contra_head_t(caption_tokens_pooled) 
            feat_t = F.normalize(feat_t,dim=-1)
            batch[key] = feat_t

        elif key == 'feat_t_vision_caption':
            caption_tokens = self.batch_get(batch, 'vision_caption_tokens')
            input_ids = caption_tokens.input_ids
            attention_mask = caption_tokens.attention_mask
            caption_tokens = self.multimodal_encoder.bert(input_ids = input_ids,
                                            attention_mask = attention_mask).last_hidden_state
            caption_tokens_pooled = self.pool_text_for_contra(caption_tokens)
            feat_t = self.contra_head_t(caption_tokens_pooled) 
            feat_t = F.normalize(feat_t,dim=-1)
            batch[key] = feat_t

        elif key == 'feat_t_audio_caption':
            caption_tokens = self.batch_get(batch, 'audio_caption_tokens')
            input_ids = caption_tokens.input_ids
            attention_mask = caption_tokens.attention_mask
            caption_tokens = self.multimodal_encoder.bert(input_ids = input_ids,
                                            attention_mask = attention_mask).last_hidden_state
            caption_tokens_pooled = self.pool_text_for_contra(caption_tokens)
            feat_t = self.contra_head_t(caption_tokens_pooled) 
            feat_t = F.normalize(feat_t,dim=-1)
            batch[key] = feat_t

        return batch[key] 


    def forward(self, batch, task, compute_loss=True):
        batch = edict(batch)
        ### gram-27m pretraining
        #if 'vision_captions' in batch or 'audio_captions' in batch or 'omni_captions' in batch:
        #    assert compute_loss
        #    return self.forward_vast27m(batch, task)

        ### other datasets pretraining or finetuning
        output_ls = []
        task_ls = task.split('_')


        for task in task_ls:
            if task.startswith('ret'):
                ret_dict = self.forward_ret(batch, task, compute_loss=compute_loss)
                output_ls.append(ret_dict)

            elif task.startswith('cap'):
                cap_dict = self.forward_cap(batch, task, compute_loss=compute_loss)
                output_ls.append(cap_dict)

            elif task.startswith('qa'):
                qa_dict = self.forward_qa(batch, task, compute_loss=compute_loss)
                output_ls.append(qa_dict)
            
            else:
                raise NotImplementedError
    

        output_dict = {k:v for dic in output_ls for k,v in dic.items()  }
        return output_dict


    def forward_vast27m(self, batch, task):
     
        output_ls = []
        task_ls = task.split('_')


        for task in task_ls:
            if task.startswith('ret'):
                ret_dict = self.forward_ret_vast27m(batch, task)
                output_ls.append(ret_dict)

            elif task.startswith('cap'):
                cap_dict = self.forward_cap_vast27m(batch, task)
                output_ls.append(cap_dict)
            
            else:
                raise NotImplementedError
    

        output_dict = {k:v for dic in output_ls for k,v in dic.items()  }
        return output_dict

    def compute_slice_scores(self, slice_multimodal_vision_input, slice_input_ids, slice_attention_mask):
            
        slice_output = self.multimodal_encoder.bert(input_ids = slice_input_ids,
                                                    attention_mask = slice_attention_mask,
                                                    encoder_hidden_states=slice_multimodal_vision_input).last_hidden_state
        slice_scores = F.softmax(self.itm_head(slice_output[:,0]),dim=1)[:,1]

        return slice_scores


    def _hg_refine(self, feats, t_frozen=None, use_semantic=False):

        from .hypergraph import doc_incidence, mutual_knn_adj, semantic_incidence
        mask = tuple(m for m in ('V', 'A', 'S', 'D') if m in feats)
        B = feats[mask[0]].shape[0]
        device = feats[mask[0]].device
        # per-clip modality presence (zero-filled feature = modality absent); masked out of doc edge + fusion mean.
        pres = torch.stack([(feats[m].float().norm(dim=-1) > 0.5).float() for m in mask], dim=1)
        H_doc = doc_incidence(B, mask, device, present=pres).float()
        H_sem = None
        if use_semantic and self.training and self.semantic_edges and t_frozen is not None:
            mask_bin = mutual_knn_adj(t_frozen.detach(), k=self.knn_k,
                                 edge_dropout=self.edge_dropout, training=True,
                                 sim_std=self.sem_sim_std)
            with torch.cuda.amp.autocast(enabled=False):
                adj = self.hgnn.edge_attention(t_frozen.detach().float(), mask_bin) if mask_bin is not None else None
            H_sem = semantic_incidence(adj, B, mask, device)
            if H_sem is not None:
                H_sem = H_sem.float()
        with torch.cuda.amp.autocast(enabled=False):
            z32 = {m: feats[m].float() for m in feats}
            z_hat, h, h_prenorm = self.hgnn(z32, mask, H_doc, H_sem, present=pres)
        return z_hat, h, h_prenorm

    def forward_ret(self, batch, task, compute_loss=True):

        if isinstance(batch.raw_captions[0],list): #### test
            batch.raw_captions = [i for j in batch.raw_captions for i in j]
        subtasks = task.split('%')[1:]  
        if compute_loss:
            loss_dict={}
            loss_itc = []
            loss_itm = []
            loss_area = []

            #Extract text features
            feat_t = self.batch_get(batch,'feat_t')
            #Extract visual features
            feat_v = self.batch_get(batch,'feat_v')
            #Extract audio features
            feat_a = self.batch_get(batch,'feat_a')
            #Extract subtitles features
            if "raw_subtitles" in batch.keys():
                feat_s = self.batch_get(batch,'feat_s')
            #extract depth features
            if "depth_pixels" in batch.keys():
                feat_d = self.batch_get(batch,'feat_d')

            # ---- Hypergraph refinement (stage B): refine gallery (non-text) modalities BEFORE the gather,
            # so HyperGRAM's own volume loss (hybrid/hyperbolic/euclidean geometry, below) is computed on the
            # refined embeddings. feat_t stays the raw anchor (leak-free). The eval branch does NOT refine ->
            # GRAM/HyperGRAM-faithful, identical to HyperAlign's design.
            hg_h = hg_h_prenorm = None
            if self.stage == 'B':
                _f = {'V': feat_v, 'A': feat_a}
                if "raw_subtitles" in batch.keys(): _f['S'] = feat_s
                if "depth_pixels" in batch.keys():  _f['D'] = feat_d
                z_hat, hg_h, hg_h_prenorm = self._hg_refine(_f, t_frozen=feat_t, use_semantic=True)
                self._gc = getattr(self, "_gc", 0) + 1
                if self._gc % 50 == 1 and dist.get_rank() == 0:
                    print(f"[GATE] step~{self._gc}: {self.hgnn.gates.detach().float().tolist()}", flush=True)
                feat_v, feat_a = z_hat['V'], z_hat['A']        # feat_t UNCHANGED (raw)
                if 'S' in z_hat: feat_s = z_hat['S']
                if 'D' in z_hat: feat_d = z_hat['D']

            feat_t_all = concat_all_gather(feat_t)
            feat_v_all = concat_all_gather(feat_v)
            feat_a_all = concat_all_gather(feat_a)
            if "raw_subtitles" in batch.keys():
                feat_s_all = concat_all_gather(feat_s)
            if "depth_pixels" in batch.keys():
                feat_d_all = concat_all_gather(feat_d)
            #TODO extract other k features
            
            caption_tokens = self.batch_get(batch, 'caption_tokens')
            input_ids, attention_mask = caption_tokens.input_ids, caption_tokens.attention_mask
            input_ids_collate = concat_all_gather(input_ids)
            attention_mask_collate = concat_all_gather(attention_mask)


            # Volume loss computation (volume ITC)
            # PMRL mode selection
            if self.geometry_mode == 'pmrl':
                # Pure PMRL mode (no volume)
                return self._forward_ret_pmrl(
                    batch, feat_t, feat_v, feat_a,
                    feat_t_all, feat_v_all, feat_a_all,
                    input_ids, attention_mask,
                    input_ids_collate, attention_mask_collate
                )

            elif self.geometry_mode == 'pmrl_volume':
                # PMRL + Volume mode
                return self._forward_ret_pmrl_volume(
                    batch, feat_t, feat_v, feat_a,
                    feat_t_all, feat_v_all, feat_a_all,
                    input_ids, attention_mask,
                    input_ids_collate, attention_mask_collate
                )

            elif self.geometry_mode == 'hybrid_pmrl':
                # Hybrid PMRL mode
                return self._forward_ret_hybrid_pmrl(
                    batch, feat_t, feat_v, feat_a,
                    feat_t_all, feat_v_all, feat_a_all,
                    input_ids, attention_mask,
                    input_ids_collate, attention_mask_collate
                )

            # Choose between Euclidean, Hyperbolic, and Hybrid volume computation
            elif self.geometry_mode == 'hyperbolic':
                from utils.hyperbolic_volume import hyperbolic_volume2, hyperbolic_volume3, hyperbolic_volume4, hyperbolic_volume5

                # Get curvatures (support both shared and per-modality)
                if hasattr(self, 'curvatures'):
                    curvs = self.curvatures
                else:
                    curvs = self.curvature

                # Apply gradient clipping for hyperbolic mode
                if self.training:
                    # Clip gradients to prevent explosion
                    torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)

                    # Monitor and clamp curvature if learnable
                    if self.learn_curvature and hasattr(self, 'curvature'):
                        # Keep curvature in reasonable range [0.1, 10.0]
                        with torch.no_grad():
                            self.curvature.data = torch.clamp(self.curvature.data, min=0.1, max=10.0)

                #Volume (Text, batch_all)
                if "raw_subtitles" in batch.keys():
                    if "depth_pixels" in batch.keys():
                        volume = hyperbolic_volume5(feat_t, feat_v_all, feat_a_all, feat_s_all, feat_d_all, curvs)
                        volume = volume / self.contra_temp
                    else:
                        volume = hyperbolic_volume4(feat_t, feat_v_all, feat_a_all, feat_s_all, curvs)
                        volume = volume / self.contra_temp
                else:
                    volume = hyperbolic_volume3(feat_t, feat_v_all, feat_a_all, curvs)
                    volume = volume / self.contra_temp

                #AreaT (Video,batch_all)
                if "raw_subtitles" in batch.keys():
                    if "depth_pixels" in batch.keys():
                        volumeT = hyperbolic_volume5(feat_t_all, feat_v, feat_a, feat_s, feat_d, curvs).T
                        volumeT = volumeT / self.contra_temp
                    else:
                        volumeT = hyperbolic_volume4(feat_t_all, feat_v, feat_a, feat_s, curvs).T
                        volumeT = volumeT / self.contra_temp
                else:
                    volumeT = hyperbolic_volume3(feat_t_all, feat_v, feat_a, curvs).T
                    volumeT = volumeT / self.contra_temp
            elif self.geometry_mode == 'hybrid':
                # Hybrid Euclidean-Hyperbolic volume computation
                from utils.hyperbolic_volume import hybrid_volume2, hybrid_volume3, hybrid_volume4, hybrid_volume5

                # Get curvatures and normalize weights
                if hasattr(self, 'curvatures'):
                    curvs = self.curvatures
                else:
                    curvs = self.curvature

                # Normalize weights to sum to 1
                weight_sum = torch.abs(self.euclidean_weight) + torch.abs(self.hyperbolic_weight)
                euclidean_weight = torch.abs(self.euclidean_weight) / weight_sum
                hyperbolic_weight = torch.abs(self.hyperbolic_weight) / weight_sum

                # Volume (Text, batch_all)
                if "raw_subtitles" in batch.keys():
                    if "depth_pixels" in batch.keys():
                        volume = hybrid_volume5(feat_t, feat_v_all, feat_a_all, feat_s_all, feat_d_all,
                                              euclidean_weight, hyperbolic_weight, curvs)
                        volume = volume / self.contra_temp
                    else:
                        volume = hybrid_volume4(feat_t, feat_v_all, feat_a_all, feat_s_all,
                                              euclidean_weight, hyperbolic_weight, curvs)
                        volume = volume / self.contra_temp
                else:
                    volume = hybrid_volume3(feat_t, feat_v_all, feat_a_all,
                                          euclidean_weight, hyperbolic_weight, curvs)
                    volume = volume / self.contra_temp

                # VolumeT (Video, batch_all)
                if "raw_subtitles" in batch.keys():
                    if "depth_pixels" in batch.keys():
                        volumeT = hybrid_volume5(feat_t_all, feat_v, feat_a, feat_s, feat_d,
                                               euclidean_weight, hyperbolic_weight, curvs).T
                        volumeT = volumeT / self.contra_temp
                    else:
                        volumeT = hybrid_volume4(feat_t_all, feat_v, feat_a, feat_s,
                                               euclidean_weight, hyperbolic_weight, curvs).T
                        volumeT = volumeT / self.contra_temp
                else:
                    volumeT = hybrid_volume3(feat_t_all, feat_v, feat_a,
                                           euclidean_weight, hyperbolic_weight, curvs).T
                    volumeT = volumeT / self.contra_temp
            else:
                # Original Euclidean volume computation
                #Volume (Text, batch_all)
                if "raw_subtitles" in batch.keys():
                    if "depth_pixels" in batch.keys():
                        volume = volume_computation5(feat_t,feat_v_all,feat_a_all,feat_s_all,feat_d_all)
                        volume = volume / self.contra_temp
                    else:
                        volume = volume_computation4(feat_t,feat_v_all,feat_a_all,feat_s_all)
                        volume = volume / self.contra_temp
                else:
                    volume = volume_computation3(feat_t,feat_v_all,feat_a_all)
                    volume = volume / self.contra_temp
                #AreaT (Video,batch_all)
                if "raw_subtitles" in batch.keys():
                    if "depth_pixels" in batch.keys():
                        volumeT = volume_computation5(feat_t_all,feat_v,feat_a,feat_s,feat_d).T
                        volumeT = volumeT / self.contra_temp
                    else:
                        volumeT = volume_computation4(feat_t_all,feat_v,feat_a,feat_s).T
                        volumeT = volumeT / self.contra_temp
                else:
                    volumeT = volume_computation3(feat_t_all,feat_v,feat_a).T
                    volumeT = volumeT / self.contra_temp
            rank = dist.get_rank()
            bs = feat_t.size(0)
            targets = torch.linspace(rank * bs, rank * bs + bs - 1, bs, dtype=int).to(volume.device)

            loss = (
                    F.cross_entropy(-volume, targets, label_smoothing=0.1) #d2a
                    + F.cross_entropy(-volumeT, targets, label_smoothing=0.1) #a2d
            ) / 2

            loss_area.append(loss)


            #   AREA VID ITM 
            #vid_sim = feat_t @ feat_v_all.T
            #vid_simT = feat_v @ feat_t_all.T
                     
            condition_feats = self.batch_get(batch, f'condition_feats_va')#self.batch_get(batch, f'condition_feats_v')
            condition_feats_collate = all_gather_with_grad(condition_feats)
            with torch.no_grad():
                weights_t2cond = F.softmax(-(volume), dim=1) + 1e-4
                weights_t2cond[:, rank * bs : rank * bs + bs].fill_diagonal_(0)
                weights_cond2t = F.softmax(-(volumeT), dim=1) + 1e-4
                weights_cond2t[:, rank * bs : rank * bs + bs].fill_diagonal_(0)

            condition_feats_neg = []
            for b in range(bs): 
                neg_idx = torch.multinomial(weights_t2cond[b], 1).item()
                condition_feats_neg.append(condition_feats_collate[neg_idx])
            condition_feats_neg = torch.stack(condition_feats_neg, dim=0)

            text_ids_neg = []
            text_atts_neg = []
            for b in range(bs):
                neg_idx = torch.multinomial(weights_cond2t[b], 1).item()
                text_ids_neg.append(input_ids_collate[neg_idx])
                text_atts_neg.append(attention_mask_collate[neg_idx])

            text_ids_neg = torch.stack(text_ids_neg, dim=0)
            text_atts_neg = torch.stack(text_atts_neg, dim=0)
        
            input_ids_1 = torch.cat((input_ids, input_ids, text_ids_neg),dim=0)
            attention_mask_1 = torch.cat((attention_mask, attention_mask, text_atts_neg),dim=0)
            
            condition_feats = torch.cat((condition_feats,condition_feats_neg,condition_feats),dim=0)
            output = self.multimodal_encoder.bert(input_ids = input_ids_1,
                                        attention_mask = attention_mask_1,
                                        encoder_hidden_states=condition_feats).last_hidden_state
            batch_size = condition_feats_neg.shape[0]
            logits = self.itm_head(output[:,0].half())
            ground_truth = torch.zeros(batch_size*3).long().cuda()
            ground_truth[:batch_size] = 1
            loss = F.cross_entropy(logits,ground_truth) #itm (dtm)
            loss_itm.append(self.itm_ratio * loss)

            

            for task in subtasks:


                # #### compute_itc
                # assert task in ['tv','ta','tva','tvs','tvas']
                # #feat_cond = self.batch_get(batch,f'feat_{task[1:]}')
                # #feat_cond_all = concat_all_gather(feat_cond)
                # sim_cond2t = torch.matmul(feat_cond, feat_t_all.permute(1,0))
                # sim_cond2t = sim_cond2t / self.contra_temp
                # sim_t2cond = torch.matmul(feat_t, feat_cond_all.permute(1,0))
                # sim_t2cond = sim_t2cond / self.contra_temp  # [batch_size, batch_size*num_gpu]
                # rank = dist.get_rank()
                # bs = feat_t.size(0)
                # targets = torch.linspace(rank * bs, rank * bs + bs - 1, bs, dtype=int).to(feat_cond.device)
                # loss = (
                #     F.cross_entropy(sim_cond2t, targets, label_smoothing=0.1)
                #     + F.cross_entropy(sim_t2cond, targets, label_smoothing=0.1)
                # ) / 2
                
                loss_itc.append(torch.tensor(0))#*loss)

                #### compute_itm
         
                # condition_feats = self.batch_get(batch, f'condition_feats_{task[1:]}')
                # condition_feats_collate = all_gather_with_grad(condition_feats)
                # with torch.no_grad():
                #     weights_t2cond = F.softmax(sim_t2cond, dim=1) + 1e-4
                #     weights_t2cond[:, rank * bs : rank * bs + bs].fill_diagonal_(0)
                #     weights_cond2t = F.softmax(sim_cond2t, dim=1) + 1e-4
                #     weights_cond2t[:, rank * bs : rank * bs + bs].fill_diagonal_(0)

                # condition_feats_neg = []
                # for b in range(bs): 
                #     neg_idx = torch.multinomial(weights_t2cond[b], 1).item()
                #     condition_feats_neg.append(condition_feats_collate[neg_idx])
                # condition_feats_neg = torch.stack(condition_feats_neg, dim=0)

                # text_ids_neg = []
                # text_atts_neg = []
                # for b in range(bs):
                #     neg_idx = torch.multinomial(weights_cond2t[b], 1).item()
                #     text_ids_neg.append(input_ids_collate[neg_idx])
                #     text_atts_neg.append(attention_mask_collate[neg_idx])

                # text_ids_neg = torch.stack(text_ids_neg, dim=0)
                # text_atts_neg = torch.stack(text_atts_neg, dim=0)
        
                # input_ids_1 = torch.cat((input_ids, input_ids, text_ids_neg),dim=0)
                # attention_mask_1 = torch.cat((attention_mask, attention_mask, text_atts_neg),dim=0)
            
                # condition_feats = torch.cat((condition_feats,condition_feats_neg,condition_feats),dim=0)
                # output = self.multimodal_encoder.bert(input_ids = input_ids_1,
                #                             attention_mask = attention_mask_1,
                #                             encoder_hidden_states=condition_feats).last_hidden_state
                # batch_size = condition_feats_neg.shape[0]
                # logits = self.itm_head(output[:,0].half())
                # ground_truth = torch.zeros(batch_size*3).long().cuda()
                # ground_truth[:batch_size] = 1
                # loss = F.cross_entropy(logits,ground_truth)
                #loss_itm.append(torch.tensor(0))#*(self.itm_ratio * loss))

            loss_itc = sum(loss_itc)/len(loss_itc)
            loss_dict['loss_itc'] = loss_itc          
            loss_itm = sum(loss_itm)/len(loss_itm)
            loss_dict['loss_itm'] = loss_itm
            loss_area = sum(loss_area)/len(loss_area)
            loss_dict['loss_area'] = loss_area

            # ---- hypergraph auxiliaries (stage B); loss_area above is the graph's retrieval loss ----
            if self.stage == 'B':
                u = {'T': batch['u_t'], 'V': batch['u_v'], 'A': batch['u_a']}
                if "raw_subtitles" in batch.keys(): u['S'] = batch['u_s']
                if "depth_pixels" in batch.keys():  u['D'] = batch['u_d']
                # loss_doc: text anchor vs doc-embedding h, arity-2 Gramian volume (Euclidean, geometry-independent)
                h_all = all_gather_with_grad(hg_h)
                ztl_all = concat_all_gather(feat_t).float()
                vol_t2h = volume_computation2(feat_t.float(), h_all) / self.contra_temp
                vol_h2t = volume_computation2(hg_h, ztl_all) / self.contra_temp
                loss_doc = (F.cross_entropy(-vol_t2h, targets, label_smoothing=0.1)
                            + F.cross_entropy(-vol_h2t, targets, label_smoothing=0.1)) / 2
                # loss_reg: VICReg-style variance hinge on pre-norm feats + h_prenorm
                def _vh(x):
                    std = torch.sqrt(x.float().var(dim=0) + 1e-4)
                    return F.relu(1.0 - std).mean()
                loss_reg = sum(_vh(u[m]) for m in u) + _vh(hg_h_prenorm)
                loss_dict['loss_doc'] = self.w_doc * loss_doc
                loss_dict['loss_reg'] = self.w_reg * loss_reg

            return loss_dict
          
        else:

            evaluation_dict = {}
            feat_t = self.batch_get(batch,'feat_t')
            evaluation_dict['feat_t'] = feat_t 
            feat_v = self.batch_get(batch,'feat_v')
            evaluation_dict['feat_v'] = feat_v 
            feat_a = self.batch_get(batch,'feat_a')
            evaluation_dict['feat_a'] = feat_a
            
            if "raw_subtitles" in batch.keys(): 
                feat_s = self.batch_get(batch,'feat_s')
                evaluation_dict['feat_s'] = feat_s
            if "depth_pixels" in batch.keys():
                feat_d = self.batch_get(batch,'feat_d')
                evaluation_dict['feat_d'] = feat_d
            

            caption_tokens = self.batch_get(batch,'caption_tokens')
            evaluation_dict['input_ids'] = caption_tokens.input_ids
            evaluation_dict['attention_mask'] = caption_tokens.attention_mask
            for task in subtasks:
                #### compute_itc
                assert task in ['tv','ta','tva','tvs','tvas','tvasd']
                # feat_cond = self.batch_get(batch,f'feat_{task[1:]}')
                # evaluation_dict[f'feat_cond_{task}'] = feat_cond

                condition_feats = self.batch_get(batch, f'condition_feats_{task[1:]}')
                evaluation_dict[f'condition_feats_{task}'] = condition_feats

            return evaluation_dict

    def forward_cap(self, batch, task, compute_loss=True):
        subtasks = task.split('%')[1:]

        if compute_loss:

            loss_dict = {}
            loss_ls_cap = []

            caption_tokens = self.batch_get(batch, 'caption_tokens')
            input_ids, attention_mask = caption_tokens.input_ids, caption_tokens.attention_mask
            input_ids, txt_labels = self.text_masker(input_ids, 0.6)
            
            seq_len = attention_mask.shape[1]
            attention_mask = attention_mask.unsqueeze(1).expand(-1, seq_len, -1).clone()
            attention_mask[:, : seq_len, : seq_len] = torch.tril(attention_mask[:, : seq_len, : seq_len])

            for task in subtasks:
                assert task in ['tv','ta','tva','tvs','tvas']
                condition_feats = self.batch_get(batch, f'condition_feats_{task[1:]}')
                output = self.multimodal_encoder(input_ids = input_ids,
                                                attention_mask = attention_mask,
                                                encoder_hidden_states=condition_feats,
                                                labels = txt_labels)
                loss_ls_cap.append(output.loss)

            loss_cap = sum(loss_ls_cap)/len(loss_ls_cap)
            loss_dict['loss_cap'] = loss_cap
            return loss_dict

        else:
            evaluation_dict = {}
            for task in subtasks:
                assert task in ['tv','ta','tva','tvs','tvas']
                condition_feats = self.batch_get(batch, f'condition_feats_{task[1:]}')

                batch_size = condition_feats.shape[0]
                if self.config.captioner_mode:
                    batch_size *=self.config.generate_nums

                init_input_ids = torch.ones(batch_size, 1).long().cuda().fill_(self.multimodal_encoder.tokenizer.bos_token_id)
                init_attention_mask = init_input_ids.new_ones(batch_size, 1, 1)
                
                if self.config.captioner_mode:
                    condition_feats = condition_feats.unsqueeze(1).expand(-1, self.config.generate_nums,-1,-1).reshape(-1,*condition_feats.shape[1:])
                    outputs = self.multimodal_encoder.generate( input_ids=init_input_ids,
                                                            attention_mask=init_attention_mask,
                                                            do_sample = True,
                                                            top_k = 10,
                                                            encoder_hidden_states=condition_feats,
                                                            max_new_tokens=self.max_caption_len,
                                                            eos_token_id=self.multimodal_encoder.tokenizer.sep_token_id,
                                                            pad_token_id=self.multimodal_encoder.tokenizer.pad_token_id) 

                                                        
                else:
                    outputs = self.multimodal_encoder.generate( input_ids=init_input_ids,
                                        attention_mask=init_attention_mask,
                                        encoder_hidden_states=condition_feats,
                                        max_new_tokens=self.max_caption_len,
                                        num_beams=self.beam_size,
                                        eos_token_id=self.multimodal_encoder.tokenizer.sep_token_id,
                                        pad_token_id=self.multimodal_encoder.tokenizer.pad_token_id,
                                        length_penalty=0.6) 
                                                        
                outputs_newgen = outputs[:,1:]
                captions = self.multimodal_encoder.tokenizer.batch_decode(outputs_newgen, skip_special_tokens=True)
                evaluation_dict[f'generated_captions_{task}'] = captions

            return evaluation_dict



    def forward_qa(self, batch, task, compute_loss=True):
        subtasks = task.split('%')[1:]
        raw_questions = batch.raw_questions
        raw_answers = batch.raw_answers
      
        if isinstance(raw_questions[0],list): #### test
            # raw_batch_size = len(raw_questions)
            num_questions = [len(i) for i in raw_questions]
            raw_questions = [j for d in raw_questions for j in d]

        question_tokens = self.multimodal_encoder.tokenizer(raw_questions,
                                                            padding="max_length",
                                                            truncation=True,
                                                            max_length=self.max_caption_len,
                                                            return_tensors="pt").to(torch.device('cuda'))

        question_tokens_ids, question_tokens_mask = question_tokens.input_ids, question_tokens.attention_mask

        if compute_loss:

            loss_dict = {}
            loss_ls_qa = []

            answer_tokens = self.multimodal_encoder.tokenizer(raw_answers,
                                                    padding="max_length",
                                                    truncation=True,
                                                    max_length=10,
                                                    return_tensors="pt")
    
            answer_tokens = answer_tokens.to(torch.device('cuda'))
            answer_tokens_ids, answer_tokens_mask = answer_tokens.input_ids, answer_tokens.attention_mask
            input_ids, txt_labels = self.text_masker(answer_tokens_ids, 0.99)
            input_ids = torch.cat((question_tokens_ids,input_ids),dim=1)
            attention_mask = torch.cat((question_tokens_mask,answer_tokens_mask),dim=1)
            dummy_labels = (-100*torch.ones_like(question_tokens_ids)).cuda()
            txt_labels = torch.cat((dummy_labels,txt_labels),dim=1)

            #### part-causal attention mask
            question_len = question_tokens_ids.shape[1]
            seq_len = attention_mask.shape[1]
            attention_mask = attention_mask.unsqueeze(1).expand(-1, seq_len, -1).clone()
            attention_mask[:, question_len: seq_len, question_len: seq_len] = torch.tril(attention_mask[:, question_len: seq_len, question_len: seq_len])
            attention_mask[:, :question_len, question_len:seq_len] = 0


            for task in subtasks:
                assert task in ['tv','ta','tva','tvs','tvas']
                condition_feats = self.batch_get(batch, f'condition_feats_{task[1:]}')
                output = self.multimodal_encoder(input_ids = input_ids,
                                attention_mask = attention_mask,
                                encoder_hidden_states=condition_feats,
                                labels = txt_labels)
                loss_ls_qa.append(output.loss)
            
            loss_qa = sum(loss_ls_qa)/len(loss_ls_qa)
            loss_dict['loss_qa'] = loss_qa
            return loss_dict
        
        else:
            evaluation_dict = {} 
            batch_size = sum(num_question)
            init_input_ids = torch.ones(batch_size, 1).long().cuda().fill_(self.multimodal_encoder.tokenizer.bos_token_id)
            init_input_ids = torch.cat((question_tokens['input_ids'],init_input_ids),dim=1)
            question_len = question_tokens['input_ids'].shape[1]
            seq_len = init_input_ids.shape[1]
            attention_mask = question_tokens['attention_mask'].unsqueeze(1).expand(-1, question_len, -1).clone()
            init_attention_mask = self.multimodal_encoder.update_attention_mask(attention_mask)


            for task in subtasks:
                assert task in ['tv','ta','tva','tvs','tvas']
                condition_feats = self.batch_get(batch, f'condition_feats_{task[1:]}')
            
                condition_feats_expand = []
                for i in range(condition_feats.shape[0]):
                    condition_feats_expand.append( condition_feats[i:i+1].expand(num_questions[i],-1,-1))
                condition_feats = torch.cat(condition_feats_expand,dim=0)
                batch_size = condition_feats.shape[0]
                       
                outputs = self.multimodal_encoder.generate( input_ids=init_input_ids,
                                                            attention_mask=init_attention_mask,
                                                            encoder_hidden_states=condition_feats,
                                                            max_new_tokens=10,
                                                            num_beams=self.beam_size,
                                                            eos_token_id=self.multimodal_encoder.tokenizer.sep_token_id,
                                                            pad_token_id=self.multimodal_encoder.tokenizer.pad_token_id) 
                
                outputs_newgen = outputs[:,seq_len:]
                answers = self.multimodal_encoder.tokenizer.batch_decode(outputs_newgen, skip_special_tokens=True)
                print(answers)
                evaluation_dict[f'generated_answers_{task}'] = answers


            return evaluation_dict




    def forward_cap_vast27m(self, batch, task):


        subtasks = task.split('%')[1:]
        loss_dict = {}
        loss_ls_cap = []

        for task in subtasks:
            assert task in ['tv','ta','tva','tvs','tvas']
            if task == 'tv':
                caption_tokens = self.batch_get(batch, 'vision_caption_tokens')

            elif task == 'ta':
                caption_tokens = self.batch_get(batch, 'audio_caption_tokens')
                
            else:
                caption_tokens = self.batch_get(batch, 'omni_caption_tokens')

            input_ids, attention_mask = caption_tokens.input_ids, caption_tokens.attention_mask
            input_ids, txt_labels = self.text_masker(input_ids, 0.6)
            seq_len = attention_mask.shape[1]
            attention_mask = attention_mask.unsqueeze(1).expand(-1, seq_len, -1).clone()
            attention_mask[:, : seq_len, : seq_len] = torch.tril(attention_mask[:, : seq_len, : seq_len])
            condition_feats = self.batch_get(batch, f'condition_feats_{task[1:]}')
            output = self.multimodal_encoder(input_ids = input_ids,
                                            attention_mask = attention_mask,
                                            encoder_hidden_states=condition_feats,
                                            labels = txt_labels)
            loss_ls_cap.append(output.loss)

        loss_cap = sum(loss_ls_cap)/len(loss_ls_cap)
        loss_dict['loss_cap'] = loss_cap
        return loss_dict


    def forward_ret_vast27m(self, batch, task):
        

        subtasks = task.split('%')[1:]  
        if compute_loss:
            loss_dict={}
            loss_itc = []
            loss_itm = []
 
            for task in subtasks:
                #### compute_itc
                assert task in ['tv','ta','tva','tvs','tvas']
                if task == 'tv':
                    feat_t = self.batch_get(batch,'feat_t_vision_caption')
                    caption_tokens = self.batch_get(batch, 'vision_caption_tokens')
                elif task == 'ta':
                    feat_t = self.batch_get(batch,'feat_t_audio_caption')
                    caption_tokens = self.batch_get(batch, 'audio_caption_tokens')
                else:
                    feat_t = self.batch_get(batch,'feat_t_omni_caption')
                    caption_tokens = self.batch_get(batch, 'omni_caption_tokens', txt_len = self.max_omni_caption_len)
               
                feat_t_all = concat_all_gather(feat_t) 

                input_ids, attention_mask = caption_tokens.input_ids, caption_tokens.attention_mask
                input_ids_collate = concat_all_gather(input_ids)
                attention_mask_collate = concat_all_gather(attention_mask)

                feat_cond = self.batch_get(batch,f'feat_{task[1:]}')
                feat_cond_all = concat_all_gather(feat_cond)
                sim_cond2t = torch.matmul(feat_cond, feat_t_all.permute(1,0))
                sim_cond2t = sim_cond2t / self.contra_temp
                sim_t2cond = torch.matmul(feat_t, feat_cond_all.permute(1,0))
                sim_t2cond = sim_t2cond / self.contra_temp  # [batch_size, batch_size*num_gpu]
                rank = dist.get_rank()
                bs = feat_t.size(0)
                targets = torch.linspace(rank * bs, rank * bs + bs - 1, bs, dtype=int).to(feat_cond.device)
                loss = (
                    F.cross_entropy(sim_cond2t, targets, label_smoothing=0.1)
                    + F.cross_entropy(sim_t2cond, targets, label_smoothing=0.1)
                ) / 2
                
                loss_itc.append(loss)

                #### compute_itm
         
                condition_feats = self.batch_get(batch, f'condition_feats_{task[1:]}')
                condition_feats_collate = all_gather_with_grad(condition_feats)
                with torch.no_grad():
                    weights_t2cond = F.softmax(sim_t2cond, dim=1) + 1e-4
                    weights_t2cond[:, rank * bs : rank * bs + bs].fill_diagonal_(0)
                    weights_cond2t = F.softmax(sim_cond2t, dim=1) + 1e-4
                    weights_cond2t[:, rank * bs : rank * bs + bs].fill_diagonal_(0)

                condition_feats_neg = []
                for b in range(bs): 
                    neg_idx = torch.multinomial(weights_t2cond[b], 1).item()
                    condition_feats_neg.append(condition_feats_collate[neg_idx])
                condition_feats_neg = torch.stack(condition_feats_neg, dim=0)

                text_ids_neg = []
                text_atts_neg = []
                for b in range(bs):
                    neg_idx = torch.multinomial(weights_cond2t[b], 1).item()
                    text_ids_neg.append(input_ids_collate[neg_idx])
                    text_atts_neg.append(attention_mask_collate[neg_idx])

                text_ids_neg = torch.stack(text_ids_neg, dim=0)
                text_atts_neg = torch.stack(text_atts_neg, dim=0)
        
                input_ids_1 = torch.cat((input_ids, input_ids, text_ids_neg),dim=0)
                attention_mask_1 = torch.cat((attention_mask, attention_mask, text_atts_neg),dim=0)
            
                condition_feats = torch.cat((condition_feats,condition_feats_neg,condition_feats),dim=0)
                output = self.multimodal_encoder.bert(input_ids = input_ids_1,
                                            attention_mask = attention_mask_1,
                                            encoder_hidden_states=condition_feats).last_hidden_state
                batch_size = condition_feats_neg.shape[0]
                logits = self.itm_head(output[:,0].half())
                ground_truth = torch.zeros(batch_size*3).long().cuda()
                ground_truth[:batch_size] = 1
                loss = F.cross_entropy(logits,ground_truth)
                loss_itm.append(self.itm_ratio * loss)

            loss_itc = sum(loss_itc)/len(loss_itc)
            loss_dict['loss_itc'] = loss_itc
            loss_itm = sum(loss_itm)/len(loss_itm)
            loss_dict['loss_itm'] = loss_itm

            return loss_dict


    # PMRL Helper Methods

    def _forward_ret_pmrl(self, batch, feat_t, feat_v, feat_a,
                          feat_t_all, feat_v_all, feat_a_all,
                          input_ids, attention_mask,
                          input_ids_collate, attention_mask_collate):
        """
        Pure PMRL forward pass (Version 1 - for ablation study)

        Loss: L^M + λ₁·L^M' + λ₂·L_IM
        No volume loss, only PMRL components
        """
        bs = feat_t.size(0)
        rank = dist.get_rank()

        # Build embeddings dict
        embeddings_dict = {
            'text': feat_t,
            'video': feat_v,
            'audio': feat_a
        }
        if "raw_subtitles" in batch.keys():
            feat_s = self.batch_get(batch, 'feat_s')
            embeddings_dict['subtitle'] = feat_s
        if "depth_pixels" in batch.keys():
            feat_d = self.batch_get(batch, 'feat_d')
            embeddings_dict['depth'] = feat_d

        # Compute PMRL L^M and L^M'
        pmrl_dict = self.pmrl_loss(embeddings_dict, return_components=True)

        # ITM Loss (L_IM)
        # Use cosine similarity for hard negative mining (instead of volume)
        sim_t2v = feat_t @ feat_v_all.T / self.contra_temp
        sim_v2t = feat_v @ feat_t_all.T / self.contra_temp

        # Get condition features
        condition_feats = self.batch_get(batch, 'condition_feats_va')
        condition_feats_collate = all_gather_with_grad(condition_feats)

        # Sample hard negatives
        with torch.no_grad():
            weights_t2cond = F.softmax(sim_t2v, dim=1) + 1e-4
            weights_t2cond[:, rank * bs : rank * bs + bs].fill_diagonal_(0)
            weights_cond2t = F.softmax(sim_v2t, dim=1) + 1e-4
            weights_cond2t[:, rank * bs : rank * bs + bs].fill_diagonal_(0)

        # Sample negatives
        condition_feats_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_t2cond[b], 1).item()
            condition_feats_neg.append(condition_feats_collate[neg_idx])
        condition_feats_neg = torch.stack(condition_feats_neg, dim=0)

        text_ids_neg = []
        text_atts_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_cond2t[b], 1).item()
            text_ids_neg.append(input_ids_collate[neg_idx])
            text_atts_neg.append(attention_mask_collate[neg_idx])

        text_ids_neg = torch.stack(text_ids_neg, dim=0)
        text_atts_neg = torch.stack(text_atts_neg, dim=0)

        # Forward through multimodal encoder + ITM head
        input_ids_1 = torch.cat((input_ids, input_ids, text_ids_neg), dim=0)
        attention_mask_1 = torch.cat((attention_mask, attention_mask, text_atts_neg), dim=0)
        condition_feats_cat = torch.cat((condition_feats, condition_feats_neg, condition_feats), dim=0)

        output = self.multimodal_encoder.bert(
            input_ids=input_ids_1,
            attention_mask=attention_mask_1,
            encoder_hidden_states=condition_feats_cat
        ).last_hidden_state

        itm_logits = self.itm_head(output[:, 0].half())
        itm_labels = torch.zeros(bs * 3).long().cuda()
        itm_labels[:bs] = 1

        L_IM = F.cross_entropy(itm_logits, itm_labels)

        # Combine losses
        loss_dict = {
            'loss_pmrl_M': pmrl_dict['pmrl_L_M'],
            'loss_pmrl_Mprime': pmrl_dict['pmrl_L_Mprime'],
            'loss_pmrl_IM': self.pmrl_loss.lambda2 * L_IM,
            'loss_itc': torch.tensor(0.0, device=feat_t.device),  # Compatibility
            'loss_itm': self.itm_ratio * L_IM,
            'loss_area': torch.tensor(0.0, device=feat_t.device)  # Compatibility
        }

        return loss_dict


    def _forward_ret_pmrl_volume(self, batch, feat_t, feat_v, feat_a,
                                 feat_t_all, feat_v_all, feat_a_all,
                                 input_ids, attention_mask,
                                 input_ids_collate, attention_mask_collate):
        """
        PMRL + Volume forward pass (Version 2 - RECOMMENDED)

        Loss: L_volume + L^M + λ₁·L^M' + λ₂·L_IM
        Combines GRAM volume with PMRL SVD alignment
        """
        bs = feat_t.size(0)
        rank = dist.get_rank()

        # 1. Compute GRAM Volume Loss
        if "raw_subtitles" in batch.keys():
            feat_s = self.batch_get(batch, 'feat_s')
            feat_s_all = concat_all_gather(feat_s)

            if "depth_pixels" in batch.keys():
                feat_d = self.batch_get(batch, 'feat_d')
                feat_d_all = concat_all_gather(feat_d)
                volume = volume_computation5(feat_t, feat_v_all, feat_a_all, feat_s_all, feat_d_all)
                volumeT = volume_computation5(feat_t_all, feat_v, feat_a, feat_s, feat_d).T
            else:
                volume = volume_computation4(feat_t, feat_v_all, feat_a_all, feat_s_all)
                volumeT = volume_computation4(feat_t_all, feat_v, feat_a, feat_s).T
        else:
            volume = volume_computation3(feat_t, feat_v_all, feat_a_all)
            volumeT = volume_computation3(feat_t_all, feat_v, feat_a).T

        volume = volume / self.contra_temp
        volumeT = volumeT / self.contra_temp

        targets = torch.linspace(rank * bs, rank * bs + bs - 1, bs, dtype=int).to(volume.device)

        loss_volume = (
            F.cross_entropy(-volume, targets, label_smoothing=0.1) +
            F.cross_entropy(-volumeT, targets, label_smoothing=0.1)
        ) / 2

        # 2. Compute PMRL Losses
        embeddings_dict = {
            'text': feat_t,
            'video': feat_v,
            'audio': feat_a
        }
        if "raw_subtitles" in batch.keys():
            embeddings_dict['subtitle'] = feat_s
        if "depth_pixels" in batch.keys():
            embeddings_dict['depth'] = feat_d

        pmrl_dict = self.pmrl_loss(embeddings_dict, return_components=True)

        # 3. ITM Loss (use volume-based negative mining)
        condition_feats = self.batch_get(batch, 'condition_feats_va')
        condition_feats_collate = all_gather_with_grad(condition_feats)

        with torch.no_grad():
            weights_t2cond = F.softmax(-volume, dim=1) + 1e-4
            weights_t2cond[:, rank * bs : rank * bs + bs].fill_diagonal_(0)
            weights_cond2t = F.softmax(-volumeT, dim=1) + 1e-4
            weights_cond2t[:, rank * bs : rank * bs + bs].fill_diagonal_(0)

        # Sample negatives
        condition_feats_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_t2cond[b], 1).item()
            condition_feats_neg.append(condition_feats_collate[neg_idx])
        condition_feats_neg = torch.stack(condition_feats_neg, dim=0)

        text_ids_neg = []
        text_atts_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_cond2t[b], 1).item()
            text_ids_neg.append(input_ids_collate[neg_idx])
            text_atts_neg.append(attention_mask_collate[neg_idx])

        text_ids_neg = torch.stack(text_ids_neg, dim=0)
        text_atts_neg = torch.stack(text_atts_neg, dim=0)

        # Forward through ITM
        input_ids_1 = torch.cat((input_ids, input_ids, text_ids_neg), dim=0)
        attention_mask_1 = torch.cat((attention_mask, attention_mask, text_atts_neg), dim=0)
        condition_feats_cat = torch.cat((condition_feats, condition_feats_neg, condition_feats), dim=0)

        output = self.multimodal_encoder.bert(
            input_ids=input_ids_1,
            attention_mask=attention_mask_1,
            encoder_hidden_states=condition_feats_cat
        ).last_hidden_state

        itm_logits = self.itm_head(output[:, 0].half())
        itm_labels = torch.zeros(bs * 3).long().cuda()
        itm_labels[:bs] = 1

        L_IM = F.cross_entropy(itm_logits, itm_labels)

        # 4. Combine All Losses
        loss_dict = {
            'loss_area': loss_volume,
            'loss_itm': self.itm_ratio * L_IM,
            'loss_pmrl_M': pmrl_dict['pmrl_L_M'],
            'loss_pmrl_Mprime': pmrl_dict['pmrl_L_Mprime'],
            'loss_pmrl_IM': self.pmrl_loss.lambda2 * L_IM,
            'loss_itc': torch.tensor(0.0, device=feat_t.device)  # Compatibility
        }

        return loss_dict


    def _forward_ret_hybrid_pmrl(self, batch, feat_t, feat_v, feat_a,
                                 feat_t_all, feat_v_all, feat_a_all,
                                 input_ids, attention_mask,
                                 input_ids_collate, attention_mask_collate):
        """
        Hybrid PMRL forward pass (Version 3 - Full integration)

        Loss: (w_euc·L_volume_euc + w_hyp·L_volume_hyp) + L^M + λ₁·L^M' + λ₂·L_IM
        Combines hybrid volume with PMRL alignment
        """
        bs = feat_t.size(0)
        rank = dist.get_rank()

        # Get curvatures
        if hasattr(self, 'curvatures'):
            curvs = self.curvatures
        else:
            curvs = self.curvature

        # Normalize hybrid weights
        weight_sum = torch.abs(self.euclidean_weight) + torch.abs(self.hyperbolic_weight)
        euclidean_weight = torch.abs(self.euclidean_weight) / weight_sum
        hyperbolic_weight = torch.abs(self.hyperbolic_weight) / weight_sum

        # 1. Compute Hybrid Volume Loss
        from utils.hyperbolic_volume import hybrid_volume3, hybrid_volume4, hybrid_volume5

        if "raw_subtitles" in batch.keys():
            feat_s = self.batch_get(batch, 'feat_s')
            feat_s_all = concat_all_gather(feat_s)

            if "depth_pixels" in batch.keys():
                feat_d = self.batch_get(batch, 'feat_d')
                feat_d_all = concat_all_gather(feat_d)
                volume = hybrid_volume5(feat_t, feat_v_all, feat_a_all, feat_s_all, feat_d_all,
                                       euclidean_weight, hyperbolic_weight, curvs)
                volumeT = hybrid_volume5(feat_t_all, feat_v, feat_a, feat_s, feat_d,
                                        euclidean_weight, hyperbolic_weight, curvs).T
            else:
                volume = hybrid_volume4(feat_t, feat_v_all, feat_a_all, feat_s_all,
                                       euclidean_weight, hyperbolic_weight, curvs)
                volumeT = hybrid_volume4(feat_t_all, feat_v, feat_a, feat_s,
                                        euclidean_weight, hyperbolic_weight, curvs).T
        else:
            volume = hybrid_volume3(feat_t, feat_v_all, feat_a_all,
                                   euclidean_weight, hyperbolic_weight, curvs)
            volumeT = hybrid_volume3(feat_t_all, feat_v, feat_a,
                                    euclidean_weight, hyperbolic_weight, curvs).T

        volume = volume / self.contra_temp
        volumeT = volumeT / self.contra_temp

        targets = torch.linspace(rank * bs, rank * bs + bs - 1, bs, dtype=int).to(volume.device)

        loss_volume = (
            F.cross_entropy(-volume, targets, label_smoothing=0.1) +
            F.cross_entropy(-volumeT, targets, label_smoothing=0.1)
        ) / 2

        # 2. Compute PMRL Losses in Hybrid Space (Strategy 5)
        embeddings_dict = {
            'text': feat_t,
            'video': feat_v,
            'audio': feat_a
        }
        if "raw_subtitles" in batch.keys():
            embeddings_dict['subtitle'] = feat_s
        if "depth_pixels" in batch.keys():
            embeddings_dict['depth'] = feat_d

        # Pass hybrid parameters to PMRL for hybrid space SVD
        pmrl_dict = self.pmrl_loss(
            embeddings_dict,
            return_components=True,
            euclidean_weight=euclidean_weight.item(),
            hyperbolic_weight=hyperbolic_weight.item(),
            curvatures=curvs
        )

        # 3. ITM Loss (use hybrid volume-based negatives)
        condition_feats = self.batch_get(batch, 'condition_feats_va')
        condition_feats_collate = all_gather_with_grad(condition_feats)

        with torch.no_grad():
            weights_t2cond = F.softmax(-volume, dim=1) + 1e-4
            weights_t2cond[:, rank * bs : rank * bs + bs].fill_diagonal_(0)
            weights_cond2t = F.softmax(-volumeT, dim=1) + 1e-4
            weights_cond2t[:, rank * bs : rank * bs + bs].fill_diagonal_(0)

        # Sample negatives
        condition_feats_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_t2cond[b], 1).item()
            condition_feats_neg.append(condition_feats_collate[neg_idx])
        condition_feats_neg = torch.stack(condition_feats_neg, dim=0)

        text_ids_neg = []
        text_atts_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_cond2t[b], 1).item()
            text_ids_neg.append(input_ids_collate[neg_idx])
            text_atts_neg.append(attention_mask_collate[neg_idx])

        text_ids_neg = torch.stack(text_ids_neg, dim=0)
        text_atts_neg = torch.stack(text_atts_neg, dim=0)

        # Forward through ITM
        input_ids_1 = torch.cat((input_ids, input_ids, text_ids_neg), dim=0)
        attention_mask_1 = torch.cat((attention_mask, attention_mask, text_atts_neg), dim=0)
        condition_feats_cat = torch.cat((condition_feats, condition_feats_neg, condition_feats), dim=0)

        output = self.multimodal_encoder.bert(
            input_ids=input_ids_1,
            attention_mask=attention_mask_1,
            encoder_hidden_states=condition_feats_cat
        ).last_hidden_state

        itm_logits = self.itm_head(output[:, 0].half())
        itm_labels = torch.zeros(bs * 3).long().cuda()
        itm_labels[:bs] = 1

        L_IM = F.cross_entropy(itm_logits, itm_labels)

        # 4. Combine All Losses
        loss_dict = {
            'loss_area': loss_volume,
            'loss_itm': self.itm_ratio * L_IM,
            'loss_pmrl_M': pmrl_dict['pmrl_L_M'],
            'loss_pmrl_Mprime': pmrl_dict['pmrl_L_Mprime'],
            'loss_pmrl_IM': self.pmrl_loss.lambda2 * L_IM,
            'loss_itc': torch.tensor(0.0, device=feat_t.device)  # Compatibility
        }

        return loss_dict
