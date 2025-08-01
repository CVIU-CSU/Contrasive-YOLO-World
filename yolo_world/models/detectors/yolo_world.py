# Copyright (c) Tencent Inc. All rights reserved.
from typing import List, Tuple, Union
import torch
import torch.nn as nn
from torch import Tensor
from mmdet.structures import OptSampleList, SampleList
from mmyolo.models.detectors import YOLODetector
from mmyolo.registry import MODELS
from mmdet.models.utils import multi_apply
import torch.nn as nn
import torch.nn.functional as F
from mmengine.dist import get_dist_info


@MODELS.register_module()
class YOLOWorldDetector(YOLODetector):
    """Implementation of YOLOW Series"""

    def __init__(self,
                 *args,
                 mm_neck: bool = False,
                 num_train_classes=80,
                 num_test_classes=80,
                 **kwargs) -> None:
        self.mm_neck = mm_neck
        self.num_train_classes = num_train_classes
        self.num_test_classes = num_test_classes
        
        super().__init__(*args, **kwargs)
        
        self.adapter = nn.Sequential(nn.Linear(512, 256),
                                         nn.ReLU(True),
                                         nn.Linear(256, 512))
        
        

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> Union[dict, list]:
        """Calculate losses from a batch of inputs and data samples."""

        self.bbox_head.num_classes = self.num_train_classes

        img_feats, txt_feats, distance_loss = self.extract_text_feat(batch_inputs,
                                                 batch_data_samples)
        losses = self.bbox_head.loss(img_feats, txt_feats, batch_data_samples)
        
        _, world_size = get_dist_info()
        distance_loss = {'loss_distance': distance_loss*world_size}
        losses.update(distance_loss)
        
        return losses

    def predict(self,
                batch_inputs: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = True) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.
        """
        
        img_feats, txt_feats, _ = self.extract_text_feat(batch_inputs,
                                                 batch_data_samples)
        self.bbox_head.num_classes = txt_feats[0].shape[0]
        results_list = self.bbox_head.predict(img_feats,
                                              txt_feats,
                                              batch_data_samples,
                                              )
        
        batch_data_samples = self.add_pred_to_datasample(
            batch_data_samples, results_list)

        return batch_data_samples

    def reparameterize(self, texts: List[List[str]]) -> None:
        # encode text embeddings into the detector
        self.texts = texts
        self.text_feats = self.backbone.forward_text(texts)

    def _forward(
            self,
            batch_inputs: Tensor,
            batch_data_samples: OptSampleList = None) -> Tuple[List[Tensor]]:
        """Network forward process. Usually includes backbone, neck and head
        forward without any post-processing.
        """
        img_feats, txt_feats = self.extract_text_feat(batch_inputs,
                                                 batch_data_samples)
        results = self.bbox_head.forward(img_feats, txt_feats)
        return results

    
    def extract_text_feat(
            self, batch_inputs: Tensor,
            batch_data_samples: SampleList) -> Tuple[Tuple[Tensor], Tensor]:
        """Extract features."""
        
        txt_feats = None
        if batch_data_samples is None:
            texts = self.texts
            txt_feats = self.text_feats
        elif isinstance(batch_data_samples, dict) and 'texts' in batch_data_samples:
            texts = batch_data_samples['texts']
        elif isinstance(batch_data_samples, list) and hasattr(batch_data_samples[0], 'texts'):
            texts = [data_sample.texts for data_sample in batch_data_samples]
        elif hasattr(self, 'text_feats'):
            texts = self.texts
            txt_feats = self.text_feats
        else:
            raise TypeError('batch_data_samples should be dict or list.')
        
        if txt_feats is not None:
            # forward image only
            img_feats = self.backbone.forward_image(batch_inputs)
        else:
            img_feats, txt_feats = self.backbone(batch_inputs, texts)
        
        txt_feats, distance_loss = multi_apply(
            self.adapter_distance_feats,
            txt_feats
        )  
        
        txt_feats = torch.stack(txt_feats, dim=0)
        distance_loss = sum(distance_loss)
        
        if self.with_neck:
            if self.mm_neck:
                img_feats = self.neck(img_feats, txt_feats)            
            else:
                img_feats = self.neck(img_feats)
        
        return img_feats, txt_feats, distance_loss
    
    def kl_loss(self, text_features, target_sim):
   
        similarity = torch.matmul(text_features,text_features.T)

        similarity = F.softmax(similarity / 0.1, dim=-1)  
        target_sim = F.softmax(target_sim / 0.1, dim=-1)
        
        loss = F.kl_div(similarity.log(), target_sim, reduction='batchmean')
        return loss
    
  
    def adapter_distance_feats(self, text_feat):
        
        text_feat_norm = F.normalize(text_feat, dim=-1, p=2)
        target_sim = torch.matmul(text_feat_norm, text_feat_norm.T)
        mask = ~torch.eye(text_feat.shape[0], dtype=torch.bool, device=text_feat.device)
        target_sim[mask] *= 0.7
        
        text_feat = self.adapter(text_feat) + text_feat
        text_feat = nn.functional.normalize(text_feat, dim=-1, p=2)
        
        distance_loss = self.kl_loss(text_feat, target_sim)
        
        return text_feat, distance_loss


@MODELS.register_module()
class YOLOWorldPromptDetector(YOLODetector):
    """Implementation of YOLO World Series"""

    def __init__(self,
                 *args,
                 mm_neck: bool = False,
                 num_train_classes=80,
                 num_test_classes=80,
                 prompt_dim=512,
                 num_prompts=80,
                 embedding_path='',
                 freeze_prompt=False,
                 use_mlp_adapter=False,
                 **kwargs) -> None:
        self.mm_neck = mm_neck
        self.num_training_classes = num_train_classes
        self.num_test_classes = num_test_classes
        self.prompt_dim = prompt_dim
        self.num_prompts = num_prompts
        self.freeze_prompt = freeze_prompt
        self.use_mlp_adapter = use_mlp_adapter
        super().__init__(*args, **kwargs)

        if len(embedding_path) > 0:
            import numpy as np
            self.embeddings = torch.nn.Parameter(
                torch.from_numpy(np.load(embedding_path)).float())
        else:
            # random init
            embeddings = nn.functional.normalize(
                torch.randn((num_prompts, prompt_dim)),dim=-1)
            self.embeddings = nn.Parameter(embeddings)

        if self.freeze_prompt:
            self.embeddings.requires_grad = False
        else:
            self.embeddings.requires_grad = True

        if use_mlp_adapter:
            self.adapter = nn.Sequential(nn.Linear(prompt_dim, prompt_dim * 2),
                                         nn.ReLU(True),
                                         nn.Linear(prompt_dim * 2, prompt_dim))
        else:
            self.adapter = None

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> Union[dict, list]:
        """Calculate losses from a batch of inputs and data samples."""
        self.bbox_head.num_classes = self.num_training_classes
        img_feats, txt_feats = self.extract_feat(batch_inputs,
                                                 batch_data_samples)
        losses = self.bbox_head.loss(img_feats, txt_feats, batch_data_samples)
        return losses

    def predict(self,
                batch_inputs: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = True) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.
        """

        img_feats, txt_feats = self.extract_feat(batch_inputs,
                                                 batch_data_samples)

        self.bbox_head.num_classes = self.num_test_classes
        results_list = self.bbox_head.predict(img_feats,
                                              txt_feats,
                                              batch_data_samples,
                                              rescale=True)

        batch_data_samples = self.add_pred_to_datasample(
            batch_data_samples, results_list)
        return batch_data_samples

    def _forward(
            self,
            batch_inputs: Tensor,
            batch_data_samples: OptSampleList = None) -> Tuple[List[Tensor]]:
        """Network forward process. Usually includes backbone, neck and head
        forward without any post-processing.
        """
        img_feats, txt_feats = self.extract_feat(batch_inputs,
                                                 batch_data_samples)
        results = self.bbox_head.forward(img_feats, txt_feats)
        return results

    def extract_feat(
            self, batch_inputs: Tensor,
            batch_data_samples: SampleList) -> Tuple[Tuple[Tensor], Tensor]:
        """Extract features."""
        # only image features
        img_feats, _ = self.backbone(batch_inputs, None)
        # use embeddings
        txt_feats = self.embeddings[None]
        if self.adapter is not None:
            txt_feats = self.adapter(txt_feats) + txt_feats
            txt_feats = nn.functional.normalize(txt_feats, dim=-1, p=2)
        txt_feats = txt_feats.repeat(img_feats[0].shape[0], 1, 1)

        if self.with_neck:
            if self.mm_neck:
                img_feats = self.neck(img_feats, txt_feats)
            else:
                img_feats = self.neck(img_feats)
        return img_feats, txt_feats
