import numpy as np
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.linear import Linear
from torch.nn.modules.pooling import AdaptiveAvgPool2d, AdaptiveMaxPool2d

import timm
from timm.models.efficientnet import tf_efficientnet_b4_ns, tf_efficientnet_b3_ns, \
    tf_efficientnet_b5_ns, tf_efficientnet_b2_ns, tf_efficientnet_b6_ns, tf_efficientnet_b7_ns, tf_efficientnet_b0_ns

from torchlibrosa.stft import Spectrogram, LogmelFilterBank
from torchlibrosa.augmentation import SpecAugmentation

from pytorch_utils import do_mixup, interpolate, pad_framewise_output

encoder_params = {
    "resnest50d" : {
        "features" : 2048,
        "init_op"  : partial(timm.models.resnest50d, pretrained=True, in_chans=1)
    },
    "densenet201" : {
        "features": 1920,
        "init_op": partial(timm.models.densenet201, pretrained=True)
    },
    "dpn92" : {
        "features": 2688,
        "init_op": partial(timm.models.dpn92, pretrained=True)
    },
    "dpn131": {
        "features": 2688,
        "init_op": partial(timm.models.dpn131, pretrained=True)
    },
    "tf_efficientnet_b0_ns": {
        "features": 1280,
        "init_op": partial(tf_efficientnet_b0_ns, pretrained=True, drop_path_rate=0.2, in_chans=1)
    },
    "tf_efficientnet_b3_ns": {
        "features": 1536,
        "init_op": partial(tf_efficientnet_b3_ns, pretrained=True, drop_path_rate=0.2, in_chans=1)
    },
    "tf_efficientnet_b2_ns": {
        "features": 1408,
        "init_op": partial(tf_efficientnet_b2_ns, pretrained=True, drop_path_rate=0.2, in_chans=1)
    },
    "tf_efficientnet_b4_ns": {
        "features": 1792,
        "init_op": partial(tf_efficientnet_b4_ns, pretrained=True, drop_path_rate=0.2, in_chans=1)
    },
    "tf_efficientnet_b5_ns": {
        "features": 2048,
        "init_op": partial(tf_efficientnet_b5_ns, pretrained=True, drop_path_rate=0.2, in_chans=1)
    },
    "tf_efficientnet_b6_ns": {
        "features": 2304,
        "init_op": partial(tf_efficientnet_b6_ns, pretrained=True, drop_path_rate=0.2, in_chans=1)
    },
    "vggish": {
    "features": 512,
  #  "init_op": partial(tf_efficientnet_b6_ns, pretrained=True, drop_path_rate=0.2, in_chans=1)
    },
}


class AudioClassifier(nn.Module):
    def __init__(self, encoder, sample_rate, window_size, hop_size, mel_bins, fmin, fmax, classes_num):
        super().__init__()

        window = 'hann'
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None
        self.encoder_name = encoder
        
        # Spectrogram extractor
        self.spectrogram_extractor = Spectrogram(n_fft=window_size, hop_length=hop_size, 
            win_length=window_size, window=window, center=center, pad_mode=pad_mode, 
            freeze_parameters=True)

        # Logmel feature extractor
        self.logmel_extractor = LogmelFilterBank(sr=sample_rate, n_fft=window_size, 
            n_mels=mel_bins, fmin=fmin, fmax=fmax, ref=ref, amin=amin, top_db=top_db, 
            freeze_parameters=True)

        # Spec augmenter
        self.spec_augmenter = SpecAugmentation(time_drop_width=64, time_stripes_num=2, 
            freq_drop_width=8, freq_stripes_num=2)
        
        if self.encoder_name == "vggish":
            self.encoder = torch.hub.load('harritaylor/torchvggish', 'vggish')
        else:
            self.encoder = encoder_params[encoder]["init_op"]()
        
        self.avg_pool = AdaptiveAvgPool2d((1, 1))
        self.dropout = Dropout(0.3)
        self.fc = Linear(encoder_params[encoder]['features'], classes_num)
    
    def forward(self, input, spec_aug=False, mixup_lambda=None):
        #print(input.type())
        x = self.spectrogram_extractor(input.float()) # (batch_size, 1, time_steps, freq_bins)
        x = self.logmel_extractor(x) # (batch_size, 1, time_steps, mel_bins)

        #if spec_aug:
        #    x = self.spec_augmenter(x)
        if self.training:
            x = self.spec_augmenter(x)
        
        # Mixup on spectrogram
        if mixup_lambda is not None:
            x = do_mixup(x, mixup_lambda)
            #pass
        
        if self.encoder_name == "vggish":
            x = self.encoder.features(x)
        else:
            x = self.encoder.forward_features(x)
        x = self.avg_pool(x).flatten(1)
        x = self.dropout(x)
        x = self.fc(x)
        
        return x
