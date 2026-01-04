import importlib
import torch
import torch.nn as nn
import torch.nn.functional as F
from buildingblocks_2d import Encoder, Decoder, FinalConv, DoubleConv, ExtResNetBlock, SingleConv
from utils import create_feature_maps

class UNet2D(nn.Module):
    def __init__(self, in_channels=18, out_channels=17, f_maps=64, layer_order='crg', num_groups=8, num_classes=17,**kwargs):
        super(UNet2D, self).__init__()
        self.use_unary = (in_channels > 1)
        if isinstance(f_maps, int):
            f_maps = create_feature_maps(f_maps, number_of_fmaps=4)

        encoders = []
        for i, out_feature_num in enumerate(f_maps):
            if i == 0:
                encoder = Encoder(in_channels, out_feature_num, apply_pooling=False, basic_module=DoubleConv,
                                  conv_layer_order=layer_order, num_groups=num_groups)
            else:
                encoder = Encoder(f_maps[i - 1], out_feature_num, basic_module=DoubleConv,
                                  conv_layer_order=layer_order, num_groups=num_groups)
            encoders.append(encoder)
        self.encoders = nn.ModuleList(encoders)

        decoders = []
        reversed_f_maps = list(reversed(f_maps))
        for i in range(len(reversed_f_maps) - 1):
            in_feature_num = reversed_f_maps[i] + reversed_f_maps[i + 1]
            out_feature_num = reversed_f_maps[i + 1]
            decoder = Decoder(in_feature_num, out_feature_num, basic_module=DoubleConv,
                              conv_layer_order=layer_order, num_groups=num_groups)
            decoders.append(decoder)
        self.decoders = nn.ModuleList(decoders)

        self.final_conv = nn.Conv2d(f_maps[0], out_channels, 1)
        self.final_activation = nn.Softmax(dim=1)

    def forward(self, img, **kwargs):
        print(f"UNet2D input shape: {img.shape}")
        x = img
        encoders_features = []
        for encoder in self.encoders:
            x = encoder(x)
            encoders_features.insert(0, x)
        encoders_features = encoders_features[1:]

        for decoder, encoder_features in zip(self.decoders, encoders_features):
            x = decoder(encoder_features, x)
        x = self.final_conv(x)
        print(f"UNet2D output shape: {x.shape}")

        if not self.training:
            x = self.final_activation(x)
        return x

class ResidualUNet2D(nn.Module):
    def __init__(self, in_channels=18, out_channels=17, f_maps=32, conv_layer_order='cbr', num_groups=8, num_classes=17, **kwargs):
        super(ResidualUNet2D, self).__init__()
        self.use_unary = (in_channels > 1)
        if isinstance(f_maps, int):
            f_maps = create_feature_maps(f_maps, number_of_fmaps=5)

        encoders = []
        for i, out_feature_num in enumerate(f_maps):
            if i == 0:
                encoder = Encoder(in_channels, out_feature_num, apply_pooling=False, basic_module=ExtResNetBlock,
                                  conv_layer_order=conv_layer_order, num_groups=num_groups)
            else:
                encoder = Encoder(f_maps[i - 1], out_feature_num, basic_module=ExtResNetBlock,
                                  conv_layer_order=conv_layer_order, num_groups=num_groups)
            encoders.append(encoder)
        self.encoders = nn.ModuleList(encoders)

        decoders = []
        reversed_f_maps = list(reversed(f_maps))
        for i in range(len(reversed_f_maps) - 1):
            decoder = Decoder(reversed_f_maps[i], reversed_f_maps[i + 1], basic_module=ExtResNetBlock,
                              conv_layer_order=conv_layer_order, num_groups=num_groups)
            decoders.append(decoder)
        self.decoders = nn.ModuleList(decoders)

        self.final_conv = nn.Conv2d(f_maps[0], out_channels, 1)
        self.final_activation = nn.Softmax(dim=1)

    def forward(self, img, **kwargs):
        print(f"ResidualUNet2D input shape: {img.shape}")
        x = img
        encoders_features = []
        for encoder in self.encoders:
            x = encoder(x)
            encoders_features.insert(0, x)
        encoders_features = encoders_features[1:]

        for decoder, encoder_features in zip(self.decoders, encoders_features):
            x = decoder(encoder_features, x)
        x = self.final_conv(x)
        print(f"ResidualUNet2D output shape: {x.shape}")

        if not self.training:
            x = self.final_activation(x)
        return x

class TagsUNet2D(nn.Module):
    def __init__(self, in_channels=18, out_channels=17, output_heads=1, conv_layer_order='crg', init_channel_number=32, num_classes=17, **kwargs):
        super(TagsUNet2D, self).__init__()
        self.use_unary = (in_channels > 1)
        num_groups = min(init_channel_number // 2, 32)

        self.encoders = nn.ModuleList([
            Encoder(in_channels, init_channel_number, apply_pooling=False, conv_layer_order=conv_layer_order, num_groups=num_groups),
            Encoder(init_channel_number, 2 * init_channel_number, conv_layer_order=conv_layer_order, num_groups=num_groups),
            Encoder(2 * init_channel_number, 4 * init_channel_number, conv_layer_order=conv_layer_order, num_groups=num_groups),
            Encoder(4 * init_channel_number, 8 * init_channel_number, conv_layer_order=conv_layer_order, num_groups=num_groups)
        ])

        self.decoders = nn.ModuleList([
            Decoder(4 * init_channel_number + 8 * init_channel_number, 4 * init_channel_number, conv_layer_order=conv_layer_order, num_groups=num_groups),
            Decoder(2 * init_channel_number + 4 * init_channel_number, 2 * init_channel_number, conv_layer_order=conv_layer_order, num_groups=num_groups),
            Decoder(init_channel_number + 2 * init_channel_number, init_channel_number, conv_layer_order=conv_layer_order, num_groups=num_groups)
        ])

        self.final_heads = nn.ModuleList([FinalConv(init_channel_number, out_channels, num_groups=num_groups) for _ in range(output_heads)])
        self.final_activation = nn.Softmax(dim=1)

    def forward(self, x):
        print(f"TagsUNet2D input shape: {x.shape}")
        encoders_features = []
        for encoder in self.encoders:
            x = encoder(x)
            encoders_features.insert(0, x)
        encoders_features = encoders_features[1:]

        for decoder, encoder_features in zip(self.decoders, encoders_features):
            x = decoder(encoder_features, x)

        tags = [final_head(x) for final_head in self.final_heads]
        x = tags[0]
        print(f"TagsUNet2D output shape: {x.shape}")

        if not self.training:
            x = self.final_activation(x)
        return x

def get_model(config):
    def _model_class(class_name):
        m = importlib.import_module('networks.model')
        clazz = getattr(m, class_name)
        return clazz

    assert 'model' in config, 'Could not find model configuration'
    model_config = config['model']
    model_config['out_channels'] = config.get('num_classes', 17)
    model_class = _model_class(model_config['name'])
    return model_class(**model_config)