import math
import torch
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.functional as F
from networks import graph

class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)

    def forward(self, input, adj=None, relu=False):
        support = torch.matmul(input, self.weight)
        if adj is not None:
            output = torch.matmul(adj, support)
        else:
            output = support
        if self.bias is not None:
            output = output + self.bias
        if relu:
            return F.relu(output)
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class Featuremaps_to_Graph(nn.Module):
    def __init__(self, input_channels, hidden_layers, nodes):
        super(Featuremaps_to_Graph, self).__init__()
        self.pre_fea = Parameter(torch.FloatTensor(input_channels, nodes))
        self.weight = Parameter(torch.FloatTensor(input_channels, hidden_layers))
        self.nodes = nodes
        self.reset_parameters()

    def reset_parameters(self):
        for ww in self.parameters():
            torch.nn.init.xavier_uniform_(ww)

    def forward(self, input):
        if input.dim() != 4:
            raise ValueError(f"Expected 4D input (N, C, H, W), got {input.dim()}D")
        n, c, h, w = input.size()
        input1 = input.view(n, c, h * w).transpose(1, 2)  # n x hw x c
        fea_node = torch.matmul(input1, self.pre_fea)  # n x hw x nodes
        weight_node = torch.matmul(input1, self.weight)  # n x hw x hidden_layer
        fea_node = F.softmax(fea_node, dim=1)  # n x hw x nodes
        graph_node = F.relu(torch.matmul(fea_node.transpose(1, 2), weight_node))  # n x nodes x hidden_layer
        return graph_node

class My_Featuremaps_to_Graph(nn.Module):
    def __init__(self, input_channels, hidden_layers, nodes):
        super(My_Featuremaps_to_Graph, self).__init__()
        self.pre_fea = Parameter(torch.FloatTensor(input_channels, nodes))
        self.weight = Parameter(torch.FloatTensor(input_channels, hidden_layers))
        self.nodes = nodes
        self.reset_parameters()

    def reset_parameters(self):
        for ww in self.parameters():
            torch.nn.init.xavier_uniform_(ww)

    def forward(self, input):
        if input.dim() != 4:
            raise ValueError(f"Expected 4D input (N, C, H, W), got {input.dim()}D")
        n, c, h, w = input.size()
        input1 = input.view(n, c, h * w).transpose(1, 2)  # n x hw x c
        fea_node = torch.matmul(input1, self.pre_fea)  # n x hw x nodes
        fea_logit = fea_node.transpose(1, 2).view(n, self.nodes, h, w)  # n x nodes x h x w
        weight_node = torch.matmul(input1, self.weight)  # n x hw x hidden_layer
        fea_node = F.softmax(fea_node, dim=1)  # n x hw x nodes
        fea_sum = torch.sum(fea_node, dim=1).unsqueeze(1).expand(n, h * w, self.nodes)
        fea_node = torch.div(fea_node, fea_sum + 1e-8)  # Normalize
        graph_node = F.relu(torch.matmul(fea_node.transpose(1, 2), weight_node))  # n x nodes x hidden_layer
        return graph_node, fea_logit

class Graph_to_Featuremaps_savemem(nn.Module):
    def __init__(self, input_channels, output_channels, hidden_layers, nodes):
        super(Graph_to_Featuremaps_savemem, self).__init__()
        self.node_fea_for_res = Parameter(torch.FloatTensor(input_channels, 1))
        self.node_fea_for_hidden = Parameter(torch.FloatTensor(hidden_layers, 1))
        self.weight = Parameter(torch.FloatTensor(hidden_layers, output_channels))
        self.nodes = nodes
        self.reset_parameters()

    def reset_parameters(self):
        for ww in self.parameters():
            torch.nn.init.xavier_uniform_(ww)

    def forward(self, input, res_feature):
        if res_feature.dim() != 4:
            raise ValueError(f"Expected 4D res_feature (N, C, H, W), got {res_feature.dim()}D")
        batchi, channeli, hi, wi = res_feature.size()
        if input.dim() == 3:
            input = input.unsqueeze(0)  # 1 x batch x nodes x hidden
        _, batch, nodes, hidden = input.size()
        if batch != batchi:
            raise ValueError(f"Batch size mismatch: input {batch}, res_feature {batchi}")
        input1 = input.transpose(0, 1).expand(batch, hi * wi, nodes, hidden)  # batch x hi*wi x nodes x hidden
        res_feature_after_view = res_feature.view(batch, channeli, hi * wi).transpose(1, 2)  # batch x hi*wi x channeli
        res_feature_after_view1 = res_feature_after_view.unsqueeze(2).expand(batch, hi * wi, nodes, channeli)  # batch x hi*wi x nodes x channeli
        new_node1 = torch.matmul(res_feature_after_view1, self.node_fea_for_res)  # batch x hi*wi x nodes x 1
        new_node2 = torch.matmul(input1, self.node_fea_for_hidden)  # batch x hi*wi x nodes x 1
        new_node = new_node1 + new_node2  # batch x hi*wi x nodes x 1
        new_weight = torch.matmul(input, self.weight)  # 1 x batch x nodes x channel
        new_node = new_node.view(batch, hi * wi, nodes)  # batch x hi*wi x nodes
        new_node = F.softmax(new_node, dim=-1)  # Normalize
        feature_out = torch.matmul(new_node, new_weight)  # 1 x batch x hi*wi x channel
        feature_out = feature_out.transpose(2, 3).contiguous().view(res_feature.size())  # batch x channeli x hi x wi
        return F.relu(feature_out)

class My_Graph_to_Featuremaps(nn.Module):
    def __init__(self, hidden_layers, output_channels, dimension=2):
        super(My_Graph_to_Featuremaps, self).__init__()
        if dimension != 2:
            raise ValueError("Only 2D dimension is supported")
        self.conv = nn.Conv2d(hidden_layers, output_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, graph, fea_logit):
        if fea_logit.dim() != 4:
            raise ValueError(f"Expected 4D fea_logit (N, C, H, W), got {fea_logit.dim()}D")
        fea_prob = F.softmax(fea_logit, dim=1)  # batch x nodes x h x w
        batch, nodes, h, w = fea_prob.size()
        fea_prob = fea_prob.view(batch, nodes, h * w).transpose(1, 2)  # batch x hw x nodes
        fea_map = torch.matmul(fea_prob, graph)  # batch x hw x hidden_layer
        fea_map = fea_map.transpose(1, 2).view(batch, -1, h, w)  # batch x hidden_layers x h x w
        fea_map = self.conv(fea_map)  # batch x output_channels x h x w
        fea_map = self.bn(fea_map)
        return self.relu(fea_map)