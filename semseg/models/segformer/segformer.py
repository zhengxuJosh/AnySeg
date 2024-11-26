import torch
import torch.nn as nn
import torch.nn.functional as F 
from .seghead import SegFormerHead
from . import MixT
from .MixT import mit_b0, mit_b1, mit_b2, mit_b4

class Seg(nn.Module):
    def __init__(self, backbone, num_classes=25, embedding_dim=768, pretrained=None, modals=None):
        super().__init__()
        self.modals = modals
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.feature_strides = [4, 8, 16, 32]
        
        if backbone == 'mit_b0':
            self.encoder = mit_b0()
            if pretrained:
                state_dict = torch.load('/hpc2hdd/home/xzheng287/DELIVER/semseg/models/segformer/mit_b0.pth')
                state_dict.pop('head.weight')
                state_dict.pop('head.bias')
                self.encoder.load_state_dict(state_dict,strict=False)
        if backbone == 'mit_b1':
            self.encoder = mit_b1()
            if pretrained:
                state_dict = torch.load('/hpc2hdd/home/xzheng287/DELIVER/semseg/models/segformer/mit_b1.pth')
                state_dict.pop('head.weight')
                state_dict.pop('head.bias')
                self.encoder.load_state_dict(state_dict,)
        if backbone == 'mit_b4':
            self.encoder = mit_b4()
            if pretrained:
                state_dict = torch.load('/hpc2hdd/home/xzheng287/DELIVER/semseg/models/segformer/mit_b4.pth')
                state_dict.pop('head.weight')
                state_dict.pop('head.bias')
                self.encoder.load_state_dict(state_dict,)
        if backbone == 'mit_b2':
            self.encoder = mit_b2()
            if pretrained:
                state_dict = torch.load('/hpc2hdd/home/xzheng287/DELIVER/semseg/models/segformer/mit_b2.pth')
                state_dict.pop('head.weight')
                state_dict.pop('head.bias')
                self.encoder.load_state_dict(state_dict,strict=False)
        self.in_channels = self.encoder.embed_dims
        ## initilize encoder
        self.backbone = backbone
        self.embed_dim = 768
        self.decoder = SegFormerHead(feature_strides=self.feature_strides, in_channels=self.in_channels, embedding_dim=self.embedding_dim, num_classes=self.num_classes)
        # self.decoder = SegFormerHead(self.in_channels, 256 if 'B0' in backbone or 'B1' in backbone else 512, self.embed_dim, num_classes)
        
        self.classifier = nn.Conv2d(in_channels=self.in_channels[-1], out_channels=self.num_classes, kernel_size=1, bias=False)
        
        # self.mlp_img = nn.Sequential(nn.Linear(1, 8), nn.ReLU(), nn.Linear(8, 1))
        # self.mlp_depth = nn.Sequential(nn.Linear(1, 8), nn.ReLU(), nn.Linear(8, 1))
        # self.mlp_event = nn.Sequential(nn.Linear(1, 8), nn.ReLU(), nn.Linear(8, 1))
        # self.mlp_lidar = nn.Sequential(nn.Linear(1, 8), nn.ReLU(), nn.Linear(8, 1))
        # n_classes = self.num_classes
        # pools = [3,7,11]
        # self.conv0 = nn.Conv2d(n_classes, n_classes, 7, padding=3, groups=n_classes)
        # self.pool1 = nn.AvgPool2d(pools[0], stride=1, padding=pools[0]//2, count_include_pad=False)
        # self.pool2 = nn.AvgPool2d(pools[1], stride=1, padding=pools[1]//2, count_include_pad=False)
        # self.pool3 = nn.AvgPool2d(pools[2], stride=1, padding=pools[2]//2, count_include_pad=False)
        # self.conv4 = nn.Conv2d(n_classes, n_classes, 1)
        # self.sigmoid = nn.Sigmoid()

        # self.mlp_max = nn.Sequential(nn.Linear(1, 8), nn.ReLU(), nn.Linear(8, 1))
        # self.mlp_min = nn.Sequential(nn.Linear(1, 8), nn.ReLU(), nn.Linear(8, 1))
        

    # def PPooling(self, x):
    #     u = x.clone()
    #     x_in = self.conv0(x)
    #     x_1 = self.pool1(x_in)
    #     x_2 = self.pool2(x_in)
    #     x_3 = self.pool3(x_in)
    #     x_out = self.sigmoid(self.conv4(x_in + x_1 + x_2 + x_3)) * u
    #     return x_out + u

    def _forward_cam(self, x):
        
        cam = F.conv2d(x, self.classifier.weight)
        cam = F.relu(cam)
        
        return cam

    def get_param_groups(self):

        param_groups = [[], [], []] # 
        
        for name, param in list(self.encoder.named_parameters()):
            if "norm" in name:
                param_groups[1].append(param)
            else:
                param_groups[0].append(param)

        for param in list(self.decoder.parameters()):

            param_groups[2].append(param)
        
        param_groups[2].append(self.classifier.weight)

        return param_groups

    # def MLM(self, input): # ['img', 'depth', 'event', 'lidar']
    #     residual = torch.mean(input, dim=0).unsqueeze(0)
    #     num = len(self.modals)
    #     modal_list = []
    #     for i in range(num):
    #         if self.modals[i] == 'img': # 1 1760 32 32 1 # 1 1760 32 32 -> 1760 32 32 
    #             output_img = self.mlp_img(input[i].unsqueeze(0).permute(1,2,3,0)).squeeze(-1)
    #             modal_list.append(output_img)
    #             # print('output_img',output_img.size())
    #         if self.modals[i] == 'depth':
    #             output_depth = self.mlp_depth(input[i].unsqueeze(0).permute(1,2,3,0)).squeeze(-1)
    #             modal_list.append(output_depth)
    #         if self.modals[i] == 'lidar':
    #             output_li = self.mlp_lidar(input[i].unsqueeze(0).permute(1,2,3,0)).squeeze(-1)
    #             modal_list.append(output_li)
    #         if self.modals[i] == 'event':
    #             output_ev = self.mlp_event(input[i].unsqueeze(0).permute(1,2,3,0)).squeeze(-1)
    #             modal_list.append(output_ev)
 
    #     modal_tensor = torch.stack(modal_list)#.permute(3,0,1,2)
    #     #print(modal_tensor.size()) 
    #     modal_tensor = torch.mean(modal_tensor, dim=0).unsqueeze(0) 
    #     return modal_tensor

    # def AFRM(self, x, mlm): # [m, c, h, w]  [1, c, h, w]
    #     score_0 = torch.nn.functional.cosine_similarity(torch.flatten(mlm, start_dim=1, end_dim=-1), torch.flatten(x[0,:,:,:].unsqueeze(0), start_dim=1, end_dim=-1))
    #     score_1 = torch.nn.functional.cosine_similarity(torch.flatten(mlm, start_dim=1, end_dim=-1), torch.flatten(x[1,:,:,:].unsqueeze(0), start_dim=1, end_dim=-1))
    #     score_2 = torch.nn.functional.cosine_similarity(torch.flatten(mlm, start_dim=1, end_dim=-1), torch.flatten(x[2,:,:,:].unsqueeze(0), start_dim=1, end_dim=-1))
    #     score_3 = torch.nn.functional.cosine_similarity(torch.flatten(mlm, start_dim=1, end_dim=-1), torch.flatten(x[3,:,:,:].unsqueeze(0), start_dim=1, end_dim=-1))
        
    #     tensors = [x[0], x[1], x[2], x[3]]
    #     max_tensor = tensors[torch.argmax(torch.stack([score_0, score_1, score_2, score_3]))]
    #     min_tensor = tensors[torch.argmin(torch.stack([score_0, score_1, score_2, score_3]))]

    #     output = self.mlp_max(max_tensor.unsqueeze(0).permute(1,2,3,0)) + self.mlp_min(min_tensor.unsqueeze(0).permute(1,2,3,0))
        
    #     # max_indices = torch.topk(torch.stack([score_0, score_1, score_2, score_3]), k=1, dim=0).indices.squeeze()
    #     # min_indices = torch.topk(torch.stack([score_0, score_1, score_2, score_3]), k=1, dim=0, largest=False).indices.squeeze()

    #     # tensors = [x[0], x[1], x[2], x[3]]
    #     # remaining_tensors = [tensors[i] for i in range(len(tensors)) if i not in [max_indices, min_indices]]
        
    #     # p = remaining_tensors[0]
    #     # q = remaining_tensors[1]
    #     # m = (p + q) / 2
    #     # jsd_loss = (F.kl_div(p, m, log_target=True) + F.kl_div(q, m, log_target=True)) / 2

    #     return output#, jsd_loss

    def forward(self, x):
        x = torch.stack(x).float()
        m,b,c,h,w = x.shape
        x = x.reshape(m*b,c,h,w)
        _, _, height, width = x.shape

        _x = self.encoder(x)

        feature =  self.decoder(_x)
        # feature = self.PPooling(feature)
        # feature = self.MLM(feature)
        # AFRM_feature = self.AFRM(feature, MLM_feature)
        # MLM_pred = F.interpolate(MLM_feature, size=(height,width), mode='bilinear', align_corners=False)
        # AFRM_pred = F.interpolate(AFRM_feature, size=(height,width), mode='bilinear', align_corners=False)
        # # out = []
        # 
        pred = F.interpolate(feature, size=(height,width), mode='bilinear', align_corners=False)
        
        pred = pred.reshape(m,b,19,h,w)
        pred = torch.mean(pred, dim=0)

        # pred = self.PPooling(pred)
        # pred = self.MLM(pred)
        # pred_AFRM = self.AFRM(pred)

        # for i in range(m):
        #     out.append(pred[i,])
        #     #print(out[i].size())
        # out = tuple(out)
        return pred# , feature # MLM_pred, AFRM_pred

if __name__ == "__main__":
    model = Seg("mit_b2", num_classes=25, pretrained=True, modals = ['img'])
    input = [torch.zeros(1,3,512,512)] #,torch.zeros(1,3,512,512),torch.zeros(1,3,512,512),torch.ones(1,3,512,512)]
    output = model(input)
    print(output.size())
    print(afrm.size())