import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary
from scipy.spatial import Delaunay
import cv2
import numpy as np

SAMPLE_LANDMARK_SET = torch.Tensor([[0.0235, 0.3031, 0.0391, 0.4280, 0.0647, 0.5394, 0.0804, 0.6644, 0.1220, 0.7775, 0.1884, 0.8571, 0.2845, 0.9241, 0.3832, 0.9561, 0.4967, 0.9741, 0.6200, 0.9549, 0.7245, 0.9157, 0.8229, 0.8559, 0.8974, 0.7688, 0.9421, 0.6693, 0.9613, 0.5580, 0.9772, 0.4458, 0.9946, 0.3242, 0.1111, 0.3124, 0.1712, 0.2874, 0.2467,
         0.2908, 0.3289, 0.3021, 0.4113, 0.3233, 0.5722, 0.3400, 0.6540, 0.3143, 0.7359, 0.3018, 0.8201, 0.2939, 0.8965, 0.3218, 0.4910, 0.4090, 0.4882, 0.4816, 0.4838, 0.5538, 0.4769, 0.6239, 0.3907, 0.6630, 0.4351, 0.6747, 0.4799, 0.6888, 0.5328, 0.6757, 0.5807, 0.6610, 0.2025, 0.4124, 0.2587, 0.4225, 0.3121, 0.4188,
         0.3700, 0.4245, 0.3126, 0.4385, 0.2571, 0.4353, 0.6309, 0.4264, 0.6870, 0.4278, 0.7404, 0.4310, 0.8010, 0.4257, 0.7434, 0.4481, 0.6872, 0.4450, 0.3507, 0.7672, 0.3964, 0.7461, 0.4452, 0.7383, 0.4828, 0.7498, 0.5118, 0.7436, 0.5620, 0.7550, 0.6282, 0.7851, 0.5653, 0.7883, 0.5129, 0.7971, 0.4783, 0.7999, 0.4422,
         0.7927, 0.3961, 0.7731, 0.3695, 0.7706, 0.4456, 0.7699, 0.4824, 0.7771, 0.5112, 0.7724, 0.5993, 0.7823, 0.5080, 0.7598, 0.4806, 0.7639, 0.4465, 0.7547]])
    
class FaceGraph:
    def __init__(self, landmarks: torch.Tensor=SAMPLE_LANDMARK_SET):
        self.selected_indices = [
            0, 2, 5, 8, 11, 14, 16,
            36, 37, 38, 39, 40, 41,
            42, 43, 44, 45, 46, 47,
            27, 29, 31, 33, 35,
            60, 61, 63, 64, 65, 67
        ]
        self.connections = [
            (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6) ,       # jawline 
            (19, 20), (20, 22), (21, 22), (22, 23),   # nose 
            (7, 8), (8, 9), (9, 10), (10, 11), (11, 12), (7,12), # left eye 
            (13,14), (14,15), (15,16), (16,17), (17,18), (13,18), # right eye 
            (24,25), (25,26), (26,27), (27,28), (28,29), (24,29)  # mouth
        ]
        self.num_nodes = len(self.selected_indices)
        self.points = self.extract_points(landmarks)
        self.A = self.get_adjacency()

    def extract_points(self, landmarks: torch.Tensor):
        """
        landmarks: (1, 136) tensor from PFLD output
        returns: (30, 2) tensor with selected landmark points
        """
        if landmarks.shape != (1, 136):
            raise ValueError(f"Expected shape (1, 136), got {landmarks.shape}")
        points = landmarks.view(68, 2)
        selected_points = points[self.selected_indices]  # (30, 2)
        return selected_points

    def get_adjacency(self):
        A = torch.zeros((self.num_nodes, self.num_nodes))
        I = torch.eye(self.num_nodes)
        for i, j in self.connections:
            A[i, j] = 1
            A[j, i] = 1  # undirected
        A = A + I
        A = self._normalize_graph(A)
        A = A.unsqueeze(0)  # shape: (1, 30, 30)
        return A
    
    def _normalize_graph(self, A):
        deg = torch.sum(A, axis=0)
        lambda_negative_half = torch.zeros((self.num_nodes, self.num_nodes))
        for i in range(self.num_nodes):
            if deg[i] > 0:
                lambda_negative_half[i, i] = deg[i] ** (-0.5)
        normalized_A = lambda_negative_half @ A @ lambda_negative_half # L^-1/2 * A * L^-1/2
        return normalized_A
    
class STGCN(nn.Module):
    def __init__(self, in_channels, num_class, landmarks_set=None, dropout=0.1, T_frame=40, deep=False, disable_out=False, device="cuda"):
        super(STGCN, self).__init__()
        if landmarks_set:
            self.graph = FaceGraph(landmarks_set) # construct an adjacency matrix using predefined rules
        else:
            self.graph = FaceGraph()
        self.disable_out = disable_out
        A = self.graph.A.clone().detach().requires_grad_(False) # A = [1, 30, 30]
        self.register_buffer('A', A)
        self.adj = A.clone().detach().requires_grad_(False).to(device=device)
        self.T_frame = T_frame
        spatial_kernel_size = A.size(0) # 1
        temporal_kernel_size = 9
        kernel_size = (temporal_kernel_size, spatial_kernel_size) # (9,1)
        self.bn1d = nn.BatchNorm1d(in_channels * A.size(1)) # (2*30) ~ (60)
        if deep:
            self.stgcn_modules = nn.ModuleList([
                SpatioTemporalConvBlock(in_channels, 64, kernel_size, 1, dropout=0.1, residual=False),
                SpatioTemporalConvBlock(64, 64, kernel_size, 1, dropout=dropout, residual=True),
                SpatioTemporalConvBlock(64, 128, kernel_size, 1, dropout=dropout, residual=True),
                SpatioTemporalConvBlock(128, 128, kernel_size, 1, dropout=dropout, residual=True),
                SpatioTemporalConvBlock(128, 256, kernel_size, 1, dropout=dropout, residual=True),
                SpatioTemporalConvBlock(256, 256, kernel_size, 1, dropout=dropout, residual=True)
            ])
        else:
            self.stgcn_modules = nn.ModuleList([
                SpatioTemporalConvBlock(in_channels, 64, kernel_size, 1, dropout=0.1, residual=False),
                SpatioTemporalConvBlock(64, 128, kernel_size, 1, dropout=dropout, residual=True),
                SpatioTemporalConvBlock(128, 256, kernel_size, 1, dropout=dropout, residual=True),
            ])

        self.final_conv = nn.Conv2d(256, num_class, kernel_size=1)
        assert num_class >= 2
        self.binary_mode = True if num_class == 2 else False
        
    def forward(self, x:torch.Tensor): # x = [N, C, T, V]
        N, C, T, V = x.size()
        x = x.permute(0,3,1,2).contiguous() # [N, V, C, T]
        x = x.view(N, V*C, T)
        x = self.bn1d(x)
        x = x.view(N, V, C, T).permute(0,2,3,1).contiguous() # [N, C, T, V]

        # ST-GCN
        for stgcn_layer in self.stgcn_modules:
            x, _ = stgcn_layer(x, self.adj)

        x = F.avg_pool2d(x, kernel_size=(self.T_frame,30))
        x = x.view(N,-1,1,1)
        x = self.final_conv(x) # [N, 2]
        x = x.view(x.size()[0], -1)
        
        if self.disable_out == False:
            # if self.binary_mode:
            #     x = torch.sigmoid(x)
            # else:
            #     x = F.softmax(x, dim=1)
            x = F.softmax(x, dim=1)

        return x

class SpatioTemporalConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dropout=0, residual=True):
        super(SpatioTemporalConvBlock, self).__init__()
        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        padding = ((kernel_size[0] - 1) // 2, 0)  # (temporal, spatial)
        self.graph_conv = TemporalGraphConv(in_channels, out_channels, kernel_size[1])
        self.temporal_conv = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, (kernel_size[0], 1), (stride, 1), padding),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True)
        )

        if not residual :
            self.residual = lambda x: 0
        elif (in_channels==out_channels) and (stride==1):
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels)
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, A):
        res = self.residual(x)
        x, A = self.graph_conv(x, A)
        x = self.temporal_conv(x) + res

        return self.relu(x), A

class TemporalGraphConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, t_kernel_size=1, t_stride=1, t_padding=0, t_dilation=1, bias=True):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(
            in_channels,
            out_channels * kernel_size,
            kernel_size=(t_kernel_size,1),
            padding=(t_padding,0),
            stride=(t_stride,1),
            dilation=(t_dilation,1),
            bias=bias
        )

    def forward(self, x, A:torch.Tensor):
        assert A.size(0) == self.kernel_size
        x = self.conv(x)
        n, kc, t, v = x.size()
        x = x.view(n, self.kernel_size, kc // self.kernel_size, t, v)
        x = torch.einsum("nkctv,kvw->nctw", (x,A))
        return x.contiguous(), A
    
class TwoStreamSTGCN(nn.Module):
    def __init__(self, in_channel, num_class, T_frame):
        super().__init__()
        self.origin_stream = STGCN(in_channel, num_class, T_frame=T_frame, deep=False, disable_out=True)
        self.motion_stream = STGCN(in_channel, num_class, T_frame=T_frame, deep=False, disable_out=True)
        self.binary_mode = True if num_class==2 else False

    def forward(self, x):
        N, C, T, V = x.size()
        m = torch.cat((torch.cuda.FloatTensor(N, C, 1, V).zero_(),
                        x[:, :, 1:-1] - 0.5 * x[:, :, 2:] - 0.5 * x[:, :, :-2],
                        torch.cuda.FloatTensor(N, C, 1, V).zero_()), 2)

        res = self.origin_stream(x) + self.motion_stream(m)
        res = F.softmax(res, dim=1)

        # if self.binary_mode:
        #     res = torch.sigmoid(res)
        # else:
        #     res = F.softmax(res, dim=1)
        return res
    
def visualize_graph(input_img, triangles, landmarks: torch.Tensor, output_img):
    img = cv2.imread(input_img)
    w, h, _ = img.shape
    restore_wh = torch.Tensor([h, w])
    landmarks_resize = landmarks.view(-1,2) # 30,2
    landmarks_resize *= restore_wh
    landmarks_resize = landmarks_resize.numpy().astype(np.int32)

    for (x, y) in landmarks_resize:
        cv2.circle(img, (x, y), 1, (0, 255, 0), -1)

    for (v1,v2,v3) in triangles:
        cv2.line(img, landmarks_resize[v1], landmarks_resize[v2], (0, 255, 0), 1)
        cv2.line(img, landmarks_resize[v2], landmarks_resize[v3], (0, 255, 0), 1)
        cv2.line(img, landmarks_resize[v3], landmarks_resize[v1], (0, 255, 0), 1)

    cv2.imwrite(output_img, img)

if __name__ == '__main__':    
    stgcn = TwoStreamSTGCN(2,2,T_frame=60).to("cuda")
    summary(stgcn, (1,2,60,30))
    # print(stgcn.adj)
    