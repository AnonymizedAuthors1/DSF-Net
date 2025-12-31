import torch
import torch.nn as nn
from torch.nn import functional as F
from GazeSeg.models.Backbone import ViGBackbone


class GazeTrajSeg(nn.Module):
    def __init__(self, out_channels=1, nhead=8, num_layers=3, max_traj_len=10):
        super(GazeTrajSeg, self).__init__()
        self.Encoder = ViGBackbone()
        save_model = torch.load(
            'D:\myPyProject/11-GazeTrajectoryPred\GazeTrajSeg-12-10\GazeSeg\models/pretrained_models/pvig_s_82.1.pth.tar')
        model_dict = self.Encoder.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.Encoder.load_state_dict(model_dict)

        channels = [80, 160, 400, 640]
        self.traj_pos = nn.Parameter(torch.randn(1, max_traj_len, channels[0]))
        self.traj_embed = nn.Linear(3, channels[0])

        decoder_layer_shallow = nn.TransformerDecoderLayer(
            d_model=channels[0],
            nhead=nhead,
            batch_first=True
        )
        self.cross_attn_shallow = nn.TransformerDecoder(
            decoder_layer_shallow,
            num_layers=num_layers
        )
        decoder_layer_deep = nn.TransformerDecoderLayer(
            d_model=channels[2],
            nhead=nhead,
            batch_first=True
        )
        self.cross_attn_deep = nn.TransformerDecoder(
            decoder_layer_deep,
            num_layers=num_layers
        )
        self.linear = nn.Linear(channels[0], channels[2])

        decoder_layer_fusion = nn.TransformerDecoderLayer(
            d_model=channels[-1],
            nhead=nhead,
            batch_first=True
        )
        self.fusion_fun_img2traj = nn.TransformerDecoder(
            decoder_layer_fusion,
            num_layers=1
        )
        self.fusion_fun_traj2img = nn.TransformerDecoder(
            decoder_layer_fusion,
            num_layers=1
        )

        self.project_layer_640 = nn.Linear(channels[2], channels[-1])
        self.project_layer = nn.Linear(max_traj_len, 49)
        self.project_layer_49 = nn.Linear(max_traj_len + 49, 49)

        self.up1 = nn.ConvTranspose2d(channels[3], channels[3] // 2, 2, 2)
        self.conv1 = nn.Sequential(
            nn.Conv2d(channels[2] + channels[3] // 2, channels[2], 3, 1, 1), nn.BatchNorm2d(channels[2]),nn.ReLU(),
            nn.Conv2d(channels[2], channels[2], 3, 1, 1), nn.BatchNorm2d(channels[2]), nn.ReLU()
        )
        self.up2 = nn.ConvTranspose2d(channels[2], channels[2] // 2, 2, 2)
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels[1] + channels[2] // 2, channels[1], 3, 1, 1), nn.BatchNorm2d(channels[1]), nn.ReLU(),
            nn.Conv2d(channels[1], channels[1], 3, 1, 1), nn.BatchNorm2d(channels[1]), nn.ReLU()
        )
        self.up3 = nn.ConvTranspose2d(channels[1], channels[1] // 2, 2, 2)
        self.conv3 = nn.Sequential(nn.Conv2d(
            channels[0] + channels[1] // 2, channels[0], 3, 1, 1), nn.BatchNorm2d(channels[0]), nn.ReLU(),
            nn.Conv2d(channels[0], channels[0], 3, 1, 1), nn.BatchNorm2d(channels[0]), nn.ReLU(),
        )

        self.seg_layer = nn.Sequential(
            nn.Conv2d(channels[0], channels[0], 3, 1, 1), nn.BatchNorm2d(channels[0]), nn.ReLU(),
            nn.ConvTranspose2d(channels[0], channels[0] // 2, 2, 2), nn.BatchNorm2d(channels[0] // 2), nn.ReLU(),
            nn.Conv2d(channels[0] // 2, channels[0] // 2, 3, 1, 1), nn.BatchNorm2d(channels[0] // 2), nn.ReLU(),
            nn.ConvTranspose2d(channels[0] // 2, channels[0] // 4, 2, 2), nn.BatchNorm2d(channels[0] // 4), nn.ReLU(),
            nn.Conv2d(channels[0] // 4, channels[0] // 4, 3, 1, 1), nn.BatchNorm2d(channels[0] // 4), nn.ReLU()
        )
        self.seg_head = nn.Conv2d(channels[0] // 4, out_channels, 1)

        self.deep_sup_d1_fg = nn.Conv2d(channels[2], 1, 1, 1, 1)
        self.deep_sup_d1_bg = nn.Conv2d(channels[2], 1, 1, 1, 1)
        self.deep_sup_d1_uc = nn.Conv2d(channels[2], 1, 1, 1, 1)
        self.deep_sup_d2_fg = nn.Conv2d(channels[1], 1, 1, 1, 1)
        self.deep_sup_d2_bg = nn.Conv2d(channels[1], 1, 1, 1, 1)
        self.deep_sup_d2_uc = nn.Conv2d(channels[1], 1, 1, 1, 1)
        self.deep_sup_d3_fg = nn.Conv2d(channels[0], 1, 1, 1, 1)
        self.deep_sup_d3_bg = nn.Conv2d(channels[0], 1, 1, 1, 1)
        self.deep_sup_d3_uc = nn.Conv2d(channels[0], 1, 1, 1, 1)

    def sample_img_features(self, img_feat, traj):
        traj = traj[:, :, :2]
        B, C, H, W = img_feat.shape
        B2, n, _ = traj.shape
        grid = traj * 2 - 1
        grid = grid.unsqueeze(2)  # (B, n, 1, 2)
        img_feat_exp = img_feat.unsqueeze(1).repeat(1, n, 1, 1, 1)
        out = F.grid_sample(
            img_feat_exp.view(B * n, C, H, W),
            grid.view(B * n, 1, 1, 2),
            mode="bilinear",
            align_corners=True
        )
        return out.view(B, n, C)

    def forward(self, x, traj=None):
        # ------------------------------img feature------------------------------
        feature_list = self.Encoder(x)
        f1, f2, f3, f4 = feature_list[0], feature_list[1], feature_list[2], feature_list[3]
        # 80x56x56 160x28x28 400x14x14 640x7x7

        # ------------------------------trajectory feature------------------------------
        if traj is not None:
            traj_emb = self.traj_embed(traj) # 10x80
            traj_emb = traj_emb + self.traj_pos # 10x80

            # The image features at the fixtion points
            img_traj_feat_shallow = self.sample_img_features(f1, traj) # 10x80
            img_traj_feat_deep = self.sample_img_features(f3, traj) # 10x400

            # cross-attention
            traj_emb = self.cross_attn_shallow(traj_emb, img_traj_feat_shallow) # 10x80

            # linear projection
            traj_emb = self.linear(traj_emb) # 10x400
            traj_emb = self.cross_attn_deep(traj_emb, img_traj_feat_deep) # 10x400
            traj_emb = self.project_layer_640(traj_emb) # 10x640

            # ------------------------------feature fusion------------------------------
            img_seq = f4.flatten(2)  # 640x49
            img_seq = img_seq.permute(0, 2, 1)  # 49x640
            self.img_seq = img_seq
            self.traj_emb_pred = self.project_layer(traj_emb.permute(0, 2, 1)).permute(0, 2, 1)  # 49x640

            fusion_img2traj = self.fusion_fun_img2traj(img_seq, traj_emb) # 49x640
            fusion_traj2img = self.fusion_fun_traj2img(traj_emb, img_seq) # 10x640

            fusion_fusion = torch.concat([fusion_traj2img, fusion_img2traj], dim=1) # 59x640
            fusion_fusion = self.project_layer_49(fusion_fusion.permute(0, 2, 1)) # 640x49
            fusion_fusion = fusion_fusion.unflatten(-1, (7, 7))  # 640x7x7

            fusion_fusion = fusion_fusion + f4  # 640x7x7
        else:
            fusion_fusion = f4
        # ------------------------------Decoder------------------------------
        d1 = self.up1(fusion_fusion) # 320x14x14
        d1 = self.conv1(torch.cat([d1, f3], dim=1)) # 400x14x14

        d2 = self.up2(d1) # 200x28x28
        d2 = self.conv2(torch.cat([d2, f2], dim=1)) # 160x28x28

        d3 = self.up3(d2) # 80x56x56
        d3 = self.conv3(torch.cat([d3, f1], dim=1)) # 80x56x56

        d4 = self.seg_layer(d3) # 20x224x224
        seg_logits = self.seg_head(d4) # 1x224x224

        # ------------------------------Deep Super------------------------------
        self.output_d1_fg = self.deep_sup_d1_fg(d1)
        self.output_d1_bg = self.deep_sup_d1_bg(d1)
        self.output_d1_uc = self.deep_sup_d1_uc(d1)
        self.output_d2_fg = self.deep_sup_d2_fg(d2)
        self.output_d2_bg = self.deep_sup_d2_bg(d2)
        self.output_d2_uc = self.deep_sup_d2_uc(d2)
        self.output_d3_fg = self.deep_sup_d3_fg(d3)
        self.output_d3_bg = self.deep_sup_d3_bg(d3)
        self.output_d3_uc = self.deep_sup_d3_uc(d3)
        return seg_logits

    def get_output_d1(self):
        return self.output_d1_fg, self.output_d1_bg, self.output_d1_uc
    def get_output_d2(self):
        return self.output_d2_fg, self.output_d2_bg, self.output_d2_uc
    def get_output_d3(self):
        return self.output_d3_fg, self.output_d3_bg, self.output_d3_uc
    def get_img_traj_features(self):
        return self.img_seq, self.traj_emb_pred



if __name__ == '__main__':
    input = torch.randn(8, 3, 224, 224)
    model = GazeTrajSeg()
    traj = torch.randn(8, 10, 3)
    output = model(input, traj)

    print(output.shape)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)















