import torch
import torch.nn as nn
import math

class RotaryPositionEncoding(nn.Module):
    def __init__(self, feature_dim, pe_type='Rotary1D'):
        super().__init__()

        self.feature_dim = feature_dim
        self.pe_type = pe_type

    @staticmethod
    def embed_rotary(x, cos, sin):
        # x2 = torch.stack([-x[..., 1::2], x[..., ::2]], dim=-1).reshape_as(x).contiguous()

        x2 = torch.cat((-x[..., x.shape[-1] // 2 :], x[..., : x.shape[-1] // 2]), dim=-1).contiguous()
        x = x * cos + x2 * sin
        return x

    def forward(self, x_position):
        bsize, npoint = x_position.shape
        div_term = torch.exp(
            torch.arange(0, self.feature_dim, 2, device=x_position.device)
            * (-math.log(10000.0) / (self.feature_dim)))
        div_term = div_term.view(1, 1, -1) # [1, 1, d]

        sinx = torch.sin(x_position * div_term)  # [B, N, d]
        cosx = torch.cos(x_position * div_term)

        sin_pos, cos_pos = map(
            lambda feat: torch.stack([feat, feat], dim=-1).view(bsize, npoint, -1),
            [sinx, cosx]
        )
        position_code = torch.stack([cos_pos, sin_pos] , dim=-1)

        if position_code.requires_grad:
            position_code = position_code.detach()

        return position_code


# My thought: increment text rope by voxel size (0.05m) each step along each axis
# How does this scale with the xyz values? (assumption is that 3D rope expects xyz in meters)
class RotaryPositionEncoding3D(RotaryPositionEncoding):

    def __init__(self, feature_dim, pe_type='Rotary3D'):
        super().__init__(feature_dim, pe_type)

    @torch.no_grad()
    def forward(self, XYZ):
        '''
        @param XYZ: [B,N,3]
        @return:
        '''
        raise NotImplementedError
        bsize, npoint, _ = XYZ.shape
        x_position, y_position, z_position = XYZ[..., 0:1], XYZ[..., 1:2], XYZ[..., 2:3]
        div_term = torch.exp(
            torch.arange(0, self.feature_dim // 3, 2, dtype=torch.float, device=XYZ.device)
            * (-math.log(10000.0) / (self.feature_dim // 3))
        )
        div_term = div_term.view(1, 1, -1)  # [1, 1, d//6]

        sinx = torch.sin(x_position * div_term)  # [B, N, d//6]
        cosx = torch.cos(x_position * div_term)
        siny = torch.sin(y_position * div_term)
        cosy = torch.cos(y_position * div_term)
        sinz = torch.sin(z_position * div_term)
        cosz = torch.cos(z_position * div_term)

        sinx, cosx, siny, cosy, sinz, cosz = map(
            lambda feat: torch.stack([feat, feat], -1).view(bsize, npoint, -1),
            [sinx, cosx, siny, cosy, sinz, cosz]
        )

        cos_pos = torch.cat([cosx, cosy, cosz], dim=-1)
        sin_pos = torch.cat([sinx, siny, sinz], dim=-1)

        if cos_pos.requires_grad:
            cos_pos = cos_pos.detach()
        if sin_pos.requires_grad:
            sin_pos = sin_pos.detach()
        return cos_pos, sin_pos
        # position_code = torch.stack([
        #     torch.cat([cosx, cosy, cosz], dim=-1),  # cos_pos
        #     torch.cat([sinx, siny, sinz], dim=-1)  # sin_pos
        # ], dim=-1)

        # if position_code.requires_grad:
        #     position_code = position_code.detach()

        # return position_code


class RotaryPositionEncoding4D(RotaryPositionEncoding):
    def __init__(self, feature_dim, pe_type='Rotary3D', dtype=torch.float32):
        super().__init__(feature_dim, pe_type)
        self.dtype=dtype

    @torch.no_grad()
    def forward(self, TXYZ):
        '''
        @param XYZ: [B,N,4]
        @return:
        '''
        bsize, npoint, _ = TXYZ.shape
        t_position, x_position, y_position, z_position = TXYZ[..., 0:1], TXYZ[..., 1:2], TXYZ[..., 2:3], TXYZ[..., 3:4]
        with torch.autocast(device_type=TXYZ.device.type, enabled=False):
            div_term = torch.exp(
                torch.arange(0, self.feature_dim // 4, 2, dtype=torch.float, device=TXYZ.device)
                * (-math.log(10000.0) / (self.feature_dim // 4))
            )
            div_term = div_term.view(1, 1, -1)  # [1, 1, d//8]

            sint = torch.sin(t_position * div_term)
            cost = torch.cos(t_position * div_term)
            sinx = torch.sin(x_position * div_term)  # [B, N, d//8]
            cosx = torch.cos(x_position * div_term)
            siny = torch.sin(y_position * div_term)
            cosy = torch.cos(y_position * div_term)
            sinz = torch.sin(z_position * div_term)
            cosz = torch.cos(z_position * div_term)

        # sint, cost, sinx, cosx, siny, cosy, sinz, cosz = map(
        #     lambda feat: torch.stack([feat, feat], -1).view(bsize, npoint, -1),
        #     [sint, cost, sinx, cosx, siny, cosy, sinz, cosz]
        # )

            cos_pos = torch.cat([cost,cosx, cosy, cosz, cost, cosx, cosy, cosz], dim=-1)
            sin_pos = torch.cat([sint,sinx, siny, sinz, sint,sinx, siny, sinz], dim=-1)
            assert cos_pos.dtype == torch.float32
            assert sin_pos.dtype == torch.float32
            # position_code = torch.stack([
            #     torch.cat([cost,cosx, cosy, cosz, cost, cosx, cosy, cosz], dim=-1),  # cos_pos
            #     torch.cat([sint,sinx, siny, sinz, sint,sinx, siny, sinz], dim=-1)  # sin_pos
            # ], dim=-1)

            if cos_pos.requires_grad:
                cos_pos = cos_pos.detach()
            if sin_pos.requires_grad:
                sin_pos = sin_pos.detach()
        return cos_pos.unsqueeze(0).to(self.dtype), sin_pos.unsqueeze(0).to(self.dtype)