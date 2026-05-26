import torch
from torch import nn


class ViTFPN(nn.Module):

    def __init__(self, out_res="res3", out_dim=256,
                 features=[96, 192, 384, 768], vit_feat_dim=768):
        super().__init__()
        self.out = out_res

        # Trick to avoid params that are not used
        res_names = []
        for name in ["res5", "res4", "res3", "res2"]:
            res_names.append(name)
            if name == out_res:
                break
        res_names = set(res_names)

        # Pyramid layers: form a "pyramid" out of ViT features
        # res2 is the deepest
        self.process = nn.ModuleDict({
            'res2': nn.Sequential(
                nn.Conv2d(
                    in_channels=vit_feat_dim,
                    out_channels=features[0],
                    kernel_size=1,
                    stride=1,
                    padding=0,
                ),
                nn.ReLU(inplace=True),  # Add ReLU activation
                nn.ConvTranspose2d(
                    in_channels=features[0],
                    out_channels=features[0],
                    kernel_size=4,
                    stride=4,
                    padding=0,
                    bias=True,
                    dilation=1,
                    groups=1,
                ),
                nn.ReLU(inplace=True),  # Add ReLU activation
                nn.Conv2d(
                    features[0],
                    out_dim,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False,
                    groups=1
                )
            ) if "res2" in res_names else nn.Identity(),
            'res3': nn.Sequential(
                nn.Conv2d(
                    in_channels=vit_feat_dim,
                    out_channels=features[1],
                    kernel_size=1,
                    stride=1,
                    padding=0,
                ),
                nn.ReLU(inplace=True),  # Add ReLU activation
                nn.ConvTranspose2d(
                    in_channels=features[1],
                    out_channels=features[1],
                    kernel_size=2,
                    stride=2,
                    padding=0,
                    bias=True,
                    dilation=1,
                    groups=1,
                ),
                nn.ReLU(inplace=True),  # Add ReLU activation
                nn.Conv2d(
                    features[1],
                    out_dim,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False,
                    groups=1
                )
            ) if "res3" in res_names else nn.Identity(),
            'res4': nn.Sequential(
                nn.Conv2d(
                    in_channels=vit_feat_dim,
                    out_channels=features[2],
                    kernel_size=1,
                    stride=1,
                    padding=0,
                ),
                nn.ReLU(inplace=True),  # Add ReLU activation
                nn.Conv2d(
                    features[2],
                    out_dim,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False,
                    groups=1
                )
            ) if "res4" in res_names else nn.Identity(),
            'res5': nn.Sequential(
                nn.Conv2d(
                    in_channels=vit_feat_dim,
                    out_channels=features[3],
                    kernel_size=1,
                    stride=1,
                    padding=0,
                ),
                nn.ReLU(inplace=True),  # Add ReLU activation
                nn.Conv2d(
                    in_channels=features[3],
                    out_channels=features[3],
                    kernel_size=3,
                    stride=2,
                    padding=1,
                ),
                nn.ReLU(inplace=True),  # Add ReLU activation
                nn.Conv2d(
                    features[3],
                    out_dim,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False,
                    groups=1
                )
            )
        })

        # Merging layers: merge "scales"
        self.refine = nn.ModuleDict({
            'res2': (
                _make_fusion_block(out_dim, use_bn=False)
                if "res2" in res_names else nn.Identity()
            ),
            'res3': (
                _make_fusion_block(out_dim, use_bn=False)
                if "res3" in res_names else nn.Identity()
            ),
            'res4': (
                _make_fusion_block(out_dim, use_bn=False)
                if "res4" in res_names else nn.Identity()
            ),
            'res5': (
                _make_fusion_block(out_dim, use_bn=False)
                if "res5" in res_names else nn.Identity()
            )
        })
    
    def forward(self, x):
        """Forward on a dict x={'res5': (B, C, H, W), ...}."""
        res_names = ["res5", "res4", "res3", "res2"]
        # Get the pyramid
        pyr_feats = []
        for name in res_names:
            pyr_feats.append(self.process[name](x[name]))
            if name == self.out:
                break
        # Merge features
        for i in range(len(pyr_feats)):
            if i == 0:
                out = self.refine[res_names[i]](pyr_feats[i])
            else:
                out = self.refine[res_names[i]](out, pyr_feats[i])
        return [out]


class ResidualConvUnit(nn.Module):
    """Residual convolution module."""

    def __init__(self, features, activation, bn):
        """Init.

        Args:
            features (int): number of features
        """
        super().__init__()

        self.bn = bn

        self.groups = 1

        self.conv1 = nn.Conv2d(
            features,
            features,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=not self.bn,
            groups=self.groups,
        )

        self.conv2 = nn.Conv2d(
            features,
            features,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=not self.bn,
            groups=self.groups,
        )

        if self.bn == True:
            self.bn1 = nn.BatchNorm2d(features)
            self.bn2 = nn.BatchNorm2d(features)

        self.activation = activation

    def forward(self, x):
        """Forward pass.

        Args:
            x (tensor): input

        Returns:
            tensor: output
        """

        out = self.activation(x)
        out = self.conv1(out)
        if self.bn == True:
            out = self.bn1(out)

        out = self.activation(out)
        out = self.conv2(out)
        if self.bn == True:
            out = self.bn2(out)

        if self.groups > 1:
            out = self.conv_merge(out)

        return out + x


class FeatureFusionBlock(nn.Module):
    """Feature fusion block."""

    def __init__(
        self,
        features,
        activation,
        deconv=False,
        bn=False,
        expand=False,
        align_corners=True,
    ):
        super().__init__()

        self.deconv = deconv
        self.align_corners = align_corners

        self.groups = 1

        self.expand = expand
        out_features = features
        if self.expand == True:
            out_features = features // 2

        self.out_conv = nn.Conv2d(
            features,
            out_features,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
            groups=1
        )

        self.resConfUnit1 = ResidualConvUnit(features, activation, bn)
        self.resConfUnit2 = ResidualConvUnit(features, activation, bn)

    def forward(self, *xs):
        """Forward pass.

        Returns:
            tensor: output
        """
        output = xs[0]

        if len(xs) == 2:
            output = output + self.resConfUnit1(xs[1])

        output = self.resConfUnit2(output)

        output = nn.functional.interpolate(
            output, scale_factor=2, mode="bilinear",
            align_corners=self.align_corners
        )

        output = self.out_conv(output)

        return output


def _make_fusion_block(features, use_bn):
    return FeatureFusionBlock(
        features,
        nn.ReLU(False),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True,
    )


if __name__ == "__main__":
    model = ViTFPN(out_res="res4").cuda()
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    # If you feed a 256x256 image to a ViT, the output dim is usually 16x16
    B = 128
    x = {
        'res5': torch.randn(B, 768, 16, 16).cuda(),
        'res4': torch.randn(B, 768, 16, 16).cuda(),
        'res3': torch.randn(B, 768, 16, 16).cuda(),
        'res2': torch.randn(B, 768, 16, 16).cuda()
    }
    from time import time
    t = time()
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for _ in range(100):
            out = model(x)
    print(time() - t)
    for i, feat in enumerate(out):
        print(f"Output feature {i}: {feat.shape}")
