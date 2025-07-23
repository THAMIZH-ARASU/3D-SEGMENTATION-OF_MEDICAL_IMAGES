from models.transformers.d_former.d_former_3d import DFormer3D
from models.transformers.d_former.utils.final_patch_expand import FinalPatchExpand_X4
from torch import nn

class SegNetwork(nn.Module):

    def __init__(self, num_classes,
                 in_chan=1,
                 embed_dim=96,
                 depths=[2, 2, 6, 2],
                 num_heads=[3, 6, 12, 24],
                 patch_size=(2, 4, 4),
                 group_size=(2, 8, 8),
                 deep_supervision=True,
                 pretrain=None):
        super(SegNetwork, self).__init__()

        self._deep_supervision = deep_supervision
        self.do_ds = deep_supervision
        self.num_classes = num_classes

        self.upscale_logits_ops = []

        self.upscale_logits_ops.append(lambda x: x)
        self.model_down = DFormer3D(pretrained=pretrain,
                                        in_chans=in_chan,
                                        group_size=group_size,
                                        patch_size=patch_size,
                                        depths=depths,
                                        num_heads=num_heads)

        self.final = []
        for i in range(len(depths) - 1):
            self.final.append(nn.Sequential(FinalPatchExpand_X4(embed_dim * 2 ** i, patch_size=patch_size),
                                            nn.Conv3d(in_channels=embed_dim * 2 ** i, out_channels=14, kernel_size=1, bias=False)))
        self.final = nn.ModuleList(self.final)

    def forward(self, x):

        seg_outputs = []
        out = self.model_down(x)

        for i in range(len(out)):
            seg_outputs.append(self.final[-(i + 1)](out[i]))

        if self._deep_supervision and self.do_ds:
            return tuple([seg_outputs[-1]] + [i(j) for i, j in
                                              zip(list(self.upscale_logits_ops)[::-1], seg_outputs[:-1][::-1])])
        else:
            return seg_outputs[-1]