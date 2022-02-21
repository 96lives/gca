from models.unet import Res16UNet14C, Res16UNet18B, Res16UNet34C

BACKBONES = {
    Res16UNet14C.name: Res16UNet14C,
    Res16UNet18B.name: Res16UNet18B,
    Res16UNet34C.name: Res16UNet34C,
}