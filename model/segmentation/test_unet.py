from ptflops import get_model_complexity_info

from model.segmentation.unet import UNet

if __name__ == '__main__':
    print('================================================================================')
    print('UNet, ResNet, 512x512')
    print('================================================================================')
    model = UNet(pretrained=True)
    flops, params = get_model_complexity_info(model, (3, 512, 512), verbose=True)

    print('{:<30}  {:<8}'.format('Computational complexity: ', flops))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))

    print('================================================================================')
    print('UNet, MobileNet, 512x512')
    print('================================================================================')
    model = UNet(backbone='mobilenet', pretrained=True)
    flops, params = get_model_complexity_info(model, (3, 512, 512), verbose=True)

    print('{:<30}  {:<8}'.format('Computational complexity: ', flops))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))
