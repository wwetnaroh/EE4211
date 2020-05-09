from mmdet.apis import init_detector
from mmdet.apis import inference_detector
from mmdet.apis import show_result

# 模型配置文件
config_file = '../../configs/faster_rcnn_r101_fpn_1x.py'

# 预训练模型文件
checkpoint_file = '../../models/faster_rcnn_r101_fpn_1x_20181129-d1468807.pth'

# 通过模型配置文件与预训练文件构建模型
model = init_detector(config_file, checkpoint_file, device='cuda:0')

# 测试单张图片并进行展示
img = 'test1.jpg'
result = inference_detector(model, img)
show_result(img, result, model.CLASSES)
