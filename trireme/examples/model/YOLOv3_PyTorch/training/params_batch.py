TRAINING_PARAMS = \
{
    "model_params": {
        "backbone_name": "darknet_53",
        "backbone_pretrained": "~/Desktop/Schoolwork/BDD/sat/app/src/python/model/YOLOv3_PyTorch/weights/darknet53_weights_pytorch.pth", #  set empty to disable
    },
    "yolo": {
        "anchors": [[[116, 90], [156, 198], [373, 326]],
                    [[30, 61], [62, 45], [59, 119]],
                    [[10, 13], [16, 30], [33, 23]]],
        "classes": 80,
    },
    "lr": {
        "backbone_lr": 0.001/16,    # normalized by batch size
        "other_lr": 0.01/16,        # normalized by batch size
        "freeze_backbone": False,   #  freeze backbone wegiths to finetune
        "decay_gamma": 0.1,
        "decay_step": 20,           #  decay lr in every ? epochs
    },
    "optimizer": {
        "type": "sgd",
        "weight_decay": 4e-05,
    },
    "batch_size": 16,
    "epochs": 1,
    "img_h": 416,
    "img_w": 416,
    "parallels": [0,1,2,3],                         #  config GPU device
    "working_dir": "/tmp/sat/",              #  replace with your working dir
    "evaluate_type": "", 
    "try": 0,
    "export_onnx": False,
}
