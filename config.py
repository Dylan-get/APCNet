CONFIG = {}

CONFIG['dataset'] = dict(
    root='/home/wht/transformer/largeData/datasets/cityscapes'
)

CONFIG['model'] = {
    'resnet': dict(
        resnet_pretrained_root='/home/wht/transformer/other/APCNet/architectures/pretrained_model/resnet101_v1c-e67eebb6.pdparams'
    )
}

CONFIG['train_batch_size'] = 6
CONFIG['val_batch_size'] = 1
CONFIG['test_batch_size'] = 1
CONFIG['num_workers'] = 16
CONFIG['optimizers'] = {
    'backbone': dict(type='SGD', lr=1e-3, momentum=0.9, weight_decay=0.0005),
    'APCHead': dict(type='SGD', lr=1e-2, momentum=0.9, weight_decay=0.0005),
    'FCNHead': dict(type='SGD', lr=1e-2, momentum=0.9, weight_decay=0.0005),
}
