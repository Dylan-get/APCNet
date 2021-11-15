
from .apcnet_paddle import APCHead
from .resnet_paddle import resnet101
from .fcnhead_paddle import FCNHead



def getApcNet(config=None):
    models={}
    models['backbone'],msg1=resnet101(config=config)
    models['APCHead']=APCHead()
    models['FCNHead']=FCNHead()
    return models,msg1