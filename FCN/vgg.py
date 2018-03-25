
import os
import os.path as osp 
#import fcn
import requests
import base64
import urllib
import hashlib

# import torch
# import torchvision

def VGG16(pretrained = False):
  model = torchvision.models.vgg16(pretrained=False)

  if not pretrained:
    return model
  model_file = _get_vgg16_pretrained_model()
  state_dict = torch.load(model_file)
  model.load_state_dict(state_dict)
  return model


def _get_vgg16_pretrained_model():

  file_path = osp.join(os.getcwd(), 'pretrain_model','vgg16_from_caffe.pth')
  
  if not os.path.exists(file_path):
    try:
      #pdb.set_trace()
      urllib.request.urlretrieve('http://drive.google.com/uc?id=0B9P1L--7Wd2vLTJZMXpIRkVVRFk',\
        file_path)
    except Exception as e:
          print(e)

  md5 = hashlib.md5()
  vgg16_md5 = 'aa75b158f4181e7f6230029eb96c1b13'

  # with open('vgg16_from_caffe.pth','rb') as f:
  #   while True:
  #     data = f.read(64 * 2048)
  #     if not data:
  #       break
  #     md5.update(data)  
  #   retmd5 = base64.b64encode(md5.digest())
  
  return file_path

if __name__ == '__main__':
  _get_vgg16_pretrained_model()