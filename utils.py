import os
import glob
import random
import matplotlib.pyplot as plt

from PIL import Image
import scipy.misc
import scipy.ndimage
import numpy as np
import matplotlib.pyplot as plt

from read_nc import get_data
import tensorflow as tf
import configparser

def get_names(config):
  return os.listdir(config.input)

def is_zero(l):
  for i in l:
    if max(i) > 10:
      return True
  return False

def preprocess_reconstruct(config):

  input_ = get_data(config.input)
  feature_ = get_data(config.feature)
  dex = np.isnan(feature_)
  feature_[dex] = 0
  feature_ = feature_/3000.
  dex = np.isnan(input_)
  input_[dex] = 0
  
  #img_ = get_data(config.input)
  scale0w = config.input_size_w/input_.shape[-2]
  scale0j = config.input_size_j/input_.shape[-1]
  img_ = scipy.ndimage.zoom(input_,(1,scale0w,scale0j))

  scale1w = config.input_size_w/feature_.shape[-2]
  scale1j = config.input_size_j/feature_.shape[-1]
  feature_ = scipy.ndimage.zoom(feature_,(1,scale1w,scale1j))

  shap = img_.shape
  image_ = np.zeros((shap[0],shap[1],shap[2],2))
  image_[:, :, :, 0] = img_
  image_[:, :, :, 1] = feature_
  return image_
                         

def preprocess(config):
  input_ = get_data(config.label)
  feature_ = get_data(config.feature)
  dex = np.isnan(feature_)
  feature_[dex] = 0
  feature_ = feature_/3000.
  
  dex = np.isnan(input_)
  input_[dex] = 0
  label_ = input_
  
  # filter the data
  print("before filter input size is:")
  print(input_.shape)
  input_ = np.array(list(filter(is_zero, input_)))
  print("after filter the input size is:")
  print(input_.shape)

  label_ = input_
  print("after filter the label size is:")
  print(label_.shape)

  scale0w = config.input_size_w/input_.shape[-2]
  scale0j = config.input_size_j/input_.shape[-1]
  img_ = scipy.ndimage.zoom(input_,(1,scale0w,scale0j))
  
  img_ = np.maximum(img_,0)
  img_ = np.where(img_<0,0,img_)
  img_ = np.where(img_>9999,0,img_)

  scale1w = config.input_size_w/feature_.shape[-2]
  scale1j = config.input_size_j/feature_.shape[-1]
  feature_ = scipy.ndimage.zoom(feature_,(1,scale1w,scale1j))

  scale2w = config.output_size_w/label_.shape[-2]
  scale2j = config.output_size_j/label_.shape[-1]
  label_ = scipy.ndimage.zoom(label_, (1,scale2w,scale2j))
  label_ = np.maximum(label_,0)
  label_=label_.reshape(-1,config.output_size_w,config.output_size_j,1)

  shap = img_.shape
  image_ = np.zeros((shap[0],shap[1],shap[2],2))
  image_[:, :, :, 0] = img_
  image_[:, :, :, 1] = feature_
  return image_, label_


def parse_config_test(config_path,section, option):
  cp = configparser.ConfigParser()
  
  if not os.path.exists(config_path):
    print("NO configuration:{0} !".format(config_path))
    exit()
  cp.read(config_path)
  return cp.get(section, option)



#if __name__=='__main__':
# parse_config()


