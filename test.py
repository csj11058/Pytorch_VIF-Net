import os
import torch
import argparse
import numpy as np
import torchvision
from loss import *
import torch.nn as nn
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision import transforms
from data import *
from model import *

def test(model_path,index):
	model=torch.load(model_path)
	model=model.cuda()
	model.eval()
	for i in range(0,60):
		input1,input2=load_train_data('./data/TNO/',i)
		output=model(input1,input2)
		image_save(output,'./test/'+str(index)+'_'+str(i)+'.jpg')

if __name__=='__main__':
	modelname=os.listdir('./model/')
	for i in range(0,len(modelname)):
		test('./model/'+modelname[i],i)