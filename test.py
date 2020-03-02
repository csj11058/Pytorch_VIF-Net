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
	_,_,testdata=load_train_data('./data/TNO/',1,1)
	model=torch.load(model_path)
	model=model.cuda()
	model.eval()
	# for i in range(0,testdata):
	input1,input2,_=load_train_data('./data/TNO/',0,1)
	output=model(input1,input2)
	image_save(output,'./test/'+str(index)+'_'+str(0)+'.jpg')

if __name__=='__main__':
	modelname=os.listdir('./model/')
	for i in range(0,len(modelname)):
		test('./model/'+modelname[i],i)