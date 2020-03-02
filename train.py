import os
import math
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

parser = argparse.ArgumentParser(description = 'train')

parser.add_argument('--epoch', type = int, default = 2500, help = 'config file')
parser.add_argument('--bs', type = int, default = 2, help = 'config file')
parser.add_argument('--lr', type = float, default = 0.1, help = 'config file')
parser.add_argument('--test', type = bool, default = True, help = 'config file')

args = parser.parse_args()

def train():
	epoch=args.epoch
	batch_size=args.bs
	lr=args.lr
	criterion = SSIM_Loss()#nn.MSELoss()#
	model=VIFNet_bn().cuda()
	_,_,traindata=load_train_data('./data/TNO/',1,1)
	# model=torch.load('./model/model300.pth')
	opt=torch.optim.Adam(model.parameters(),lr)
	for i in range(1,epoch+1):
		allloss=0
		with tqdm(total=math.ceil(traindata/batch_size)) as train_bar:
			for j in range(0,math.ceil(traindata/batch_size)):
				train_bar.update(1)
				input1,input2,_=load_train_data('./data/TNO/',j,batch_size)
				output=model(input1,input2)
				loss = criterion(input1,input2,output)
				opt.zero_grad()
				loss.backward()
				opt.step()
				allloss=allloss+loss
				# image_save(output,'./output/'+str(i)+'_'+str(j)+'.jpg')
				if i//800-i/800==0:
					lr=lr/10
					opt=torch.optim.Adam(model.parameters(),lr)
				train_bar.set_description('epoch:%s loss:%.5f'%(i,allloss/(j+1)))
		torch.save(model,'./model/model'+str(i)+'.pth')

if __name__=='__main__':
	train()
