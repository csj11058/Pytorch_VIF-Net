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

parser = argparse.ArgumentParser(description = 'train')

parser.add_argument('--epoch', type = int, default = 50, help = 'config file')
parser.add_argument('--lr', type = float, default = 0.0001, help = 'config file')
parser.add_argument('--test', type = bool, default = True, help = 'config file')

args = parser.parse_args()

def train():
	epoch=args.epoch
	lr=args.lr
	criterion = SSIM_Loss()
	model=VIFNet_bn().cuda()
	opt=torch.optim.Adam(model.parameters(),lr)
	with tqdm(total=epoch) as train_bar:
		for i in range(1,epoch+1):
			allloss=0
			train_bar.update(1)
			for j in range(0,60):
				input1,input2=load_train_data('./data/TNO/',j)
				output=model(input1,input2)
				loss = criterion(input1,input2,output)
				opt.zero_grad()
				loss.backward()
				opt.step()
				allloss=allloss+loss
				if i//100-i/100==0:
					lr=lr/10
					opt=torch.optim.Adam(model.parameters(),lr)
			train_bar.set_description('epoch:%s loss:%.5f'%(i,allloss.item()/60))
			torch.save(model,'./model/model'+str(i)+'.pth')

if __name__=='__main__':
	train()