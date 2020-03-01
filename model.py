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

class Feature_extraction_block(nn.Module):
	def __init__(self):
		super(Feature_extraction_block, self).__init__()
		self.C1=nn.Sequential(
			nn.Conv2d(in_channels=1,out_channels=16,
						kernel_size=3,stride=1,padding=1),
			nn.ReLU())
		self.D1=nn.Sequential(
			nn.Conv2d(in_channels=16,out_channels=16,
						kernel_size=3,stride=1,padding=1),
			nn.ReLU())
		self.D2=nn.Sequential(
			nn.Conv2d(in_channels=32,out_channels=16,
						kernel_size=3,stride=1,padding=1),
			nn.ReLU())
		self.D3=nn.Sequential(
			nn.Conv2d(in_channels=48,out_channels=16,
						kernel_size=3,stride=1,padding=1),
			nn.ReLU())

	def forward(self,x):
		c1=self.C1(x)
		d1=self.D1(c1)
		d2=torch.cat((c1,d1),1)
		d2=self.D2(d2)
		d3=torch.cat((c1,d1,d2),1)
		d3=self.D3(d3)
		out=torch.cat((c1,d1,d2,d3),1)
		return out

class Feature_extraction_bn_block(nn.Module):
	def __init__(self):
		super(Feature_extraction_bn_block, self).__init__()
		self.C1=nn.Sequential(
			nn.Conv2d(in_channels=1,out_channels=16,
						kernel_size=3,stride=1,padding=1),
			nn.BatchNorm2d(16),
			nn.ReLU())
		self.D1=nn.Sequential(
			nn.Conv2d(in_channels=16,out_channels=16,
						kernel_size=3,stride=1,padding=1),
			nn.BatchNorm2d(16),
			nn.ReLU())
		self.D2=nn.Sequential(
			nn.Conv2d(in_channels=32,out_channels=16,
						kernel_size=3,stride=1,padding=1),
			nn.BatchNorm2d(16),
			nn.ReLU())
		self.D3=nn.Sequential(
			nn.Conv2d(in_channels=48,out_channels=16,
						kernel_size=3,stride=1,padding=1),
			nn.BatchNorm2d(16),
			nn.ReLU())

	def forward(self,x):
		c1=self.C1(x)
		d1=self.D1(c1)
		d2=torch.cat((c1,d1),1)
		d2=self.D2(d2)
		d3=torch.cat((c1,d1,d2),1)
		d3=self.D3(d3)
		out=torch.cat((c1,d1,d2,d3),1)
		return out

class Feature_fusion_block(nn.Module):
	def __init__(self):
		super(Feature_fusion_block, self).__init__()

	def forward(self,IA,IB):
		out=torch.cat((IA,IB),1)
		return out

class Feature_reconstruction_block(nn.Module):
	def __init__(self):
		super(Feature_reconstruction_block, self).__init__()
		self.C2=nn.Sequential(
			nn.Conv2d(in_channels=128,out_channels=128,
						kernel_size=3,stride=1,padding=1),
			nn.ReLU())
		self.C3=nn.Sequential(
			nn.Conv2d(in_channels=128,out_channels=64,
						kernel_size=3,stride=1,padding=1),
			nn.ReLU())
		self.C4=nn.Sequential(
			nn.Conv2d(in_channels=64,out_channels=32,
						kernel_size=3,stride=1,padding=1),
			nn.ReLU())
		self.C5=nn.Sequential(
			nn.Conv2d(in_channels=32,out_channels=16,
						kernel_size=3,stride=1,padding=1),
			nn.ReLU())
		self.C6=nn.Sequential(
			nn.Conv2d(in_channels=16,out_channels=1,
						kernel_size=3,stride=1,padding=1),
			nn.ReLU())

	def forward(self,x):
		x=self.C2(x)
		x=self.C3(x)
		x=self.C4(x)
		x=self.C5(x)
		x=self.C6(x)
		return x

class Feature_reconstruction_bn_block(nn.Module):
	def __init__(self):
		super(Feature_reconstruction_bn_block, self).__init__()
		self.C2=nn.Sequential(
			nn.Conv2d(in_channels=128,out_channels=128,
						kernel_size=3,stride=1,padding=1),
			nn.BatchNorm2d(128),
			nn.ReLU())
		self.C3=nn.Sequential(
			nn.Conv2d(in_channels=128,out_channels=64,
						kernel_size=3,stride=1,padding=1),
			nn.BatchNorm2d(64),
			nn.ReLU())
		self.C4=nn.Sequential(
			nn.Conv2d(in_channels=64,out_channels=32,
						kernel_size=3,stride=1,padding=1),
			nn.BatchNorm2d(32),
			nn.ReLU())
		self.C5=nn.Sequential(
			nn.Conv2d(in_channels=32,out_channels=16,
						kernel_size=3,stride=1,padding=1),
			nn.BatchNorm2d(16),
			nn.ReLU())
		self.C6=nn.Sequential(
			nn.Conv2d(in_channels=16,out_channels=1,
						kernel_size=3,stride=1,padding=1),
			nn.BatchNorm2d(1),
			nn.ReLU())

	def forward(self,x):
		x=self.C2(x)
		x=self.C3(x)
		x=self.C4(x)
		x=self.C5(x)
		x=self.C6(x)
		return x

class VIFNet(nn.Module):
	def __init__(self):
		super(VIFNet, self).__init__()
		self.Feature_extraction_block=Feature_extraction_block()
		self.Feature_fusion_block=Feature_fusion_block()
		self.Feature_reconstruction_block=Feature_reconstruction_block()

	def forward(self,IA,IB):
		IA=self.Feature_extraction_block(IA)
		IB=self.Feature_extraction_block(IB)
		x=self.Feature_fusion_block(IA,IB)
		x=self.Feature_reconstruction_block(x)
		return x

class VIFNet_bn(nn.Module):
	def __init__(self):
		super(VIFNet_bn, self).__init__()
		self.Feature_extraction_block=Feature_extraction_bn_block()
		self.Feature_fusion_block=Feature_fusion_block()
		self.Feature_reconstruction_block=Feature_reconstruction_bn_block()

	def forward(self,IA,IB):
		IA=self.Feature_extraction_block(IA)
		IB=self.Feature_extraction_block(IB)
		x=self.Feature_fusion_block(IA,IB)
		x=self.Feature_reconstruction_block(x)
		return x

if __name__=='__main__':
	epoch=1000
	criterion = SSIM_Loss()
	input1=torch.rand((1,1,320,320)).cuda()*100
	input2=torch.rand((1,1,320,320)).cuda()*100
	output=torch.rand((1,1,320,320)).cuda()*100
	model=VIFNet().cuda()
	opt=torch.optim.Adam(model.parameters(),0.1)
	with tqdm(total=epoch) as train_bar:
		for i in range(0,epoch):
			train_bar.update(1)
			output=model(input1,input2)
			loss = criterion(input1,input2,output)
			opt.zero_grad()
			loss.backward()
			opt.step()
			print(loss)
			torch.save(model,'model.pth')