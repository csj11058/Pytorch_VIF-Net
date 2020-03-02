import torch
from data import *
import torch.nn as nn
from torch.nn import functional as F

class TV_Loss(nn.Module):
	def __init__(self):
		super(TV_Loss,self).__init__()

	def forward(self,IA,IF):
		r=IA-IF
		batch_size=r.shape[0]
		h=r.shape[2]
		w=r.shape[3]
		tv1=torch.pow((r[:,:,1:,:]-r[:,:,:h-1,:]),2).mean()
		tv2=torch.pow((r[:,:,:,1:]-r[:,:,:,:w-1]),2).mean()
		return tv1+tv2

class SSIM_Loss(nn.Module):
	def __init__(self,lam=1000,size=11,stride=11,C=9e-4,use_ssim=True,use_tv=True):
		super().__init__()
		self.lam=lam
		self.size=size
		self.use_ssim=use_ssim
		self.use_tv=use_tv
		self.stride=stride
		self.C=C
		self.TV=TV_Loss()

	def forward(self,IA,IB,IF):

		if self.use_tv:
			TV1=self.TV(IA,IF)
			TV2=self.TV(IB,IF)
		if self.use_ssim:

			window=torch.ones((1,1,self.size,self.size))/(self.size*self.size)
			window=window.cuda()

			mean_IA=F.conv2d(IA,window,stride=self.stride)
			mean_IB=F.conv2d(IB,window,stride=self.stride)
			mean_IF=F.conv2d(IF,window,stride=self.stride)

			mean_IA_2=F.conv2d(torch.pow(IA,2),window,stride=self.stride)
			mean_IB_2=F.conv2d(torch.pow(IB,2),window,stride=self.stride)
			mean_IF_2=F.conv2d(torch.pow(IF,2),window,stride=self.stride)

			var_IA=mean_IA_2-torch.pow(mean_IA,2)
			var_IB=mean_IB_2-torch.pow(mean_IB,2)
			var_IF=mean_IF_2-torch.pow(mean_IF,2)

			mean_IAIF=F.conv2d(IA*IF,window,stride=self.stride)
			mean_IBIF=F.conv2d(IB*IF,window,stride=self.stride)

			sigma_IAIF=mean_IAIF-mean_IA*mean_IF
			sigma_IBIF=mean_IBIF-mean_IB*mean_IF

			C=torch.ones(sigma_IAIF.shape)*self.C
			C=C.cuda()

			ssim_IAIF=(2*sigma_IAIF+C)/(var_IA+var_IF+C)
			ssim_IBIF=(2*sigma_IBIF+C)/(var_IB+var_IF+C)

			score=ssim_IAIF*(mean_IA>mean_IB)+ssim_IBIF*(mean_IA<=mean_IB)
			ssim=1-torch.mean(score)

		if self.use_ssim and self.use_tv:
			return self.lam*ssim+TV1
		elif self.use_ssim and not self.use_tv:
			return self.lam*ssim
		elif not self.use_ssim and self.use_tv:
			return (TV1+TV2)/2
		else:
			return 0

if __name__=='__main__':
	criterion = SSIM_Loss()
	input=torch.rand((2,1,320,320))
	input=input.cuda()
	loss=criterion(input,input,input)
	print(loss.item())