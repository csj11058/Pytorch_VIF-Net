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

loader = transforms.Compose([transforms.ToTensor()])  
unloader = transforms.ToPILImage()

def image_save(tensor,name):
	image=tensor.cpu().clone()
	image=image.squeeze(0)
	image=unloader(image)
	image.save(name)

def image_loader(image_name,shape=320):
	img = Image.open(image_name)#.convert('RGB')
	image = img.resize((shape,shape))
	image = loader(image).unsqueeze(0)
	image=image.to('cuda', torch.float32)
	if image.shape[1]>1:
		image=image[:,0,:,:]
		image=torch.reshape(image,(1,1,shape,shape))
	return image

def load_train_data(path,batch):
	dirname=os.listdir(path)
	imgname=[]
	for i in dirname:
		img=os.listdir(path+i)
		img=[path+i+'/'+img[0],path+i+'/'+img[1]]
		imgname.append(img)
	for i in range(batch,batch+1):
		if i==batch:
			train_data1=image_loader(imgname[i][0])
			train_data2=image_loader(imgname[i][1])
		else:
			data1=image_loader(imgname[i][0])
			data2=image_loader(imgname[i][1])
			train_data1=torch.cat((train_data1,data1),0)
			train_data2=torch.cat((train_data2,data2),0)
	train_data1=train_data1*255
	train_data2=train_data2*255
	return train_data1.cuda(),train_data2.cuda()

if __name__=='__main__':
	img1,img2=load_train_data()
	print(img1.shape,img2.shape)