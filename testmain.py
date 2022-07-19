# 训练+测试


import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import numpy as np
import re
import string
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# ---------------- start  add code -------------------------

#   ----  read pos.txt file, and return a 2d-list  ----------
def readtxt(path):
	poshandle = open(path + '/pos.txt', encoding="utf-8")  # open pos.txt
	postext = poshandle.read()
	poslist = postext.split("\n")
	posarr = [[j for j in range(10)] for i in range(len(poslist))]
	cnt1 = 0
	
	for siglist in poslist:
		if len(siglist) > 1:
			siglistsplit = siglist.split(",", 1)
			pathpng = path + '/' + str(int(float(siglistsplit[0]))) + '.png'
			imgpixs = cv2.imread(pathpng)
			if not (imgpixs[0, 639, 0] - 224 == 0 and imgpixs[0, 1279, 0] - 255 == 0):
				continue
			
			siglistsplit[0] = pathpng
			posarr[cnt1] = siglistsplit
			cnt1 += 1
	out = posarr[0:cnt1]
	return out


# -----------------------------------------------------------------------------------
# -----------  create label.txt, containing PNG file path and labels -------------------------
def createlabel(path):
	folders = os.listdir(path)  # 列出dirname下的目录和文件
	traintxt = open(path + 'label.txt', 'w+')
	lenfolders = len(folders)
	cnt = 1
	for folder in folders:
		print('--- Pending --- ', cnt, ' out of ', lenfolders)
		cnt += 1
		if folder[-4:len(folder)] == '.txt':
			continue
		folderdir = path + folder
		
		#      read pos.txt in folderdir
		posarr = readtxt(folderdir)
		for posarrsig in posarr:
			posarrsigstr = str(posarrsig)
			posarrsigstr = posarrsigstr.replace('[', '')
			posarrsigstr = posarrsigstr.replace(']', '')
			posarrsigstr = posarrsigstr.replace(',', ' ')
			posarrsigstr = posarrsigstr.replace("'", '')
			traintxt.write(posarrsigstr + '\n')
	
	traintxt.close()


# ------------------------------------------------


def default_loader(path):
	return Image.open(path).convert('RGB')


class MyDataset(Data.Dataset):
	def __init__(self, txt, transform=None, target_transform=None, loader=default_loader):
		super(MyDataset, self).__init__()
		
		fhandle = open(txt, 'r')
		imgs = []
		for line in fhandle:
			line = line.strip('\n')
			line = line.rstrip('\n')
			
			words = line.split()
			imgs.append((words[0] + ' ' + words[1], words[2:len(words)]))
		
		self.imgs = imgs
		self.transform = transform
		self.target_transform = target_transform
		self.loader = loader
		# *************************** #使用__getitem__()对数据进行预处理并返回想要的信息**********************
	
	def __getitem__(self, index):  # 这个方法是必须要有的，用于按照索引读取每个元素的具体内容
		fn, label = self.imgs[index]
		# fn是图片path #fn和label分别获得imgs[index]也即是刚才每行中word[0]和word[1]的信息
		img = self.loader(fn)
		# 按照路径读取图片
		if self.transform is not None:
			img = self.transform(img)
			# 数据标签转换为Tensor
		return img, label
		# return回哪些内容，那么我们在训练时循环读取每个batch时，就能获得哪些内容
		# **********************************  #使用__len__()初始化一些需要传入的参数及数据集的调用**********************
	
	def __len__(self):
		# 这个函数也必须要写，它返回的是数据集的长度，也就是多少张图片，要和loader的长度作区分
		return len(self.imgs)


class MyDatasetRnn(Data.Dataset):
	def __init__(self, txt, transform=None, target_transform=None, loader=default_loader):
		super(MyDatasetRnn, self).__init__()
		
		fhandle = open(txt, 'r')
		imgs = []
		for line in fhandle:
			line = line.strip('\n')
			line = line.rstrip('\n')
			
			words = line.split('|')
			imgs.append((words[0:-1], words[-1]))
		
		self.imgs = imgs
		self.transform = transform
		self.target_transform = target_transform
		self.loader = loader
		# *************************** #使用__getitem__()对数据进行预处理并返回想要的信息**********************
	
	def __getitem__(self, index):  # 这个方法是必须要有的，用于按照索引读取每个元素的具体内容
		fn, label = self.imgs[index]
		# fn是图片path #fn和label分别获得imgs[index]也即是刚才每行中word[0]和word[1]的信息
		seq_len = 20
		imgs = []
		if fn.__len__() >= seq_len:
			for i in range(seq_len):
				img = self.loader(fn[fn.__len__() - 20 + i])
				if self.transform is not None:
					img = self.transform(img)
				imgs.append(img)
		else:
			for i in range(seq_len):
				if i < seq_len - fn.__len__():
					img = self.loader(fn[0])
					if self.transform is not None:
						img = self.transform(img)
				else:
					img = self.loader(fn[i - 1 - (seq_len - fn.__len__())])
					if self.transform is not None:
						img = self.transform(img)
				imgs.append(img)
		
		return imgs, label.split(' ')
		# return回哪些内容，那么我们在训练时循环读取每个batch时，就能获得哪些内容
		# **********************************  #使用__len__()初始化一些需要传入的参数及数据集的调用**********************
	
	def __len__(self):
		# 这个函数也必须要写，它返回的是数据集的长度，也就是多少张图片，要和loader的长度作区分
		return len(self.imgs)


# 用class类来建立CNN模型
# CNN流程：卷积(Conv2d)-> 激励函数(ReLU)->池化(MaxPooling)->
#        卷积(Conv2d)-> 激励函数(ReLU)->池化(MaxPooling)->
#        展平多维的卷积成的特征图->接入全连接层(Linear)->输出

class CNN(nn.Module):  # 我们建立的CNN继承nn.Module这个模块
	def __init__(self):
		super(CNN, self).__init__()
		# 建立第一个卷积(Conv2d)-> 激励函数(ReLU)->池化(MaxPooling)
		self.conv1 = nn.Sequential(
			# 第一个卷积con2d
			nn.Conv2d(  # 输入图像大小(3,1280,512)
				in_channels=3,  # 输入图片的高度，因为minist数据集是灰度图像只有一个通道
				out_channels=16,  # n_filters 卷积核的高度
				kernel_size=5,  # filter size 卷积核的大小 也就是长x宽=5x5
				stride=1,  # 步长
				padding=2,  # 想要con2d输出的图片长宽不变，就进行补零操作 padding = (kernel_size-1)/2
			),  # 输出图像大小(16,1280,512)
			
			nn.BatchNorm2d(16, 1e-5, 0.1, True, True, None, None),
			# 激活函数
			nn.ReLU(),
			# 池化，下采样
			nn.MaxPool2d(kernel_size=2),  # 在2x2空间下采样
			# 输出图像大小(16,640,256)
		)
		# 建立第二个卷积(Conv2d)-> 激励函数(ReLU)->池化(MaxPooling)
		self.conv2 = nn.Sequential(
			# 输入图像大小(16,640,256)
			nn.Conv2d(  # 也可以直接简化写成nn.Conv2d(16,32,5,1,2)
				in_channels=16,
				out_channels=32,
				kernel_size=5,
				stride=1,
				padding=2
			),
			# 输出图像大小 (32,640,256)
			nn.BatchNorm2d(32, 1e-5, 0.1, True, True, None, None),
			nn.ReLU(),
			nn.MaxPool2d(2),
			# 输出图像大小(32,320,128)
		)
		# 建立第三个卷积(Conv2d)-> 激励函数(ReLU)->池化(MaxPooling)
		self.conv3 = nn.Sequential(
			# 输入图像大小(32,320,128)
			nn.Conv2d(
				in_channels=32,
				out_channels=64,
				kernel_size=5,
				stride=1,
				padding=2
			),
			# 输出图像大小 (64,320,128)
			nn.BatchNorm2d(64, 1e-5, 0.1, True, True, None, None),
			nn.ReLU(),
			nn.MaxPool2d(2),
			# 输出图像大小(64,160,64)
		)
		# 建立第四个卷积(Conv2d)-> 激励函数(ReLU)->池化(MaxPooling)
		self.conv4 = nn.Sequential(
			# 输入图像大小(64,160,64)
			nn.Conv2d(
				in_channels=64,
				out_channels=128,
				kernel_size=5,
				stride=1,
				padding=2
			),
			# 输出图像大小 (128,160,64)
			nn.BatchNorm2d(128, 1e-5, 0.1, True, True, None, None),
			nn.ReLU(),
			nn.MaxPool2d(2),
			# 输出图像大小(128,,80,32)
		)
		# 建立第五个卷积(Conv2d)-> 激励函数(ReLU)->池化(MaxPooling)
		self.conv5 = nn.Sequential(
			# 输入图像大小(128,,80,32)
			nn.Conv2d(
				in_channels=128,
				out_channels=128,
				kernel_size=5,
				stride=1,
				padding=2
			),
			# 输出图像大小 (128,80,32)
			nn.BatchNorm2d(128, 1e-5, 0.1, True, True, None, None),
			nn.ReLU(),
			nn.MaxPool2d(2),
			# 输出图像大小(128,,40,16)
		)
		# 建立第六个卷积(Conv2d)-> 激励函数(ReLU)->池化(MaxPooling)
		self.conv6 = nn.Sequential(
			# 输入图像大小(128,,40,16)
			nn.Conv2d(
				in_channels=128,
				out_channels=32,
				kernel_size=5,
				stride=1,
				padding=2
			),
			# 输出图像大小 (32,40,16)
			nn.BatchNorm2d(32, 1e-5, 0.1, True, True, None, None),
			nn.ReLU(),
			nn.MaxPool2d(2),
			# 输出图像大小(32,,20,8)
		)
		# 建立全卷积连接层
		self.fc = nn.Sequential(
			nn.Linear(32 * 20 * 8, 1000),
			nn.Linear(1000, 100),
			nn.Linear(100, 3),
		)
		# 输出是10个类
	
	# 下面定义x的传播路线
	def forward(self, x):
		x = self.conv1(x)  # x先通过conv1
		x = self.conv2(x)  # 再通过conv2
		x = self.conv3(x)
		x = self.conv4(x)
		x = self.conv5(x)
		x = self.conv6(x)
		# 把每一个批次的每一个输入都拉成一个维度，即(batch_size,32*7*7)
		# 因为pytorch里特征的形式是[bs,channel,h,w]，所以x.size(0)就是batchsize
		x = x.view(x.size(0), -1)  # view就是把x弄成batchsize行个tensor
		output = self.fc(x)
		return output


class RNN(nn.Module):  # 我们建立的 RNN
	def __init__(self, input_size, hidden_size, num_layers, seq_len):
		super(RNN, self).__init__()
		
		self.hidden_size = hidden_size
		self.num_layers = num_layers
		
		self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
		
		# 建立全卷积连接层
		self.fc = nn.Sequential(
			nn.Linear(seq_len * hidden_size, 100),
			nn.Linear(100, 100),
			nn.Linear(100, 3),
		)
		# 输出是10个类
	
	# 下面定义x的传播路线
	def forward(self, x):
		out, _ = self.lstm(x)
		# 把每一个批次的每一个输入都拉成一个维度，即(batch_size,32*7*7)
		# 因为pytorch里特征的形式是[bs,channel,h,w]，所以x.size(0)就是batchsize
		out = out.view(out.size(0), -1)  # view就是把x弄成batchsize行个tensor
		output = self.fc(out)
		return output


def createlabelrnn(path):
	rnnhandle = open(path + 'rnnlabel.txt', 'w+')
	
	cnt = 1
	trainhandle = open(path + 'label.txt', encoding="utf-8")  # open label.txt
	traintext = trainhandle.read()
	trainlist = traintext.split("\n")
	lentrain = len(trainlist)
	labarr = [i for i in range(len(trainlist))]
	cnt1 = 0
	orderNum = -1
	
	pathsum = []
	seqlen = 0
	for i in range(lentrain):
		if len(trainlist[i]) > 1:
			
			trainlistsplit = trainlist[i].split("  ")
			trainlistsplitorder = trainlistsplit[1].split(' ', 1)
			orderread = float(trainlistsplitorder[0])
			if orderNum == -1:
				pathsum = trainlistsplit[0]
				seqlen += 1
			elif not orderread == orderNum:
				labarr[cnt1] = pathsum + '|' + str(int(seqlen)) + ' ' + trainlistsplitorder[1]
				cnt1 += 1
				seqlen = 0
				pathsum = trainlistsplit[0]
			else:
				pathsum = pathsum + '|' + trainlistsplit[0]
				seqlen += 1
			
			orderNum = orderread
	
	labvalarr = labarr[0:cnt1]
	k = 0
	for labvalarrsig in labvalarr:
		k += 1
		if not k == labvalarr.__len__():
			rnnhandle.write(labvalarrsig + '\n')
		else:
			rnnhandle.write(labvalarrsig)
	rnnhandle.close()


# *******************************************************************************************************************
# ---  processing labels of train and test data----

trainTxtPath = './pybulletData/train/'
testTxtPath = './pybulletData/test/'
DataProcessTrain = 0
if (DataProcessTrain == 1):
	createlabel(trainTxtPath)
	createlabel(testTxtPath)
#   --------------------------------------------------

train_data = MyDataset(txt=trainTxtPath + 'label.txt', transform=torchvision.transforms.ToTensor())
test_data = MyDataset(txt=testTxtPath + 'label.txt', transform=torchvision.transforms.ToTensor())

DataProcessRnn = 0
if (DataProcessRnn == 1):
	createlabelrnn(trainTxtPath)
	createlabelrnn(testTxtPath)

train_dataRnn = MyDatasetRnn(txt=trainTxtPath + 'rnnlabel.txt', transform=torchvision.transforms.ToTensor())
test_dataRnn = MyDatasetRnn(txt=testTxtPath + 'rnnlabel.txt', transform=torchvision.transforms.ToTensor())

# ---------------- stop  add code -------------------------

torch.manual_seed(1)  # 使用随机化种子使神经网络的初始化每次都相同

# 超参数
EPOCH = 1  # 训练整批数据的次数
BATCH_SIZE = 8
LR = 0.001  # 学习率
SEQ_LEN = 20

# 批训练 2个samples， 3  channel，1280*512 (2,3,1280,512)
train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = Data.DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=True)

train_loader_rnn = Data.DataLoader(dataset=train_dataRnn, batch_size=1, shuffle=True)
test_loader_rnn = Data.DataLoader(dataset=test_dataRnn, batch_size=1, shuffle=True)

cnn = CNN()
rnn = RNN(3, 20, 2, SEQ_LEN)
print(cnn)

# -----------开始训练- CNN ----------
loss_func = nn.MSELoss()
trainEn = 0
if trainEn:
	testlossmean = []
	mseloss = []
	# 优化器选择Adam
	optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)
	# 损失函数
	
	# loss_func = nn.CrossEntropyLoss()  # 目标标签是one-hotted
	for epoch in range(EPOCH):
		for step, (b_x, b_y) in enumerate(train_loader):
			# if step > 1000:
			#     break
			output = cnn(b_x)  # 先将数据放到cnn中计算output
			
			labval = torch.from_numpy(np.array(b_y[1:4], dtype=np.float64))
			loss = loss_func(output.T.float(), labval.float())  # 输出和真实标签的loss，二者位置不可颠倒
			optimizer.zero_grad()  # 清除之前学到的梯度的参数
			loss.backward()  # 反向传播，计算梯度
			optimizer.step()  # 应用梯度
			mseloss.append(loss.data.numpy())
			# mselossmean = np.mean(mseloss)
			
			if step % 50 == 0:
				testloss = []
				for stepeval, (b_x, b_y) in enumerate(test_loader):
					if stepeval > 30:
						break
					output = cnn(b_x)  # 先将数据放到cnn中计算output
					
					labval = torch.from_numpy(np.array(b_y[1:4], dtype=np.float64))
					loss = loss_func(output.T.float(), labval.float())  # 输出和真实标签的loss，二者位置不可颠倒
					testloss.append(loss.data.numpy())
				testlossmean.append(np.mean(testloss))
				print('Epoch: ', epoch, ' | Step: ', step, '| train loss: %.4f' % mseloss[-1],
					  ' | test loss:  %.4f' % testlossmean[-1])
			if step % 1000 == 999:
				torch.save(cnn.state_dict(), str(step) + '.pkl')
	
	torch.save(cnn.state_dict(), 'cnn2.pkl')  # 保存模型
	plt.figure()
	plt.plot(testlossmean)
	plt.show()
# -----------------------------------------------------------------------------------------

# 加载模型 CNN ，调用时需将前面训练及保存模型的代码注释掉，否则会再训练一遍
evaluatecnn = 0
if evaluatecnn == 1:
	cnn.load_state_dict(torch.load('999.pkl'))
	cnn.eval()
	# print 10 predictions from test data
	evaloss = []
	reslabcmp = []
	for stepeval, (b_x, b_y) in enumerate(test_loader):
		output = cnn(b_x)  # 先将数据放到cnn中计算output
		if stepeval > 50:
			break
		labval = torch.from_numpy(np.array(b_y[1:4], dtype=np.float64))
		loss = loss_func(output.T.float(), labval.float())  # 输出和真实标签的loss，二者位置不可颠倒
		evaloss.append(loss.data.numpy())
		
		res = output.detach().numpy()
		lab = np.array(b_y[1:4], dtype=np.float64).T
		reslab = np.concatenate((res, lab), axis=1)
		if stepeval == 0:
			reslabcmp = reslab
		else:
			reslabcmp = np.concatenate((reslabcmp, reslab))
	
	plt.figure()
	plt.scatter(reslabcmp[:, 0], reslabcmp[:, 3], color='b')
	plt.scatter(reslabcmp[:, 1], reslabcmp[:, 4], color='r')
	plt.scatter(reslabcmp[:, 2], reslabcmp[:, 5], color='k')
	plt.plot([0, 1], [0, 1])
	plt.show()
	evalossmean = np.mean(evaloss)
	print(' Average loss of the evaluation is %.4f' % evalossmean)

# --------------------------------------------------------------------------------------------------------
#   ----    train Rnn   ----
loss_funcRnn = nn.MSELoss()
trainRnn = 0
if trainRnn == 1:
	testlossmean = []
	mseloss = []
	# 优化器选择Adam
	optimizerRnn = torch.optim.Adam(rnn.parameters(), lr=LR)
	# 损失函数
	
	# loss_func = nn.CrossEntropyLoss()  # 目标标签是one-hotted
	
	cnn.load_state_dict(torch.load('cnnlast.pkl'))
	for p in cnn.parameters():
		p.requires_grad = False
	
	for epoch in range(1):
		for step, (b_x, b_y) in enumerate(train_loader_rnn):
			# if step > 1000:
			#     break
			for i in range(SEQ_LEN):
				poscnn = cnn(b_x[i])
				if i == 0:
					cnnSeq = poscnn
				else:
					cnnSeq = torch.cat((cnnSeq, poscnn))
			cnnSeq = cnnSeq.unsqueeze(0)
			outRnn = rnn(cnnSeq)  # 先将数据放到cnn中计算output
			
			labval = torch.from_numpy(np.array(b_y[4:7], dtype=np.float64))
			loss = loss_funcRnn(outRnn.T.float(), labval.float())  # 输出和真实标签的loss，二者位置不可颠倒
			optimizerRnn.zero_grad()  # 清除之前学到的梯度的参数
			loss.backward()  # 反向传播，计算梯度
			optimizerRnn.step()  # 应用梯度
			mseloss.append(loss.data.numpy())
			# mselossmean = np.mean(mseloss)
			
			if step % 50 == 0:
				testloss = []
				for stepeval, (b_x, b_y) in enumerate(test_loader_rnn):
					if stepeval > 30:
						break
					for i in range(SEQ_LEN):
						poscnn = cnn(b_x[i])
						if i == 0:
							cnnSeq = poscnn
						else:
							cnnSeq = torch.cat((cnnSeq, poscnn))
					cnnSeq = cnnSeq.unsqueeze(0)
					outRnn = rnn(cnnSeq)  # 先将数据放到cnn中计算output
					
					labval = torch.from_numpy(np.array(b_y[4:7], dtype=np.float64))
					loss = loss_funcRnn(outRnn.T.float(), labval.float())  # 输出和真实标签的loss，二者位置不可颠倒
					testloss.append(loss.data.numpy())
				testlossmean.append(np.mean(testloss))
				print('Epoch: ', epoch, ' | Step: ', step, '| train loss: %.4f' % mseloss[-1],
					  ' | test loss:  %.4f' % testlossmean[-1])
			if step % 1000 == 999:
				torch.save(rnn.state_dict(), str(step) + 'rnn' + '.pkl')
	torch.save(rnn.state_dict(), 'rnnlast.pkl')  # 保存模型
	plt.figure()
	plt.plot(testlossmean)
	plt.show()

# 加载模型 RNN ，调用时需将前面训练及保存模型的代码注释掉，否则会再训练一遍
evaluateRnn = 1
if evaluateRnn == 1:
	cnn.load_state_dict(torch.load('cnnlast.pkl'))
	cnn.eval()
	
	rnn.load_state_dict(torch.load('rnnlast.pkl'))
	rnn.eval()
	# print 10 predictions from test data
	evaloss = []
	reslabcmp = []
	for stepeval, (b_x, b_y) in enumerate(test_loader_rnn):
		for i in range(SEQ_LEN):
			poscnn = cnn(b_x[i])
			if i == 0:
				cnnSeq = poscnn
			else:
				cnnSeq = torch.cat((cnnSeq, poscnn))
		cnnSeq = cnnSeq.unsqueeze(0)
		outRnn = rnn(cnnSeq)  # 先将数据放到cnn中计算output
		
		if stepeval > 50:
			break
		labval = torch.from_numpy(np.array(b_y[4:7], dtype=np.float64))
		loss = loss_funcRnn(outRnn.T.float(), labval.float())  # 输出和真实标签的loss，二者位置不可颠倒
		evaloss.append(loss.data.numpy())
		
		res = outRnn.detach().numpy()
		lab = np.array(b_y[4:7], dtype=np.float64).T
		reslab = np.concatenate((res, lab), axis=1)
		if stepeval == 0:
			reslabcmp = reslab
		else:
			reslabcmp = np.concatenate((reslabcmp, reslab))
	
	plt.figure()
	plt.scatter(reslabcmp[:, 0], reslabcmp[:, 3], color='b')
	plt.scatter(reslabcmp[:, 1], reslabcmp[:, 4], color='r')
	plt.scatter(reslabcmp[:, 2], reslabcmp[:, 5], color='k')
	plt.plot([0, 1], [0, 1])
	plt.show()
	evalossmean = np.mean(evaloss)
	print(' Average loss of the evaluation is %.4f' % evalossmean)

# --------------------------------------------------------------------------------------------------------
