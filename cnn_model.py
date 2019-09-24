import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch.autograd import Variable
import math


class TextCNN(nn.Module):
	def __init__(self, args,embed_weights):
		super(TextCNN, self).__init__()
		embed_size = args.embed_size
		max_length = args.max_length
		kernel_size = args.kernel_size
		in_channel = 1
		kernel_num = args.kernel_num
		dropout = args.dropout
		class_num = args.class_num

		self.embed = nn.Embedding.from_pretrained(Variable(torch.FloatTensor(embed_weights)), freeze=True)
		#self.init_embed(embed_weights)
		#
		self.convs = nn.ModuleList([nn.Conv2d(in_channel, kernel_num, (K, embed_size)) for K in kernel_size])
		self.dropout = nn.Dropout(dropout)
		self.fc = nn.Linear(len(kernel_size)*kernel_num, class_num)

	def conv_and_pool(self, x, conv):
		x = x.unsqueeze(1)
		x = conv(x)
		x = F.relu(x).squeeze(3)
		x = F.max_pool1d(x, x.size(2)).squeeze(2)
		return x

	def init_embed(self, embed_matrix):
		self.embed.weight = nn.Parameter(torch.Tensor(embed_matrix))
		self.embed.weight.requires_grad = False

	def forward(self, x):
		x = self.embed(x)

		#x = Variable(x)
		#x = x.unsqueeze(1)
		#x = x.permute(0, 2, 1)
		x = [self.conv_and_pool(x, conv) for conv in self.convs]

		x = torch.cat(x, 1)
		#print('after cat',x.data.shape)
		x = self.dropout(x)
		pro = self.fc(x)
		return pro

	def init_weight(self):
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
				m.weight.data.normal_(0, math.sqrt(2. / n))
				if m.bias is not None:
					m.bias.data.zero_()
			elif isinstance(m, nn.Linear):
				m.weight.data.normal_(0, 0.01)
				m.bias.data.zero_()
