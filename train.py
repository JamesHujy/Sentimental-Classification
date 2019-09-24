import os
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import pickle
import tqdm
from sklearn.metrics import f1_score


def train(model, args, train_iter, test_iter, label_weight):
	if torch.cuda.is_available():
		model.cuda()
	print(model)
	optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
	'''
	weight = [0.8, 1, 1, 0.7, 0.9, 1, 1, 1]
	weight = np.array(weight)
	weight = torch.from_numpy(weight).float()
	print(weight)
	if torch.cuda.is_available():
		weight = weight.cuda()
	'''
	if args.classify:
		criterion = nn.CrossEntropyLoss(weight=None)
	else:
		criterion = nn.MSELoss()
	steps = 0
	losslist = []
	accuracy = []
	F1_score_list = []
	cor_list = []

	args.save_dir = args.save_dir + args.nn_kind + '_' + str(args.hidden_size) + '_' + str(args.lr) + '_'
	for epoch in range(1, args.epochs+1):
		model.train()
		epochloss = 0

		for feature, label, x_length in train_iter:

			feature = Variable(feature.long())
			if args.classify:
				label = Variable(label.long())
			else:
				label = Variable(label.float())

			x_length = Variable(x_length.long())
			if torch.cuda.is_available():
				feature, label, x_length = feature.cuda(), label.cuda(), x_length.cuda()

			optimizer.zero_grad()
			if args.nn_kind == 'cnn':
				prob = model(feature)
			else:
				prob = model(feature, x_length)
			if not args.classify:
				prob = F.softmax(prob, dim=1)
			loss = criterion(prob, label.squeeze())
			epochloss = loss.item()
			loss.backward()
			optimizer.step()
			steps += 1
			if steps % args.logger_interval == 0:
				print('epoch{}: steps[{}] - loss: {:.6f}'.format(epoch, steps, loss.item()))

		model.eval()
		num_rights = 0
		num_all = 0
		pre_list = []
		label_true = []
		cor_temp_list = []
		print('testing...')
		for feature, label, x_length in tqdm.tqdm(test_iter):
			feature = Variable(feature.long())

			if torch.cuda.is_available():
				feature = feature.cuda()
			if args.nn_kind == 'cnn':
				predict = model(feature).cpu().detach().numpy()[0]
			else:
				predict = model(feature, x_length).cpu().detach().numpy()[0]
			score = max(predict)
			label_pre = np.where(predict == score)[0][0]
			if not args.classify:
				cor_temp_list.append(np.corrcoef(label, predict)[0][1])
				label = np.argmax(label)
			else:
				label = label.item()
			label_true.append(label)
			if label_pre == label:
				num_rights += 1
			num_all += 1
			pre_list.append(label_pre)

		sum = 0
		for item in cor_temp_list:
			sum += item
		print('acc:',num_rights / num_all)
		F1_score = f1_score(label_true, pre_list, labels=[0, 1, 2, 3, 4, 5, 6, 7], average='macro')
		accuracy.append(num_rights/num_all)
		F1_score_list.append(F1_score)
		cor_list.append(sum/num_all)
		print('F1:', F1_score)
		print('cor:', sum/num_all)
		losslist.append(epochloss)
		save(model, args.save_dir, epoch)

	with open('./data/loss_list_'+args.nn_kind+str(args.hidden_size)+str(args.lr)+'.pkl', 'wb') as f:
		pickle.dump(losslist, f)

	plt.figure()
	plt.plot(range(1, args.epochs+1), losslist, label='loss')
	plt.savefig('./data/loss_'+args.nn_kind+str(args.hidden_size)+str(args.lr)+'.png')
	plt.legend()
	plt.figure()
	plt.plot(range(1, args.epochs+1), accuracy, label='accuracy')
	plt.plot(range(1, args.epochs + 1), F1_score_list, label='accuracy')
	plt.plot(range(1, args.epochs + 1), cor_list, label='accuracy')
	plt.savefig('./data/accuracy_' + args.nn_kind + str(args.hidden_size) + str(args.lr) + '.png')
	print('highest_accuracy:', max(accuracy))


def test(model, args, test_iter, path):
	try:
		print('loading the neural network')
		model.load_state_dict(torch.load(path))
	except:
		print('No such file')
	model.eval()
	print(model)
	num_rights = 0
	num_all = 0
	for feature, label in tqdm.tqdm(test_iter):
		feature = Variable(feature.long())
		predict = model(feature).cpu().detach().numpy()[0]
		score = max(predict)
		label_pre = np.where(predict == score)[0][0]
		if label_pre == label.item():
			num_rights += 1
		num_all += 1
	print(num_rights)
	print(num_rights/num_all)


def save(model, save_dir, epoch):
	if not os.path.isdir(save_dir):
		os.makedirs(save_dir)
	save_path = '{}epoch{}.pt'.format(save_dir, epoch)
	torch.save(model.state_dict(), save_path)




