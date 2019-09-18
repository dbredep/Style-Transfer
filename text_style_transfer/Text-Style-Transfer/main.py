import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
def makeVocab(file, min_freq=1, special=('<pad>', '<bos>', '<eos>', '<unk>')):
    f = open(file, 'r')
    vocab = {}
    count = len(special)
    for i in range(count):
        vocab[special[i]] = i

    freq_dict = {}
    for l in f:
        ls = l.rstrip().split()
        for item in ls[1:]:
            if item in freq_dict.keys():
                freq_dict[item] += 1
            else:
                freq_dict[item] = 1

    sorted_dict = sorted(freq_dict.items(), key=lambda item: item[1], reverse=True)

    for item in sorted_dict:
        word, freq = item
        if freq < min_freq:
            break
        vocab[word] = count
        count += 1

    print('Vocabulary size is {}.'.format(count))

    id2word = {value:key for key, value in vocab.items()}

    return vocab, id2word

def makeDataset(file, vocab, max_len=15):
    f = open(file, 'r')
    labels = {'democratic': 0, 'republican': 1}
    target = []

    lines = f.readlines()
    N = len(lines)
    print('File contains {} lines.'.format(N))

    sen_lens = []

    dataset = np.zeros((N, max_len))
    for i in range(N):
        ls = lines[i].rstrip().split()
        label, sen = ls[0], ls[1:]
        sen_lens.append(len(sen))
        sen_len = min(max_len - 2, len(sen))
        dataset[i][0] = vocab['<bos>']
        for j in range(sen_len):
            if sen[j] not in vocab.keys():
                dataset[i][j+1] = vocab['<unk>']
            else:
                dataset[i][j+1] = vocab[sen[j]]
        dataset[i][sen_len+1] = vocab['<eos>']
        target.append(labels[label])

    target = np.array(target)

    print('Mean sentence length: ', np.array(sen_lens).mean())
    print('Padding percentage: ', np.where(dataset==0)[0].shape[0] / (N * max_len))

    return dataset, target


def check_dataset(dataset, id2word, batch_first=False):
    if batch_first:
        pass
    else:
        L, N = dataset.shape
        for i in range(N):
            for j in range(L):
                print(id2word[dataset[i][j]], ' ', end='')
            print('')




vocab, id2word = makeVocab('classtrain.txt')
dataset, target = makeDataset('classtrain.txt', vocab)

np.savez('data', data=dataset, target=target)

#check_dataset(dataset, id2word)

class LSTMModel(nn.Module):
    def __init__(self, batch_size):
        super(LSTMModel, self).__init__()

        self.batch_size = batch_size

        self.emb = nn.Embedding(num_embeddings=len(vocab), embedding_dim=100, padding_idx=0)
        self.lstm = nn.LSTM(input_size=100, hidden_size=100,
                            num_layers=3, bidirectional=True)

        self.initialize(self.batch_size)

        self.linear = nn.Linear(200, 2)

    def initialize(self, batch_size):
        self.hc0 = (torch.rand(3 * 2, batch_size, 100).cuda(),
                   torch.rand(3 * 2, batch_size, 100).cuda())

    def forward(self, x):
        x, hc = self.lstm(self.emb(torch.t(x)), self.hc0)
        x = x[-1]
        return self.linear(x)



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

opt = {'lr': 1e-3, 'batch_size': 32}

# Prepare dataloader
data = np.load('data.npz')['data']
target = np.load('data.npz')['target']


trainset = torch.utils.data.TensorDataset(torch.from_numpy(data[:72000]).long(), torch.from_numpy(target[:72000]).long())
valset = torch.utils.data.TensorDataset(torch.from_numpy(data[72000:]).long(), torch.from_numpy(target[72000:]).long())

trainloader = torch.utils.data.DataLoader(dataset = trainset, batch_size = opt['batch_size'], shuffle = True, num_workers = 2)
validloader = torch.utils.data.DataLoader(dataset = valset, batch_size = opt['batch_size'], shuffle = False, num_workers = 2)

model = LSTMModel(opt['batch_size'])
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)

model.to(device)

optimizer = optim.Adam(model.parameters(), lr=opt['lr'],
                      weight_decay=5e-7)
criterion = torch.nn.CrossEntropyLoss()

num_epoch = 50

interval = 100

for epoch in range(1, num_epoch):
    print('| - at epoch {}'.format(epoch))
    model.train()

    losses = []
    acces = []

    for i, (data, target) in enumerate(trainloader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)

        pred = torch.argmax(output, dim=1)
        acc = np.where((pred - target).cpu().data.numpy() == 0)[0].shape[0] / output.size(0)

        loss = criterion(output, target)

        #losses.append(loss.data)
        losses.append(loss.item())
        acces.append(acc)

        if i % interval == 0:
            print('step {}, loss {}, accuracy {}'.format(i, loss.data, acc))

        loss.backward()
        optimizer.step()

    print('epoch {}, loss {}, accuracy {}'.format(epoch, np.array(losses).mean(), np.array(acces).mean()))



