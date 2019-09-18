import torch
import torch.nn as nn

class FeatureSpaceClassifier(nn.Module):
    def __init__(self, hid_size, num_classes):
        super(FeatureSpaceClassifier, self).__init__()
        self.linear = nn.Linear(hid_size, num_classes)

    def forward(self, x):
        return self.linear(x)

class TokenSpaceClassifier(nn.Module):
    def __init__(self, config, inputs):
        super(TokenSpaceClassifier, self).__init__()

        self.vocab_size = inputs.size()
        self.embedding_size = config.embedding_size

        self.num_filters = config.num_filters
        self.pooling_window_size = config.sequence_length - config.filter_size + 1
        self.filter_size = config.filter_size
        self.strides = (1, 1)
        self.num_classes = config.num_classes

        self.dropout = config.dropout

        self.embedding = nn.Embedding(self.vocab_size,
                                     self.embedding_size,
                                      padding_idx=0)
        self.conv1 = nn.Conv2d(in_channels=self.embedding_size,
                               out_channels=self.num_filters,
                               kernel_size=(self.filter_size, 1),
                               stride=self.strides,
                               bias=True)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool3d(kernel_size=(1, self.pooling_window_size, 1, 1), stride=(1, 1, 1, 1))
        self.dropout = nn.Dropout(self.dropout)
        self.linear = nn.Linear(self.num_filters, self.num_classes - 1)

    def load_pretrained_vectors(self, config):
        if config.pre_word_vecs_enc is not None:
            pretrained = torch.load(config.pre_word_vecs_enc)
            self.embedding.weight.data.copy_(pretrained)

    def forward(self, input):
        ## src size is seq_size x batch_size x vocab_size. Most cases (50 x 64 x v)
        ## matrix multiply instead of lookup
        emb = torch.mm(input.view(-1, input.size(2)), self.word_lut.weight)
        emb = emb.view(-1, input.size(1), self.word_vec_size)
        emb = emb.transpose(0, 1)
        emb = emb.transpose(1, 2)
        emb = emb.unsqueeze(-1)
        h_conv = self.conv1(emb)
        h_relu = self.relu1(h_conv)
        h_max = self.maxpool1(h_relu)
        h_flat = h_max.view(-1, self.num_filters)
        h_drop = self.dropout(h_flat)
        out = self.linear(h_drop)
        return out
