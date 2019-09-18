import numpy as np
import torch
import torch.nn as nn


class LinearContentEncoder(nn.Module):
    def __init__(self, embedding_dim, hid_size):
        super(LinearContentEncoder, self).__init__()
        self.layers = [nn.Linear(embedding_dim, hid_size[0])]
        for i in range(1, len(hid_size)):
            self.layers.append(nn.Linear(hid_size[i - 1], hid_size[i]).cuda())

    def forward(self, x):
        # input x is of shape (N, L, E)
        x = torch.mean(x, dim=1).cuda()

        for layer in self.layers:
            x = layer(x)

        return x

class LSTMContentEncoder(nn.Module):
    def __init__(self, embedding_dim, hid_size, num_layers,
                 batch_first, dropout, bidirectional):
        super(LSTMContentEncoder, self).__init__()
        self.embedding_dim = embedding_dim
        self.hid_size = hid_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.dropout = dropout
        self.bidirectional = bidirectional
        if self.bidirectional:
            self.h0_size = self.num_layers * 2
            self.c0_size = self.num_layers * 2
        else:
            self.h0_size = self.num_layers
            self.c0_size = self.num_layers

        self.lstm = nn.LSTM(self.embedding_dim,
                             self.hid_size,
                             self.num_layers,
                             batch_first=self.batch_first,
                             dropout=self.dropout,
                             bidirectional=self.bidirectional)

    def forward(self, x):
        if self.batch_first:
            batch_size = x.shape[0]
        else:
            batch_size = x.shape[1]
        h0 = torch.randn(self.h0_size, batch_size, self.hid_size).cuda()
        c0 = torch.randn(self.c0_size, batch_size, self.hid_size).cuda()
        output, (h, c) = self.lstm(x, (h0, c0))
        return self.lstm(x)

class TransformerContentEncoder(nn.Module):
    def __init__(self, embedding_dim, num_heads, num_layers):
        super(TransformerContentEncoder, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_head = num_heads
        self.num_layers = num_layers
        self.encoder_layer = nn.TransformerEncoderLayer(self.embedding_dim, self.num_heads)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, self.num_layers)

    def forward(self, x):
        # input x is of shape (N, L, E) for batch_first mode, while (L, N, E) for non batch_first mode
        return self.transformer_encoder(x)

class StyleInvariantEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, config):
        super(StyleInvariantEncoder, self).__init__()

        assert vocab_size > 1, 'VocabSize should be greater than 1.'
        self.vocab_size = vocab_size

        assert embedding_dim > 1, 'EmbeddingDim should be greater than 1.'
        self.embedding_dim = embedding_dim

        self.embedding = nn.Embedding(num_embeddings=self.vocab_size,
                                      embedding_dim=self.embedding_dim)
        self.model_types = (
            'Linear',
            'LSTM',
            'Transformer'
        )
        assert 'model_type' in config.keys(), 'Configuration Missing: model_type.'
        assert config['model_type'] in self.model_types, 'ModelType not supported. Please choose' \
                                              ' from Linear, LSTM and Transformer.'
        model_type = config['model_type']

        self.config = config

        if model_type == 'Linear':
            assert 'hid_size' in config.keys(), "Configuration Missing: hid_size."
            self.hid_size = config['hid_size']
            assert isinstance(self.hid_size, tuple) or isinstance(self.hid_size, list), "Configuration Error:" \
                                                                              " hid_size should be tuple or list"
            assert len(self.hid_size) > 0, "The LinearModel should have at least one layer."
            self.model = LinearContentEncoder(self.embedding_dim, self.hid_size)

        elif model_type == 'LSTM':
            # LSTM configuration includes:
            # {'model_type', 'hid_size', 'num_layers', 'batch_first', 'dropout', 'bidirectional'}
            assert 'hid_size' in config.keys(), "Configuration Missing: hid_size."
            self.hid_size = config['hid_size']
            assert isinstance(self.hid_size, int), "Configuration Error: hid_size should be int."

            assert 'num_layers' in config.keys(), "Configuration Missing: num_layers."
            self.num_layers = config['num_layers']
            assert isinstance(self.num_layers, int), "Configuration Error: num_layers should be int."

            if 'batch_first' in config.keys():
                self.batch_first = config['batch_first']
                assert isinstance(self.batch_first, bool), "Configuration Error: batch_first should be boolean."
            else:
                # batch_first default is False.
                self.batch_first = False

            if 'dropout' in config.keys():
                self.dropout = config['dropout']
                assert isinstance(self.dropout, float) and 0 <= self.dropout < 1, \
                    "Configuration Error: dropout should be float within [0, 1)."
            else:
                # dropout default is 0.
                self.dropout = 0

            if 'bidirectional' in config.keys():
                self.bidirectional = config['bidirectional']
                assert isinstance(self.bidirectional, bool), "Configuration Error: bidirectional should be boolean."
            else:
                # bidirectional default is False.
                self.bidirectional = False

            self.model = LSTMContentEncoder(self.embedding_dim, self.hid_size, self.num_layers,
                 self.batch_first, self.dropout, self.bidirectional)

        elif model_type == 'Transformer':
            assert 'num_heads' in config.keys(), "Configuration Missing: num_heads."
            self.num_heads = config['num_heads']
            assert isinstance(self.num_heads, int), "Configuration Error: num_heads should be int."

            assert 'num_layers' in config.keys(), "Configuration Missing: num_layers."
            self.num_layers = config['num_layers']
            assert isinstance(self.num_layers, int), "Configuration Error: num_layers should be int."

            self.model = TransformerContentEncoder(self.embedding_dim, self.num_heads, self.num_layers)
        else:
            self.model = None

    def forward(self, x):
        x = self.embedding(x)
        return self.model(x)