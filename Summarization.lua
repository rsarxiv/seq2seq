require 'torch'
require 'nn'
require 'rnn'
require 'hdf5'
require 'cltorch'
require 'clnn'

Summarization = {}

torch.include('Summarization', 'Seq2Seq.lua')
torch.include('Summarization', 'Dataset.lua')

return Summarization