require 'torch'
require 'nn'
require 'rnn'
require 'cltorch'
require 'clnn'
require 'hdf5'

Summarization = {}

torch.include('Summarization', 'Seq2Seq.lua')
torch.include('Summarization', 'Dataset.lua')

return Summarization