--[[

Example of "coupled" separate encoder and decoder networks, e.g. for sequence-to-sequence networks.

]]--

require 'Summarization'
require 'xlua'

cmd = torch.CmdLine()
cmd:text('Options:')
cmd:option('--learningRate', 0.1, 'learning Rate')
cmd:option('--momentum', 0.9, 'momentum')
cmd:option('--cuda', false, 'use CUDA')
cmd:option('--opencl', false, 'use opencl')
cmd:option('--hiddenSize', 100, 'number of hidden units in LSTM')
cmd:option('--maxEpochs', 10, 'max Epochs')
cmd:option('--batchSize',20, 'batchSize')
cmd:option('--lrfactor',0.1,'learningRate change factor')
cmd:option('--datafile','seq2seq.hdf5','source,target data from python')
cmd:option('--vocabfile','vocab.dict','vocab dict file from python')

cmd:text()
opt = cmd:parse(arg)
local minMeanError = nil

-- Dataset --
print("Loading Dataset...")
dataset = Summarization.Dataset(opt)
vocabSize = dataset.vocab.size

-- Model --
print("Building Model...")
model = Summarization.Seq2Seq(vocabSize,opt.hiddenSize)
model.criterion = nn.SequencerCriterion(nn.ClassNLLCriterion())
model.learningRate = opt.learningRate
model.momentum = opt.momentum

-- Enabled CUDA and OpenCL
if opt.cuda then
  require 'cutorch'
  require 'cunn'
  model:cuda()
elseif opt.opencl then
  require 'cltorch'
  require 'clnn'
  model:cl()
end

-- Training --
print("Training...")
for epoch=1,opt.maxEpochs do
   print("Epoch " .. epoch)
   local errors = torch.Tensor(dataset.batchNum):fill(0)
   for batchId=0,dataset.batchNum-1 do
      encInSeq,decInSeq,decOutSeq = dataset:nextBatch(batchId)
      if opt.cuda then
         encInSeq = encInSeq:cuda()
         decInSeq = decInSeq:cuda()
         decOutSeq = decOutSeq:cuda()
      elseif opt.opencl then
         encInSeq = encInSeq:cl()
         decInSeq = decInSeq:cl()
         decOutSeq = decOutSeq:cl()
      end
      err = model:train(encInSeq,decInSeq,decOutSeq)
      errors[batchId+1] = err
      xlua.progress(batchId+1,dataset.batchNum)
      print("err " .. err)
   end
   print("Epoch ".. epoch .." mean error " .. errors:mean())
     -- Save the model if it improved.
   if minMeanError == nil or errors:mean() < minMeanError then
      print("\n(Saving model ...)")
      torch.save("model.t7", model)
      minMeanError = errors:mean()
   end
   model.learningRate = model.learningRate * opt.lrfactor
end
