--[[

Example of "coupled" separate encoder and decoder networks, e.g. for sequence-to-sequence networks.

]]--

require 'Summarization'
require 'xlua'

cmd = torch.CmdLine()
cmd:text('Options:')
cmd:option('--learningRate', 0.1, 'learning Rate')
cmd:option('--momentum', 0.9, 'momentum')
cmd:option('--gpuid',-1,'which gpu to use. -1 = use CPU')
cmd:option('--opencl',1,'use OpenCL (instead of CUDA)')
cmd:option('--seed',123,'seed')
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
-- initialize cunn/cutorch for training on the GPU and fall back to CPU gracefully
if opt.gpuid >= 0 and opt.opencl == 0 then
    local ok, cunn = pcall(require, 'cunn')
    local ok2, cutorch = pcall(require, 'cutorch')
    if not ok then print('package cunn not found!') end
    if not ok2 then print('package cutorch not found!') end
    if ok and ok2 then
        print('using CUDA on GPU ' .. opt.gpuid .. '...')
        cutorch.setDevice(opt.gpuid + 1) -- note +1 to make it 0 indexed! sigh lua
        cutorch.manualSeed(opt.seed)
    else
        print('If cutorch and cunn are installed, your CUDA toolkit may be improperly configured.')
        print('Check your CUDA toolkit installation, rebuild cutorch and cunn, and try again.')
        print('Falling back on CPU mode')
        opt.gpuid = -1 -- overwrite user setting
    end
end

-- initialize clnn/cltorch for training on the GPU and fall back to CPU gracefully
if opt.gpuid >= 0 and opt.opencl == 1 then
    local ok, cunn = pcall(require, 'clnn')
    local ok2, cutorch = pcall(require, 'cltorch')
    if not ok then print('package clnn not found!') end
    if not ok2 then print('package cltorch not found!') end
    if ok and ok2 then
        print('using OpenCL on GPU ' .. opt.gpuid .. '...')
        cltorch.setDevice(opt.gpuid + 1) -- note +1 to make it 0 indexed! sigh lua
        torch.manualSeed(opt.seed)
    else
        print('If cltorch and clnn are installed, your OpenCL driver may be improperly configured.')
        print('Check your OpenCL driver installation, check output of clinfo command, and try again.')
        print('Falling back on CPU mode')
        opt.gpuid = -1 -- overwrite user setting
    end
end

-- Training --
print("Training...")
for epoch=1,opt.maxEpochs do
   print("Epoch " .. epoch)
   local errors = torch.Tensor(dataset.batchNum):fill(0)
   for batchId=0,dataset.batchNum-1 do
      encInSeq,decInSeq,decOutSeq = dataset:nextBatch(batchId)
      if opt.gpuid >= 0 and opt.opencl == 0 then
         encInSeq = encInSeq:cuda()
         decInSeq = decInSeq:cuda()
         decOutSeq = decOutSeq:cuda()
      elseif opt.gpuid >= 0 and opt.opencl == 1 then
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
