--[[

Dataset Class

]]--

local Dataset = torch.class("Summarization.Dataset")

function Dataset:__init(opt)
	local dataset = hdf5.open(opt.datafile,"r")
	self.source = dataset:read("source"):all()
	self.target  = dataset:read("target"):all()
	self.batchNum = math.floor(self.source:size(1) / opt.batchSize)
	self.batchSize = opt.batchSize
	self.vocab = self:buildVocab(opt)
end

function Dataset:buildVocab(opt)
	local id2word = {}
	local word2id = {}
	local f = torch.DiskFile(opt.vocabfile, "r")
  	f:quiet()
  	local word =  f:readString("*l") -- read file by line
  	while word ~= '' do
      	id2word[#id2word+1] = word
      	word2id[word] = #id2word
      	word = f:readString("*l")
  	end
  	return {["id2word"]=id2word,["word2id"]=word2id,["size"]=#id2word}
end

function Dataset:nextBatch(batchId)
   local encInSeq = {}
   local decInSeq = {}
   local decOutSeq = {}
   for i = batchId * self.batchSize + 1, (batchId + 1) * self.batchSize do
      table.insert(encInSeq,torch.totable(self.source[i]))
      table.insert(decInSeq,torch.totable(self.target[i]:sub(1,-2)))
      table.insert(decOutSeq,torch.totable(self.target[i]:sub(2,-1)))
   end
   return torch.Tensor(encInSeq),torch.Tensor(decInSeq),torch.Tensor(decOutSeq)
end