require 'torch'
require 'cutorch'
require 'paths'
require 'xlua'
require 'optim'

local opts = paths.dofile('opts.lua')
print(arg)
opt = opts.parse(arg)
print(opt)

torch.setdefaulttensortype('torch.FloatTensor')

cutorch.setDevice(opt.GPU) -- by default, use GPU 1
torch.manualSeed(opt.manualSeed)

print('Saving everything to: ' .. opt.save)
os.execute('mkdir -p ' .. opt.save)

paths.dofile('data.lua')

print "done"