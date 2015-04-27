require 'torch'
require 'image'
require 'fbcunn'
require 'xlua'    -- xlua provides useful tools, like progress bars
require 'optim'  

function co_run(queue)
  while true do
  	tmp = queue:pop()
    if tmp ~= nil then
      tmp()
    end
		coroutine.yield()
  end
end

function newQueue () --queue 
	local _self = {data = {}, first = nil, size = 0}

	local pop = function(self)
    if _self.size == 0 then
      return nil
    else
      _self.size = _self.size - 1
      local ret = _self.data[1]
      table.remove(_self.data, 1)
      return ret
    end
  end

	local push = function(self, f)
		table.insert(_self.data, f)
		_self.size = _self.size + 1
	end

	return {pop = pop, push = push, size = function() return _self.size end}
end

function split(str, delim)
    local res = {}
    local pattern = string.format("([^%s]+)%s", delim, delim)
    for line in str:gmatch(pattern) do
        table.insert(res, line)
    end
    return res
end

function file_exists(file)
  local f = io.open(file, "rb")
  if f then f:close() end
  return f ~= nil
end

function calculate_mean_std(data)
	local channels = {'r','g','b'}
	local rep = {mean = {}, std = {}}

	for i,channel in ipairs(channels) do
		rep.mean[i] = data[{ {},i,{},{} }]:mean()
		rep.std[i] =  data[{ {},i,{},{} }]:std()
	end
	return rep
end

function lines_from(file, size, resize)
  if not file_exists(file) then return {} end
  local batchSize = 1000 --arbitrary 
  --local db = torch.Tensor(batchSize, 3, resize, resize):double()
  local part = {
  db = torch.Tensor(batchSize, 3, resize, resize):double(),
  labels = torch.Tensor(batchSize)
  }
  --local labels = torch.Tensor(size)
  local i = 1
  local count = 0
  mean = {0,0,0}
  std = {0,0,0}
  labels = {}
  queue = newQueue()
  pool = {}
  for i=1,150 do
    table.insert(pool, coroutine.create(co_run))
  end

	for line in io.lines(file) do
      local tmp = i
	  	queue:push(function ()
          print(tmp, line)
			  	local img = image.scale(image.load(line), resize, resize)
			  	part.db[tmp - (batchSize * count)] = img
			  	local line = split(line, "/")
			  	if labels[line[#line]] == nil then 
			  		labels[line[#line]] = #labels
			  	end
			  	part.labels[tmp - (batchSize * count)] = labels[line[#line]]
		end)
	    i = i + 1
		if i % batchSize == 0 and i / batchSize > 0 then
			while queue:size() > 0 do
        local k = 0
				for j=1,#pool do
					if coroutine.status(pool[j]) == 'suspended' then
						coroutine.resume(pool[j], queue)
          elseif coroutine.status(pool[j]) == 'dead' then
           print(k, j)
          else
            print(coroutine.status(pool[j]))
					end
				end
			end

			local tmp = calculate_mean_std(part.db)
			for j=1,#mean do
				mean[j] = (mean[j] + tmp.mean[j])/2
				std[j] = (std[j] + tmp.std[j])/2
			end
      print(mean)
      print(std)
			torch.save("part" .. count .. '.t7', part)
			count = count + 1
			part.db:zero()
			part.labels:zero()
      print("Done part", count)
		end
	end

	local tmp = calculate_mean_std(part.db)
	for j=1,#mean do
		mean[j] = (mean[j] + tmp.mean[j])/2
		std[j] = (std[j] + tmp.std[j])/2
	end
	torch.save("part-" .. count .. '.t7', part)

  local trainData = {
    labels = labels, --hack
    size = function() return (#trainData.data)[1] end,
    batchSize = batchSize,
    part = count,
    prefix = "part-"
 	}
  return trainData
end

function generate_db(file, size, resize, name)
	local db_train = lines_from(file, size, resize)
	mean = {}
	std = {}

	for i,channel in ipairs(channels) do
		mean[i] = db_train.data[{ {},i,{},{} }]:mean()
		std[i] = db_train.data[{ {},i,{},{} }]:std()
		db_train.data[i]:add(-mean[i])
		db_train.data[i]:div(std[i])
	end
	neighborhood = image.gaussian1D(7)
	normalization = nn.SpatialContrastiveNormalization(1, neighborhood):double()

	for i,channel in ipairs(channels) do
		 trainMean = db_train.data[{ {},i }]:mean()
		 trainStd = db_train.data[{ {},i }]:std()

		 print('training data, '..channel..'-channel, mean: ' .. trainMean)
		print('training data, '..channel..'-channel, standard deviation: ' .. trainStd)

	end
	if name == nil then
		name = "database.t7"
	end
	torch.save(name, db_train)
	print("Saved to ", name)
	return db_train
end

function getModel(nClasses)
  require 'fbcunn'
  features = nn.ModelParallel(2)
   local fb1 = nn.Sequential() -- branch 1
   fb1:add(nn.SpatialConvolutionMM(3,48,11,11,4,4,2,2))       -- 224 -> 55
   fb1:add(nn.ReLU())
   fb1:add(nn.SpatialMaxPooling(3,3,2,2))                   -- 55 ->  27
   fb1:add(nn.SpatialConvolutionMM(48,128,5,5,1,1,2,2))       --  27 -> 27
   fb1:add(nn.ReLU())
   fb1:add(nn.SpatialMaxPooling(3,3,2,2))                   --  27 ->  13
   fb1:add(nn.SpatialConvolutionMM(128,192,3,3,1,1,1,1))      --  13 ->  13
   fb1:add(nn.ReLU())
   fb1:add(nn.SpatialConvolutionMM(192,192,3,3,1,1,1,1))      --  13 ->  13
   fb1:add(nn.ReLU())
   fb1:add(nn.SpatialConvolutionMM(192,128,3,3,1,1,1,1))      --  13 ->  13
   fb1:add(nn.ReLU())
   fb1:add(nn.SpatialMaxPooling(3,3,2,2))                   -- 13 -> 6

   local fb2 = fb1:clone() -- branch 2
   for k,v in ipairs(fb2:findModules('nn.SpatialConvolutionMM')) do
      v:reset() -- reset branch 2's weights
   end

   features:add(fb1)
   features:add(fb2)

   -- 1.3. Create Classifier (fully connected layers)
   local classifier = nn.Sequential()
   classifier:add(nn.View(256*6*6))
   classifier:add(nn.Dropout(0.5))
   classifier:add(nn.Linear(256*6*6, 4096))
   classifier:add(nn.Threshold(0, 1e-6))
   classifier:add(nn.Dropout(0.5))
   classifier:add(nn.Linear(4096, 4096))
   classifier:add(nn.Threshold(0, 1e-6))
   classifier:add(nn.Linear(4096, nClasses))
   classifier:add(nn.LogSoftMax())

   -- 1.4. Combine 1.1 and 1.3 to produce final model
   local model = nn.Sequential():add(features):add(classifier)
   return model
end


function train(model, parameters, gradParameters, optimState, optimMethod, criterion, trainData)

   -- epoch tracker
   epoch = epoch or 1

   -- local vars
   local time = sys.clock()

   -- set model to training mode (for modules that differ in training and testing, like Dropout)
   model:training()

   -- shuffle at each epoch
   --shuffle = torch.randperm(trsize)
   batchSize = 50
   -- do one epoch
   print('==> doing epoch on training data:')
   print("==> online epoch # " .. epoch .. ' [batchSize = ' .. batchSize .. ']')
    for t = 1,trainData:size(),batchSize do
      -- disp progress

      -- create mini batch
      local inputs = {}
      local targets = {}
      for i = t,math.min(t+batchSize-1,trainData:size()) do
         -- load new sample
         local input = trainData.data[i]
         local target = trainData.labels[i]
         input = input:cuda()
         table.insert(inputs, input)
         table.insert(targets, target)
      end

      -- create closure to evaluate f(X) and df/dX
      local feval = function(x)
                       -- get new parameters
                       if x ~= parameters then
                          parameters:copy(x)
                       end

                       -- reset gradients
                       gradParameters:zero()

                       -- f is the average of all criterions
                       local f = 0

                       -- evaluate function for complete mini batch
                       for i = 1,#inputs do
                          -- estimate f
                          local output = model:forward(inputs[i])
                          local err = criterion:forward(output, targets[i])
                          f = f + err

                          -- estimate df/dW
                          local df_do = criterion:backward(output, targets[i])
                          model:backward(inputs[i], df_do)

                          -- update confusion
                          confusion:add(output, targets[i])
                       end

                       -- normalize gradients and f(X)
                       gradParameters:div(#inputs)
                       f = f/#inputs

                       -- return f and df/dX
                       return f,gradParameters
                    end

      -- optimize on current mini-batch
--         optimMethod(feval, parameters, optimState)
   end

   -- time taken
   time = sys.clock() - time
   time = time / trainData:size()
   print("\n==> time to learn 1 sample = " .. (time*1000) .. 'ms')

   -- print confusion matrix
   print(confusion)

   -- save/log current net
   --local filename = paths.concat(opt.save, 'model.net')
   --os.execute('mkdir -p ' .. sys.dirname(filename))
   --print('==> saving model to '..filename)
   torch.save("model.t7", model)

   -- next epoch
   confusion:zero()
   epoch = epoch + 1
end

function run(dataset)
	classes = {'1','2','3','4','5','6','7','8','9','0'}
	confusion = optim.ConfusionMatrix(classes)
	criterion = nn.MultiMarginCriterion():cuda()
	model = getModel(#classes):cuda()
	parameters,gradParameters = model:getParameters()

	optimState = {
      learningRate = 1e-3,
      weightDecay = 0,
      momentum = 0,
      learningRateDecay = 1e-7
   }
   optimMethod = optim.sgd
	while true do
	   train(model, parameters, gradParameters, optimState, optimMethod, criterion, dataset)
	end
end


generate_db(arg[1], tonumber(arg[2]), tonumber(arg[3]), arg[4])
