require 'torch'
require 'image'
require 'fbcunn'

function file_exists(file)
  local f = io.open(file, "rb")
  if f then f:close() end
  return f ~= nil
end

function lines_from(file, size, resize)
  if not file_exists(file) then return {} end
  lines = {}
  print (size, resize)
  db = torch.Tensor(size, 3, resize, resize):double()
  local i = 1
  for line in io.lines(file) do 
  	img = image.scale(image.load(line), resize, resize)
  	db[i] = img 
    lines[#lines + 1] = line
    i = i + 1
  end
  return db
end



function generate_db(file, size, resize, name)
	local db_train = lines_from(file, size, resize)
	channels = {'r','g','b'}
	mean = {}
	std = {}

	for i,channel in ipairs(channels) do
		mean[i] = db_train[{ {},i,{},{} }]:mean()
		std[i] = db_train[{ {},i,{},{} }]:std()
		db_train[i]:add(-mean[i])
		db_train[i]:div(std[i])
	end
	neighborhood = image.gaussian1D(7)
	normalization = nn.SpatialContrastiveNormalization(1, neighborhood):double()

	for i,channel in ipairs(channels) do
		 trainMean = db_train[{ {},i }]:mean()
		 trainStd = db_train[{ {},i }]:std()

		 print('training data, '..channel..'-channel, mean: ' .. trainMean)
		print('training data, '..channel..'-channel, standard deviation: ' .. trainStd)

	end
	if name == nil then
		name = "database.t7"
	end
	torch.save(name, db_train)
	print("Saved to ", name)
end

generate_db(arg[1], tonumber(arg[2]), tonumber(arg[3]), arg[4])
