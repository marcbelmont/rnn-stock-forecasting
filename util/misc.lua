--------------------
-- misc utilities --
--------------------

function clone_list(tensor_list, zero_too)
    -- utility function. todo: move away to some utils file?
    -- takes a list of tensors and returns a list of cloned tensors
    local out = {}
    for k, v in pairs(tensor_list) do
        out[k] = v:clone()
        if zero_too then out[k]:zero() end
    end
    return out
end

function print_predictions(prediction)
    local result = {}
    local index_max, max
    for i = 1, prediction:size(2) do
        local value = prediction[1][i]
        if max == nil or value > max then
            max = value
            index_max = i
        end
        table.insert(result, string.format("%2.1f", value))
    end
    print(table.concat(result, " "), index_max)
end

function count_occurences(text, s)
    local counter = 0
    for w in string.gmatch(s, text) do
        counter = counter + 1
    end
    return counter
end

function tensor_string(tensor)
    local result = ""
    for i = 1, tensor:size(1) do
        result = result .. tensor[i]
    end
    return result
end

function repeatability(sample, training)
    local occurences = count_occurences(tensor_string(training), sample)
    return 100 * occurences * training:size(1) / #sample
end

function string_tensor(str)
    local t = {}
    for x in string.gmatch(str, "%S+") do
        table.insert(t, tonumber(x))
    end
    return torch.Tensor(t)
end

function file_append(path, str)
    local file = io.open(path, "a")
    io.output(file)
    io.write(str)
    io.close(file)
end

function read_all(path)
    local file = io.open(path, "r")
    local x = file:read()
    io.close(file)
    return x
end

function save_model(opt, protos, epoch)
    local checkpoint = {}
    checkpoint.protos = protos
    checkpoint.opt = opt
    checkpoint.epoch = epoch

    -- save to file
    os.execute("mkdir -p " .. opt.checkpoint_dir)
    local path = opt.checkpoint_dir .. string.format("model-%s.t7", epoch)
    torch.save(path, checkpoint)
end

function load_model(path)
    local checkpoint = torch.load(path)
    torch.manualSeed(checkpoint.opt.seed)
    local protos = checkpoint.protos
    protos.rnn:evaluate() -- put in eval mode so that dropout works properly
    return protos, checkpoint.opt
end

function log_result(opt, time, t_losses, v_losses)
    local best, epoch = torch.min(v_losses, 1)
    print(string.format(
              "OK %s %.1fs. Best %s %s.",
              opt.name,
              time,
              best[1],
              epoch[1]))
    chart({{"Training", t_losses, '-'},
            {"Validation", v_losses, '-'}}, "Loss", "learning")
end

function printf(x, ...)
    print(string.format(x, unpack({...})))
end

function table_str(t)
    local result = {}
    for _, v in ipairs(t) do
        table.insert(result, string.format("%.3f", v))
    end
    return table.concat(result, ", ")
end

function tensor_str(t)
    return table_str(torch.totable(t:squeeze()))
end

counter = 0
function chart(data, title, type)
    local dir = "/tmp/torch/visualisation"
    os.execute("mkdir -p " .. dir)
    local file_name = string.format("%s/%s%s%s.png", dir, type, os.date("%s"), counter)
    gnuplot.pngfigure(file_name)
    gnuplot.raw("set terminal png medium size 800,300")
    gnuplot.title(title)
    if type == "hist" then
        gnuplot.hist(data)
    else
        gnuplot.plot(unpack(data))
    end

    gnuplot.plotflush()
    counter = counter + 1
end

---------------------------
-- Input / output layers --
---------------------------

function input_val(x)
    local limit = 5
    x[{{}, 1}] = x[{{}, 1}]:clamp(-limit, limit)

    local mean = x[{{}, 1}]:mean()
    local std = x[{{}, 1}]:std()
    x[{{}, 1}] = x[{{}, 1}]:add(-mean):div(std)

    local mean = x[{{}, 2}]:mean()
    local std = x[{{}, 2}]:std()
    x[{{}, 2}] = x[{{}, 2}]:add(-mean):div(std * 100)
    return x
end

function input_val2(x)
    local limit = 8
    local precision = 2
    x:clamp(-limit, limit):mul(precision):round()
    x:add(torch.Tensor{precision * limit}:round()[1] + 1)
    return x
end

-- outputs
function output_val(x)
    if x < 0 then
        return 1
    else
        return 2
    end
end

function output_val2(x)
    if x < 0 then
        return 1
    elseif x == 0 then
        return 2
    else
        return 3
    end
end

function output_val3(x)
    local limit = .5
    if x <= -limit then
        return 1
    elseif x > -limit and x < limit then
        return 2
    else
        return 3
    end
end
