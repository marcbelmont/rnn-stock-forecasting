require 'torch'
require 'nn'
require 'nngraph'
require 'optim'
require 'lfs'
require "gnuplot"

require 'util.OneHot'
require 'util.misc'

local csv2tensor = require 'csv2tensor'
local LSTM = require 'models.LSTM'
local model_utils = require 'util.model_utils'

------------------
-- Command line --
------------------

function init_cmd()
    cmd = torch.CmdLine()

    -- model params
    cmd:option('-rnn_size', 128, 'size of LSTM internal state')
    cmd:option('-num_layers', 2, 'number of layers in the LSTM')

    -- optimization
    cmd:option('-learning_rate', 2e-3, 'learning rate')
    cmd:option('-learning_rate_decay', 0.97, 'learning rate decay')
    cmd:option('-learning_rate_decay_after', 10, 'in number of epochs, when to start decaying the learning rate')
    cmd:option('-decay_rate', 0.95, 'decay rate for rmsprop')
    cmd:option('-dropout', 0, 'dropout for regularization, used after each RNN hidden layer. 0 = no dropout')
    cmd:option('-seq_length', 50, 'number of timesteps to unroll for')
    cmd:option('-batch_size', 50, 'number of sequences to train on in parallel')
    cmd:option('-grad_clip', 5, 'clip gradients at this value')
    cmd:option('-max_epochs', 10, 'number of times with go through the dataset')

    -- bookkeeping
    cmd:option('-seed', 123, 'torch manual random number generator seed')
    cmd:option("-checkpoint_dir", "/tmp/torch/checkpoints/", "where we save models")
    cmd:option("-training", "/tmp/torch/training_data.txt", "training")
    cmd:option("-validation", "/tmp/torch/validation.txt", "validation")
    cmd:option("-name", "NA", "Name of the experiment")
    return cmd:parse(arg)
end

-------------------
-- Training data --
-------------------

function load_data(opt, data_set)
    local data = csv2tensor.load(data_set)
    local ydata = data[{{}, 1}]:clone()
    ydata:sub(1,-2):copy(data[{{},1}]:sub(2,-1))
    ydata[-1] = data[1][1]
    -- reformat data
    data = input_val(data)

    -- reformat ydata
    local num_types = data:max()
    ydata:apply(output_val)
    local lines = math.floor(data:size(1) / opt.seq_length / opt.batch_size) * opt.batch_size
    if lines == 0 then
        assert("Not enough data!")
    end
    local training_x = data:resize(lines, opt.seq_length, 2)
    local training_y = ydata:resize(lines, opt.seq_length)
    -- Each batch must be of dimension (batch_size, seq_length)
    return training_x:split(opt.batch_size, 1), training_y:split(opt.batch_size, 1), num_types
end

local opt = init_cmd()
local batch_index = 0
local training_x, training_y, num_types = load_data(opt, opt.training)

------------------
-- Define model --
------------------

function init_model(opt, input_size)
    torch.manualSeed(opt.seed)
    local protos = {}
    protos.rnn = LSTM.lstm(input_size, opt.rnn_size, opt.num_layers, opt.dropout, output_val(1000))
    protos.criterion = nn.ClassNLLCriterion()

    -- the initial state of the cell/hidden states
    local init_state = {}
    for L = 1, opt.num_layers do
        local h_init = torch.zeros(opt.batch_size, opt.rnn_size)
        table.insert(init_state, h_init:clone())
        table.insert(init_state, h_init:clone())
    end

    -- put the above things into one flattened parameters tensor
    local params, grad_params = model_utils.combine_all_parameters(protos.rnn)

    -- initialization
    if do_random_init then
        params:uniform(-0.08, 0.08) -- small numbers uniform
    end

    -- make a bunch of clones after flattening, as that reallocates memory
    local clones = {}
    for name, proto in pairs(protos) do
        clones[name] = model_utils.clone_many_times(proto, opt.seq_length)
    end

    return params, grad_params, init_state, clones, protos
end

local params, grad_params, init_state, clones, protos = init_model(opt, num_types)
local init_state_global = clone_list(init_state)

-------------------
-- Evaluate loss --
-------------------

function feval(x)
    if x ~= params then
        params:copy(x)
    end
    grad_params:zero()

    ---------------
    -- Get batch --
    ---------------
    batch_index = batch_index + 1
    if batch_index > #training_x then
        batch_index = 1
    end
    local x = training_x[batch_index]
    local y = training_y[batch_index]

    -------------
    -- Forward --
    -------------

    local rnn_state = {[0] = init_state_global}
    local predictions = {} -- softmax outputs
    local loss = 0
    for t=1, opt.seq_length do
        clones.rnn[t]:training() -- make sure we are in correct mode (this is cheap, sets flag)
        local lst = clones.rnn[t]:forward{x[{{}, t}], unpack(rnn_state[t-1])}
        rnn_state[t] = {}
        for i=1, #init_state do
            table.insert(rnn_state[t], lst[i]) -- extract the state, without the output
        end
        predictions[t] = lst[#lst] -- last element is the prediction
        loss = loss + clones.criterion[t]:forward(predictions[t], y[{{}, t}])
    end
    loss = loss / opt.seq_length

    --------------
    -- Backward --
    --------------

    -- initialize gradient at time t to be zeros (there's no influence from future)
    local drnn_state = {[opt.seq_length] = clone_list(init_state, true)} -- true also zeros the clones
    for t=opt.seq_length, 1, -1 do
        -- backprop through loss, and softmax/linear
        local doutput_t = clones.criterion[t]:backward(predictions[t], y[{{}, t}])
        table.insert(drnn_state[t], doutput_t)
        local dlst = clones.rnn[t]:backward({x[{{}, t}], unpack(rnn_state[t-1])}, drnn_state[t])
        drnn_state[t-1] = {}
        for k,v in pairs(dlst) do
            if k > 1 then -- k == 1 is gradient on x, which we dont need
                -- note we do k-1 because first item is dembeddings, and then follow the
                -- derivatives of the state, starting at index 2. I know...
                drnn_state[t-1][k-1] = v
            end
        end
    end

    ----------
    -- Misc --
    ----------

    -- transfer final state to initial state (BPTT)
    init_state_global = rnn_state[#rnn_state] -- NOTE: I don't think this needs to be a clone, right?
    -- clip gradient element-wise
    grad_params:clamp(-opt.grad_clip, opt.grad_clip)
    return loss, grad_params
end

------------------
-- Optimization --
------------------

function optimize(opt, params, iterations)
    local optim_state = {learningRate = opt.learning_rate, alpha = opt.decay_rate}
    local print_time = 0
    local epoch = 1
    local timer_global = torch.Timer()
    local t_losses = {}
    local v_losses = {}
    local counter = 1
    local best = 1000
    local best_epoch
    for i = 1, iterations do
        local timer = torch.Timer()
        local _, loss = optim.rmsprop(feval, params, optim_state)
        -- Print stats every n seconds
        if timer_global:time().real - print_time > 2  then
            -- print(string.format("             Loss %6.12f, time/batch = %.2fs, ",
                                -- loss[1],
                                -- timer:time().real))
            print_time = timer_global:time().real
        end

        -- Calculate loss on validation set every epoch
        t_losses[counter] = loss[1]
        if batch_index == 1 then
            v_losses[counter] = eval_loss(opt, clones, init_state)

            -- Early stopping
            if v_losses[counter] < best then
                best = v_losses[counter]
                best_epoch = epoch
                -- Save model
                save_model(opt, protos, epoch)
            end
            if epoch > 2 * best_epoch  and epoch > 5 then
                break
            end

            epoch = epoch + 1
            -- Decay learning rate every n epochs
            if epoch % opt.learning_rate_decay_after == 0 then
                optim_state.learningRate = optim_state.learningRate * opt.learning_rate_decay
            end
        else
            v_losses[counter] = v_losses[counter - 1]
        end
        counter = counter + 1
    end
    os.execute(string.format("cp %smodel-%s.t7 %s/model-best.t7", opt.checkpoint_dir, best_epoch, opt.checkpoint_dir))
    return {timer_global:time().real, torch.Tensor(t_losses), torch.Tensor(v_losses)}
end

--------------
-- Evaluate --
--------------

-- Evaluate loss on validation set
local val_x, val_y, _ = load_data(opt, opt.validation)
function eval_loss(opt, clones, init_state)
    local loss = 0
    local rnn_state = {[0] = init_state}
    local n = #val_x

    local hack = 1 -- TODO last element of last batch causes mutations

    -- iterate over batches
    for i = 1, n do
        local x = val_x[i]
        local y = val_y[i]

        -- forward pass
        for t = 1, opt.seq_length - hack do
            clones.rnn[t]:evaluate() -- for dropout proper functioning
            local lst = clones.rnn[t]:forward{x[{{}, t}], unpack(rnn_state[t - 1])}
            rnn_state[t] = {}
            for i=1, #init_state do
                table.insert(rnn_state[t], lst[i])
            end
            prediction = lst[#lst]
            loss = loss + clones.criterion[t]:forward(prediction, y[{{}, t}])
        end

        -- carry over lstm state
        rnn_state[0] = rnn_state[#rnn_state]
    end

    loss = loss / (opt.seq_length - hack) / n
    return loss
end



-----------
-- Start --
-----------

log_result(opt, unpack(optimize(
    opt,
    params,
    opt.max_epochs * #training_x)))
