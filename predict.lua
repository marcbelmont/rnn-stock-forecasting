require 'torch'
require 'nn'
require 'nngraph'
require "gnuplot"

require 'util.OneHot'
require 'util.misc'
local csv2tensor = require 'csv2tensor'

------------------
-- Command line --
------------------

function init_cmd()
    cmd = torch.CmdLine()
    cmd:option('-model', "", 'model to use')
    return cmd:parse(arg)
end

local opt = init_cmd()

function prediction_score(opt, protos)
    protos.rnn:evaluate()

    -- Initial states
    local current_state = {}
    for L = 1, opt.num_layers do
        -- c and h for all layers
        local h_init = torch.zeros(1, opt.rnn_size)
        table.insert(current_state, h_init:clone())
        table.insert(current_state, h_init:clone())
    end
    local state_size = #current_state
    local data_in_raw = csv2tensor.load(opt.validation)
    local data_in = input_val(data_in_raw:clone())
    local data_out = data_in_raw[{{}, 1}]:clone()
    data_out:apply(output_val)

    -- Predict
    local prediction, input
    local result = torch.Tensor(data_in:size(1) - opt.seq_length, 4)
    local result_index = 1
    local predictions = torch.Tensor(data_in:size(1) - opt.seq_length)
    data_in:resize(data_in:size(1), 1, 2)
    for i = 1, data_in:size(1) do
        input = data_in[i]
        -- After seeding the pattern n times check predictions
        if i > opt.seq_length then
            -- Get probability of the correct value
            local probs = torch.exp(prediction:squeeze()) -- RNN predictions
            result[result_index][1] = probs[data_out[i]] -- Correct prediction
            result[result_index][2] = data_out[i] -- Correct group
            result[result_index][3] = probs:max() -- RNN first choice
            result[result_index][4] = data_in_raw[i][1] -- RNN first choice
            result_index = result_index + 1
        end

        -- Get next prediction
        local lst = protos.rnn:forward{input, unpack(current_state)}
        current_state = {}
        for i = 1, state_size do
            table.insert(current_state, lst[i])
        end
        prediction = lst[#lst] -- last element holds the log probabilities
    end
    return result
end

function display(result, opt)
    local y, i = torch.sort(result, 1, true)
    local best_proba = y[10][1]

    local num_correct = 0
    local num_total = 0
    local top_true = {}
    local top_false = {}
    for i = 1, result:size(1) do
        for j = 1, output_val(1000) do
            if result[i][2] == j then
                if result[i][3] > best_proba then
                    if result[i][1] == result[i][3] then
                        num_correct = num_correct + 1
                        table.insert(top_true, result[i][4])
                    else
                        table.insert(top_false, result[i][4])
                    end
                    num_total = num_total + 1
                end
            end
        end
    end

    printf("Top %s predictions %.2f%% true (%.2f%%), Mean %.2f%% [%.2f%%, %.2f%%]",
           num_total,
           num_correct / num_total * 100,
           best_proba * 100,
           (torch.Tensor(top_false):sum() + torch.Tensor(top_true):sum()) / num_total,
           torch.Tensor(top_false):mean(),
           torch.Tensor(top_true):mean())

    -- histogram
    chart(result[{{}, 1}], "Correct predictions", "hist")
    chart(result[{{}, 3}], "RNN top predictions", "hist")
end

local protos, model_opt = load_model(opt.model)
display(prediction_score(model_opt, protos), model_opt)
