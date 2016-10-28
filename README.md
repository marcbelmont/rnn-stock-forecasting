Experiment to model the price of a stock using an LSTM / stacked LSTMs.

### Requirements

You need torch to run this code. You will also need a bunch of packages.
```
$ luarocks install gnuplot
$ luarocks install nngraph
$ luarocks install csv2tensor
$ luarocks install optim
$ luarocks install nn
```

### How to run the code

Train the model:
`luajit train.lua -batch_size 16 -dropout 0.5 -learning_rate 0.02  -learning_rate_decay 0.97 -rnn_size 128 -num_layers 2 -seq_length 16 -max_epochs 40 -checkpoint_dir /tmp/checkpoints/ -training sample_data/training_data.txt -validation sample_data/validation.txt

And for predictions:
`luajit predict.lua -model /tmp/checkpoints/model-best.t7`

The input data is a csv file. Column 1 is the change of price in percentage and column 2 is the volume.
