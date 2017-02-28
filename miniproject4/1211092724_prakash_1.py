#!/usr/bin/env python
from yann.network import network
import time
import sys

run = 1
dataset = sys.argv[1]

momentum = "nesterov"
optim = "rmsprop"
lr = (0.01,0.001,0.0001)

# create a network
net = network()
dataset_params = {"dataset": dataset, "id": 'svhn', "n_classes": 10}
net.add_layer(type = "input", id = "input", dataset_init_args = dataset_params)

# adding layers

net.add_layer(type = "conv_pool",
	origin = "input",
	id = "conv_pool_1",
	num_neurons = 20,
	filter_size = (5,5),
	pool_size = (2,2),
	batch_norm = True,
	activation = 'relu',
	)


net.add_layer(type = "conv_pool",
	origin = "conv_pool_1",
	id = "conv_pool_2",
	num_neurons = 40,
	filter_size = (3,3),
	pool_size = (2,2),
	batch_norm = True,
	activation = 'relu',
	)

net.add_layer(type = "conv_pool",
	origin = "conv_pool_2",
	id = "conv_1",
	num_neurons = 60,
	filter_size = (3,3),
	pool_size = (1,1),
	batch_norm = True,
	activation = 'relu',
	)

net.add_layer(type = "conv_pool",
	origin = "conv_1",
	id = "conv_2",
	num_neurons = 150,
	filter_size = (3,3),
	pool_size = (1,1),
	batch_norm = True,
	activation = 'relu',
	)

net.add_layer(type = "dot_product", 
	origin = "conv_2",
	id = "dot_product_1", 
	num_neurons = 600, 
	regularize = True, 
	activation = 'relu'
	)


net.add_layer(type = "dot_product", 
	origin = "dot_product_1",
	id = "dot_product_2", 
	num_neurons = 600, 
	regularize = True, 
	activation = 'relu'
	)

net.add_layer(type = "classifier",
	id = "softmax",
	origin = "dot_product_2",
	num_classes = 10,
	activation = "softmax"
	)

net.add_layer(type = "objective",
	id = "nil",
	origin = "softmax")

net.pretty_print()


id = momentum+'-'+optim

optimizer_params = { 
	"momentum_type" 	: momentum,
	"momentum_params" 	: (0.5,0.95, 20),
	"regularization" 	: (0.0001,0.0002),
	"optimizer_type" 	: optim,
	"id"				: id
	}

net.add_module(type = 'optimizer', params = optimizer_params )

learning_rates = lr


if run is 1:
	net.cook( optimizer = id,
			objective_layer = 'nil',
			datastream = 'svhn',
			classifier = 'softmax'
			)

	net.train( epochs = (10,10),
		validate_after_epochs = 2,
		training_accuracy = True,
		learning_rates = learning_rates,
		show_progress = True,
		early_terminate = True)

	net.test()

	print "------------------",id,lr,"------------------"
	print "================================================"


