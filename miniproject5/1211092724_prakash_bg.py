#!/usr/bin/env python
from yann.network import network
from yann.utils.pickle import load 
import time
import sys

#run = 1
dataset = sys.argv[1]
momentum = "nesterov"
optim = "rmsprop"
lr = (0.01,0.005,0.001)

params = load('network_dataset_90998.pkl')

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
	input_params = params['conv_pool_1'],
	learnable  = False
	)

net.add_layer(type = "conv_pool",
	origin = "conv_pool_1",
	id = "conv_pool_2",
	num_neurons = 50,
	filter_size = (3,3),
	pool_size = (2,2),
	batch_norm = True,
	activation = 'relu',
	input_params = params['conv_pool_2'],
	learnable  = False
	)

net.add_layer(type = "dot_product", 
	origin = "conv_pool_2",
	id = "dot_product_1",
	num_neurons = 800, 
	regularize = True, 
	activation = 'relu',
	dropout_rate = 0.5,
	input_params = params['dot_product_1'],
	learnable  = False
	)


net.add_layer(type = "dot_product", 
	origin = "dot_product_1",
	id = "dot_product_2",
	num_neurons = 800, 
	regularize = True, 
	activation = 'relu',
	dropout_rate = 0.5,
	input_params = params['dot_product_2'],
	learnable  = False
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
	"momentum_params" 	: (0.6	,0.95, 30),
	"regularization" 	: (0.0001,0.0001),
	"optimizer_type" 	: optim,
	"id"				: id
	}

net.add_module(type = 'optimizer', params = optimizer_params )

learning_rates = lr



net.cook( optimizer = id,
		objective_layer = 'nil',
		datastream = 'svhn',
		classifier = 'softmax'
		)

net.train( epochs = (15,15),
	validate_after_epochs = 2,
	training_accuracy = True,
	learning_rates = learning_rates,
	show_progress = True,
	early_terminate = True)

net.test()


# print "------------------",count,id,lr,"------------------"
# print "================================================"
#count += 1
# time.sleep(150)


		 