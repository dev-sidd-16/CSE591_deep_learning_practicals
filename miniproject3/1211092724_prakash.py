from yann.network import network
import time
import sys

dataset = sys.argv[1]

momentum = "nesterov"
optim = "rmsprop"
lr = (0.01,0.001,0.0001)

# create a network
net = network()
dataset_params = {"dataset": dataset, "id": 'mnist', "n_classes": 10}
net.add_layer(type = "input", id = "input", dataset_init_args = dataset_params)

# adding layers

net.add_layer(type = "dot_product", 
	origin = "input", 
	id = "dot_product_1", 
	num_neurons = 800, 
	regularize = True, 
	activation = 'relu'
	)

net.add_layer(type = "dot_product",
	origin = "dot_product_1",
	id = "dot_product_2",
	num_neurons = 800,
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
	"momentum_params" 	: (0.6,0.95, 30),
	"regularization" 	: (0.0001,0.0002),
	"optimizer_type" 	: optim,
	"id"				: id
	}

net.add_module(type = 'optimizer', params = optimizer_params )

learning_rates = lr

net.cook( optimizer = id,
		objective_layer = 'nil',
		datastream = 'mnist',
		classifier = 'softmax'
		)

net.train( epochs = (20,20),
	validate_after_epochs = 2,
	training_accuracy = True,
	learning_rates = learning_rates,
	show_progress =False,
	early_terminate = True)

net.test()

print "------------------",id,lr,"------------------"
print "================================================"


