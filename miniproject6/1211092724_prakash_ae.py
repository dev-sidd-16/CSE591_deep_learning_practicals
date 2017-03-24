#!/usr/bin/env python
"""
TODO:

    * Need a validation and testing thats better than just measuring rmse. Can't find something 
      great.
    * Loss increases after 3 epochs.
    
"""
from yann.network import network


def autoencoder(dataset= None, expNo = 0, verbose = 1):
    """
    This function is a demo example of a sparse autoencoder. 
    This is an example code. You should study this code rather than merely run it.  

    Args: 
        dataset: Supply a dataset.    
        verbose: Similar to the rest of the dataset.
    """
    exp_no = str(expNo) + '_autoencoders_tutorial_' + dataset
    root_dir = './experiments/' + exp_no
    dataset_params  = {
                            "dataset"   : dataset,
                            "type"      : 'x',
                            "id"        : 'data'
                    }

    visualizer_params = {
                    "root"       : root_dir,
                    "frequency"  : 1,
                    "sample_size": 32,
                    "rgb_filters": False,
                    "debug_functions" : False,
                    "debug_layers": True,  
                    "id"         : 'main'
                        }  
                      
    # intitialize the network    
    optimizer_params =  {        
                "momentum_type"       : 'polyak',             
                "momentum_params"     : (0.5, 0.95, 20),      
                "regularization"      : (0.0001, 0.0001),       
                "optimizer_type"      : 'adagrad',                
                "id"                  : "main"
                    }
    net = network(   borrow = True,
                     verbose = verbose )                       

    net.add_module ( type = 'datastream', 
                     params = dataset_params,
                     verbose = verbose )
    
    net.add_module ( type = 'visualizer',
                     params = visualizer_params,
                     verbose = verbose 
                    ) 
    net.add_module ( type = 'optimizer',
                     params = optimizer_params,
                     verbose = verbose )
    # add an input layer 
    net.add_layer ( type = "input",
                    id = "input",
                    verbose = verbose, 
                    origin = 'data', # if you didnt add a dataset module, now is 
                                                 # the time. 
                    mean_subtract = False )

    net.add_layer ( type = "flatten",
                    origin = "input",
                    id = "flatten",
                    verbose = verbose
                    )

    net.add_layer ( type = "dot_product",
                    origin = "flatten",
                    id = "encoder",
                    num_neurons = 64,
                    activation = 'relu',
                    verbose = verbose
                    )

    net.add_layer ( type = "dot_product",
                    origin = "encoder",
                    id = "decoder",
                    num_neurons = 784,
                    activation = 'relu',
                    input_params = [net.dropout_layers['encoder'].w.T, None],
                    # Use the same weights but transposed for decoder. 
                    learnable = False,                    
                    # because we don't want to learn the weights of somehting already used in 
                    # an optimizer, when reusing the weights, always use learnable as False                    
                    verbose = verbose
                    )           

    # We still need to learn the newly created biases in the decoder layer, so add them to the 
    # Learnable parameters list before cooking

    net.active_params.append(net.dropout_layers['decoder'].b)

    net.add_layer ( type = "unflatten",
                    origin = "decoder",
                    id = "unflatten",
                    shape = (28,28,1),
                    verbose = verbose
                    )

    net.add_layer ( type = "merge",
                    origin = ("input","unflatten"),
                    id = "merge",
                    layer_type = "error",
                    error = "rmse",
                    verbose = verbose)

    net.add_layer ( type = "objective",
                    id = "obj",
                    origin = "merge", # this is useless anyway.
                    layer_type = 'value',
                    objective = net.layers['merge'].output,
                    datastream_origin = 'data', 
                    verbose = verbose
                    )          

    learning_rates = (0.001, 0.1, 0.001)  
    net.cook( objective_layers = ['obj'],
              datastream = 'data',
              learning_rates = learning_rates,
              verbose = verbose
              )

    # from yann.utils.graph import draw_network
    # draw_network(net.graph, filename = 'autoencoder.png')    
    net.pretty_print()

    net.train( epochs = (10, 10), 
               validate_after_epochs = 1,
               training_accuracy = True,
               show_progress = True,
               early_terminate = False,
               verbose = verbose)
                    
def convolutional_autoencoder ( dataset= None, expNo=0, verbose = 1 ):
    """
    This function is a demo example of a deep convolutional autoencoder. 
    This is an example code. You should study this code rather than merely run it.  
    This is also an example for using the deconvolutional layer or the transposed fractional stride
    convolutional layers.

    Args: 
        dataset: Supply a dataset.    
        verbose: Similar to the rest of the dataset.
    """
    codewords = [64, 128, 256, 512, 1024]

    exp_no = str(expNo) + '_autoencoders_cnn_tutorial_' + dataset
    root_dir = './experiments/' + exp_no
    dataset_params  = {
                            "dataset"   : dataset,
                            "type"      : 'x',
                            "id"        : 'data'
                    }
    print dataset_params

    visualizer_params = {
                    "root"       : root_dir,
                    "frequency"  : 1,
                    "sample_size": 32,
                    "rgb_filters": True,
                    "debug_functions" : False,
                    "debug_layers": True,  
                    "id"         : 'main'
                        }  
    print visualizer_params           
    # intitialize the network    
    optimizer_params =  {        
                "momentum_type"       : 'nesterov',             
                "momentum_params"     : (0.65, 0.95, 30),      
                "regularization"      : (0.0001, 0.0001),       
                "optimizer_type"      : 'rmsprop',                
                "id"                  : "main"
                    }

    print optimizer_params
    net = network(   borrow = True,
                     verbose = verbose )                       

    net.add_module ( type = 'datastream', 
                     params = dataset_params,
                     verbose = verbose )
    
    net.add_module ( type = 'visualizer',
                     params = visualizer_params,
                     verbose = verbose 
                    ) 
    net.add_module ( type = 'optimizer',
                     params = optimizer_params,
                     verbose = verbose )
    # add an input layer 
    net.add_layer ( type = "input",
                    id = "input",
                    verbose = verbose, 
                    origin = 'data', # if you didnt add a dataset module, now is 
                                                 # the time. 
                    mean_subtract = True )

    
    net.add_layer ( type = "conv_pool",
                    origin = "input",
                    id = "conv",
                    num_neurons = 20,
                    filter_size = (5,5),
                    pool_size = (1,1),
                    activation = 'relu',
                    regularize = True,   
                    #stride = (2,2),                          
                    verbose = verbose
                    )
    
    net.add_layer ( type = "flatten",
                    origin = "conv",
                    id = "flatten",
                    verbose = verbose
                    )

    net.add_layer ( type = "dot_product",
                    origin = "flatten",
                    id = "hidden-encoder1",
                    num_neurons = 1200,
                    activation = 'relu',
                    dropout_rate = 0.5,                    
                    regularize = True,
                    verbose = verbose
                    )

    net.add_layer ( type = "dot_product",
                    origin = "hidden-encoder1",
                    id = "hidden-encoder2",
                    num_neurons = 256,
                    activation = 'relu',
                    dropout_rate = 0.5,                        
                    regularize = True,
                    verbose = verbose
                    )

    net.add_layer ( type = "dot_product",
                    origin = "hidden-encoder2",
                    id = "encoder",
                    num_neurons = 128,
                    activation = 'relu',
                    dropout_rate = 0.5,                        
                    regularize = True,
                    verbose = verbose
                    )

    net.add_layer ( type = "dot_product",
                    origin = "encoder",
                    id = "decoder",
                    num_neurons = 256,
                    activation = 'relu',
                    input_params = [net.dropout_layers['encoder'].w.T, None],
                    # Use the same weights but transposed for decoder. 
                    learnable = False,                    
                    # because we don't want to learn the weights of somehting already used in 
                    # an optimizer, when reusing the weights, always use learnable as False   
                    dropout_rate = 0.5,                                         
                    verbose = verbose
                    )           

    net.add_layer ( type = "dot_product",
                    origin = "decoder",
                    id = "hidden-decoder2",
                    num_neurons = 1200,
                    activation = 'relu',
                    input_params = [net.dropout_layers['hidden-encoder2'].w.T, None],
                    # Use the same weights but transposed for decoder. 
                    learnable = False,                    
                    # because we don't want to learn the weights of somehting already used in 
                    # an optimizer, when reusing the weights, always use learnable as False    
                    dropout_rate = 0.5,                                        
                    verbose = verbose
                    )                                            

    net.add_layer ( type = "dot_product",
                    origin = "hidden-decoder2",
                    id = "hidden-decoder1",
                    num_neurons = net.layers['flatten'].output_shape[1],
                    activation = 'relu',
                    input_params = [net.dropout_layers['hidden-encoder1'].w.T, None],
                    # Use the same weights but transposed for decoder. 
                    learnable = False,                    
                    # because we don't want to learn the weights of somehting already used in 
                    # an optimizer, when reusing the weights, always use learnable as False    
                    dropout_rate = 0.5,                                        
                    verbose = verbose
                    )                                            

    net.add_layer ( type = "unflatten",
                    origin = "hidden-decoder1",
                    id = "unflatten",
                    shape = (net.layers['conv'].output_shape[2],
                             net.layers['conv'].output_shape[3],
                             20),
                    verbose = verbose
                    )

    net.add_layer ( type = "deconv",
                    origin = "unflatten",
                    id = "deconv",
                    num_neurons = 20,
                    filter_size = (5,5),
                    pool_size = (1,1),
                    output_shape = (28,28,1),
                    activation = 'relu',
                    input_params = [net.dropout_layers['conv'].w, None],        
                    learnable = False,              
                    #stride = (2,2),
                    verbose = verbose
                    )

    # We still need to learn the newly created biases in the decoder layer, so add them to the 
    # Learnable parameters list before cooking

    net.active_params.append(net.dropout_layers['hidden-decoder2'].b)
    net.active_params.append(net.dropout_layers['hidden-decoder1'].b)
    net.active_params.append(net.dropout_layers['decoder'].b)    
    net.active_params.append(net.dropout_layers['deconv'].b)

    net.add_layer ( type = "merge",
                    origin = ("input","deconv"),
                    id = "merge",
                    layer_type = "error",
                    error = "rmse",
                    verbose = verbose)

    net.add_layer ( type = "objective",
                    id = "obj",
                    origin = "merge", # this is useless anyway.
                    layer_type = 'value',
                    objective = net.layers['merge'].output,
                    datastream_origin = 'data', 
                    verbose = verbose
                    )          

    learning_rates = (0, 0.01, 0.001)  
    net.cook( objective_layers = ['obj'],
              datastream = 'data',
              learning_rates = learning_rates,
              verbose = verbose
              )

    # from yann.utils.graph import draw_network
    # draw_network(net.graph, filename = 'autoencoder.png')    
    net.pretty_print()
    net.train( epochs = (10, 10), 
               validate_after_epochs = 1,
               training_accuracy = True,
               show_progress = True,
               early_terminate = False,
               verbose = verbose)

if __name__ == '__main__':
    import sys
    
    expNo = sys.argv[2]
    dataset = None  

    if len(sys.argv) > 1:
        if sys.argv[1] == 'create_dataset':
            from yann.special.datasets import cook_mnist_normalized as cook_mnist  
            data = cook_mnist (verbose = 2)
            dataset = data.dataset_location()
        else:
            dataset = sys.argv[1]
    else:
        print "provide dataset"
    
    if dataset is None:
        print " creating a new dataset to run through"
        from yann.special.datasets import cook_mnist_normalized as cook_mnist  
        data = cook_mnist (verbose = 2)
        dataset = data.dataset_location()

    #autoencoder ( dataset, expNo, verbose = 2 )
    convolutional_autoencoder ( dataset , expNo, verbose = 2 )