#sample_submission.py
import numpy as np


printtrue = 0
    

def activation(x, act, deriv = False):
    if act is 'i':
        if deriv == True:
            return 1
        return x

    if act is 'g':
        if deriv == True:
            return -2*x*np.exp(-1*(np.power(x,2)))
        return np.exp(-1*(np.power(x,2)))

    if act is 'r':
        if deriv == True:
            return 0 if x<0 else 1
        return np.maximum(0,x)

    if act is 's':
        if deriv == True:
            return x * (1 - x)
        return 1/(1 + np.exp(-x))

    if act is 'soft':
        if deriv == True:
            return 1/(1 + np.exp(-x))
        return np.log(1 + np.exp(x))

    if act is 't':
        if deriv == True:
            return 1 - np.power(x,2)
        return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

    if act is 'r':
        if deriv == True:
            return 0 if x<0 else 1
        return np.maximum(0,x)

def calc_acc(valid_label,validation):

    return (np.sum(np.asarray(validation == valid_label, dtype='int'),axis=0) / float(valid_label.shape[0])) * 100

class xor_net(object):
    """
    This is a sample class for miniproject 1.

    Args:
        data: Is a tuple, ``(x,y)``
              ``x`` is a two or one dimensional ndarray ordered such that axis 0 is independent 
              data and data is spread along axis 1. If the array had only one dimension, it implies
              that data is 1D.
              ``y`` is a 1D ndarray it will be of the same length as axis 0 or x.   
                          
    """
    def __init__(self, data, labels):
        self.x = data
        self.y = labels

        train = 0.85
        valid = 0.15

        train_data = self.x[:int(train*len(self.x)),:]
        train_label = self.y[:int(train*len(self.y)):]

        valid_data = self.x[int(train*len(self.x)): , :]
        valid_label = self.y[int(train*len(self.y)): ]

        input_dim = train_data.shape[1]
        self.output_dim = 2
        
        # Number of nodes in the hidden layer
        self.hidden_dim = 10
        self.epochs = 200001

        self.alpha = 1e-5
        self.beta = 1e-8 

        self.act = 's'

        w1 = np.random.randn(input_dim, self.hidden_dim)
        b1 = np.zeros((self.hidden_dim))

        w2 = np.random.randn(self.hidden_dim, self.output_dim)
        b2 = np.zeros((self.output_dim))

        #print self.x.shape, self.y.shape        
        #print 'Main Training data shape: ',train_data.shape, train_label.shape        
        #print 'Main Validation data shape: ',valid_data.shape, valid_label.shape  
        #print 'Main W1, b1, W2, b2 shape: ', w1.shape, b1.shape, w2.shape, b2.shape  
        
        w1, b1, w2, b2 = self.train_model(train_data,train_label, w1,b1,w2,b2, self.alpha, self.beta, self.epochs)

        #print '======================================================================='
        #print 'Trained W1, b1, W2, b2 shape: ', w1.shape, b1.shape, w2.shape, b2.shape  
        
        self.params = [(w1,b1),(w2,b2)]  # [(w,b),(w,b)]         

        #print 'Hyper-parameters: epochs = ', self.epochs,' alpha = ', self.alpha, ' beta = ', self.beta, ' hidden dims = ', self.hidden_dim, ' activation = ', self.act

        training = self.get_predictions(train_data)
        #print 'Training accuracy = ', calc_acc(train_label,training), '%'

        validation = self.get_predictions(valid_data)
        #print 'Validation accuracy = ', calc_acc(valid_label,validation), '%'

    def calc_loss(self, X, y, w1, b1, w2, b2):
        z_1 = np.dot(X, w1) + b1
        a_1 = activation(z_1, self.act)

        z_2 = np.dot(a_1,w2) + b2
        
        score = np.exp(z_2)
        model = score / np.sum(score,axis = 1, keepdims=True)

        # Calculate loss
        logprobs = -np.log(model[range(len(X)), y])
        loss = np.sum(logprobs)

        loss = loss + (0.01*(np.sum(np.square(w1)) + np.sum(np.square(w2))))

        return loss/ (len(X))



    def train_model(self,train_data,train_label, w1,b1,w2,b2, alpha, beta, epochs):
        num_data = len(train_data)
        for i in range(epochs):

            # Forward Propagation
            z_1 = np.dot(train_data, w1) + b1
            #print 'z_1 shape: ', z_1.shape

            a_1 = activation(z_1, self.act)
            #print 'a_1 shape: ', a_1.shape

            z_2 = np.dot(a_1,w2) + b2
            #print 'z_2 shape: ', z_2.shape
            
            score = np.exp(z_2)
            model = score / np.sum(score,axis = 1, keepdims=True)

            # Backward Propagation, weight updates

            del_3 = score
            del_3[range(num_data), train_label] -= 1
            #del_3 /= num_data

            dw2 = np.dot(a_1.T, del_3)
            db2 = np.sum(del_3)

            if self.act is not 'r':
                del_2 = np.dot(del_3,w2.T) * activation(a_1, self.act, deriv=True)
            # backprop with relu non-linearity    
            else:
                del_2 = np.dot(del_3,w2.T)
                del_2[a_1 <= 0] = 0
            
            #del_2 = np.dot(del_3,w2.T) * activation(a_1, self.act, deriv=True)
            dw1 = np.dot(train_data.T, del_2)
            db1 = np.sum(del_2)


            # Regularizing
            dw2 = dw2 + (beta * w2)
            dw1 = dw1 + (beta * w1)

            # Weight update
            w2 = w2 - (alpha*dw2)
            b2 = b2 - (alpha*db2)
            w1 = w1 - (alpha*dw1)
            b1 = b1 - (alpha*db1)

            if printtrue and i % 10000 == 0:
                #alpha =  alpha - (i/1000)*1e-9
                print 'In Epoch :', i,' Loss  = ', self.calc_loss(train_data, train_label, w1, b1, w2, b2)
    
        return w1,b1,w2,b2

    def get_params (self):
        """ 
        Method that should return the model parameters.

        Returns:
            tuple of numpy.ndarray: (w, b). 

        Notes:
            This code will return an empty list for demonstration purposes. A list of tuples of 
            weoghts and bias for each layer. Ordering should from input to outputt

        """
        return self.params

    def get_predictions (self, x):
        """
        Method should return the outputs given unseen data

        Args:
            x: array similar to ``x`` in ``data``. Might be of different size.

        Returns:    
            numpy.ndarray: ``y`` which is a 1D array of predictions of the same length as axis 0 of 
                            ``x`` 
        Notes:
            Temporarily returns random numpy array for demonstration purposes.                            
        """        
        # Here is where you write a code to evaluate the data and produce predictions.
        weights = self.get_params()
        #print weights
        w1 = weights[0][0]
        b1 = weights[0][1]
        w2 = weights[1][0]
        b2 = weights[1][1]

        z_1 = np.dot(x, w1) + b1
        a_1 = activation(z_1,self.act)

        z_2 = np.dot(a_1,w2) + b2
        
        score = np.exp(z_2)
        model = score / np.sum(score,axis = 1, keepdims=True)

        return np.argmax(model, axis = 1)
        #return np.random.randint(low =0, high = 2, size = x.shape[0])

class mlnn(xor_net):
    """
    At the moment just inheriting the network above. 
    """
    def __init__ (self, data, labels):
        #super(mlnn,self).__init__(data, labels)
        self.x = data / data.max()

        #print self.x.max(), self.x.min()
        #self.x = np.divide(data, 255.)
        self.y = labels

        train = 0.85
        valid = 0.15

        train_data = self.x[:int(train*len(self.x)),:]
        train_label = self.y[:int(train*len(self.y)):]

        valid_data = self.x[int(train*len(self.x)): , :]
        valid_label = self.y[int(train*len(self.y)): ]

        input_dim = train_data.shape[1]
        self.output_dim = 2
        
        # Number of nodes in the hidden layer
        self.hidden_dim = 100
        self.epochs = 100001

        self.alpha = 1e-5
        self.beta = 1e-9

        self.act = 's'

        w1 = np.random.randn(input_dim, self.hidden_dim)
        b1 = np.zeros((self.hidden_dim))

        w2 = np.random.randn(self.hidden_dim, self.output_dim)
        b2 = np.zeros((self.output_dim))

        #print self.x.shape, self.y.shape        
        #print 'Main Training data shape: ',train_data.shape, train_label.shape        
        #print 'Main Validation data shape: ',valid_data.shape, valid_label.shape  
        #print 'Main W1, b1, W2, b2 shape: ', w1.shape, b1.shape, w2.shape, b2.shape  
        
        w1, b1, w2, b2 = self.train_model(train_data,train_label, w1,b1,w2,b2, self.alpha, self.beta, self.epochs)

        #print '======================================================================='
        #print 'Trained W1, b1, W2, b2 shape: ', w1.shape, b1.shape, w2.shape, b2.shape  
        
        self.params = [(w1,b1),(w2,b2)]  # [(w,b),(w,b)]         

        #print 'Hyper-parameters: epochs = ', self.epochs,' alpha = ', self.alpha, ' beta = ', self.beta, ' hidden dims = ', self.hidden_dim, ' activation = ', self.act  
        
        training = self.get_predictions(train_data)
        #print 'Training accuracy = ', calc_acc(train_label,training), '%'

        validation = self.get_predictions(valid_data)
        #print 'Validation accuracy = ', calc_acc(valid_label,validation), '%'


if __name__ == '__main__':
    pass 
