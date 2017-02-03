#sample_submission.py
import numpy as np
import matplotlib.pyplot as plt

class regressor(object):
    """
    This is a sample class for miniproject 1.

    Args:
        data: Is a tuple, ``(x,y)``
              ``x`` is a two or one dimensional ndarray ordered such that axis 0 is independent 
              data and data is spread along axis 1. If the array had only one dimension, it implies
              that data is 1D.
              ``y`` is a 1D ndarray it will be of the same length as axis 0 or x.   
                          
    """
    def __init__(self, data):
        self.x, self.y = data 

        #print self.x.shape, self.y.shape
        self.w, self.b  = np.random.rand(self.x.shape[1],1), np.random.rand(1)
        #print self.w.shape
               
        #y = self.get_predictions(self.x)
        #er = np.sqrt (np.sum((y-self.y) ** 2))/len (self.x)
        #print er
        bias = np.ones((self.x.shape[0], 1))
        self.x = np.concatenate([self.x, bias], axis = 1)
        #print self.x.shape
        # Here is where your training and all the other magic should happen. 
        # Once trained you should have these parameters with ready. 
        #print 'in:',self.x.shape[0],self.x.shape[1]
        self.w = np.concatenate ((self.w, np.random.rand(1,1)))
        #self.b = np.random.rand(1)

        alpha = 1e-6* 2
        beta = 1e-8 * 9 
        eps = 1000 * 10


        self.w, self.cost = self.grad_descent(self.x, self.y, self.w, alpha, beta, eps)
        self.b = self.w[-1]
        self.w = self.w[0:-1]
        #print self.w.shape, self.b.shape, self.x.shape
        #plt.plot(self.cost)
        #plt.show()

    def error(self, x, y, w):
        #print '7. error x size, w size, y size: ',x.shape, w.shape, y.shape
        #print '###############################################'    
        q = (np.multiply(x, w.T) - y)
        ans = np.sqrt(np.sum(q ** 2))  / (x.size)

        return ans

        
    def grad_descent(self, x, y, w, alpha, beta, eps):
        #print 'initial w size, type: ', w.shape, w.dtype
        #print '###############################################'
        temp = np.matrix(np.zeros(w.shape))
        #print 'Initial Temp size, type: ', temp.shape, temp.dtype
        #print '###############################################'
        dims = int(w.ravel().shape[0])
        #print 'Dimensions:',dims
        #print '###############################################'
        cost = np.zeros(eps)
        #cost = []
        #print 'Cost size: ',cost.shape
        #print '###############################################'

        for i in range(eps):
            #print '1. start x size, w size, y size: ',x.shape, w.shape,y.shape
            #print '###############################################' 
            #er = (np.multiply(x, w.T) - y)
            er = -2/len(x)*((np.dot(x.T, y)) - np.dot(np.dot(x.T,x), w))
            #print '3. er size, type: ', er.shape, er.dtype
            #print '###############################################'
            
            w = w - (alpha * (er) + (beta * np.dot (w.T, w)))
            #print '5. w size, type: ', w.shape, w.dtype
            #print '###############################################'
            #print '6. outer error size, x size, w size: ',i,':',er.shape, x.shape, w.shape
            #print '###############################################'
            cost[i] = self.error(x, y, w)
            #print i, ':', cost[i]
            if cost[i] < 0.015 and i > 3000:
                #print 'iteration: ', i
                break
            #print '###############################################'


        return w, cost



    def get_params (self):
        """ 
        Method that should return the model parameters.

        Returns:
            tuple of numpy.ndarray: (w, b). 

        Notes:
            This code will return a random numpy array for demonstration purposes.

        """
        return (self.w, self.b)

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
        #print 'i', x.shape, self.w.shape
        ans = np.dot(x, self.w) + self.b
        return ans
        #return np.random.rand(self.x.shape[0])

if __name__ == '__main__':
    pass 
