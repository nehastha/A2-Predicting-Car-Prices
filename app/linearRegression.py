import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mlflow

from sklearn.model_selection import KFold

class LinearRegression(object):
    
    #in this class, we add cross validation as well for some spicy code....
    kfold = KFold(n_splits=3)


    def __init__(self, regularization, lr=0.001, method='batch', num_epochs=500, batch_size=50, cv=kfold, polynomial= True, degree= 3, weight='zeros', momentum=0.0):
        self.lr         = lr
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.method     = method
        self.cv         = cv
        self.regularization = regularization
        self.weight = weight
        self.degree = degree
        self.momentum = momentum
        self.polynomial = polynomial
        self.prev_step = 0
    
    def mse(self, ytrue, ypred):
        return ((ypred - ytrue) ** 2).sum() / ytrue.shape[0]
    
    ## r2 = 1: The model explains all the variance in the target variable (perfect fit).
    ## r2 = 0: The model does not explain any variance in the target variable (no better than predicting the mean).
    ## r2 < 0: The model performs worse than a horizontal line (mean of the target variable).
    def r2(self, ytrue, ypred):
        ss_total = ((ytrue - ytrue.mean()) ** 2).sum()
        ss_residual = ((ytrue - ypred) ** 2).sum()
        return 1 - (ss_residual / ss_total)
    
    # function to compute average mse for all kfold_scores
    def mseMean(self): # changing avgMse to mseMean
        return np.sum(np.array(self.kfold_scores))/len(self.kfold_scores)
    
    # function to compute average r2 for all kfold_scores
    def r2Mean(self): # changing avgr2 to r2Mean
        return np.sum(np.array(self.kfold_r2))/len(self.kfold_r2)
    
    def fit(self, X_train, y_train):

        # Ensures that feature names are stored for later use.
        self.columns = X_train.columns

        if self.polynomial == True: 
            X_train = self._transform_features(X_train) 
            print(X_train.shape)
            print("Using Polynomial")
        else:
            X_train = X_train.to_numpy()
            print("Using Linear")
       

        y_train = y_train.to_numpy() #Converts the target variable (y_train) into a NumPy array for compatibility with ML models.

        #create a list of kfold scores and r2
        self.kfold_scores = list()
        self.kfold_r2 = list()
        
        #reset val lossß
        self.val_loss_old = np.inf

        #kfold.split in the sklearn.....
        #5 splits
        for fold, (train_idx, val_idx) in enumerate(self.cv.split(X_train)):
            
            X_cross_train = X_train[train_idx]
            y_cross_train = y_train[train_idx]
            X_cross_val   = X_train[val_idx]
            y_cross_val   = y_train[val_idx]
            
            if(self.weight == 'zeros'):
                self.theta = np.zeros(X_cross_train.shape[1]) 
            elif(self.weight == 'xavier'):
                # number of samples
                m = X_cross_train.shape[0] 

                # calculating the range for the weight
                lower, upper = -(1.0 / np.sqrt(m)), (1.0 / np.sqrt(m)) 
                num = np.random.rand(X_cross_train.shape[1]) 
                scaled = lower + num * (upper - lower)
                self.theta = scaled
            else:
                print("Weight Initialization Method Is Invalid")
                return
            
            #one epoch will exhaust the WHOLE training set
            with mlflow.start_run(run_name=f"Fold-{fold}", nested=True):
                
                params = {"method": self.method, "lr": self.lr, "reg": type(self).__name__}
                mlflow.log_params(params=params)
                
                for epoch in range(self.num_epochs):
                
                    #with replacement or no replacement
                    #with replacement means just randomize
                    #with no replacement means 0:50, 51:100, 101:150, ......300:323
                    #shuffle your index to make a random model so that it does not have ordered data
                    perm = np.random.permutation(X_cross_train.shape[0])
                            
                    #shuffle based on permutations
                    X_cross_train = X_cross_train[perm]
                    y_cross_train = y_cross_train[perm]
                    
                    #stochastic
                    if self.method == 'stochastic':
                        for batch_idx in range(X_cross_train.shape[0]):
                            X_method_train = X_cross_train[batch_idx].reshape(1, -1) 
                            y_method_train = y_cross_train[batch_idx].reshape(1, ) 
                            train_loss = self._train(X_method_train, y_method_train)

                    #mini_batch        
                    elif self.method == 'mini_batch':
                        for batch_idx in range(0, X_cross_train.shape[0], self.batch_size):
                            #batch_idx = 0, 50, 100, 150
                            X_method_train = X_cross_train[batch_idx:batch_idx+self.batch_size, :]
                            y_method_train = y_cross_train[batch_idx:batch_idx+self.batch_size]
                            train_loss = self._train(X_method_train, y_method_train)

                    #batch 
                    else:
                        X_method_train = X_cross_train
                        y_method_train = y_cross_train
                        train_loss = self._train(X_method_train, y_method_train)

                    mlflow.log_metric(key="train_loss", value=train_loss, step=epoch)

                    #predict yhat_val with validation data
                    yhat_val = self.predict(X_cross_val) 
                    
                    #calculate mse loss and r2 
                    val_loss_new = self.mse(y_cross_val, yhat_val) # This line calculates the mean squared error (MSE) between the true labels (y_cross_val) and the predicted labels (yhat_val) for the cross-validation set.
                    val_r2_new = self.r2(y_cross_val, yhat_val) # This line calculates the R² (coefficient of determination) score between the true labels (y_cross_val) and the predicted labels (yhat_val) for the cross-validation data.
                    
                    #log to mlflow
                    mlflow.log_metric(key="val_loss", value=val_loss_new, step=epoch) # This line logs the calculated validation loss (MSE) (val_loss_new) into MLflow, a platform for managing machine learning experiments.
                    mlflow.log_metric(key="val_r2", value=val_r2_new, step=epoch) # Similar to the previous line, this logs the validation R² score (val_r2_new) into MLflow for the current epoch.
                    
                    #early stopping
                    if np.allclose(val_loss_new, self.val_loss_old):
                        break
                    self.val_loss_old = val_loss_new
            
                 #add to the kfold list
                self.kfold_scores.append(val_loss_new)
                self.kfold_r2.append(val_r2_new)
                
                print(f"Fold {fold}: MSE:{val_loss_new}")
                print(f"Fold {fold}: r2:{val_r2_new}")
        
    def _transform_features(self, X):
        # transforms input features to include polynomial degree where highest degree is taken
        X_poly = np.column_stack([X ** (self.degree)])        
        return X_poly
                 
    def _train(self, X, y):
        yhat = self.predict(X) #predicts target values with weight
        m    = X.shape[0]

        if self.regularization:
            grad = (1/m) * X.T @(yhat - y) + self.regularization.derivation(self.theta) #compute with regularization
        else:
            grad = (1/m) * X.T @ (yhat - y) #compute without regularization


        #momentum
        if(self.momentum >= 0 and self.momentum <= 1): #if true momentum is applied to gradient descent 
            gra = self.lr * grad #momentum implemented
            self.theta = self.theta - gra + self.momentum * self.prev_step
            self.prev_step = gra
        else:
            self.theta = self.theta - self.lr * grad #mommentum range is between (0, 1)
        return self.mse(y, yhat) #return the MSE loss
    
    def predict(self, X):
        return X @ self.theta  #===>(m, n) @ (n, ) #predicts the targeted values using trained regression model and matrix multiplication
    
    def _coef(self):
        return self.theta[1:]  #remind that theta is (w0, w1, w2, w3, w4.....wn)
                               #w0 is the bias or the intercept
                               #w1....wn are the weights / coefficients / theta
    def _bias(self):
        return self.theta[0]
    
    def features(self, X, features):
        return plt.barh(X.columns, features)
    
    def feature_importance(self, width=5, height=5):
    # Create a DataFrame with coefficients and feature names, if available
        if self.theta is not None and self.columns is not None:
            coefs = pd.DataFrame(data=self.theta, columns=['Coefficients'],index=self.columns) 
            coefs.plot(kind="barh", figsize=(width, height)) # Create a horizontal bar plot
            plt.title("Feature Importance") # Set the title of the plot
            plt.show()  # Display the plot
            print(coefs)
        else:
            print("Coefficients or feature names are not available to create the graph.")

class LassoPenalty:
    
    def __init__(self, l):
        self.l = l # lambda value
        
    def __call__(self, theta): #__call__ allows us to call class as method
        return self.l * np.sum(np.abs(theta))
        
    def derivation(self, theta):
        return self.l * np.sign(theta)
    
class RidgePenalty:
    
    def __init__(self, l):
        self.l = l
        
    def __call__(self, theta): #__call__ allows us to call class as method
        return self.l * np.sum(np.square(theta))
        
    def derivation(self, theta):
        return self.l * 2 * theta
        
class Lasso(LinearRegression):
    
    def __init__(self, method, lr, l,momentum, weight, polynomial, degree):
        self.regularization = LassoPenalty(l)
        
        super().__init__(self.regularization, lr, method, momentum=momentum, weight=weight, polynomial=polynomial, degree=degree)
    
    def mseMean(self):
        return np.sum(np.array(self.kfold_scores)) / len(self.kfold_scores)
        
class Ridge(LinearRegression):
    
    def __init__(self, method, lr, l,momentum, weight, polynomial, degree):
        self.regularization = RidgePenalty(l)
        super().__init__(self.regularization, lr, method, momentum=momentum, weight=weight, polynomial=polynomial, degree=degree)

    def mseMean(self):
        return np.sum(np.array(self.kfold_scores)) / len(self.kfold_scores)
    
class Normal(LinearRegression):  
    def __init__(self, method, lr, l, momentum, weight, polynomial, degree):
        self.regularization = None
        super().__init__(self.regularization, lr, method, momentum=momentum, weight=weight, polynomial=polynomial, degree=degree)
   
    def mseMean(self):
        return np.sum(np.array(self.kfold_scores)) / len(self.kfold_scores)    