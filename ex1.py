
# coding: utf-8

# # Machine Learning Exercise 1 - Linear Regression

# ## Linear regression with one variable

# In[1]:


import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


data = pd.read_csv('ex1data1.txt', header=None, names=['Population', 'Profit'])
data.head()


# In[3]:


data.describe()


# In[4]:


ax = data.plot(kind='scatter', x='Population', y='Profit', title='Scatter plot of training data', figsize=(8,4),grid=True);
ax.set_xlabel('Population of city in 10,000s')
ax.set_ylabel('Profit in $10,000s')


# ## Gradient Descent

# First, you create a function to compute the cost of a given solution (characterized by the parameters beta):

# In[5]:


def compute_cost(X, y, beta):
    sumitem = np.power(((X * beta.T) - y), 2)
    return (np.sum(sumitem) / (2 * len(X)))
    
    
    


# We store each example as a row in the X matrix. To take into account the intercept term (\beta0), we add an additional first column to X and set it to all ones. This allows us to treat \beta0 as simply another 'feature'.

# In[6]:


data.insert(0, 'beta zero', 1)


# Now let's do some variable initialization

# In[7]:


# set X (training data) and y (target variable)
cols = data.shape[1]
X = data.iloc[:,0:cols-1]
y = data.iloc[:,cols-1:cols]


# Now, you need to guarantee that X (training set) and y (target variable) are correct.

# In[8]:


X.head()


# In[9]:


y.head()


# The cost function is expecting numpy matrices so we need to convert X and y before we can use them. We also need to initialize beta.

# In[10]:


X = np.matrix(X.values)
y = np.matrix(y.values)
beta = np.matrix(np.array([0,0]))


# Here's what beta looks like.

# In[11]:


beta


# Let's take a quick look at the shape of our matrices.

# In[12]:


X.shape, beta.shape, y.shape


# Now let's compute the cost for our initial solution (0 values for beta).

# In[13]:


compute_cost(X, y, beta)


# Now, you are asked to define a function to perform gradient descent on the parameters beta

# In[14]:


def gradient_descent(X, y, theta, alpha, iters):
    '''
    alpha: learning rate
    iters: number of iterations
    OUTPUT:
    theta: learned parameters
    cost:  a vector with the cost at each training iteration
    '''
    temp       = np.matrix(np.zeros(theta.shape))
    parameters = int(theta.ravel().shape[1])
    cost       = np.zeros(iters)
    
    for i in range(iters):
        #Calcula hb(xi)-yi, que eh o erro do modelo
        erro = (X * theta.T) - y

        for j in range(parameters):
            #Calcula o erro para cada amostra
            erroamos = np.multiply(erro, X[:,j])
            temp[0,j] = theta[0,j] - ((alpha / len(X)) * np.sum(erroamos))

        theta = temp
        cost[i] = compute_cost(X, y, theta)
        #print(cost[i])
        
    return theta, cost


# Initialize some additional variables - the learning rate alpha, and the number of iterations to perform

# In[15]:


alpha = 0.01
iters = 1500


# Now let's run the gradient descent algorithm to fit our parameters theta to the training set.

# In[16]:


g, cost = gradient_descent(X, y, beta, alpha, iters)
g


# Finally we can compute the cost (error) of the trained model using our fitted parameters.

# In[17]:


compute_cost(X, y, g)


# Now let's plot the linear model along with the data to visually see how well it fits.

# In[18]:


x = np.linspace(data.Population.min(), data.Population.max(), 100)
f = g[0, 0] + (g[0, 1] * x)

fig, ax = plt.subplots(figsize=(8,4))
ax.plot(x, f, 'r', label='Prediction')
ax.scatter(data.Population, data.Profit, label='Traning Data')
ax.legend(loc=2)
ax.set_xlabel('Population of city in 10,000s')
ax.set_ylabel('Profit in $10,000s')
ax.set_title('Predicted Profit vs. Population Size')
ax.grid(True)


# Looks pretty good! Remember that the gradient decent function also outputs a vector with the cost at each training iteration, we can plot it as well. 
# 
# Since the cost always decreases - this is an example of a convex optimization problem.

# In[19]:


fig, ax = plt.subplots(figsize=(8,4))
ax.plot(np.arange(iters), cost, 'r')
ax.set_xlabel('Iterations')
ax.set_ylabel('Cost')
ax.set_ylim(4.0)
ax.set_title('Error vs. Training Epoch')
ax.grid(True)


# Now, we will show a contour plot that presents beta0 against beta1 and the outcome of J. First, we set values for beta0 and beta1

# In[20]:


beta0_vals = np.linspace(-10, 10, 100)
beta1_vals = np.linspace(-1, 4, 100)


# Now, initialize J values to a matrix of 0's

# In[21]:


j_vals = np.zeros([len(beta0_vals), len(beta1_vals)])


# In[22]:


for i in range(len(beta0_vals)):
    for j in range(len(beta1_vals)):
        t = np.matrix(np.array([beta0_vals[i], beta1_vals[j]]))
        j_vals[i,j] = compute_cost(X, y, t)


# In[23]:


plt.contour(beta0_vals, beta1_vals, j_vals.T, np.logspace(-2, 3, 20));


# In[24]:


plt.scatter(g[0,0],g[0,1],)
plt.contour(beta0_vals, beta1_vals, j_vals.T, np.logspace(-2, 3, 20));


# Now, in 3D

# In[25]:


beta0_mesh, beta1_mesh = np.meshgrid(beta0_vals, beta1_vals)
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(beta0_mesh, beta1_mesh, j_vals.T);


# ## Linear regression with multiple variables

# From now on, you will use the second dataset, i.e., ex1data2.txt. This is a housing price dataset with 2 variables (size of the house in square feet and number of bedrooms) and a target (price of the house). You are asked to use the techniques already applied to analyze that data set.

# In[26]:


data2 = pd.read_csv('ex1data2.txt', header=None, names=['Size', 'Bedrooms', 'Price'])
data2.head()


# For this task we add another pre-processing step - normalizing the features.

# Notice that the scale of the values for each feature is vastly large. A house will typically have 2-5 bedrooms, but may have anywhere from hundreds to thousands of square feet. If we use the features as they are in the dataset, the 'size' feature would too much wheighted and would end up dwarfing any contributions from the 'number of bedrooms' feature. To fix this, we need to do something called 'feature normalization'. That is, we need to adjust the scale of the features to level the playing field. One way to do this is by subtracting from each value in a feature the mean of that feature, and then dividing by the standard deviation.

# In[27]:


data2 = (data2 - data2.mean()) / data2.std()
data2.head()


# Given that you were asked to implement both cost function and gradient descent using matrix operations, your previously implementations will work just fine in the multivariate dataset. Hence, you need now insert the 'ones' column as before and separate the X's and the y's.

# Conduct the rest of this exercise by repeating the experiments conducted in the simple linear dataset...

# In[28]:


data2.insert(0, 'beta zero', 1)


# In[29]:


# set X (training data) and y (target variable)
cols2 = data2.shape[1]
X2 = data2.iloc[:,0:cols2-1]
y2 = data2.iloc[:,cols2-1:cols2]


# In[30]:


X2.head()


# In[31]:


y2.head()


# In[32]:


X2 = np.matrix(X2.values)
y2 = np.matrix(y2.values)
beta2 = np.matrix(np.array([0,0,0]))


# In[33]:


#Tamanho aumenta pela mudanca na quantidade de features
beta2 


# In[34]:


X2.shape, beta2.shape, y2.shape


# In[35]:


compute_cost(X2, y2, beta2)


# In[36]:


alpha2 = 0.01
iters2 = 1500


# In[37]:


g2, cost2 = gradient_descent(X2, y2, beta2, alpha2, iters2)
g2


# In[38]:


compute_cost(X2, y2, g2)


# In[39]:


fig2, ax2 = plt.subplots(figsize=(8,4))
ax2.plot(np.arange(iters2), cost2, 'r')
ax2.set_xlabel('Iterations')
ax2.set_ylabel('Cost')
ax2.set_ylim(0.05)
ax2.set_title('Error vs. Training Epoch')
ax2.grid(True)

