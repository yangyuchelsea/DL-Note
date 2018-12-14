# Assignment 1

## Content for code folds
![alt text](https://github.com/yangyuchelsea/cs231n-note/blob/master/Assignment%201/assignment%201.png)

## Dataset: CIFAR10

* 50,000 training data
* 10,000 test data
* shape of data: 32,32,3

* 10 classes:
plane,car,bird,cat,deer,dog,frog,horse,ship,truck

![alt text](https://github.com/yangyuchelsea/cs231n-note/blob/master/Assignment%201/data-eg.png)


## To do list

### Q1: KNN

#### Need to modify: 

* knn.ipynb 
* k_nearest_neighbor.py


#### Processing:

1. subsample the data for this part:

   * 5000 training data
   * 500 test data
   
2. flatten the data:

```
X_train = np.reshape(X_train, (X_train.shape[0], -1))
X_test = np.reshape(X_test, (X_test.shape[0], -1))
```

   * shape of data: 1,3072
   
3. compute the L2 distance:

   * what the distance matrix looks like: (500,5000)
   
   * three approaches: 
   
     1️⃣ two loops: 
   
       ```
        for i in range(num_test): #500
          for j in range(num_train): #5000
            dists[i][j] = np.sqrt(np.sum(np.square(X[i] - self.X_train[j])))
       ```
       
     2️⃣ one loop:
       
       ```
        for i in range(num_test): #500
          dists[i] = np.sqrt(np.sum(np.square(X[i] - self.X_train), axis = 1))
       ```
         
        📎 Why could achieve it: np.sum : sum of array elements over a given axis, axis = 1: sum of row
          
        Example from [Docs](https://docs.scipy.org/doc/numpy-1.9.0/reference/generated/numpy.sum.html)
          
        ```
        >>> np.sum([[0, 1], [0, 5]], axis=0)
        array([0, 6])
        >>> np.sum([[0, 1], [0, 5]], axis=1)
        array([1, 5])
        ```
        
     3️⃣ no loop:
     
     
        ![alt text](https://github.com/yangyuchelsea/cs231n-note/blob/master/Assignment%201/no-loop.png)



4. prediction
   
   * input: distance matrix and label of training data
   * hints:
   
     1️⃣ [numpy.argsort](https://docs.scipy.org/doc/numpy/reference/generated/numpy.argsort.html): Returns the indices that would sort an array.
    
        ```
        x = np.array([3, 1, 2])
        >>> np.argsort(x)
        array([1, 2, 0])
        ```
    
     2️⃣ [numpy.bincount](https://docs.scipy.org/doc/numpy/reference/generated/numpy.bincount.html): Count number of occurrences of each value in array of non-negative ints.
   
        ```
        >>> np.bincount(np.array([0, 1, 1, 3, 2, 1]))
        array([1, 3, 1, 1, 0, 0, 0])
        ```
      
     3️⃣ [np.argmax](https://docs.scipy.org/doc/numpy/reference/generated/numpy.argmax.html): Returns the indices of the maximum values along an axis.
   
   * code：
   
     for each test sample: 
     
       1️⃣ get the label of K nearest neighbors
       
       2️⃣ count the frequency of each label(vote) 
       
       3️⃣ then select the most frequency one 

        ```
         for i in range(num_test):
            closest_y = []
            closest_y = self.y_train[np.argsort(dists[i])[:k]]
            y_pred[i] = np.argmax(np.bincount(closest_y)) 
        ```

Now, we completed k_nearest_neighbor.py, see the architecure of the class:

![alt-text](https://github.com/yangyuchelsea/cs231n-note/blob/master/Assignment%201/Class-KNearestNeighbor.png)

* how to use the class:

```
classifier = KNearestNeighbor()
classifier.train(training_data, training_label)
prediction = classifier.predict(test_data, k=k,num_loops = 0)
```

5. 5-fold cross validation
   
save accuracy in a dictionary, keys are difference choice of k.
   

1️⃣ split the dataset
   
  [np.array_split](https://docs.scipy.org/doc/numpy/reference/generated/numpy.array_split.html):Split an array into multiple sub-arrays.
   
   ```
   X_train_folds = np.array_split(X_train, 5) #each fold shape: (1000, 3072), and combine as a list
   y_train_folds = np.array_split(y_train, 5) #shape: (5,1000)
   ```
   
2️⃣ set dault keys of dictionary(optional):
   
   ```
    k_to_accuracies = {}
    for k_ in k_choices:
      k_to_accuracies.setdefault(k_, [])
   ```
   
3️⃣ for each time, select a fold as validation, and train the rest part.
   
   * hint: 
     * [np.vstack](https://docs.scipy.org/doc/numpy/reference/generated/numpy.vstack.html):Stack arrays in sequence vertically (row wise)
     
     * [np.hstack](https://docs.scipy.org/doc/numpy/reference/generated/numpy.hstack.html#numpy.hstack): column wise
     * [np.concatenate](https://docs.scipy.org/doc/numpy/reference/generated/numpy.concatenate.html): Join a sequence of arrays along an existing axis.axis = 0: column wise
     
     
     ```
      for i in range(num_folds):
        X_val_train = np.vstack(X_train_folds[0:i] + X_train_folds[i+1:])
        y_val_train = np.concatenate(y_train_folds[0:i] + y_train_folds[i+1:]) # or np.hstack
     ```
    
   
4️⃣ calculate the accuracy
   
   ```
   accuracy = float(np.sum(y_val_pred == y_train_folds[i])) / len(y_val_pred)
   ```
   
5️⃣ calcuate the average accuracy for each k_choice and select the best one
   
   ```
    average_acc = []
    for k in sorted(k_to_accuracies):
      average_acc.append(np.mean(k_to_accuracies[k])) 
    best_k = k_choices[np.argmax(average_acc)]
   ```
   

### Additional: PCA(code from my assignment for COMP5318 in USYD)

* only use numpy, code in the end of knn.ipynb

1. eigenvalue decomposition

* input: training_data and t
* processing:
  * normalization
  * compute the Scatter Matrix
  * calculate eigenvectors and eigenvalue
* output: transformed_training_data,and its dimention

2. SVD

* input: training_data and t
* processing:
  * normalization
  * perform Singular Value Decomposition
* output: transformed_training_data,and its dimention


         

### Q2: SVM

Need to modify: 

* svm.ipynb 
* linear_svm.py
* gradient_check.py

#### processing

1. subsample the data for this part:

   * 49000 training data
   * 1000 validation data
   * 1000 test data
   
   
   [np.random.choice](https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.random.choice.html): Generates a random sample from a given 1-D array
   
    ```
    Generate a uniform random sample from np.arange(5) of size 3 without replacement:
    >>> np.random.choice(5, 3, replace=False)
    array([3,1,0])
    ```
2. flatten the data:

```
X_train = np.reshape(X_train, (X_train.shape[0], -1))
X_test = np.reshape(X_test, (X_test.shape[0], -1))
```

   * shape of data: 1,3072
   
3. Preprocessing: normalization

  * (1) compute the image mean based on the training data
    ```
    mean_image = np.mean(X_train, axis=0)
    ```
    
  * (2) subtract the mean image from train and test data
     ```
     X_train -= mean_image
     ```
  * (3) append the bias dimension of ones (i.e. bias trick) so that our SVM
     ```
     X_train = np.hstack([X_train, np.ones((X_train.shape[0], 1))])
     ```
     
     * shape of data: 1,3073
  
4. Loss function(two approaches: we completed linear_svm.py)

* Inputs:
  - W: A numpy array of shape (D, C) containing weights. (3073,10)
  - X: A numpy array of shape (N, D) containing a minibatch of data. (49000,3073)
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means (49000,1)
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

* Returns
  - loss as single float
  - gradient with respect to weights W (an array of same shape as W)

1️⃣ svm loss naive

  * initialize the gradient as zero
  * compute the loss and the gradient
  
    ```
      loss = 0.0
      for i in range(num_train):
      scores = X[i] @ W
      for j in range(num_classes):
        if j != y[i]:
          margin = np.maximum(0, scores[j] - scores[y[i]] + delta)
          dW[:,j] += X[i].T
        loss += margin
     ```

    [Great interprate of gradient](https://mlxai.github.io/2017/01/06/vectorized-implementation-of-svm-loss-and-gradient-update.html)
    
  
  * average
  * add regularization to the loss
  
  ```
  loss += reg * np.sum(W * W)
  dW += reg * W
  ```
 

2️⃣ svm loss vectorized(better)

   * different with 1️⃣:
      * correct class scores
      ```
      correct_class_scores = scores[range(num_train), list(y)].reshape(-1,1) #(N, 1)
      ```
      * margin
      ```
      margins[range(num_train), list(y)] = 0
      ```
      
      * loss
      ```
      np.sum(margins)
      ```
      
      
5. LinearSVM( use SVM classifier)

  * Classes and Inheritance: [liaoxuefeng](https://www.liaoxuefeng.com/wiki/001374738125095c955c1e6d8bb493182103fac9270762a000/001386820044406b227b3e751cc4d5190420d17a2dc6353000)

  ```
  class LinearSVM(LinearClassifier):
  """ A subclass that uses the Multiclass SVM loss function """

  def loss(self, X_batch, y_batch, reg):
    return svm_loss_vectorized(self.W, X_batch, y_batch, reg)
  ```
  
  * how to use the Class
  
  ```
    # training part
    for rs in regularization_strengths:
      for lr in learning_rates:
          svm = LinearSVM()
          loss_hist = svm.train(X_train, y_train, lr, rs, num_iters=3000)
          y_train_pred = svm.predict(X_train)
          train_accuracy = np.mean(y_train == y_train_pred)
          y_val_pred = svm.predict(X_val)
          val_accuracy = np.mean(y_val == y_val_pred)

          if val_accuracy > best_val:
              best_val = val_accuracy
              best_svm = svm           

    # Evaluate the best svm on test set
    y_test_pred = best_svm.predict(X_test)
    test_accuracy = np.mean(y_test == y_test_pred)
    print('linear SVM on raw pixels final test set accuracy: %f' % test_accuracy)
    
    
    #get the weight of best svm 
    w = best_svm.W[:-1,:] # strip out the bias
  ```
  
        
7. Visualize the learned weights for each class

  * hint: 
    * [numpy.matrix.squeeze](https://docs.scipy.org/doc/numpy/reference/generated/numpy.matrix.squeeze.html): return a possibly reshaped matrix
    * w: (3072, 10) -> reshape: (32, 32, 3, 10)
    * Rescale the weights to be between 0 and 255
    ```
    wimg = 255.0 * (w[:, :, :, i].squeeze() - w_min) / (w_max - w_min)
    ```
    
 * Describe what your visualized SVM weights look like, and offer a brief explanation for why they look they way that they do.
 
    * green part in mind mapping
    * higher score: weight should be more like to sample
  
      

### Q3: Softmax 

Need to modify: 

* softmax.ipynb 

### Q4: Two-Layer Neural Network

* two_layer_net.ipynb 

### Q5: Image Features

Need to modify: 

* features.ipynb 
