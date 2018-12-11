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

### Q3: Softmax 

Need to modify: 

* softmax.ipynb 

### Q4: Two-Layer Neural Network

* two_layer_net.ipynb 

### Q5: Image Features

Need to modify: 

* features.ipynb 
