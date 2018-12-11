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
