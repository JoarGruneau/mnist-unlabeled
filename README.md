# mnist-unlabeled
# Description
This describes an unsupervised clustering network that achieves ~86-88% on the test set after 300 epochs.
The network is trained in a completely unsupervised fashion.
Ten images of different labels are used during training to determine the centers of each class the labels are only used to assign a class to each cluster for testing purposes.

## Loss function
Let x be the input image and aug(x) the augmentation transform. The encoder takes an input image and maps it to the surface of a n-dimensional sphere.
Let x_labeled be our 10 labeled samples and Enc(x_labeled) = c be the mapping of our labeled samples onto the feature sphere.
We will refer to c as the class centers for each cluster. We will update c when the mapping Enc(x_labeled) changes but c should be considered as fixed class centers and not a function of Enx().
The decoder takes the coordinates on the sphere and tries to reconstruct the original image.
Let's define the mapping onto the sphere as Enc(x) and the reconstruction as Dec(Enc(x)), then can then describe the three parts of our loss function as.

feature_loss = MSE(Enc(x), Enc(aug(x))

We want that the encoding of x and the encoding of the augmentation of x to be near each other on the surface of the sphere.

cluster_loss = (distance(Enc(x), c) + distance(Enc(aug(x)), c))/(2*dimensions of the sphere)

This is the class average distance from x mapped onto the sphere to all the class centers. 
This loss will be minimum when Enc(x) and Enx(aug(x)) lies in a class center.
To minimize this loss ensures that our class centers will be the center of a cluster with features similar to itself on the sphere.
Note that the value of this loss will increase as the class centers initially are spread out onto the sphere but this is not important since c is not a function of Enc(),and all it means is that the absolute minimum value of the loss function is growing while the class centers are seperating.

reconstruction_loss = MSE(Enc(aug(x)), x)

We want our mapping on the sphere to only contain the most basic information to reconstruct it back to the right class.
We therefor compare the reconstruction of the augmentation of x to x itself.

We can ther compute the total loss as:

loss = a * feature_loss + b * cluster_loss + c * reconstruction_loss
Here a, b, c are constants with values 0.01, 1, 0.01

# Run code
python main.py
