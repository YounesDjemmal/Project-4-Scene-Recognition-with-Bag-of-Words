import numpy as np
import torch
from scipy import stats
from scipy.spatial.distance import cdist 


from proj4_code.feature_matching.SIFTNet import get_siftnet_features
from proj4_code.utils import generate_sample_points


def pairwise_distances(X, Y):
    """
    This method will be very similar to the pairwise_distances() function found
    in sklearn (https://scikit-learn.org/stable/modules/generated/sklearn
    .metrics.pairwise_distances.html)
    However, you are NOT allowed to use any library functions like this
    pairwise_distances or pdist from scipy to do the calculation!

    The purpose of this method is to calculate pairwise distances between two
    sets of vectors. The distance metric we will be using is 'euclidean',
    which is the square root of the sum of squares between every value.
    (https://en.wikipedia.org/wiki/Euclidean_distance)

    Useful functions:
    -   np.linalg.norm()

    Args:
    -   X: N x d numpy array of d-dimensional features arranged along N rows
    -   Y: M x d numpy array of d-dimensional features arranged along M rows

    Returns:
    -   D: N x M numpy array where d(i, j) is the distance between row i of
    X and row j of Y
    """
    N, d_y = X.shape
    M, d_x = Y.shape
    assert d_y == d_x

    # D is the placeholder for the result
    D = None

    #############################################################################
    # TODO: YOUR CODE HERE
    #############################################################################
    # t1 = X - Y
    # t2 = np.square(t1)
    # D = np.sqrt(np.sum(t2))

    return np.linalg.norm(X[:, None, :] - Y[None, :, :], axis=-1)
    #############################################################################
    #                             END OF YOUR CODE
    #############################################################################

def cosine_distance(a, b):
    if a.shape != b.shape:
        raise RuntimeError("array {} shape not match {}".format(a.shape, b.shape))
    if a.ndim==1:
        a_norm = np.linalg.norm(a)
        b_norm = np.linalg.norm(b)
    elif a.ndim==2:
        a_norm = np.linalg.norm(a, axis=1, keepdims=True)
        b_norm = np.linalg.norm(b, axis=1, keepdims=True)
    else:
        raise RuntimeError("array dimensions {} not right".format(a.ndim))
    similiarity = np.dot(a, b.T)/(a_norm * b_norm) 
    dist = 1. - similiarity
    return dist

def manhattan_dist(A, B):
    return np.abs(A[:,0,None] - B[:,0]) + np.abs(A[:,1,None] - B[:,1])


def nearest_neighbor_classify(train_image_feats,
                              train_labels,
                              test_image_feats,
                              k=3):
    """
    This function will predict the category for every test image by finding
    the training image with most similar features. Instead of 1 nearest
    neighbor, you can vote based on k nearest neighbors which can increase the
    performance.

    Useful functions:
    -   D = pairwise_distances(X, Y) computes the distance matrix D between
    all pairs of rows in X and Y. This is the method you implemented above.
        -  X is a N x d numpy array of d-dimensional features arranged along
        N rows
        -  Y is a M x d numpy array of d-dimensional features arranged along
        N rows
        -  D is a N x M numpy array where d(i, j) is the distance between
        row i of X and row j of Y
    - np.argsort()
    - scipy.stats.mode()

    Args:
    -   train_image_feats:  N x d numpy array, where d is the dimensionality
    of the feature representation
    -   train_labels: N element list, where each entry is a string
    indicating the ground truth category for each training image
    -   test_image_feats: M x d numpy array, where d is the dimensionality
    of the feature representation.
    -   k: the k value in kNN, indicating how many votes we need to check
    for the label

    Returns:
    -   pred_labels: M element list, where each entry is a string indicating
    the predicted category for each testing image
    """

    pred_labels = []

    #############################################################################
    # TODO: YOUR CODE HERE
    #############################################################################
    print('tnaket2')
    D = cdist(test_image_feats, train_image_feats, metric= 'correlation')
    sorted_indices = np.argsort(D, axis= 1)
    knns = sorted_indices[:,0:k]
    pred_labels = np.array(train_labels)[knns.astype(int)]
    pred_labels = stats.mode(pred_labels,axis=1)[0].flatten()
    pred_labels = pred_labels.tolist()
    print(pred_labels)
    #############################################################################
    #                             END OF YOUR CODE
    #############################################################################
    return pred_labels


def kmeans(feature_vectors, k, max_iter=100):
    """
    Implement the k-means algorithm in this function. Initialize your centroids
    with random *unique* points from the input data, and repeat over the
    following process:
    1. calculate the distances from data points to the centroids
    2. assign them labels based on the distance - these are the clusters
    3. re-compute the centroids from the labeled clusters

    Please note that you are NOT allowed to use any library functions like
    vq.kmeans from scipy or kmeans from vlfeat to do the computation!

    Useful functions:
    -   np.random.randint
    -   np.linalg.norm
    -   pairwise_distances - implemented above
    -   np.argmin

    Args:
    -   feature_vectors: the input data collection, a Numpy array of shape (
    N, d)
            where N is the number of features and d is the dimensionality of
            the features
    -   k: the number of centroids to generate, of type int
    -   max_iter: the total number of iterations for k-means to run, of type
    int

    Returns:
    -   centroids: the generated centroids for the input feature_vectors,
    a Numpy
            array of shape (k, d)
    """

    # dummy centroids placeholder
    centroids = None
    np.random.seed(42)
    #############################################################################
    # TODO: YOUR CODE HERE
    #############################################################################

    feature_vectors = feature_vectors.astype(np.float32)
    n,d = np.shape(feature_vectors)
    centroids = np.zeros((k,d))
    cluster_idx = np.zeros((n,1))
    for i in range(k) : 
        centroids[i] = feature_vectors[np.random.randint(0,n)]
        
    for it in range(max_iter):
        dists = pairwise_distances(feature_vectors,centroids)         
        cluster_idx = [np.argmin(dists[i]) for i in range(n)]
        cluster_idx = np.array(cluster_idx)

        #Update center
        old_centers = centroids
        K, D = np.shape(old_centers)
        centroids = np.copy(old_centers)
        cluster_idx = np.array(cluster_idx)
        i= 0
        counter = 0
        for k in range(K):
            assignment_k = np.argwhere(cluster_idx == k)
            average_K = 0
            if np.size(assignment_k) > 0:
                average_K = np.mean(feature_vectors[assignment_k,:], axis= 0)
                centroids[i] = average_K[0]
            i+=1
    

    return centroids

    #############################################################################
    #                             END OF YOUR CODE
    #############################################################################
    return centroids


def build_vocabulary(image_arrays, vocab_size=50, stride=20, max_iter=10):
    """
    This function will generate the vocabulary which will be further used
    for bag of words classification.

    To generate the vocab you first randomly sample features from the
    training set. Get SIFT features for the images using
    get_siftnet_features() method. This method takes as input the image
    tensor and x and y coordinates as arguments.
    Now cluster sampled features from all images using kmeans method
    implemented by you previously and return the vocabulary of words i.e.
    cluster centers.

    Points to note:
    *   To save computation time, you don't necessarily need to
    sample from all images, although it would be better to do so.
    *   Sample the descriptors from each image to save memory and
    speed up the clustering.
    *   For testing, you may experiment with larger
    stride so you just compute fewer points and check the result quickly.
    *   The default vocab_size of 50 is sufficient for you to get a
    decent accuracy (>40%), but you are free to experiment with other values.

    Useful functions:
    -   torch.from_numpy(img_array) for converting a numpy array to a torch
    tensor for siftnet
    -   torch.view() for reshaping the torch tensor
    -   use torch.type() or np.array(img_array).astype() for typecasting
    -   generate_sample_points() from utils.py for sampling interest points

    Useful note:
    - You will first need to convert each array in image_arrays into a float type array. Then convert each of these image arrays into a 4-D torch tensor by using torch.from_numpy(img_array) and then reshaping it to (1 x 1 x H x W) where H and W are the image height and width. You can use tensor views i.e torch.view to do the reshaping.

    Args:
    -   image_arrays: list of images in Numpy arrays, in grayscale
    -   vocab_size: size of vocabulary
    -   stride: the stride of your SIFT sampling

    Returns:
    -   vocab: This is a (vocab_size, dim) Numpy array (vocabulary). Where
    dim is the length of your SIFT descriptor. Each row is a cluster
    center/visual word.
    """

    dim = 128  # length of the SIFT descriptors that you are going to compute.
    vocab = None

    #############################################################################
    # TODO: YOUR CODE HERE
    #############################################################################
    # size = int(len(image_arrays)/10)
    # indices = np.arange(0, len(image_arrays), 1, dtype=int)
    # sample_indices = np.random.choice(indices,size= size, replace = False)
    # sample = np.take(image_arrays,sample_indices, axis = 0)
    fvs = []
    i = 0
    for img in image_arrays:
    # for img in sample:
        img = np.array(img).astype(np.float32)
        torch_img = torch.from_numpy(img)
        torch_img = torch_img.view(1, 1, img.shape[0],img.shape[1])
        torch_img = torch_img.type(torch.float32)
        xv, yv = generate_sample_points(img.shape[0],img.shape[1],stride)
        fv = get_siftnet_features(torch_img, xv, yv)
        fvs.append(fv)
        i+=1
    fvs = np.vstack(fvs).astype(np.float32)
    vocab = kmeans(fvs, vocab_size, max_iter)
    #############################################################################
    #                             END OF YOUR CODE
    #############################################################################

    return vocab


def kmeans_quantize(raw_data_pts, centroids):
    """
    Implement the k-means quantization in this function. Given the input
    data and the centroids, assign each of the data entry to the closest
    centroid.

    Useful functions:
    -   pairwise_distances
    -   np.argmin

    Args:
    -   feature_vectors: the input data collection, a Numpy array of shape (
    N, d) where N is the number of input data, and d is the dimension of it,
    given the standard SIFT descriptor, d  = 128
    -   centroids: the generated centroids for the input feature_vectors,
    a Numpy
            array of shape (k, D)

    Returns:
    -   indices: the index of the centroid which is closest to the data points,
            a Numpy array of shape (N, )

    """
    cluster_idx = None

    #############################################################################
    # TODO: YOUR CODE HERE
    #############################################################################

    dists = pairwise_distances(raw_data_pts,centroids) 
    n,_ = np.shape(raw_data_pts)
    cluster_idx = np.zeros((n,1))
    cluster_idx = [np.argmin(dists[i]) for i in range(n)]
    cluster_idx = np.array(cluster_idx)

    #############################################################################
    #                             END OF YOUR CODE
    #############################################################################
    return cluster_idx


def get_bags_of_sifts(image_arrays, vocabulary, stride=5):
    """
    You will want to construct SIFT features here in the same way you
    did in build_vocabulary() (except for possibly changing the sampling
    rate) and then assign each local feature to its nearest cluster center
    and build a histogram indicating how many times each cluster was used.
    Don't forget to normalize the histogram, or else a larger image with more
    SIFT features will look very different from a smaller version of the same
    image.

    Useful functions:
    -   torch.from_numpy(img_array) for converting a numpy array to a torch
    tensor for siftnet. 
    -   torch.view() for reshaping the torch tensor
    -   use torch.type() or np.array(img_array).astype() for typecasting
    -   generate_sample_points() from utils.py for sampling interest points
    -   get_siftnet_features() from SIFTNet: you can pass in the image
    tensor in grayscale, together with the sampled x and y positions to
    obtain the SIFT features
    -   np.histogram() : easy way to help you calculate for a particular
    image, how is the visual words span across the vocab. Check https://numpy.org/doc/stable/reference/generated/numpy.histogram.html for examples on how to use a histogram on an input array.
    -   np.linalg.norm() for normalizing the histogram


    Useful note:
    - You will first need to convert each array in image_arrays into a float type array. Then convert each of these image arrays into a 4-D torch tensor by using torch.from_numpy(img_array) and then reshaping it to (1 x 1 x H x W) where H and W are the image height and width. You can use tensor views i.e torch.view to do the reshaping.

    Args:
    -   image_arrays: A list of N PIL Image objects
    -   vocabulary: A numpy array of dimensions: vocab_size x 128 where each
    row is a kmeans centroid or visual word.
    -   stride: same functionality as the stride in build_vocabulary().

    Returns:
    -   image_feats: N x d matrix, where d is the dimensionality of the
    feature representation. In this case, d will equal the number of
    clusters or equivalently the number of entries in each image's histogram
    (vocab_size) below.
    """
    # load vocabulary
    vocab = vocabulary

    vocab_size = len(vocab)
    num_images = len(image_arrays)

    feats = np.zeros((num_images, vocab_size))

    # size = int(len(image_arrays)/10)
    # indices = np.arange(0, len(image_arrays), 1, dtype=int)
    # sample_indices = np.random.choice(indices,size= size, replace = False)
    # sample = np.take(image_arrays,sample_indices, axis = 0)

    for i, img in enumerate(image_arrays):
        img = np.array(img).astype(float)
        torch_img = torch.from_numpy(img)
        torch_img = torch_img.view(1, 1, img.shape[0],img.shape[1])
        torch_img = torch_img.type(torch.float32)
        xv, yv = generate_sample_points(img.shape[0],img.shape[1],stride)
        fv = get_siftnet_features(torch_img, xv, yv)
        histogram = np.zeros(vocab_size)
        distances = pairwise_distances(fv, vocab)  
        closest_vocabulary = np.argsort(distances, axis=1)[:,0]
        indices, counts = np.unique(closest_vocabulary, return_counts=True)
        histogram[indices] += counts
        histogram = histogram / np.linalg.norm(histogram)
        feats[i] = histogram

    #############################################################################
    # TODO: YOUR CODE HERE
    #############################################################################

    

    #############################################################################
    #                             END OF YOUR CODE
    #############################################################################

    return feats
