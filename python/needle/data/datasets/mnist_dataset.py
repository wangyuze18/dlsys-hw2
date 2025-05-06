from typing import List, Optional
from ..data_basic import Dataset
import numpy as np
import gzip

class MNISTDataset(Dataset):
    def __init__(
        self,
        image_filename: str,
        label_filename: str,
        transforms: Optional[List] = None,
    ):
        ### BEGIN YOUR SOLUTION
        super().__init__(transforms)
        self.X,self.y = parse_mnist(image_filename,label_filename)
        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        ### BEGIN YOUR SOLUTION
        x = self.apply_transforms(self.X[index].reshape(28, 28, -1))
        return x.reshape(-1, 28*28), self.y[index]
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        ### BEGIN YOUR SOLUTION
        return len(self.y)
        ### END YOUR SOLUTION


def parse_mnist(image_filename, label_filename):
    """ Read an images and labels file in MNIST format.  See this page:
    http://yann.lecun.com/exdb/mnist/ for a description of the file format.

    Args:
        image_filename (str): name of gzipped images file in MNIST format
        label_filename (str): name of gzipped labels file in MNIST format

    Returns:
        Tuple (X,y):
            X (numpy.ndarray[np.float32]): 2D numpy array containing the loaded 
                data.  The dimensionality of the data should be 
                (num_examples x input_dim) where 'input_dim' is the full 
                dimension of the data, e.g., since MNIST images are 28x28, it 
                will be 784.  Values should be of type np.float32, and the data 
                should be normalized to have a minimum value of 0.0 and a 
                maximum value of 1.0 (i.e., scale original values of 0 to 0.0 
                and 255 to 1.0).

            y (numpy.ndarray[dtype=np.uint8]): 1D numpy array containing the
                labels of the examples.  Values should be of type np.uint8 and
                for MNIST will contain the values 0-9.
    """
    ### BEGIN YOUR CODE
    with gzip.open(image_filename,'rb') as f:
      content = f.read()
      magic_number_image = int.from_bytes(content[:4])
      number_of_images = int.from_bytes(content[4:8])
      rows = int.from_bytes(content[8:12])
      columns = int.from_bytes(content[12:16])
      X = np.frombuffer(content[16:],dtype=np.uint8).astype(dtype=np.float32).reshape(number_of_images,rows*columns)/255.0
    with gzip.open(label_filename,'rb') as f:
      content = f.read()
      magic_number_label = int.from_bytes(content[:4])
      number_of_labels = int.from_bytes(content[4:8])
      y = np.frombuffer(content[8:],dtype=np.uint8).reshape(number_of_labels)
    return (X,y)
    ### END YOUR CODE