# import libs
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T


class DQN(nn.Module):

    """
        Our model will be a convolutional neural network that takes in the difference between the
        current and previous screen patches. It has two outputs, representing Q(s,left) and Q(s,right)
        (where s is the input to the network). In effect, the network is trying to predict the expected
        return of taking each action given the current input.

        ...

        Attributes
        ----------
        conv1 : nn.Sequential
            A sequential model that receives a input of size 3 and has a output of size 16, also has a BatchNorm and Dropout.
        conv2 : nn.Sequential
            A sequential model that receives a input of size 3 and has a output of size 16, also has a BatchNorm and Dropout.
        conv3 : nn.Sequential
            A sequential model that receives a input of size 3 and has a output of size 16, also has a BatchNorm and Dropout.
        head  : nn.Linear
            A linear layer that outputs the model prediction.

        Methods
        -------
        conv2d_size_out()
            Number of Linear input connections depends on output of conv2d layers and therefore the input image size, so compute it.
        forward()
            Called with either one element to determine next action, or a batch during optimization.

    """
    def __init__(self, h, w, outputs):
        
        super(DQN, self).__init__()

        self.conv1 = nn.Sequential(
                        nn.Conv2d(3, 16, kernel_size=5, stride=2),
                        nn.BatchNorm2d(16),
                        nn.ReLU(inplace=True),
                        nn.Dropout(0.2),
                    )

        self.conv2 = nn.Sequential(
                        nn.Conv2d(16, 32, kernel_size=5, stride=2),
                        nn.BatchNorm2d(32),
                        nn.ReLU(inplace=True),
                        nn.Dropout(0.2),
                    )

        self.conv3 = nn.Sequential(
                        nn.Conv2d(32, 32, kernel_size=5, stride=2),
                        nn.BatchNorm2d(32),
                        nn.ReLU(inplace=True),
                        nn.Dropout(0.2),    
                    )
 
        
        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            """
                Number of Linear input connections depends on output of conv2d layers,
                and therefore the input image size, so compute it.

                :input: size        -  a single number or a tuple
                :input: kernel_size -  refers to the widthxheight of the filter mask, a single number or a tuple
                :input: stride      -  controls the stride for the cross-correlation, a single number or a tuple

                :return: the size of the conv output (--> (size - (kernel_size - 1) - 1) // stride  + 1 <--)

            """
            return (size - (kernel_size - 1) - 1) // stride  + 1

        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32
        
        self.head = nn.Linear(linear_input_size, outputs)

    
    def forward(self, x):
        """
            Called with either one element to determine next action, or a batch during optimization.

            :input: x - the input data that go trough the model to predict an action.
            :return: tensor([[left0exp,right0exp]...])

        """
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        
        return self.head(x.view(x.size(0), -1))
