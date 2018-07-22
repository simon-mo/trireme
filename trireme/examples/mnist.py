# From https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import asyncio
import sys
import logging

logger = logging.getLogger(__name__)


class MNISTNet(nn.Module):
    def __init__(self):
        super(MNISTNet, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class MnistTrainActor:
    def __init__(self, lr=0.01):
        self.net = MNISTNet()
        self.optimizer = optim.SGD(self.net.parameters(), lr=lr)

        self.forward_result = {}
        self.model_state_id = 0

        self.criterion = nn.MSELoss()

    async def __call__(self, input_batch):
        # Routing logic
        # There will be four batches
        cancel_batch = []
        re_forward_batch = []
        backward_batch = []
        new_forward_batch = []

        for inp in input_batch:
            if inp["path"] == "infer":
                new_forward_batch.append(inp)

            elif inp["path"] == "train":
                if int(inp["model state id"]) != self.model_state_id:
                    re_forward_batch.append(inp)
                backward_batch.append(inp)

            elif inp["path"] == "cancel":
                if int(inp["model state id"]) == self.model_state_id:
                    cancel_batch.append(inp)

        batch_info = f"""
        Cancel {len(cancel_batch)};
        Re-forward {len(re_forward_batch)};
        Backward {len(backward_batch)};
        Forward {len(new_forward_batch)}"""
        logger.info(f"MNIST NN is processing {batch_info}")

        self._handle_cancel(cancel_batch)
        self._handle_forward(re_forward_batch)
        await asyncio.sleep(0)
        self._handle_backward(backward_batch)
        await asyncio.sleep(0)
        return self._handle_forward(new_forward_batch)

    def _handle_cancel(self, input_batch):
        for inp in input_batch:
            self.forward_result.pop(inp["object id"])

    def _handle_backward(self, input_batch):
        if len(input_batch) == 0:
            return None

        # TODO(simon): tune learning rate according to batch size
        # TODO(simon): step per n batch size, not every invocation

        # Prepare Batch
        output_tensors = []
        target_tensors = []
        for inp in input_batch:
            forward_tensor = self.forward_result[inp["object id"]]
            forward_tensor = forward_tensor.unsqueeze(0)
            output_tensors.append(forward_tensor)

            target = torch.zeros(1, 10)
            label_digit = int(inp["label"])
            target[0, label_digit] = label_digit
            target_tensors.append(target)

        # Batch backward
        output = torch.cat(output_tensors)
        target = torch.cat(target_tensors)
        loss = self.criterion(output, target)
        loss.backward()

        self.optimizer.step()
        self.net.zero_grad()

        # Clear Cache
        self.model_state_id += 1
        self.forward_result = {}

    def _handle_forward(self, input_batch):
        if len(input_batch) == 0:
            return []

        # Convert to tensors
        tensors = []
        for inp in input_batch:
            bytes_input = inp["input"]

            # The following input processing step can go into:
            # - downloader
            # - another middleware
            # - trainer
            np_input = np.frombuffer(bytes_input, dtype=np.float32).reshape(1, 32, 32)
            tensor = torch.Tensor(np_input)
            tensor = tensor.unsqueeze(0)
            tensors.append(tensor)
        batch_inp = torch.cat(tensors)

        forward_res = self.net(batch_inp)

        returns = []
        for i, inp in enumerate(input_batch):
            # cache forward result
            self.forward_result[inp["object id"]] = forward_res[i]

            result_digit = int(torch.argmax(forward_res[i]))
            returns.append(
                {
                    "object id": inp["object id"],
                    "prediction": result_digit,
                    "model state id": self.model_state_id,
                }
            )
        return returns


if __name__ == "__main__":
    """Sanity Check to make sure the network is working. 
       For real server example, see test.py
    """

    data = datasets.MNIST(
        root="data",
        train=True,
        download=True,
        transform=transforms.Compose([transforms.Scale(32), transforms.ToTensor()]),
    )
    net = MNISTNet()
    optimizer = optim.SGD(net.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    for inp, label in data:
        inp = inp.unsqueeze(0)  # 1 sample batch
        output = net(inp)

        target = torch.zeros(1, 10)
        target[0, label] = 1

        loss = criterion(output, target)

        pred = torch.argmax(output)
        if int(label) == int(pred):
            print(".", end=" ")
        else:
            print("x", end=" ")

        sys.stdout.flush()

        net.zero_grad()
        loss.backward()
        optimizer.step()
