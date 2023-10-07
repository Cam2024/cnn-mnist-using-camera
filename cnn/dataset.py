import torchvision
import torchvision.transforms as transforms

# 定义数据预处理
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

# 加载训练数据集
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)

# 加载测试数据集
testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
