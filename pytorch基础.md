#### pytprch基础语法
pytorch的最基本的操作是Tensor<br>
不同数据类型的Tensor：torch.FloatTensor torch.DoubleTensor torch.ShortTensor torch.IntTensor toch.longTensor
* 创建一个三行两列给定元素的矩阵
```
a = torch.Tensor([[2,3],[4,8],[7,9]])
print('a is:{}'.format(a))

out:
a is:tensor([[2., 3.],
        [4., 8.],
        [7., 9.]])
```
note:torch.Tensor默认的就是torch.FloatTensor
* 创建全为0的空Tensor或者取一个正态分布作为随机初始值
```
a = torch.zeros((3,2))
print('zero Tensor:{}'.format(a))

out:
zero Tensor:tensor([[0., 0.],
        [0., 0.],
        [0., 0.]])
        
b = torch.randn((3,2))
print('normal randon is:{}'.format(b))

out:
normal randon is:tensor([[ 0.1332, -2.1152],
        [ 0.3776,  0.2737],
        [-1.5427, -0.4175]])
```

note:torch.zeros((3,2))和torch.randn((3,2))中参数是一个tuple，表示创建的矩阵的维度的大小

* Tensor与numpy之间的互相转化
```
#将Tensor转化为numpy
a = torch.Tensor([[1,2],[3,4],[5,6]])
numpy_b = a.numpy()

print('Tensor a is:{}'.format(a))
print('numpy numpy_b is:{}'.format(numpy_b))

out:
Tensor a is:tensor([[1., 2.],
        [3., 4.],
        [5., 6.]])

numpy numpy_b is:[[1. 2.]
 [3. 4.]
 [5. 6.]]
 
#将numpy转化为Tensor
a = np.array([[1,2],[3,4],[5,6]])
tensor_b = torch.from_numpy(a)
 
print('numpy a is:{}'.format(a))
print('Tensor tensor_b is:{}'.format(tensor_b))
 
out:
numpy a is:[[1 2]
[3 4]
[5 6]]
 
Tensor tensor_b is:tensor([[1, 2],
       [3, 4],
       [5, 6]], dtype=torch.int32)

#将tensor的类型转换  eg:int32->float
tensor_b = tensor_b.float()
print('Tensor tensor_b is:{}'.format(tensor_b))
 
out:
Tensor tensor_b is:tensor([[1., 2.],
        [3., 4.],
        [5., 6.]])
```
### 变量（Variable）
variable是神经网络计算图中的特有的概念，Variable提供了自动求导的功能<br>
variable的三个重要组成属性：data grad grad_fn<br>
data存储variable的tensor数值<br>
grad_fn表示如何得到这个variable的，如加减乘除法等
grad表示这个Variable反向传播梯度

```
# Create Variable
from torch.autograd import Variable#导入Variable

#创建三个变量 都是一维的
x = Variable(torch.Tensor([1]), requires_grad=True)
w = Variable(torch.Tensor([2]), requires_grad=True)
b = Variablr(torch.Tensor([3]), requires_grad=True)

#构建计算图
y = w*x+b

#计算梯度
y.backward()
print(x.grad)
print(w.grad)
print(b.grad)

out:
tensor([2.])
tensor([1.])
tensor([1.])
```
note:当我们构建Variable时，需要传入参数requires_grad=True ，这个参数表示是否对该变量进行求导，默认值为False<br>
y.backward()即自动对所有的参数进行自动求导。等价于y.backward(torch.FloatTensor([1])),对于标量而言，里面的参数可以省略不写，对于向量而言，里面必须写参数。

```
#矩阵求导
x = Variable(torch.randn(3),requires_grad=True)
y = x*2
print(y)
y.backward(torch.FloatTensor([1,1,1]))
print(x.grad)

out:
tensor([ 0.6738, -2.6373,  0.4540], grad_fn=<MulBackward>)
tensor([2., 2., 2.])
```
note:backward里面的参数值表示各个分量的梯度的权重
