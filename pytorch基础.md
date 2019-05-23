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
