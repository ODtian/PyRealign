# 图像配准代码优化详解

本文对比分析了同一图像配准项目的两个版本：初始版本与最终优化版本，详细整理了每项优化措施的具体代码变更、所采用的技术工具、原理以及性能提升的原因。

---

## 优化一：多进程并行计算（MPI）

###  相关库：`mpi4py`

###  优化说明：
初始版本完全串行处理每张图像，效率低；最终版本引入 MPI，将图像帧的处理任务在多个进程中并行执行。

### 初始版本代码片段：
```python
for picture in range(self.shape[0]):
    new[picture,:, :, :] = self.get_new_img(
        self.img_data_gauss[picture,:,:,:],
        self.parameter[picture],
        picture
    )
```
### 最终代码版本
```python
unit = self.shape[0] // size
remain = self.shape[0] % size
start = rank * unit + remain if rank != 0 else 0
end = (rank + 1) * unit + remain

seperate_img = np.empty((end - start, ...) )
for picture in range(start, end):
    seperate_img[picture-start, :, :, :] = self.get_new_img(...)

comm.Gather(seperate_img, gathered_img, root=0)
```
### 加速原理
使用 mpi4py 将计算分发到多个进程（可分布于多核心或多节点），有效利用 CPU 多核资源，加速线性帧数增长所带来的计算开销。

## 优化二：JIT 加速与并行循环（Numba）
### 相关库： `numba`
### 优化说明：
三重 for-loop 是图像处理中的瓶颈，初始版本直接使用原生 Python 实现，执行效率极低。最终版本使用 @njit(parallel=True) 修饰函数，编译为高性能本地代码，并开启线程级并行。
### 初始版本代码片段：
```python
def get_new_img(...):
    for i in range(self.shape[1]):
        for j in range(self.shape[2]):
            for k in range(self.shape[3]):
                interpo_pos = ...
                new_img[i, j, k] = ...
```
### 最终版本
```python
 @njit(parallel=True)
def fast_get_new_img(resource, invM, shape1, shape2, shape3):
    new_img = np.empty((shape1, shape2, shape3))
    for i in prange(shape1):
        for j in range(shape2):
            for k in range(shape3):
                pos = invM @ np.array([i, j, k, 1.0])
                ...
```
### 加速原理
@njit: 使用 Numba 的 JIT 编译器将 Python 函数编译为本地代码。
prange: 开启多线程并行循环，加快大规模像素点处理速度。

## 优化三：矩阵逆计算复用
### 优化说明：
刚体变换矩阵 M 每次在像素级调用 np.linalg.inv() 会极大拖慢计算。最终版本在外部计算一次 invM 后传入内部使用。
### 初始版本代码片段
```python
for i in range(...):
    M = self.rigid(q)
    interpo_pos = np.linalg.inv(M) @ [i, j, k, 1]
```
### 最终版本
```python
M = self.rigid(q)
invM = np.linalg.inv(M) if np.linalg.det(M) > 1e-10 else np.linalg.pinv(M)
...
for i in prange(...):
    pos = invM @ np.array([i, j, k, 1])
```
### 加速原理
矩阵逆是一项昂贵操作，将其提至循环外只算一次，大幅减少冗余计算。

## 优化四：并行数据划分与汇总
### 优化说明：
所有点都放进一个数组在每个进程中处理会造成内存浪费与冲突。最终版本每个进程只处理 start ~ end 范围的数据，最后用 comm.Gather 汇总。
### 初始版本代码片段
```python
bi = np.zeros(self.x*self.y*self.z)
b_diff = np.zeros((self.x*self.y*self.z, 7))
```
### 最终版本
bi = np.zeros(end - start)
b_diff = np.zeros((end - start, 7))
...
comm.Gather(bi, gathered_bi, root=0)
### 加速原理
减少每个进程占用内存，只处理自身任务，避免共享状态带来的冲突。

## 优化五：主进程控制 IO 操作
### 优化说明：
初始版本所有进程都可能执行绘图和保存操作，会造成文件冲突和冗余。最终版本只允许 rank==0 的主进程执行。
### 初始版本代码片段
```python
self.save_img(...)
self.draw_parameter()
```
### 最终版本
```python
if rank == 0:
    self.save_img(...)
    self.draw_parameter()
```
### 加速原理
确保唯一写出源，避免多线程写冲突，提升稳定性。

## 优化六：插值与导数函数封装
### 优化说明：
多次使用 Evaluate() 和 EvaluateDerivative() 可封装成函数，利于统一替换为 Numba/GPU 版本。
### 最终版本
```python
def use_inter_evaluate(...):
    return interpolator.Evaluate(point) - reference[n_i][n_j][n_k]

def use_inter_evaluate_derivative(...):
    return interpolator.EvaluateDerivative(point)
```
### 加速原理
提高代码复用性与维护性，未来便于进一步向 GPU 转移或批量向量化。

## 优化七：条件判断合并精简
### 优化说明：
判断 det(M)==0 多处重复，最终统一为 <= 1e-10，逻辑更清晰。
### 加速原理
简化判断逻辑，提升代码可读性，稍有性能提升（尤其在 JIT 中）。

## 优化八：性能分析模块集成
### 优化说明：
最终版本引入 cProfile 分析器，在主进程可选启动性能分析。
### 最终版本代码片段：
```python
if rank == 0 and PROFILE:
    profiler.enable()
...
if rank == 0 and PROFILE:
    profiler.disable()
    profiler.dump_stats("profile.prof")

```