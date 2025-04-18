import numpy as np
from math import cos, sin
import itk
import nibabel as nib
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import time
from mpi4py import MPI
import cProfile
from numba import njit, prange

profiler = cProfile.Profile()

PROFILE = False  # 是否开启性能分析

# MPI初始化
comm = MPI.COMM_WORLD
rank = comm.Get_rank()  # 获取当前进程的rank
size = comm.Get_size()  # 获取总进程数


@njit(parallel=True)
def fast_get_new_img(resource, invM, shape1, shape2, shape3):
    new_img = np.empty((shape1, shape2, shape3), dtype=resource.dtype)
    for i in prange(shape1):
        for j in range(shape2):
            for k in range(shape3):
                pos = invM @ np.array([i, j, k, 1.0])
                x, y, z = pos[0], pos[1], pos[2]
                if 0 <= x < shape1 and 0 <= y < shape2 and 0 <= z < shape3:
                    new_img[i, j, k] = resource[int(x), int(y), int(z)]
                else:
                    new_img[i, j, k] = resource[i, j, k]
    return new_img


class Realign:
    def __init__(self, path, sigma=1, iteration=1, order=1, step = 2, alpha = 1, beta = 1.2):
        '''初始化(initialization)\n
            输入参数(parameter):\n
            path:文件路径(file path)
        '''
        # 读取数据(load data)
        self.path = path
        self.img = itk.imread(path)
        self.img_data = itk.GetArrayFromImage(self.img)
        # if rank == 0:
            # self.img_data = itk.GetArrayFromImage(self.img)
        # else:
        #     self.img_data = None
        # self.img_data = comm.bcast(self.img_data)
        # 记录数据信息(record data information)
        self.shape = self.img_data.shape
        spacing = self.img.GetSpacing()  # 体素间距(voxel spacing)
        spacing_array = np.array(spacing)  
        spacing_array = spacing_array[::-1]  # 倒序(reverse order)itk[读取的数据与nibabel读取的数据的坐标轴顺序不同]
        self.affine = np.diag(spacing_array)  # 仿射矩阵(affine matrix)
        # 默认参数设置及预处理(default parameter setting and preprocessing)
        self.iteration = iteration  # 迭代次数(iteration times)
        self.interpolator = itk.BSplineInterpolateImageFunction.New(self.img)  # B样条插值器(B-spline interpolation)
        self.interpolator.SetSplineOrder(order)# 设置B样条插值阶数(set B-spline interpolation order)
        self.img_data_gauss = gaussian_filter(self.img_data, sigma)# 高斯平滑(gaussian smooth)
        self.img_gauss = itk.image_from_array(self.img_data_gauss)

        self.step = step  # 选点间隔(point interval)
        self.x, self.y, self.z = int(self.shape[1] /self.affine[1,1]//self.step) , int(self.shape[2]/self.affine[2,2]//self.step) , int(self.shape[3]/self.affine[3,3]//self.step)
        # 初始旋转平移参数(rotation and translation parameters)
        self.parameter = np.zeros((self.shape[0],7))
        
        self.alpha = alpha # 迭代过程的收敛系数 
        self.beta = beta # 每张图的初始缩小系数 
        
        # print("图片载入成功,请耐心等待:) \nimage loaded successfully,please wait patiently")
    
    def set_gussian(self, sigma):
        '''
        设置高斯平滑参数(set gaussian parameter)\n
        输入参数(parameter):\n
        sigma:高斯核的标准差(sigma of gaussian kernel)
        '''
        self.img_data = gaussian_filter(self.img_data, sigma)
        
    def set_iteration(self, iteration):
        '''
        设置迭代次数(set iteration times)\n
        输入参数(parameter):\n
        iteration:迭代次数(iteration times)'''
        self.iteration = iteration

    def set_order(self, order):
        '''
        设置B样条插值阶数(set B-spline interpolation order)\n
        输入参数(parameter):\n
        order:B样条插值阶数(B-spline interpolation order)'''
        self.interpolator.SetSplineOrder(order)

    def set_step(self, step):
        '''
        设置选点间隔(set point interval)\n
        输入参数(parameter):\n
        step:选点间隔(point interval)'''
        self.step = step
        self.x, self.y, self.z = int(self.shape[1] /self.affine[1,1]//self.step) , int(self.shape[2]/self.affine[2,2]//self.step) , int(self.shape[3]/self.affine[3,3]//self.step)
        
    def set_alpha(self, alpha):
        '''
        设置迭代过程的收敛系数(set the parameter of constriction during iteration)(0< alpha < 1)\n
        输入参数(parameter):\n
        alpha:迭代过程的收敛系数(the parameter of constriction during iteration)
        '''
        self.alpha = alpha
        
    def set_beta(self, beta):
        '''
        设置每张图的初始缩小系数(set the initial reduction parameter of every graph)(beta > 1)\n
        输入参数(parameter):\n
        beta:每张图的初始缩小系数(the initial reduction parameter of every graph)
        '''
        self.beta = beta

    def v2d(self):
        '''
        体素坐标到笛卡尔坐标的转换,并将坐标原点移到图像中心\n
        (voxel to descartes coordinate system\n
        Move the coordinate origin to the image center)\n
        输出(output):\n
        descartes:坐标系转换矩阵(coordinate system transformation matrix)
        '''
        descartes = np.zeros((4, 4))
        descartes[:3, :3] = self.affine[1:, 1:]
        shape1 = np.asarray(self.shape)
        shape1=np.append(shape1[1:],0)
        descartes[:, 3] = -0.5 * shape1
        descartes[3,3]=1
        return descartes

    def rigid(self, q):
        '''
        刚体变换(Rigid body transformation)\n
        输入参数(parameter):\n
        q:旋转平移参数(rotation and translation parameters)
        输出(output):\n
        M:刚体变换矩阵(rigid body transformation matrix)'''
        # 平移矩阵(translation matrix)
        T = np.array([[1, 0, 0, q[1]],
                      [0, 1, 0, q[2]],
                      [0, 0, 1, q[3]],
                      [0, 0, 0, 1]])
        # 旋转矩阵(Rotation matrix)
        R_x = np.array([[1, 0, 0, 0],
                        [0, cos(q[4]), sin(q[4]), 0],
                        [0, -sin(q[4]), cos(q[4]), 0],
                        [0, 0, 0, 1]])
        R_y = np.array([[cos(q[5]), 0, sin(q[5]), 0],
                        [0, 1, 0, 0],
                        [-sin(q[5]), 0, cos(q[5]), 0],
                        [0, 0, 0, 1]])
        R_z = np.array([[cos(q[6]), sin(q[6]), 0, 0],
                        [-sin(q[6]), cos(q[6]), 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]])
        R = R_x @ R_y @ R_z

        M_r = T @ R
        # 坐标系转换矩阵(coordinate system transformation matrix)
        coor = self.v2d()
        return np.linalg.inv(coor) @ M_r @ coor

    def get_new_img(self, resource, q, pic_num):
        M = self.rigid(q)
        if np.linalg.det(M) <= 1e-10:
            invM = np.linalg.pinv(M)
        else:
            invM = np.linalg.inv(M)
        shape1, shape2, shape3 = resource.shape
        return fast_get_new_img(resource, invM, shape1, shape2, shape3)

    def use_inter_evaluate(self, point, interpolator, reference, n_i, n_j, n_k):
        # return (interpolator.Evaluate(point)-reference[n_i][n_j][n_k])
        return interpolator.Evaluate(point)-reference[n_i][n_j][n_k]
        
    def use_inter_evaluate_derivative(self, point, interpolator):
        return interpolator.EvaluateDerivative(point)

    def iterate(self, reference,resource, q,pic_num):
        '''
        高斯牛顿迭代(gauss-newton iterate)\n
        输入参数(parameter):\n
        resource:原始图像(raw image)\n
        reference:参考图像(reference image)\n
        q:旋转平移参数(rotation and translation parameters)\n
        输出(output):\n
        q:更新后的旋转平移参数(new rotation and translation parameters)\n
        bi:残差(residual)\n
        b_diff:偏导矩阵(derivative matrix)
        '''

        interpolator = self.interpolator  # B样条插值(B-spline interpolation)
        step = self.step  # 选点间隔(point interval)

        
        total = self.x * self.y * self.z
        unit = total // size  # 每个进程处理的图像数量
        remain = total % size  # 剩余的图像数量
        
        if rank == 0:
            start = 0
        else:
            start = rank * unit + remain
        
        end = (rank + 1) * unit + remain
        
        bi = np.zeros(end - start)  # 残差 (residual)   
        b_diff = np.zeros((end - start, 7))  # 偏导矩阵 (derivative matrix)

        M = self.rigid(q)
        M[3,3]=0.0
        if np.linalg.det(M) <= 1e-10:
            M = np.linalg.pinv(M)
        else:
            M = np.linalg.inv(M)
            
        # 处理每个进程分配的图像
        for index in range(start, end):
            i = index // (self.y * self.z)
            j = (index // self.z) % self.y
            k = index % self.z
            
            n_i = int(i*step * self.affine[1, 1]) if int(i*step * self.affine[1, 1]) else self.shape[1] - 1
            n_j = int(j*step * self.affine[2, 2]) if int(j*step * self.affine[2, 2]) else self.shape[2] - 1
            n_k = int(k*step * self.affine[3, 3]) if int(k*step * self.affine[3, 3]) else self.shape[3] - 1

            interpo_pos = M @ [n_i, n_j, n_k, 1]  # 变换后的坐标(transformed coordinate)

            point = itk.Point[itk.D, 4]()
            if 0 <= interpo_pos[0] < self.shape[1] and 0 <= interpo_pos[1] < self.shape[2] and 0 <= interpo_pos[2] < self.shape[3]:  # 判断是否在范围内
                point[0] = pic_num
                point[1] = interpo_pos[0]
                point[2] = interpo_pos[1]
                point[3] = interpo_pos[2]
            else:
                point[1] = n_i
                point[2] = n_j
                point[3] = n_k
                point[0] = pic_num
            # bi[index - start] = (interpolator.Evaluate(point)-reference[n_i][n_j][n_k])**2
            bi[index - start] = self.use_inter_evaluate(point, interpolator, reference, n_i, n_j, n_k)
            # tem=interpolator.Evaluate(point)-reference[n_i][n_j][n_k]
            # derivative = interpolator.EvaluateDerivative(point)
            derivative = self.use_inter_evaluate_derivative(point, interpolator)
            diff_x = derivative[1]
            diff_y = derivative[2]
            diff_z = derivative[3]
            (a,b,c)=self.shape[1:]
            b_diff[index - start][1:4] = diff_x, diff_y, diff_z
            # b_diff[index - start][1:4] = derivative[1:4]
            b_diff[index - start][4] = (diff_y*(-0.5*self.affine[0][0]*a*(sin(q[4])*sin(q[6]) - sin(q[5])*cos(q[4])*cos(q[6]))/self.affine[1][1] + self.affine[0][0]*(sin(q[4])*sin(q[6]) - sin(q[5])*cos(q[4])*cos(q[6]))/self.affine[1][1] - 0.5*b*(-sin(q[4])*cos(q[6]) - sin(q[5])*sin(q[6])*cos(q[4])) - sin(q[4])*cos(q[6]) - sin(q[5])*sin(q[6])*cos(q[4]) - 0.5*self.affine[2][2]*c*cos(q[4])*cos(q[5])/self.affine[1][1] + self.affine[2][2]*cos(q[4])*cos(q[5])/self.affine[1][1]) + \
                diff_z*(-0.5*self.affine[0][0]*a*(sin(q[4])*sin(q[5])*cos(q[6]) + sin(q[6])*cos(q[4]))/self.affine[2][2] + self.affine[0][0]*(sin(q[4])*sin(q[5])*cos(q[6]) + sin(q[6])*cos(q[4]))/self.affine[2][2] - 0.5*self.affine[1][1]*b*(
                    sin(q[4])*sin(q[5])*sin(q[6]) - cos(q[4])*cos(q[6]))/self.affine[2][2] + self.affine[1][1]*(sin(q[4])*sin(q[5])*sin(q[6]) - cos(q[4])*cos(q[6]))/self.affine[2][2] + 0.5*c*sin(q[4])*cos(q[5]) - sin(q[4])*cos(q[5]))) #*tem
            b_diff[index - start][5] = (diff_x*(0.5*a*sin(q[5])*cos(q[6]) - sin(q[5])*cos(q[6]) + 0.5*self.affine[1][1]*b*sin(q[5])*sin(q[6])/self.affine[0][0] - self.affine[1][1]*sin(q[5])*sin(q[6])/self.affine[0][0] - 0.5*self.affine[2][2]*c*cos(q[5])/self.affine[0][0] + self.affine[2][2]*cos(q[5])/self.affine[0][0]) +\
                diff_y*(0.5*self.affine[0][0]*a*sin(q[4])*cos(q[5])*cos(q[6])/self.affine[1][1] - self.affine[0][0]*sin(q[4])*cos(q[5])*cos(q[6])/self.affine[1][1] + 0.5*b*sin(q[4])*sin(q[6])*cos(q[5]) - sin(q[4])*sin(q[6])*cos(q[5]) + 0.5*self.affine[2][2]*c*sin(q[4])*sin(q[5])/self.affine[1][1] - self.affine[2][2]*sin(q[4])*sin(q[5])/self.affine[1][1]) +\
                diff_z*(0.5*self.affine[0][0]*a*cos(q[4])*cos(q[5])*cos(q[6])/self.affine[2][2] - self.affine[0][0]*cos(q[4])*cos(q[5])*cos(q[6])/self.affine[2][2] + 0.5*self.affine[1][1]*b*sin(
                    q[6])*cos(q[4])*cos(q[5])/self.affine[2][2] - self.affine[1][1]*sin(q[6])*cos(q[4])*cos(q[5])/self.affine[2][2] + 0.5*c*sin(q[5])*cos(q[4]) - sin(q[5])*cos(q[4])))#*tem
            b_diff[index - start][6] = (diff_x*(0.5*a*sin(q[6])*cos(q[5]) - sin(q[6])*cos(q[5]) - 0.5*self.affine[1][1]*b*cos(q[5])*cos(q[6])/self.affine[0][0] + self.affine[1][1]*cos(q[5])*cos(q[6])/self.affine[0][0]) +\
                diff_y*(-0.5*self.affine[0][0]*a*(sin(q[4])*sin(q[5])*sin(q[6]) - cos(q[4])*cos(q[6]))/self.affine[1][1] + self.affine[0][0]*(sin(q[4])*sin(q[5])*sin(q[6]) - cos(q[4])*cos(q[6]))/self.affine[1][1] - 0.5*b*(-sin(q[4])*sin(q[5])*cos(q[6]) - sin(q[6])*cos(q[4])) - sin(q[4])*sin(q[5])*cos(q[6]) - sin(q[6])*cos(q[4])) +\
                diff_z*(-0.5*self.affine[0][0]*a*(sin(q[4])*cos(q[6]) + sin(q[5])*sin(q[6])*cos(q[4]))/self.affine[2][2] + self.affine[0][0]*(sin(q[4])*cos(q[6]) + sin(q[5])*sin(q[6])*cos(q[4]))/self.affine[2][2] - 0.5*self.affine[1][1]*b*(sin(q[4])*sin(q[6]) - sin(q[5])*cos(q[4])*cos(q[6]))/self.affine[2][2] + self.affine[1][1]*(sin(q[4])*sin(q[6]) - sin(q[5])*cos(q[4])*cos(q[6]))/self.affine[2][2]))#*tem
            b_diff[index - start][0] = 1
            
            # print('tem:',tem)
            # break
            
        # 汇总所有进程处理的结果
        gathered_bi = np.zeros(unit * size)  # 创建一个空数组用于存储汇总结果
        gathered_b_diff = np.zeros((unit * size, 7))  # 创建一个空数组用于存储汇总结果
        
        if rank == 0:
            comm.Gather(bi[remain:], gathered_bi, root=0)  # 将每个进程处理的结果汇总到主进程
            comm.Gather(b_diff[remain:], gathered_b_diff, root=0)  # 将每个进程处理的结果汇总到主进程
        else:                
            comm.Gather(bi, gathered_bi, root=0)  # 将每个进程处理的结果汇总到主进程
            comm.Gather(b_diff, gathered_b_diff, root=0)  # 将每个进程处理的结果汇总到主进程
        return gathered_bi, gathered_b_diff

    def reslicing(self,name):
        '''
        用刚体变换参数将图片进行重采样对齐\n
        Resampling and aligning the image with the rigid-body transform parameter\n
        output: 对齐后的图片(Aligned image)
        '''
        self.interpolator = itk.BSplineInterpolateImageFunction.New(self.img_gauss)  # B样条插值器(B-spline interpolation)
        
        # print("开始重采样对齐(注意，需要很长时间，建议先去干点别的事)\nreslicing")
        new = np.ndarray(self.shape)
        
        unit = self.shape[0] // size  # 每个进程处理的图像数量
        remain = self.shape[0] % size  # 剩余的图像数量
        
        if rank == 0:
            start = 0
        else:
            start = rank * unit + remain
        end = (rank + 1) * unit + remain  # 每个进程处理的结束索引
        
        # 处理每个进程分配的图像
        seperate_img = np.empty(shape=(end - start,) + self.img_data_gauss.shape[1:])  # 创建一个空数组用于存储每个进程处理的结果
        for picture in range(start, end):
            seperate_img[picture-start,:, :, :] = self.get_new_img(self.img_data_gauss[picture,:, :, :], self.parameter[picture],picture)
            # print(f"rank {rank}: 进度{(picture - start)*100/seperate_img.shape[0]}%")
        # 汇总所有进程处理的结果
        gathered_img = np.empty((unit * size, ) + tuple(self.shape[1:]))  # 创建一个空数组用于存储汇总结果
        
        if rank == 0:
            comm.Gather(seperate_img[remain:], gathered_img, root=0)
        else:
            comm.Gather(seperate_img, gathered_img, root=0)  # 将每个进程处理的结果汇总到主进程
        
        # 主进程保存结果
        if rank == 0:
            self.save_img(gathered_img, name)
            print("重采样完成,请关闭图片\nreslicing complete")

        return new

    def estimate(self):
        '''
        估计刚体变换参数\n
        Estimate the rigid-body transform parameter\n
        output: 刚体变换参数(Rigid-body transform parameter)'''
        
        # print("开始估计刚体变换参数\nestimating")
        q=np.zeros(7)
        q[0] = 1
        for picture in range(1, self.shape[0]):
            # 高斯牛顿迭代
            q=q / self.beta
            for _ in range(self.iteration):
                
                q[0] = 1
                
                comm.Bcast(q, root=0)  # 广播参数到所有进程
                
                (b, diff_b) = self.iterate(self.img_data[0,:, :, :],self.img_data[picture,:,:,:], q,picture)
                
                if rank == 0:
                    tmp =diff_b.T@diff_b
                    if np.linalg.det(tmp) <= 1e-10:  # 判断矩阵是否可逆
                        # 矩阵不可逆时用伪逆
                        q -= self.alpha * np.linalg.pinv(tmp)@diff_b.T@b
                    else:
                        q -= self.alpha * np.linalg.inv(tmp)@diff_b.T@b
            q[0] = 1
            q[q>40]=0
            q[q<-40]=0
            self.parameter[picture] = q/10
        #     print(f"进度{picture*100/self.shape[0]}%", end="\r")
        # print("刚体变换参数估计完成,若需要进行reslicing，请先关闭图像\nestimation complete")
        
        if rank == 0:
            self.draw_parameter()
        self.parameter = comm.bcast(self.parameter, root=0)  # 广播参数到所有进程
        return self.parameter

    def save_img(self,img, name):
        '''
        保存图片\n
        Save image\n
        input: 图片数据(Image data), 图片名(Image name)
        '''
        img = np.transpose(img,(3,2,1,0)) # 将图片转换为nii格式就要按nii的顺序排列
        img = nib.Nifti1Image(img,self.affine)
        name = f"{name}.nii.gz"
        nib.save(img, name)
        print(f"图片{name}保存成功")
        
    def draw_parameter(self):
        '''
        绘制刚体变换参数\n
        Draw rigid-body transform parameter\n
        '''
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))
        x = np.arange(self.shape[0])
        ax1.plot(x, self.parameter[:,1], label='X', color='red')
        ax1.plot(x, self.parameter[:,2], label='Y', color='green')
        ax1.plot(x, self.parameter[:,3], label='Z', color='blue')
        ax1.set_title('translation')
        ax1.set_xlabel('image')
        ax1.set_ylabel('mm')
        ax1.legend()
        ax2.plot(x, self.parameter[:,4], label='pitch', color='purple')
        ax2.plot(x, self.parameter[:,5], label='roll', color='orange')
        ax2.plot(x, self.parameter[:,6], label='yaw', color='brown')
        ax2.set_title('rotation')
        ax2.set_xlabel('image')
        ax2.set_ylabel('degrees')
        ax2.legend()
        fig.tight_layout()
        plt.savefig("parameter.png")
        

if __name__ == "__main__":    
    # 设置numba线程数（如需自定义线程数，设置环境变量NUMBA_NUM_THREADS）
    # 例如: os.environ["NUMBA_NUM_THREADS"] = "4" 需在import numba前设置
    import os
    os.environ["NUMBA_NUM_THREADS"] = "8"  # 可取消注释并设置线程数
    # print("Numba线程数:", config.NUMBA_DEFAULT_NUM_THREADS)  # 查看当前线程数

    profiler = cProfile.Profile()
    profiler.enable()

    start = time.perf_counter()
    
    if rank == 0 and PROFILE:
        profiler.enable()

    
    ## 测验代码
    realign = Realign('./data/sub-Ey153_ses-3_task-rest_acq-EPI_run-2_bold.nii.gz')

    realign.set_step(2)
    realign.set_iteration(2)
    realign.set_order(3)
    realign.set_gussian(2)
    
    realign.set_alpha(0.15)
    realign.set_beta(1.65)
    
    # 获得刚体变换参数并绘制头动曲线
    # if rank == 0:
    realign.estimate()
    # 获得重采样后的图像
    realign.reslicing("picture")
    
    MPI.Finalize()

    if rank == 0 and PROFILE:
        profiler.disable()
        profiler.dump_stats("profile.prof")
    
    end = time.perf_counter()
    
    if rank == 0:
        print(f"总耗时{end-start}秒") 