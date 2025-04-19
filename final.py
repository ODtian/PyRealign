import numpy as np
from math import cos, sin
import itk
import nibabel as nib
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import torch

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
        
        print("图片载入成功,请耐心等待:) \nimage loaded successfully,please wait patiently")
    
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

    def get_new_img(self, resource, q,pic_num):
        '''
        获取新图像(get new image)\n
        输入参数(parameter):\n
        resource:原始图像(raw image)\n
        q:旋转平移参数(rotation and translation parameters)\n
        输出(output):\n
        new_img:新图像(new image)'''
        new_img = np.ndarray(resource.shape)
        # 对应位置的转换
        for i in range(self.shape[1]):
            for j in range(self.shape[2]):
                for k in range(self.shape[3]):
                    M = self.rigid(q)
                    if np.linalg.det(M)==0:
                        interpo_pos = np.linalg.pinv(M)@[i, j, k, 1]
                    else:
                        interpo_pos =np.linalg.inv(M)@[i, j, k, 1]
                    if 0 <= interpo_pos[0] < self.shape[1] and 0 <= interpo_pos[1] < self.shape[2] and 0 <= interpo_pos[2] < self.shape[3]:  # 判断是否在范围内
                        new_img[i, j, k] = resource[int(interpo_pos[0]), int(interpo_pos[1]), int(interpo_pos[2])]
                    else:
                        new_img[i, j, k] = resource[i, j, k]
        return new_img

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
        bi = np.zeros(self.x*self.y*self.z)    # 残差 (residual)
        # 偏导矩阵 (derivative matrix)
        b_diff = np.zeros(((self.x*self.y*self.z), 7))
        index = 0
        for i in range(self.x):
            for j in range(self.y):
                for k in range(self.z):
                    n_i = i*step  # 对应位置的转换,获得点在原图的位置(get the position of the point in the original image)
                    n_j = j*step
                    n_k = k*step
                    M = self.rigid(q)
                    M[3,3]=0.0
                    if np.linalg.det(M)==0:
                        interpo_pos = np.linalg.pinv(M)@[n_i, n_j, n_k, 1]
                    else:
                        interpo_pos =np.linalg.inv(M)@[n_i, n_j, n_k, 1]
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
                        
                    bi[index] = (interpolator.Evaluate(point)-reference[n_i][n_j][n_k])**2
                    tem=interpolator.Evaluate(point)-reference[n_i][n_j][n_k]
                    derivative = interpolator.EvaluateDerivative(point)
                    diff_x = derivative[1]
                    diff_y = derivative[2]
                    diff_z = derivative[3]
                    (a,b,c)=self.shape[1:]
                    b_diff[index][1:4] = diff_x*tem, diff_y*tem, diff_z*tem
                    b_diff[index][4] = (diff_y*(-0.5*self.affine[0][0]*a*(sin(q[4])*sin(q[6]) - sin(q[5])*cos(q[4])*cos(q[6]))/self.affine[1][1] + self.affine[0][0]*(sin(q[4])*sin(q[6]) - sin(q[5])*cos(q[4])*cos(q[6]))/self.affine[1][1] - 0.5*b*(-sin(q[4])*cos(q[6]) - sin(q[5])*sin(q[6])*cos(q[4])) - sin(q[4])*cos(q[6]) - sin(q[5])*sin(q[6])*cos(q[4]) - 0.5*self.affine[2][2]*c*cos(q[4])*cos(q[5])/self.affine[1][1] + self.affine[2][2]*cos(q[4])*cos(q[5])/self.affine[1][1]) + \
                        diff_z*(-0.5*self.affine[0][0]*a*(sin(q[4])*sin(q[5])*cos(q[6]) + sin(q[6])*cos(q[4]))/self.affine[2][2] + self.affine[0][0]*(sin(q[4])*sin(q[5])*cos(q[6]) + sin(q[6])*cos(q[4]))/self.affine[2][2] - 0.5*self.affine[1][1]*b*(
                            sin(q[4])*sin(q[5])*sin(q[6]) - cos(q[4])*cos(q[6]))/self.affine[2][2] + self.affine[1][1]*(sin(q[4])*sin(q[5])*sin(q[6]) - cos(q[4])*cos(q[6]))/self.affine[2][2] + 0.5*c*sin(q[4])*cos(q[5]) - sin(q[4])*cos(q[5])))*tem

                    b_diff[index][5] = (diff_x*(0.5*a*sin(q[5])*cos(q[6]) - sin(q[5])*cos(q[6]) + 0.5*self.affine[1][1]*b*sin(q[5])*sin(q[6])/self.affine[0][0] - self.affine[1][1]*sin(q[5])*sin(q[6])/self.affine[0][0] - 0.5*self.affine[2][2]*c*cos(q[5])/self.affine[0][0] + self.affine[2][2]*cos(q[5])/self.affine[0][0]) +\
                        diff_y*(0.5*self.affine[0][0]*a*sin(q[4])*cos(q[5])*cos(q[6])/self.affine[1][1] - self.affine[0][0]*sin(q[4])*cos(q[5])*cos(q[6])/self.affine[1][1] + 0.5*b*sin(q[4])*sin(q[6])*cos(q[5]) - sin(q[4])*sin(q[6])*cos(q[5]) + 0.5*self.affine[2][2]*c*sin(q[4])*sin(q[5])/self.affine[1][1] - self.affine[2][2]*sin(q[4])*sin(q[5])/self.affine[1][1]) +\
                        diff_z*(0.5*self.affine[0][0]*a*cos(q[4])*cos(q[5])*cos(q[6])/self.affine[2][2] - self.affine[0][0]*cos(q[4])*cos(q[5])*cos(q[6])/self.affine[2][2] + 0.5*self.affine[1][1]*b*sin(
                            q[6])*cos(q[4])*cos(q[5])/self.affine[2][2] - self.affine[1][1]*sin(q[6])*cos(q[4])*cos(q[5])/self.affine[2][2] + 0.5*c*sin(q[5])*cos(q[4]) - sin(q[5])*cos(q[4])))*tem

                    b_diff[index][6] = (diff_x*(0.5*a*sin(q[6])*cos(q[5]) - sin(q[6])*cos(q[5]) - 0.5*self.affine[1][1]*b*cos(q[5])*cos(q[6])/self.affine[0][0] + self.affine[1][1]*cos(q[5])*cos(q[6])/self.affine[0][0]) +\
                        diff_y*(-0.5*self.affine[0][0]*a*(sin(q[4])*sin(q[5])*sin(q[6]) - cos(q[4])*cos(q[6]))/self.affine[1][1] + self.affine[0][0]*(sin(q[4])*sin(q[5])*sin(q[6]) - cos(q[4])*cos(q[6]))/self.affine[1][1] - 0.5*b*(-sin(q[4])*sin(q[5])*cos(q[6]) - sin(q[6])*cos(q[4])) - sin(q[4])*sin(q[5])*cos(q[6]) - sin(q[6])*cos(q[4])) +\
                        diff_z*(-0.5*self.affine[0][0]*a*(sin(q[4])*cos(q[6]) + sin(q[5])*sin(q[6])*cos(q[4]))/self.affine[2][2] + self.affine[0][0]*(sin(q[4])*cos(q[6]) + sin(q[5])*sin(q[6])*cos(q[4]))/self.affine[2]
                                [2] - 0.5*self.affine[1][1]*b*(sin(q[4])*sin(q[6]) - sin(q[5])*cos(q[4])*cos(q[6]))/self.affine[2][2] + self.affine[1][1]*(sin(q[4])*sin(q[6]) - sin(q[5])*cos(q[4])*cos(q[6]))/self.affine[2][2]))*tem

                    b_diff[index][0] = 1
                    index += 1
        return bi, b_diff

    def reslicing(self,name):
        '''
        用刚体变换参数将图片进行重采样对齐\n
        Resampling and aligning the image with the rigid-body transform parameter\n
        output: 对齐后的图片(Aligned image)
        '''
        self.interpolator = itk.BSplineInterpolateImageFunction.New(self.img_gauss)  # B样条插值器(B-spline interpolation)
        
        print("开始重采样对齐(注意，需要很长时间，建议先去干点别的事)\nreslicing")
        new = np.ndarray(self.shape)
        for picture in range(self.shape[0]):
            new[picture,:, :, :] = self.get_new_img(self.img_data_gauss[picture,:, :, :], self.parameter[picture],picture)
            print(f"进度{picture*100/self.shape[0]}%", end="\r")
        self.save_img(new, name)
        return new

    def estimate(self):
        '''
        估计刚体变换参数\n
        Estimate the rigid-body transform parameter\n
        output: 刚体变换参数(Rigid-body transform parameter)'''
        print("开始估计刚体变换参数\nestimating")
        q=np.zeros(7)
        q[0] = 1
        for picture in range(1, self.shape[0]):
            # 高斯牛顿迭代
            q=q / self.beta
            for _ in range(self.iteration):
                
                q[0]=1
                (b, diff_b) = self.iterate(self.img_data[0,:, :, :],self.img_data[picture,:,:,:], q,picture)
                
                if np.linalg.det(diff_b.T@diff_b) == 0:
                    # 矩阵不可逆时用伪逆
                    q -= self.alpha * np.linalg.pinv(diff_b.T@diff_b)@diff_b.T@b
                else:
                    q -= self.alpha * np.linalg.inv(diff_b.T@diff_b)@diff_b.T@b
            q[0] = 1
            q[q>40]=0
            q[q<-40]=0
            self.parameter[picture] = q/10
            print(f"进度{picture*100/self.shape[0]}%", end="\r")
        print("刚体变换参数估计完成,若需要进行reslicing，请先关闭图像\nestimation complete")
        self.draw_parameter()
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
        plt.show()
        
        
class TorchRealign (torch.autograd.Function,Realign):
    def __init__(self,path):
        Realign.__init__(self,path)

#把对准图像改成了第一张
    @staticmethod
    def forward(q,ctx,picture):#ctx = self
        b, b_diff = ctx.iterate(ctx.img_data[0, :, :, :], ctx.img_data[picture, :, :, :], q, picture)
        #b_diff.save_for_backward(torch.from_numpy(q))
        return b,b_diff
    
    @staticmethod# arges = diff_b
    def backward(alpha, b, b_diff):
        q = torch.zeros(7)
        q[6] = 1
        
        if np.linalg.det(b_diff.T @ b_diff) == 0:
            # 矩阵不可逆时用伪逆
            q -= alpha * torch.from_numpy(np.linalg.pinv(b_diff.T @ b_diff) @ b_diff.T @ b)
        else:
            q -= alpha * torch.from_numpy(np.linalg.inv(b_diff.T @ b_diff) @ b_diff.T @ b)

        return q

    def TorchOptimize(self, LEARNINGTIMES):
        print("开始估计刚体变换参数(Torch 版...)..")
        q = torch.zeros(7)
        q[0] = 1
        
        for picture in range(1, self.shape[0]):
            q = q / self.beta

            for _ in range(LEARNINGTIMES):
                b, diff_b = self.forward(q, self, picture)
                q = self.backward(self.alpha, b, diff_b)
                q[0] = 1
                q[q>50]=0
                q[q<-50]=0
                
                self.parameter[picture] = q.numpy() / 10
                print(f"进度{picture * 100 / self.shape[0]}%", end="\r")

        return self.parameter

if __name__ == "__main__":
    ## 测验代码
    #  realign = Realign('sub-Ey153_ses-3_task-rest_acq-EPI_run-1_bold.nii.gz')
    realign = Realign('C:\\Users\\26698\\BME_SHT\\data\\sub-Ey153_ses-3_task-rest_acq-EPI_run-1_bold.nii.gz')

    realign.set_step(2)
    realign.set_iteration(2)
    realign.set_order(3)
    realign.set_gussian(2)
    
    realign.set_alpha(0.15)
    realign.set_beta(1.65)
    
    # 获得刚体变换参数并绘制头动曲线
#    realign.estimate()
    # 获得重采样后的图像
#    realign.reslicing("picture")


    torchrealign = TorchRealign("C:\\Users\\26698\\BME_SHT\\data\\sub-Ey153_ses-3_task-rest_acq-EPI_run-1_bold.nii")
    torchrealign.TorchOptimize(1)
    torchrealign.draw_parameter()