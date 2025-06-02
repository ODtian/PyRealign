import dataclasses
import string
import time
from functools import partial
from typing import Literal

import itk
import jax
import jax.experimental
import jax.experimental.pjit
import jax.experimental.shard_map
import jax.export
import jax.numpy as jnp
import jax.scipy as jsp
import jax.scipy.spatial
import jax.scipy.spatial.transform
import nibabel as nib
from jax.sharding import NamedSharding, PositionalSharding, PartitionSpec as P
from matplotlib import pyplot as plt

# JAX 默认使用 32 位浮点数，如果需要 64 位，可以取消注释下一行
# from jax import config
# config.update("jax_enable_x64", True)


# @partial(jax.jit, static_argnames=["degree"])
# def find_span(t: float, knots: jnp.ndarray, degree: int) -> int:

#     # n = knots.shape[-1] - degree - 2

#     # t = jnp.clip(t, knots[degree], knots[n + 1])

#     span =

#     # span = jnp.clip(span, degree, n)

#     return span


@partial(jax.jit, static_argnames=["degree"])
def B_spline_basis_functions(span_idx: int, t: float, degree: int, knots: jnp.ndarray) -> jnp.ndarray:
    # return non-empty basis weight of degree p at t
    start_span = span_idx - degree

    order = degree + 1
    basis_num = 2 * order - 1
    N = jnp.zeros(basis_num).at[degree].set(1.0)

    for p in range(1, order):
        basis_num -= 1
        # N_out = jnp.zeros(2 * degree - 1).at[degree].set(1.0)
        left_basis = N[:-1]
        right_basis = N[1:]

        t_i = knots[start_span : start_span + basis_num]
        t_i1 = knots[start_span + 1 : start_span + basis_num + 1]
        t_ip = knots[start_span + p : start_span + p + basis_num]
        t_ip1 = knots[start_span + p + 1 : start_span + p + basis_num + 1]
        # a = knots[start_span + k - 1 : start_span + k - 1 + basis_num]
        # denom = a - t_i
        # denom1 =
        denom_left = t_ip - t_i
        weight_left = jnp.where(denom_left == 0.0, 0.0, (t - t_i) / denom_left)

        denom_right = t_ip1 - t_i1
        weight_right = jnp.where(denom_right == 0.0, 0.0, (t_ip1 - t) / denom_right)

        N = left_basis * weight_left + right_basis * weight_right

    return N


@partial(jax.jit, static_argnames=["degree"])
def B(t: float, knots: jnp.ndarray, degree: int) -> jnp.ndarray:
    # valid_knots = knots[span - degree : span + degree]
    basis_num = 2 * degree + 1
    N = jnp.zeros(basis_num).at[degree].set(1.0)

    for p in range(1, degree + 1):
        basis_num -= 1
        # N_out = jnp.zeros(2 * degree - 1).at[degree].set(1.0)
        left_basis = N[:-1]
        right_basis = N[1:]

        t_i = knots[:basis_num]
        t_i1 = knots[1 : basis_num + 1]
        t_ip = knots[p : p + basis_num]
        t_ip1 = knots[p + 1 : p + basis_num + 1]

        # a = knots[start_span + k - 1 : start_span + k - 1 + basis_num]
        # denom = a - t_i
        # denom1 =
        denom_left = t_ip - t_i
        weight_left = jnp.where(denom_left == 0.0, 0.0, (t - t_i) / denom_left)

        denom_right = t_ip1 - t_i1
        weight_right = jnp.where(denom_right == 0.0, 0.0, (t_ip1 - t) / denom_right)

        N = left_basis * weight_left + right_basis * weight_right

    return N


# def BIn(
#     x: jnp.ndarray, data: jnp.ndarray, knots: jnp.ndarray, degree: int
# ) -> jnp.ndarray:
#     # n = knots.shape[0] - degree - 2

#     # if x < knots[degree] or x > knots[n + 1]:
#     #     return jnp.zeros_like(x)
#     # if degree & 1 == 0:
#     #     x -= 0.5
#     # x = jnp.clip(x, knots[degree], knots[-degree - 1])
#     span = find_span(x, degree, knots)
#     N = basis_functions(span, x, degree, knots)
#     # print(N)
#     # print(control_points[span + 1 - degree : span + 1])
#     return N @ data[span - degree : span + 1]


@jax.tree_util.register_pytree_node_class
class BSpline:
    def __init__(self, data: jnp.ndarray, degree: int, mode="clamp"):

        self.mode = mode

        # jax.scipy.interpolate.RegularGridInterpolator
        # self.data = data
        self.degree = degree
        self.order = degree + 1
        self.dim = data.ndim - 1

        # script_letter = string.ascii_letters[: data.ndim - 1]
        # self.subscript = f"{','.join(script_letter)}->{''.join(script_letter)}"
        # print(data.ndim)
        # if mode == "clamp":
        offset = self.order / 2

        if mode == "clamp":
            self.data = data
            self.knots = tuple(
                jnp.concatenate(
                    (
                        jnp.repeat(0.0, self.order),
                        jnp.arange(self.order, axis, dtype=jnp.float32) - offset,
                        jnp.repeat(axis - 1, self.order),
                    )
                )
                for axis in data.shape[:-1]
            )
        else:
            if mode is not None:
                self.data = jnp.pad(data, ((degree, degree),) * self.dim + ((0, 0),), mode=mode)
                self.knots = tuple(
                    jnp.arange(-degree, axis + 2 * degree + 1, dtype=jnp.float32) - offset
                    for axis in data.shape[:-1]
                )

            else:
                self.data = data
                self.knots = tuple(
                    jnp.arange(0, axis + self.order, dtype=jnp.float32) - offset for axis in data.shape[:-1]
                )

        self.block_shape = (self.degree + 1,) * self.dim + (self.data.shape[-1],)
        self.basis_shapes = tuple((1,) * i + (self.order,) + (1,) * (self.dim - i) for i in range(self.dim))

    def tree_flatten(self):
        return (self.data,), (
            self.mode,
            self.degree,
            self.order,
            self.dim,
            self.knots,
            self.block_shape,
            self.basis_shapes,
        )

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        obj = object.__new__(cls)
        (obj.data,) = children
        (
            obj.mode,
            obj.degree,
            obj.order,
            obj.dim,
            obj.knots,
            obj.block_shape,
            obj.basis_shapes,
        ) = aux_data
        return obj

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # return self.interplot(x)
        return BSpline._interplot(
            x,
            self.data,
            self.knots,
            self.block_shape,
            self.basis_shapes,
            self.degree,
            self.mode,
        )

    @staticmethod
    @partial(jax.jit, static_argnames=["block_shape", "basis_shapes", "degree", "mode"])
    def _interplot(
        x: jnp.ndarray,
        data: jnp.ndarray,
        knots: tuple[jnp.ndarray],
        block_shape: tuple[int],
        basis_shapes: tuple[tuple[int]],
        degree: int,
        mode: str | None,
    ):
        if mode is None:
            x = jnp.array(
                tuple(jnp.clip(xi, knots[degree], knots[-degree - 1]) for xi, knots in zip(x, knots))
            )

        spans = tuple(jnp.searchsorted(knots, xi, side="right") - 1 - degree for xi, knots in zip(x, knots))
        result = jax.lax.dynamic_slice(data, jnp.r_[spans, 0], block_shape)
        # print(result)
        for xi, span, knot, shape in zip(x, spans, knots, basis_shapes):
            valid_knot = jax.lax.dynamic_slice(knot, (span,), (2 * degree + 2,))
            # print(f"{valid_knots=}")
            # print(f"{b.reshape(shape[:-1])}")
            r = B(xi, valid_knot, degree).reshape(shape)

            # print(r)
            result *= r
            # print(f"{result=}")

            # print(f"{result=}")
        return result.sum()


def value_and_jacfwd(f, x):
    pushfwd = partial(jax.jvp, f, (x,))
    basis = jnp.eye(x.size, dtype=x.dtype)
    y, jac = jax.vmap(pushfwd, out_axes=(None, 1))((basis,))
    return y, jac


@jax.tree_util.register_pytree_node_class
class Realign:
    def __init__(self, path, sigma=1, iteration=1, degree=2, step=2, alpha=1.0, beta=1.2):
        """初始化(initialization)\n
        输入参数(parameter):\n
        path:文件路径(file path)
        """
        # 读取数据(load data)
        # self.path = path
        data = itk.imread(path)
        self.ims = jnp.array(itk.GetArrayFromImage(data))

        # 记录数据信息(record data information)
        spacing = jnp.array(data.GetSpacing())[::-1]
        # 倒序(reverse order)itk[读取的数据与nibabel读取的数据的坐标轴顺序不同]

        self.affine = jnp.diag(spacing)  # 仿射矩阵(affine matrix)

        # 默认参数设置及预处理(default parameter setting and preprocessing)
        self.iteration = iteration  # 迭代次数(iteration times)
        self.data_shape = self.ims.shape[1:]

        self.ref_interpolator = BSpline(self.ims[0][..., None], degree, mode="reflect")

        self.v2d = self.voxel_to_descartes()
        self.inv_v2d = jnp.linalg.inv(self.v2d)
        # self.interpolator = BSpline()

        # self.img_data_gauss = gaussian_filter(
        #     self.img_data, sigma
        # )  # 高斯平滑(gaussian smooth)
        # self.img_gauss = itk.image_from_array(self.img_data_gauss)

        self.step = step  # 选点间隔(point interval)
        # self.x, self.y, self.z = self.
        self.voxel_steps = self.step * spacing[1:]
        # self.voxel_steps = 1
        grids = jnp.meshgrid(
            *(jnp.arange(0, axis, voxel_step) for axis, voxel_step in zip(self.data_shape, self.voxel_steps))
        )
        self.sample_coords = jnp.stack((*grids, jnp.ones(grids[0].shape)), axis=-1).reshape(
            (-1, self.ims.ndim)
        )

        # self.x, self.y, self.z = (
        #     int(self.shape[1] / self.affine[1, 1] // self.step),
        #     int(self.shape[2] / self.affine[2, 2] // self.step),
        #     int(self.shape[3] / self.affine[3, 3] // self.step),
        # )
        # 初始旋转平移参数(rotation and translation parameters)
        # self.parameter = np.zeros((self.shape[0], 7))

        self.alpha = alpha  # 迭代过程的收敛系数
        self.beta = beta  # 每张图的初始缩小系数
        self.degree = degree
        # print(
        #     "图片载入成功,请耐心等待:) \nimage loaded successfully,please wait patiently"
        # )

    # def gaussian(self):
    #     x = jnp.stack(jnp.indices((3, 3, 3, 3)), axis=-1)
    #     # x = jnp.linspace(-3, 3, 7)
    #     pdf = jsp.stats.norm.pdf(x)
    #     jsp.signal.correlate()
    #     # window = pdf * pdf[:, None] * pdf[:, :, None]
    #     # jnp.meshg
    #     # smooth_image = jsp.signal.convolve(self., window, mode='same')

    # @jax.jit
    def iterate(self, q: jnp.ndarray, interpolator: BSpline):

        # @partial(jax.jit, static_argnames=["data", "ref", "sample_size"])
        @jax.value_and_grad
        def b_func(transform, coord, data, ref, sample_size):
            # transform = self.rigid(q)
            # # .at[3, 3].set(0.0)

            # # if jnp.linalg.det(transform) == 0:
            # #     transform_inv = jnp.linalg.pinv(transform)
            # # else:
            # #     transform_inv = jnp.linalg.inv(transform)
            # transform_inv = jnp.where(
            #     jnp.linalg.det(transform) == 0,
            #     jnp.linalg.pinv(transform),
            #     jnp.linalg.inv(transform),
            # )
            # transformed_coord = jnp.clip(
            #     (transform_inv @ coord)[:-1], jnp.zeros(3), jnp.array(sample_size)
            # )
            # transformed_coord = ( coord)[:-1]
            return data(coord[:-1]) - ref((transform @ coord)[:-1])

        @jax.value_and_grad
        def b_func2(q, _, coord, data, ref, sample_size):
            transform = self.rigid(q)
            # # .at[3, 3].set(0.0)

            # # if jnp.linalg.det(transform) == 0:
            # #     transform_inv = jnp.linalg.pinv(transform)
            # # else:
            # #     transform_inv = jnp.linalg.inv(transform)
            # transform_inv = jnp.where(
            #     jnp.linalg.det(transform) == 0,
            #     jnp.linalg.pinv(transform),
            #     jnp.linalg.inv(transform),
            # )
            # transformed_coord = jnp.clip(
            #     (transform_inv @ coord)[:-1], jnp.zeros(3), jnp.array(sample_size)
            # )
            return data(coord[:-1]) - ref((transform @ coord)[:-1])

        # jax.jacfwd()

        # (b, d), f_jvp = jax.jvp(b_func, *args)
        # return (b, d), f_jvp((jnp.ones_like(b), jnp.zeros_like(d)))[0]

        # def value_and_vjp(*args):
        #     (b, d), f_vjp = jax.vjp(b_func, *args)
        #     return (b, d), f_vjp((jnp.ones_like(b), jnp.zeros_like(d)))[0]

        # im = self.ims[data_index]
        # interpolator = BSpline(im[..., None], self.degree, mode="reflect")
        # transform, grad = jax.value_and_grad(self.rigid)(q)
        transform, jac = value_and_jacfwd(self.rigid, q)
        # transform_inv = jnp.where(
        #     jnp.linalg.det(transform) == 0,
        #     jnp.linalg.pinv(transform),
        #     jnp.linalg.inv(transform),
        # )
        jax.lax.map
        value, grad = jax.vmap(b_func, in_axes=(None, 0, None, None, None))(
            transform,
            self.sample_coords,
            interpolator,
            self.ref_interpolator,
            self.data_shape,
        )

        # value, grad = jax.vmap(b_func2, in_axes=(None, None, 0, None, None, None))(
        #     q,
        #     0,
        #     self.sample_coords,
        #     interpolator,
        #     self.ref_interpolator,
        #     self.data_shape,
        # )
        # return value, grad
        return value, jnp.einsum("bik,ijk->bj", grad, jac)
        # value, grad = jax.vmap(b_func, in_axes=(None, None, 0, None, None, None))(
        #     q,
        #     transform_inv,
        #     self.sample_coords,
        #     interpolator,
        #     self.ref_interpolator,
        #     self.data_shape,
        # )

        # value, grad = b_func(q, transform, )
        # return value, grad

    def tree_flatten(self):
        return (
            # self.path,
            self.ims,
            self.affine,
            self.v2d,
            self.inv_v2d,
            self.step,
            self.voxel_steps,
            self.sample_coords,
            self.alpha,  # 迭代过程的收敛系数
            self.beta,  # 每张图的初始缩小系数
        ), (
            self.data_shape,
            self.ref_interpolator,
            self.iteration,
            self.degree,
        )

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        obj = object.__new__(cls)
        (
            obj.ims,
            obj.affine,
            # obj.data_shape,
            # obj.ref_interpolator,
            obj.v2d,
            obj.inv_v2d,
            obj.step,
            obj.voxel_steps,
            obj.sample_coords,
            obj.alpha,  # 迭代过程的收敛系数
            obj.beta,  # 每张图的初始缩小系数
        ) = children
        (
            obj.data_shape,
            obj.ref_interpolator,
            obj.iteration,
            obj.degree,
        ) = aux_data
        return obj

    # @partial(jax.jit, static_argnames=["self"])
    @jax.jit
    def estimate(self, data_index):
        q = jnp.zeros(7).at[0].set(1.0)
        print(data_index.shape)
        im = self.ims[data_index]
        interpolator = BSpline(im[..., None], self.degree, mode="reflect")

        for _ in range(self.iteration):
            # q /= self.beta
            b, diff_b = self.iterate(q, interpolator)
            # print(b, diff_b)
            # jsp
            # jax.scipy.sparse.linalg.cg()
            # r = diff_b.T @ diff_b
            # m = (
            #     # self.alpha
            #     jnp.where(
            #         jnp.linalg.det(r) == 0,
            #         jnp.linalg.pinv(r),
            #         jnp.linalg.inv(r),
            #     )
            #     @ diff_b.T
            #     @ b
            # )
            m = jsp.linalg.solve(diff_b.T @ diff_b, diff_b.T @ b, overwrite_a=True, overwrite_b=True)
            q -= self.alpha * m
            # plt.plot(jnp.arange(self.sample_coords.shape[0]), b, "r.")
            # plt.plot(jnp.arange(7), m, "r.")
            # plt.show()

        q = q.at[0].set(1.0)
        # q[0] = 1
        # q[q > 40] = 0
        # q[q < -40] = 0
        M = self.rigid(q)
        # print(M)
        coord = jnp.stack((*jnp.indices(self.data_shape), jnp.ones(self.data_shape)), axis=-1).reshape(
            (-1, 4)
        )
        # coord = jax.vmap(lambda x: M @ x, in_axes=0)(coord)

        resampled = jax.vmap(lambda x: interpolator((M @ x)[:-1]), in_axes=0)(coord).reshape(self.data_shape)
        # print(coord)
        # print(coord)
        # return jnp.r_[q, m]
        return q, resampled

    # jax.vmap(interpolator, in_axes=0)(coord).reshape(self.data_shape)

    # , jax.vmap(interpolator, in_axes=0)(jnp.)

    def voxel_to_descartes(self):
        """
        体素坐标到笛卡尔坐标的转换,并将坐标原点移到图像中心\n
        (voxel to descartes coordinate system\n
        Move the coordinate origin to the image center)\n
        输出(output):\n
        descartes:坐标系转换矩阵(coordinate system transformation matrix)
        """
        # [
        #     [a 0 0 0]
        #     [0 b 0 0]
        #     [0 0 c 0]
        #     [0 0 0 d]
        # ]
        # [
        #     [b 0 0 0]
        #     [0 c 0 0]
        #     [0 0 d 0]
        #     [0 0 0 0]
        # ]
        # [
        #     [b 0 0 x * -0.5]
        #     [0 c 0 y * -0.5]
        #     [0 0 d z * -0.5]
        #     [0 0 0 1      ]
        # ]
        return jnp.hstack(
            (
                jnp.vstack((self.affine[1:, 1:], jnp.array((0, 0, 0)))),
                -0.5 * jnp.array((*self.data_shape, 1))[:, None],
            )
        )

    def rigid(self, q):
        # return (
        #     self.inv_v2d
        #     @ jnp.eye(4).at[:3, 3].set(q[1:4])
        #     @ jnp.eye(4)
        #     .at[:3, :3]
        #     .set(
        #         jax.scipy.spatial.transform.Rotation.from_euler(
        #             "zyx", q[4:7]
        #         ).as_matrix()
        #     )
        #     @ self.v2d
        # )
        T = jnp.array([[1, 0, 0, q[1]], [0, 1, 0, q[2]], [0, 0, 1, q[3]], [0, 0, 0, 1]])
        # 旋转矩阵(Rotation matrix)
        R_x = jnp.array(
            [
                [1, 0, 0, 0],
                [0, jnp.cos(q[4]), jnp.sin(q[4]), 0],
                [0, -jnp.sin(q[4]), jnp.cos(q[4]), 0],
                [0, 0, 0, 1],
            ]
        )
        R_y = jnp.array(
            [
                [jnp.cos(q[5]), 0, jnp.sin(q[5]), 0],
                [0, 1, 0, 0],
                [-jnp.sin(q[5]), 0, jnp.cos(q[5]), 0],
                [0, 0, 0, 1],
            ]
        )
        R_z = jnp.array(
            [
                [jnp.cos(q[6]), jnp.sin(q[6]), 0, 0],
                [-jnp.sin(q[6]), jnp.cos(q[6]), 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        )
        R = R_x @ R_y @ R_z

        M_r = T @ R
        # 坐标系转换矩阵(coordinate system transformation matrix)
        return self.inv_v2d @ M_r @ self.v2d
        # coor = self.v2d()
        # return np.linalg.inv(coor) @ M_r @ coor

    def save(self, aligned, name):
        img = jnp.transpose(aligned, (3, 2, 1, 0))  # 将图片转换为nii格式就要按nii的顺序排列
        # img = nib.Nifti1Image(img, self.affine)
        img = nib.nifti1.Nifti1Image(img, self.affine)
        name = f"{name}.nii.gz"
        nib.loadsave.save(img, name)
        print(f"图片{name}保存成功")

    def draw_parameter(self, parameter):
        """
        绘制刚体变换参数\n
        Draw rigid-body transform parameter\n
        """
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 10))
        x = jnp.arange(parameter.shape[0])
        ax1.plot(x, parameter[:, 1], label="X", color="red")
        ax1.plot(x, parameter[:, 2], label="Y", color="green")
        ax1.plot(x, parameter[:, 3], label="Z", color="blue")
        ax1.set_title("translation")
        ax1.set_xlabel("image")
        ax1.set_ylabel("mm")
        ax1.legend()
        ax2.plot(x, parameter[:, 4], label="pitch", color="purple")
        ax2.plot(x, parameter[:, 5], label="roll", color="orange")
        ax2.plot(x, parameter[:, 6], label="yaw", color="brown")
        ax2.set_title("rotation")
        ax2.set_xlabel("image")
        ax2.set_ylabel("degrees")
        ax2.legend()

        # ax3.set_title("error")
        # ax3.set_xlabel("image")
        # ax3.plot(x, parameter[:, 7], label="error", color="yellow")
        # ax3.plot(x, parameter[:, 8], label="error", color="yellow")
        # ax3.plot(x, parameter[:, 9], label="error", color="yellow")
        # ax3.plot(x, parameter[:, 10], label="error", color="yellow")
        # ax3.plot(x, parameter[:, 11], label="error", color="yellow")
        # ax3.plot(x, parameter[:, 12], label="error", color="yellow")
        # ax3.plot(x, parameter[:, 13], label="error", color="yellow")
        # ax3.legend()
        fig.tight_layout()
        plt.show()


def main_test():
    jax.config.update("jax_compilation_cache_dir", "./jax_cache")

    # 1. 定义一个对单个向量进行“昂贵”操作的函数
    #    这个函数将包含一些重复的计算。
    def expensive_operation_on_single_vector(
        vector: jnp.ndarray,
        matrix_a: jnp.ndarray,
        matrix_b: jnp.ndarray,
        num_iterations: int,
    ) -> jnp.ndarray:
        """
        对单个向量执行一系列计算密集型操作。
        num_iterations 必须是静态的，以便JIT可以展开循环。
        """
        v_current = vector
        for _ in range(num_iterations):
            # 一些任意的、但会占用计算资源的线性代数和逐元素操作
            transform_a = jnp.dot(matrix_a, v_current)
            transform_b = jnp.dot(matrix_b, v_current[::-1])  # 使用翻转的向量增加一点变化
            v_current = jnp.tanh(transform_a - transform_b)
            v_current = v_current * jnp.sin(v_current) + jnp.cos(v_current)
            # 规范化以防止值爆炸或消失 (可选，但对迭代过程有好处)
            v_current = v_current / (jnp.linalg.norm(v_current) + 1e-6)
        return jnp.sum(v_current**2)  # 返回一个标量结果

    # --- 参数配置 ---
    vector_size = 1024  # 每个向量的维度
    batch_size = 200000  # 我们要处理的向量数量 (调大这个值以更好地观察并行效果)
    iterations_per_vector = 15  # 每个向量内部的迭代次数 (增加这个值会使单次调用更耗时)

    print(
        f"配置: vector_size={vector_size}, batch_size={batch_size}, iterations_per_vector={iterations_per_vector}"
    )
    print(f"预计总操作数级别: {batch_size * iterations_per_vector * (vector_size**2) * 2} (粗略估计矩阵乘法)")
    print("-" * 30)

    # --- 数据生成 ---
    key = jax.random.PRNGKey(42)
    key_batch, key_m1, key_m2 = jax.random.split(key, 3)

    # 一批输入向量
    batch_of_vectors = jax.random.normal(key_batch, (batch_size, vector_size))

    # 用于操作的固定矩阵 (这些矩阵将在所有向量操作中重复使用)
    # 确保它们的形状与 vector_size 兼容
    fixed_matrix_a = jax.random.normal(key_m1, (vector_size, vector_size))
    fixed_matrix_b = jax.random.normal(key_m2, (vector_size, vector_size))

    # --- 方法一: Python 循环 (相对较慢) ---
    # 我们仍然 JIT 编译单次操作函数，以公平比较 JAX 的计算部分
    # 但 Python 循环本身会引入开销，并且不能在批处理维度上实现 XLA 级别的并行。
    # jitted_single_op = jax.jit(
    #     expensive_operation_on_single_vector, static_argnums=(3,)
    # )  # num_iterations 是静态的

    # print("运行方法一: Python 循环 + JIT编译的单项操作...")
    # # 预热/编译 (如果 jitted_single_op 尚未编译)
    # _ = jitted_single_op(
    #     batch_of_vectors[0], fixed_matrix_a, fixed_matrix_b, iterations_per_vector
    # ).block_until_ready()

    # start_time_loop = time.time()
    # results_loop = []
    # for i in range(batch_size):
    #     result = jitted_single_op(
    #         batch_of_vectors[i], fixed_matrix_a, fixed_matrix_b, iterations_per_vector
    #     )
    #     results_loop.append(result)

    # # 将结果列表转换为 JAX 数组并确保计算完成
    # if results_loop:  # 确保列表不为空
    #     final_results_loop = jnp.stack(results_loop)
    #     final_results_loop.block_until_ready()  # 等待所有计算完成
    # else:
    #     final_results_loop = jnp.array([])

    # end_time_loop = time.time()
    # time_py_loop = end_time_loop - start_time_loop
    # print(f"Python 循环耗时: {time_py_loop:.4f} 秒")
    # print("-" * 30)

    # --- 方法二: jax.vmap + jax.jit (向量化，期望更快) ---
    print("运行方法二: jax.vmap + jax.jit...")
    print("请在此时观察您的 CPU/GPU 利用率 (例如使用 htop 或 nvidia-smi)")

    # 1. 使用 vmap 向量化单项操作函数
    #    in_axes: 指定如何映射输入参数
    #       0: 对 batch_of_vectors 的第一个轴 (批处理轴) 进行映射
    #       None: fixed_matrix_a 和 fixed_matrix_b 不映射，广播给所有调用
    #       None: num_iterations (静态参数) 也不映射
    vmapped_expensive_op = jax.vmap(
        expensive_operation_on_single_vector,
        in_axes=(
            0,
            None,
            None,
            None,
        ),  # 对应 vector, matrix_a, matrix_b, num_iterations
        out_axes=0,  # 结果也沿着第一个轴堆叠
    )

    # 2. JIT 编译向量化后的函数
    #    num_iterations 仍然需要是静态的。
    #    当 vmapped_expensive_op 被调用时，num_iterations 会被传入。
    #    我们需要确保JIT知道这个参数是静态的。
    @partial(jax.jit, static_argnums=(3,))  # num_iterations 是第四个参数 (索引为3)
    def run_vmapped_jitted(vectors, mat_a, mat_b, num_iter_static):
        return vmapped_expensive_op(vectors, mat_a, mat_b, num_iter_static)

    # 预热/编译
    print("JIT 编译 (预热)...")
    _ = run_vmapped_jitted(
        batch_of_vectors, fixed_matrix_a, fixed_matrix_b, iterations_per_vector
    ).block_until_ready()
    print("预热完成.")

    start_time_vmap = time.time()
    results_vmap_jit = run_vmapped_jitted(
        batch_of_vectors, fixed_matrix_a, fixed_matrix_b, iterations_per_vector
    )
    results_vmap_jit.block_until_ready()  # 等待所有计算完成
    end_time_vmap = time.time()
    time_vmap_jit_total = end_time_vmap - start_time_vmap
    print(f"jax.vmap + jax.jit 耗时: {time_vmap_jit_total:.4f} 秒")
    print("-" * 30)

    # --- 结果比较 ---
    print("性能比较:")
    if time_vmap_jit_total > 0:
        speedup = time_py_loop / time_vmap_jit_total
        print(f"  vmap + jit 比 Python 循环快 {speedup:.2f} 倍")
    else:
        print("  vmap + jit 执行时间过短，无法计算有效加速比。")

    # (可选) 检查结果是否一致 (应该非常接近，除了可能的浮点精度差异)
    # diff = jnp.abs(final_results_loop - results_vmap_jit).max()
    # print(f"两种方法结果的最大差异: {diff}")

    print("\n--- 如何“看到”向量化/并行化 ---")
    print(
        "1. **执行时间**: `jax.vmap + jax.jit` 版本应该比 Python 循环版本快得多，特别是当 `batch_size` 很大时。"
    )
    print("2. **CPU/GPU 利用率**: 在 `jax.vmap + jax.jit` 版本运行时 (特别是在打印“预热完成”之后的那几秒):")
    print(
        "   - **CPU**: 打开系统监视器 (Linux/macOS: `htop`; Windows: 任务管理器)。你应该看到多个 CPU 核心的利用率显著上升。"
    )
    print(
        "   - **GPU (如果 JAX 配置为使用 GPU)**: 运行 `nvidia-smi` (NVIDIA) 或 `rocm-smi` (AMD)。你应该看到 GPU 利用率 (GPU-Util) 和内存使用量增加。"
    )
    print("3. **调整参数**: ")
    print("   - 增加 `batch_size` 会让并行化的优势更明显。")
    print(
        "   - 增加 `iterations_per_vector` 或 `vector_size` 会增加每个单独任务的计算量，这也有助于观察并行效果，因为并行开销占比会降低。"
    )


def main_map():
    jax.config.update("jax_compilation_cache_dir", "./jax_cache")
    jax.config.update("jax_optimization_level", "O3")
    jax.config.update("jax_log_compiles", True)
    # jax.config.update("jax_num_cpu_devices", 200)
    # jax.config.update("jax_memory_fitting_effort", 1.0)

    # jax.config.update("jax_explain_cache_misses", True)
    # jax.config.update("jax_log_compiles", True)
    # jax.config.update("jax_logging_level", "INFO")
    t1 = time.time()
    realign = Realign(
        r"data\sub-Ey153_ses-3_task-rest_acq-EPI_run-1_bold.nii.gz",
        # r"test.nii",
        degree=3,
        iteration=100,
        step=1,
        alpha=0.15,
        beta=1.65,
    )
    t2 = time.time()
    print(f"t2 - t1 = {t2 - t1:.4f}")

    def block_indices(indices, block_size=20):
        num_items = indices.shape[0]

        # 计算填充细节
        num_blocks = (num_items + block_size - 1) // block_size
        padded_len = num_blocks * block_size
        padding_amount = padded_len - num_items

        # 填充索引。使用第一个有效索引进行填充。
        # 来自填充索引的结果将被修剪。
        # 在访问 all_indices[0] 之前确保 all_indices 不为空
        return jnp.pad(indices, (0, padding_amount), constant_values=indices[0]).reshape(
            (num_blocks, block_size)
        )

    # JIT 编译的函数，用于使用 jax.lax.map 分块计算结果
    # @jax.jit
    def compute_results_in_blocks(indices, block_size=20):
        # data_blocks_input 的形状为 (num_blocks, block_size)

        # 将 process_single_block 应用于每个块
        # v_block_results 的形状将为 (num_blocks, block_size, 7)
        # aligned_block_results 的形状将为 (num_blocks, block_size)
        num_items = indices.shape[0]
        blocked_indices = block_indices(indices, block_size)
        estimate = jax.vmap(realign.estimate, in_axes=0)
        v_results, aligned_results = jax.lax.map(estimate, blocked_indices)

        # 重塑以展平块维度
        # 假设 realign_obj.estimate 返回 (q_shape=(7,), scalar_shape=())
        # 因此，v_item_shape_tail 为 (7,)，aligned_item_shape_tail 为 ()
        flat_v_results = v_results.reshape(-1, *v_results.shape[2:])
        flat_aligned_results = aligned_results.reshape(-1, *aligned_results.shape[2:])

        # 将结果修剪为原始项目数以移除填充
        final_v_output = flat_v_results[:num_items]
        final_aligned_output = flat_aligned_results[:num_items]

        return (
            final_v_output.block_until_ready(),
            final_aligned_output.block_until_ready(),
        )

    # 执行分块计算
    v, aligned = compute_results_in_blocks(jnp.arange(200).repeat(20), 20)
    print(v)
    print(aligned.shape)
    print(f"t3 - t2 = {time.time() - t2:.4f}")
    # realign.draw_parameter(v)


def main_shard():
    jax.config.update("jax_compilation_cache_dir", "./jax_cache")
    # jax.config.update("jax_optimization_level", "O3")
    jax.config.update("jax_log_compiles", True)
    jax.config.update("jax_num_cpu_devices", 16)

    t1 = time.time()
    realign = Realign(
        r"data\sub-Ey153_ses-3_task-rest_acq-EPI_run-1_bold.nii.gz",
        # r"test.nii",
        degree=3,
        iteration=100,
        step=1,
        alpha=0.15,
        beta=1.65,
    )
    t2 = time.time()
    print(f"t2 - t1 = {t2 - t1:.4f}")
    # sharding = PositionalSharding(jax.devices())

    # sharding = NamedSharding()
    mesh = jax.make_mesh((16,), ("x",))

    sharding = NamedSharding(mesh, P("x"))
    # G = jax.local_device_count()
    # sharded_x = jax.device_put(jnp.arange(1, 200), sharding.reshape(G, 1))
    sharded_x = jax.device_put(jnp.arange(1, 193), sharding)
    jax.debug.visualize_array_sharding(sharded_x)
    v, aligned = jax.jit(jax.vmap(realign.estimate, in_axes=0))(sharded_x)
    # print(v)
    v.block_until_ready()
    print(aligned.shape)
    print(f"t3 - t2 = {time.time() - t2:.4f}")


def main():
    jax.config.update("jax_compilation_cache_dir", "./jax_cache")
    jax.config.update("jax_optimization_level", "O3")
    # jax.config.update("jax_num_cpu_devices", 200)
    jax.config.update("jax_memory_fitting_effort", 1.0)

    # jax.config.update("jax_explain_cache_misses", True)
    # jax.config.update("jax_log_compiles", True)
    # jax.config.update("jax_logging_level", "INFO")
    realign = Realign(
        r"data\sub-Ey153_ses-3_task-rest_acq-EPI_run-1_bold.nii.gz",
        # r"test.nii",
        degree=3,
        iteration=2,
        step=1,
        alpha=0.15,
        beta=1.65,
    )
    # v = jax.vmap(realign.estimate, in_axes=0)(jnp.arange(1, 200))
    t1 = time.time()
    print(jax.devices())
    # with jax.profiler.trace("/tmp/jax-trace", create_perfetto_link=True):
    # v = realign.estimate(1).reshape((-1, 8))
    # v, aligned = jax.vmap(realign.estimate, in_axes=0)(jnp.arange(1, 16))
    # v, aligned = (
    #     jax.vmap(realign.estimate, in_axes=0).lower(jnp.arange(1, 200)).cost_analysis()
    # )
    # jax.lax.fori_loop()
    # jax.pmap(realign.estimate, in_axes=0)(jnp.arange(1, 16))
    # jax.pmap()
    # jax.vmap(realign.estimate, in_axes=0).lower(jnp.arange(1, 200)).
    # v, aligned = jax.lax.map(
    #     realign.estimate, jnp.arange(1, 200).repeat(20), batch_size=30
    # )
    with jax.profiler.trace("./jax-trace", create_perfetto_link=True):
        v, aligned = jax.jit(jax.vmap(realign.estimate, in_axes=0))(jnp.arange(1, 200))
    # v, aligned = jax.pmap(
    #     realign.estimate,
    #     in_axes=0,
    #     # lambda x: jax.vmap(realign.estimate, in_axes=0)(
    #     #     jnp.arange(x * 10, (x + 1) * 10)
    #     # )
    # )(jnp.arange(1, 200))
    # print(realign.estimate.lower(realign, 1).cost_analysis())
    # jax.lax.map
    # print(jax.make_jaxpr(realign.estimate)(1))
    # jax.scipy.interpolate.RegularGridInterpolator
    # v = jax.pmap(realign.estimate, in_axes=0)(jnp.arange(1, 16))
    # _ = jax.vmap(realign.estimate, in_axes=0)(jnp.arange(1, 200))
    # _ = jax.vmap(realign.estimate, in_axes=0)(jnp.arange(1, 200))
    # _ = jax.vmap(realign.estimate, in_axes=0)(jnp.arange(1, 200))
    # _ = jax.vmap(realign.estimate, in_axes=0)(jnp.arange(1, 200))
    # _ = jax.vmap(realign.estimate, in_axes=0)(jnp.arange(1, 200))
    # _ = jax.vmap(realign.estimate, in_axes=0)(jnp.arange(1, 200))
    # _ = jax.vmap(realign.estimate, in_axes=0)(jnp.arange(1, 200))
    # _ = jax.vmap(realign.estimate, in_axes=0)(jnp.arange(1, 200))
    # v = jax.vmap(realign.estimate, in_axes=0)(jnp.arange(1, 200)).block_until_ready()
    # v = jax.vmap(realign.estimate, in_axes=0)(jnp.arange(1, 200)).block_until_ready()
    # v = jax.vmap(realign.estimate, in_axes=0)(jnp.arange(1, 200)).block_until_ready()
    # plt.imshow(d)
    v.block_until_ready()
    # img = jnp.transpose(aligned, (3, 2, 1, 0))  # 将图片转换为nii格式就要按nii的顺序排列
    # img = nib.Nifti1Image(img, realign.affine)
    # img.to_filename("tested.nii.gz")
    # v = jax.vmap(realign.estimate, in_axes=0)(jnp.array((0, 0)))
    print(
        f"{time.time() - t1:.4f}",
    )
    # realign.draw_parameter(v)


def show():
    def flatten_voxels_to_image(voxel_data, grid_cols=None, background_value=0):
        D, H, W = voxel_data.shape

        # --- 计算网格维度 ---

        grid_cols = int(jnp.ceil(jnp.sqrt(D)))

        # 计算网格行数
        grid_rows = int(jnp.ceil(D / grid_cols))

        # --- 准备数据：填充以形成完整网格 ---
        total_grid_cells = grid_rows * grid_cols
        padding_needed = total_grid_cells - D

        # 如果需要填充
        if padding_needed > 0:
            # 创建填充块
            padding_shape = (padding_needed, H, W)
            padding_block = jnp.full(padding_shape, background_value, dtype=voxel_data.dtype)
            # 将填充块连接到原始数据后面 (沿着第一个轴)
            padded_data = jnp.concatenate([voxel_data, padding_block], axis=0)
        else:
            padded_data = voxel_data  # 不需要填充

        # --- 重新排列数据 ---
        # 1. 将填充后的数据 reshape 成 (grid_rows, grid_cols, H, W)
        #    这会将切片按网格结构分组
        grid_view = padded_data.reshape(grid_rows, grid_cols, H, W)

        # 2. 交换/转置轴的顺序，使得 H 和 W 维度相邻
        #    目标形状: (grid_rows, H, grid_cols, W)
        transposed_view = grid_view.transpose((0, 2, 1, 3))

        # 3. 最后，将前两个维度 (grid_rows, H) 和后两个维度 (grid_cols, W) 合并
        #    得到最终的 2D 图像形状: (grid_rows * H, grid_cols * W)
        flattened_image = transposed_view.reshape(grid_rows * H, grid_cols * W)

        return flattened_image

    def read_data(path):
        data = itk.imread(path)

        return data, jnp.array(itk.GetArrayFromImage(data))

    # with open("test.nii", "rb") as f:
    #     data = f.read()
    # print(data)
    data, d = read_data(r"data\sub-Ey153_ses-3_task-rest_acq-EPI_run-1_bold.nii.gz")

    fig2 = plt.figure(figsize=(10, 8))
    # ax2 = fig2.add_subplot(111, projection="3d")
    ax2 = fig2.add_subplot(111)
    cmap = plt.get_cmap("coolwarm")
    # 创建归一化器，将数据值映射到 [0, 1]
    # norm = mcolors.Normalize(vmin=min_val, vmax=max_val)

    # 将数据值通过归一化器和颜色映射转换为 RGBA 颜色
    # cmap(norm(data)) 会返回一个 (Z, Y, X, 4) 的数组
    r = flatten_voxels_to_image(d[0], 4)
    # r = d[0, 12]
    # facecolors = r / 1000
    facecolors = cmap(
        # (r - r.min())
        # / (r.max() - r.min())
        r.clip(-5000, 5000)
        / 5000
        # / 1000.0
    )  # 这里假设 d 是一个 (Z, Y, X) 的数组
    # facecolors = cmap(
    #     (r - r.min()) / (r.max() - r.min())
    #     # r
    # )  # 这里假设 d 是一个 (Z, Y, X) 的数组

    # 可选：设置透明度 (修改 alpha 通道)
    # facecolors[:, :, :, -1] = 0.7  # 设置所有体素的 alpha 值

    # r = d[0].reshape((-1,))
    # 使用 voxels 绘制
    # 第一个参数是一个布尔数组，指示哪些体素需要绘制，这里我们绘制所有
    # facecolors 参数指定每个体素表面的颜色
    # edgecolors 可以设置边框颜色
    ax2.imshow(facecolors)
    # voxels = ax2.voxels(
    #     jnp.ones(r.shape, dtype=bool),
    #     facecolors=facecolors,
    #     edgecolors="k",
    #     linewidth=0.2,
    # )
    # facecolors
    plt.show()


if __name__ == "__main__":
    # m = jax.random.randint(jax.random.PRNGKey(120), (3, 3), 0, 10)
    # bsp = BSpline(
    #     m.reshape((*m.shape, 1)),
    #     degree=1,
    #     mode="reflect",
    # )
    # bsp(jnp.array((1, 1.5)))
    # d.shape
    main()
    # show()
