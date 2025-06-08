import time
from functools import partial

import itk
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import jax.scipy.spatial
import nibabel as nib
from jax.scipy.spatial.transform import Rotation

# from jax.sharding import NamedSharding, PositionalSharding, PartitionSpec as P
from matplotlib import pyplot as plt


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


@jax.tree_util.register_pytree_node_class
class BSpline:
    def __init__(self, data: jnp.ndarray, degree: int, mode="clamp"):
        self.mode = mode

        self.degree = degree
        self.order = degree + 1
        self.dim = data.ndim - 1

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
        return (self.data, self.knots), (
            self.mode,
            self.degree,
            self.order,
            self.dim,
            self.block_shape,
            self.basis_shapes,
        )

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        obj = object.__new__(cls)
        (obj.data, obj.knots) = children
        (
            obj.mode,
            obj.degree,
            obj.order,
            obj.dim,
            obj.block_shape,
            obj.basis_shapes,
        ) = aux_data
        return obj

    @jax.jit
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.interplot(x)
        # return self.interplot(x)
        # return BSpline._interplot(
        #     x,
        #     self.data,
        #     self.knots,
        #     self.block_shape,
        #     self.basis_shapes,
        #     self.degree,
        #     self.mode,
        # )

    def interplot(self, x: jnp.ndarray):
        if self.mode is None:
            x = jnp.array(
                tuple(
                    jnp.clip(xi, knots[self.degree], knots[-self.degree - 1])
                    for xi, knots in zip(x, self.knots)
                )
            )

        spans = tuple(
            jnp.searchsorted(knots, xi, side="right") - 1 - self.degree for xi, knots in zip(x, self.knots)
        )
        result = jax.lax.dynamic_slice(self.data, jnp.array(spans + (0,)), self.block_shape)
        for xi, span, knot, shape in zip(x, spans, self.knots, self.basis_shapes):
            valid_knot = jax.lax.dynamic_slice(knot, (span,), (2 * self.degree + 2,))
            r = B(xi, valid_knot, self.degree).reshape(shape)
            result *= r
        return result.sum()


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
    def _iterate_core(self, q: jnp.ndarray, moving_interpolator: BSpline, ref_values: jnp.ndarray):

        def residual_at_one_point(q_params, single_coord, single_ref_value):
            transform = self.rigid(q_params)
            transformed_coord = (transform @ single_coord)[:-1]
            moving_value = moving_interpolator(transformed_coord)
            # .reshape()确保返回标量，避免vmap/jacfwd产生额外的维度
            return (moving_value - single_ref_value).reshape()

        # 计算残差向量 (b)
        b = jax.vmap(residual_at_one_point, in_axes=(None, 0, 0))(q, self.sample_coords, ref_values)

        # 定义一个只接受参数q并返回整个残差向量的函数

        # 计算雅可比矩阵 (J)
        jacobian_matrix = jax.jacfwd(
            lambda p: jax.vmap(residual_at_one_point, in_axes=(None, 0, 0))(p, self.sample_coords, ref_values)
        )(q)

        return b, jacobian_matrix

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

    # @partial(jax.jit, static_argnames=["self"])  # JIT整个函数
    @jax.jit
    def estimate(self, data_index: int):
        # 确保data_index是JAX可追踪的
        im = jax.lax.dynamic_slice_in_dim(self.ims, data_index, 1, axis=0)[0]
        moving_interpolator = BSpline(im[..., None], self.degree, mode="reflect")

        # 1. 预计算参考值
        ref_values = jax.vmap(self.ref_interpolator)(self.sample_coords[:, :-1])

        # 2. 定义 fori_loop 的 body
        def optimization_step(i, q_val):
            b, diff_b = self._iterate_core(q_val, moving_interpolator, ref_values)
            # 3. 使用 lstsq 求解
            m = jnp.linalg.lstsq(diff_b, b)[0]
            q_new = q_val - self.alpha * m
            return q_new

        # 初始参数
        q_initial = jnp.zeros(7).at[0].set(1.0)

        # 4. 使用 fori_loop 运行优化
        q_final = jax.lax.fori_loop(0, self.iteration, optimization_step, q_initial)

        # 强制w=1，并计算最终变换
        q_final = q_final.at[0].set(1.0)
        M = self.rigid(q_final)

        # 重采样
        all_coords = jnp.stack((*jnp.indices(self.data_shape), jnp.ones(self.data_shape)), axis=-1).reshape(
            (-1, self.ims.ndim)
        )
        resampled_flat = jax.vmap(lambda c: moving_interpolator((M @ c)[:-1]))(all_coords)
        resampled_image = resampled_flat.reshape(self.data_shape)

        return q_final, resampled_image

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
        return jnp.linalg.multi_dot(
            [
                self.inv_v2d,
                jnp.eye(4)
                .at[:3, :3]
                .set(Rotation.from_euler("zyx", q[4:7]).as_matrix())
                .at[:3, 3]
                .set(q[1:4]),
                self.v2d,
            ]
        )

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
    # jax.config.update("jax_optimization_level", "O3")
    jax.config.update("jax_enable_x64", True)
    # jax.config.update("jax_num_cpu_devices", 200)
    # jax.config.update("jax_memory_fitting_effort", 1.0)

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
    # print(jax.devices())
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
    # with jax.profiler.trace("./jax-trace", create_perfetto_link=True):
    v, aligned = jax.lax.map(realign.estimate, jnp.arange(1, 200), batch_size=20)
    # v, aligned = jax.jit(jax.vmap(realign.estimate, in_axes=0))(jnp.arange(1, 200))
    v.block_until_ready()

    # img = jnp.transpose(aligned, (3, 2, 1, 0))  # 将图片转换为nii格式就要按nii的顺序排列
    # img = nib.Nifti1Image(img, realign.affine)
    # img.to_filename("tested.nii.gz")
    # v = jax.vmap(realign.estimate, in_axes=0)(jnp.array((0, 0)))
    print(
        f"{time.time() - t1:.4f}",
    )
    realign.draw_parameter(v)


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
