// Copyright (c) Facebook, Inc. and its affiliates.
// 
// 本源码遵循MIT协议, 详见根目录下的LICENSE文件。

#include "group_points.h"
#include "utils.h"

/*
 * @brief 点云分组操作CUDA核函数的包装器声明
 * 
 * 该函数在.cu文件中实现, 用于在GPU上执行分组采样操作。
 * 
 * @param b        批量大小
 * @param c        特征通道数
 * @param n        每批次点云的点数
 * @param npoints  采样中心点数量
 * @param nsample  每个中心点的邻域采样点数
 * @param points   输入点云特征指针
 * @param idx      分组采样的索引指针
 * @param out      输出分组后的特征指针
 */
void group_points_kernel_wrapper(int b, int c, int n, int npoints, int nsample,
                                 const float *points, const int *idx,
                                 float *out);

/*
 * @brief 点云分组操作反向传播CUDA核函数的包装器声明
 * 
 * 该函数在.cu文件中实现, 用于在GPU上执行分组采样的反向传播操作。
 * 
 * @param b           批量大小
 * @param c           特征通道数
 * @param n           每批次点云的点数
 * @param npoints     采样中心点数量
 * @param nsample     每个中心点的邻域采样点数
 * @param grad_out    上游梯度指针
 * @param idx         分组采样的索引指针
 * @param grad_points 输出, 输入特征的梯度指针
 */
void group_points_grad_kernel_wrapper(int b, int c, int n, int npoints,
                                      int nsample, const float *grad_out,
                                      const int *idx, float *grad_points);

/*
 * @brief 点云分组操作主函数(PyTorch接口)
 * 
 * 根据给定的采样索引idx, 从输入点云特征points中采样, 返回分组后的特征。
 * 常用于PointNet++等点云网络的局部特征聚合阶段, 实现对每个采样中心点的邻域特征提取。
 * 
 * @param points (Tensor) 输入点云特征, 形状为[B, C, N]
 * @param idx    (Tensor) 分组采样的索引, 形状为[B, npoint, nsample]
 * @return       (Tensor) 分组后的特征, 形状为[B, C, npoint, nsample]
 */
at::Tensor group_points(at::Tensor points, at::Tensor idx) {
  // 检查输入张量是否为连续内存、类型是否正确
  CHECK_CONTIGUOUS(points);
  CHECK_CONTIGUOUS(idx);
  CHECK_IS_FLOAT(points);
  CHECK_IS_INT(idx);

  // 如果points在CUDA上, 则idx也必须在CUDA上
  if (IS_CUDA_TENSOR(points)) {
    CHECK_CUDA(idx);
  }

  // 创建输出张量, 初始化为0, 形状为[B, C, npoint, nsample]
  at::Tensor output =
      torch::zeros({points.size(0), points.size(1), idx.size(1), idx.size(2)},
                   at::device(points.device()).dtype(at::ScalarType::Float));

  // 如果输入为CUDA张量, 调用CUDA核函数包装器
  if (IS_CUDA_TENSOR(points)) {
    group_points_kernel_wrapper(points.size(0), points.size(1), points.size(2),
                                idx.size(1), idx.size(2), TENSOR_DATA_PTR(points, float),
                                TENSOR_DATA_PTR(idx, int), TENSOR_DATA_PTR(output, float));
  } else {
    // 仅支持CUDA实现, CPU暂不支持
    AT_CHECK(false, "CPU not supported");
  }

  // 返回分组后的特征
  return output;
}

/*
 * @brief 点云分组操作反向传播主函数(PyTorch接口)
 * 
 * 计算group_points操作的梯度, 将上游梯度grad_out根据采样索引idx累加回原始输入特征points的位置。
 * 用于训练时反向传播, 确保梯度正确传递到原始点云特征。
 * 
 * @param grad_out (Tensor) 上游梯度, 形状为[B, C, npoint, nsample]
 * @param idx      (Tensor) 分组采样的索引, 形状为[B, npoint, nsample]
 * @param n        (int)    原始点的数量N
 * @return         (Tensor) 输入特征points的梯度, 形状为[B, C, N]
 */
at::Tensor group_points_grad(at::Tensor grad_out, at::Tensor idx, const int n) {
  // 检查输入张量是否为连续内存、类型是否正确
  CHECK_CONTIGUOUS(grad_out);
  CHECK_CONTIGUOUS(idx);
  CHECK_IS_FLOAT(grad_out);
  CHECK_IS_INT(idx);

  // 如果grad_out在CUDA上, 则idx也必须在CUDA上
  if (IS_CUDA_TENSOR(grad_out)) {
    CHECK_CUDA(idx);
  }

  // 创建输出张量, 初始化为0, 形状为[B, C, N]
  at::Tensor output =
      torch::zeros({grad_out.size(0), grad_out.size(1), n},
                   at::device(grad_out.device()).dtype(at::ScalarType::Float));

  // 如果输入为CUDA张量, 调用CUDA核函数包装器
  if (IS_CUDA_TENSOR(grad_out)) {
    group_points_grad_kernel_wrapper(
        grad_out.size(0), grad_out.size(1), n, idx.size(1), idx.size(2),
        TENSOR_DATA_PTR(grad_out, float), TENSOR_DATA_PTR(idx, int), TENSOR_DATA_PTR(output, float));
  } else {
    // 仅支持CUDA实现, CPU暂不支持
    AT_CHECK(false, "CPU not supported");
  }

  // 返回输入特征的梯度
  return output;
}
