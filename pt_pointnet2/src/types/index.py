# filepath: /pt_pointnet2/pt_pointnet2/src/types/index.py
"""
该文件定义了项目中使用的类型和接口，确保类型安全和代码可读性。
"""

from typing import List, Tuple, Dict, Any

# 定义点云数据的类型
PointCloud = List[Tuple[float, float, float]]  # 每个点的(x, y, z)坐标

# 定义特征的类型
Features = List[float]  # 特征向量

# 定义批量数据的类型
BatchData = Dict[str, Any]  # 包含点云和其他相关信息的字典

# 定义模型输出的类型
ModelOutput = Dict[str, Any]  # 模型输出的字典，包含预测结果等信息

# 定义训练和测试的配置类型
Config = Dict[str, Any]  # 配置字典，包含超参数和其他设置