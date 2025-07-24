import xml.etree.ElementTree as ET
import torch
# PyTorch Geometric 是一个流行的GNN库，我们将使用它的Data对象来封装图数据
# 如果您尚未安装，请运行: pip install torch_geometric
from torch_geometric.nn import NNConv, global_mean_pool, GINEConv
from torch_geometric.data import Data
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv, global_add_pool, global_max_pool
from torch.nn import Linear, LayerNorm, Dropout

def pad_node_features(x, feat_dim, target_dim=35):
    """
        x: 原始节点特征 [num_nodes, feature_dim]
        返回: 填充后的节点特征 [target_dim, feature_dim]
        """
    num_nodes = x.size(0)
    padding_size = target_dim - num_nodes

    # 创建零填充的虚拟节点
    padding = torch.zeros(padding_size, feat_dim, dtype=x.dtype)

    # 拼接原始节点和虚拟节点
    padded_x = torch.cat([x, padding], dim=0)

    # 创建节点掩码 (1=真实节点, 0=虚拟节点)
    node_mask = torch.cat([
        torch.ones(num_nodes, dtype=torch.bool),
        torch.zeros(padding_size, dtype=torch.bool)
    ])

    return padded_x, node_mask

def pad_edge_features(edge_index, edge_attr, target_edges=34):
    """
    edge_index: 原始边索引 [2, num_edges]
    edge_attr: 原始边特征 [num_edges, feature_dim]
    返回: 填充后的边索引和边特征
    """
    num_edges = edge_index.size(1)
    padding_size = target_edges - num_edges

    # 创建自环的虚拟边（连接到最后一个虚拟节点）
    last_node_idx = 34  # 最后一个节点的索引
    padding_index = torch.tensor(
        [[last_node_idx], [last_node_idx]],
        dtype=torch.long
    ).repeat(1, padding_size)

    # 拼接原始边和虚拟边
    padded_index = torch.cat([edge_index, padding_index], dim=1)

    # 创建零填充的边特征
    padding_attr = torch.zeros(padding_size, edge_attr.size(1), dtype=edge_attr.dtype)
    padded_attr = torch.cat([edge_attr, padding_attr], dim=0)

    # 创建边掩码 (1=真实边, 0=虚拟边)
    edge_mask = torch.cat([
        torch.ones(num_edges, dtype=torch.bool),
        torch.zeros(padding_size, dtype=torch.bool)
    ])

    return padded_index, padded_attr, edge_mask

def parse_urdf(urdf_path):
    with open(urdf_path, 'r') as f:
        urdf_content = f.read()
        
    # 解析URDF
    root = ET.fromstring(urdf_content)

    link_inertial = {}
    for link in root.findall('link'):
        link_name = link.get('name')
        inertial_node = link.find('inertial')
        if inertial_node is None:
            continue
        inertial_matrix = inertial_node.find('inertia')
        ixx = float(inertial_matrix.get('ixx'))
        ixy = float(inertial_matrix.get('ixy'))
        ixz = float(inertial_matrix.get('ixz'))
        iyy = float(inertial_matrix.get('iyy'))
        iyz = float(inertial_matrix.get('iyz'))
        izz = float(inertial_matrix.get('izz'))
        link_inertial[link_name] = [ixx, ixy, ixz, iyy, iyz, izz]
    return link_inertial

def parse_mujoco_to_graph(xml_file_path: str, urdf_path: str) -> Data:
    """
    将一个MuJoCo XML文件解析成一个PyTorch Geometric的图数据对象。

    Args:
        xml_file_path (str): MuJoCo XML文件的路径。

    Returns:
        torch_geometric.data.Data: 一个包含节点、边和其特征的图对象。
    """
    tree = ET.parse(xml_file_path)
    root = tree.getroot()
    worldbody = root.find('worldbody')

    link_inertial = parse_urdf(urdf_path)

    # 用于存储提取出的特征
    node_features = []  # 存储每个节点的特征
    edge_index_list = []  # 存储图的连接关系
    edge_features = []  # 存储每条边的特征

    # 辅助数据结构
    body_name_to_id = {}

    # 核心步骤：递归地遍历所有<body>标签来构建图
    def recursive_body_parser(body_element, parent_id=None):
        """
        一个递归函数，用于遍历XML树，提取节点和边的信息。
        """
        # 1. 处理当前刚体 (body) -> 创建一个图节点
        current_body_name = body_element.attrib['name']
        current_body_id = len(node_features)
        body_name_to_id[current_body_name] = current_body_id

        # if body_element.attrib.get('pos') is None:
        #     raise ValueError(f"This body no pos {current_body_name}")
        body_pos = [float(x) for x in body_element.attrib.get('pos', "0 0 0").split()]
        body_quat = [float(x) for x in body_element.attrib.get('quat', "1 0 0 0").split()]
        body_feat = body_pos + body_quat
        print(current_body_name)
        # 2. 提取节点特征 (Node Features)
        # 特征可以包括：质量、惯性张量等
        inertial = body_element.find('inertial')

        if inertial is not None:
            # 获取 pos
            if inertial.attrib.get('pos') is None:
                raise ValueError(f"This inertial no pos {current_body_name}")
            pos = [float(x) for x in inertial.attrib.get('pos').split()]

            # 获取 quat
            if inertial.attrib.get('quat') is None:
                raise ValueError(f"This inertial has no quat {current_body_name}")
            quat = [float(x) for x in inertial.attrib.get('quat', "1 0 0 0").split()]

            # 获取质量
            if inertial.attrib.get('mass') is None:
                raise ValueError(f"This inertial has no mass {current_body_name}")
            mass = float(inertial.attrib.get('mass', 0))

            # 惯性张量的对角线元素
            inertial_matrix = link_inertial[current_body_name]
            inertial_feat = pos + [mass] + inertial_matrix
        else:
            # 如果没有惯性信息，则使用0填充
            inertial_feat = [0.0, 0.0, 0.0] + [1.0, 0.0, 0.0, 0.0] + [0.0] * 4
        node_features.append(body_feat + inertial_feat)

        # 3. 处理连接父刚体的关节 (joint) -> 创建一条图的边
        if parent_id is not None:
            # 添加一条从父节点到当前节点的边
            edge_index_list.append([parent_id, current_body_id])


            if len(body_element.findall('joint')) != 1:
                raise ValueError(f"joint number wrong {current_body_name}")
            # 4. 提取边特征 (Edge Features)
            # 边特征可以包括：关节类型、轴向、活动范围等
            joint = body_element.find('joint')

            if joint is not None:
                # 获取pos
                pos = [float(x) for x in joint.attrib.get('pos', "0 0 0").split()]
                # 关节轴向 (x,y,z)
                axis = [float(x) for x in joint.attrib.get('axis', '0 0 0').split()]
                # 关节活动范围 (min, max)
                joint_range = [float(x) for x in joint.attrib.get('range', '0 0').split()]
                edge_feat = pos + axis + joint_range
            else:
                # 如果没有关节信息（不常见，但为了稳健性），用0填充
                edge_feat = [0.0] * 8
            edge_features.append(edge_feat)

        # 5. 递归遍历所有子刚体
        for child_body in body_element.findall('body'):
            recursive_body_parser(child_body, parent_id=current_body_id)

    # 从根刚体开始递归 (worldbody下的第一个body)
    root_body = worldbody.find('body')
    if root_body:
        recursive_body_parser(root_body)

    # 6. 将Python列表转换为PyTorch张量
    x = torch.tensor(node_features, dtype=torch.float)

    # 6.1 将node进行padding
    padded_x, node_mask = pad_node_features(x, feat_dim=17, target_dim=35)


    # edge_index 需要是 [2, num_edges] 的形状
    # .t是转置
    # contiguous确保张量在内存中是连续的
    edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_features, dtype=torch.float)
    # 6.2 将edge进行padding
    padded_index, padded_attr, edge_mask = pad_edge_features(edge_index, edge_attr, target_edges=34)


    # 7. 创建并返回最终的图数据对象
    graph_data = Data(x=padded_x, edge_index=padded_index, edge_attr=padded_attr, node_mask=node_mask, edge_mask=edge_mask)
    return graph_data



def obtain_robot_graph():
    '''
        节点信息存储顺序: body_pos, body_quat, inertial_pos, inertial_quat, inertial_mass, inertial_diaginertial
        边信息存储顺序: joint_pos, joint_axis, joint_range
        '''
    xml_path = r"/root/projects/PBHC_g1_SingleWaistYaw/description/robots/g1/g1_23dof_lock_wrist.xml"
    urdf_path = r"/root/projects/PBHC_g1_SingleWaistYaw/description/robots/g1/g1_23dof_lock_wrist.urdf"
    robot_graph = parse_mujoco_to_graph(xml_path, urdf_path)

    print("图数据对象创建成功！")
    print("===================")
    print(f"文件名: {xml_path}")
    # print(robot_graph)
    print("===================")
    print(f"节点数量 (刚体数): {robot_graph.num_nodes}")
    print(f"节点特征张量 shape (x): {robot_graph.x.shape}")
    print("节点特征示例 (第一个节点):", robot_graph.x[0])
    print("\n")
    print(f"边的数量 (关节数): {robot_graph.num_edges}")
    print(f"边索引张量 shape (edge_index): {robot_graph.edge_index.shape}")
    print("边索引示例 (前5条边):", robot_graph.edge_index[:, :8])
    print("\n")
    print(f"边特征张量 shape (edge_attr): {robot_graph.edge_attr.shape}")
    print("边特征示例 (第一条边):", robot_graph.edge_attr[7])

    return robot_graph


class RobotGraphEncoder(torch.nn.Module):
    def __init__(self,
                 node_dim=17,
                 edge_dim=8,
                 hidden_dim=128,
                 num_heads=4,
                 num_layers=3,
                 out_dim=128):
        super().__init__()

        # 初始投影层
        self.node_proj = Linear(node_dim, hidden_dim)
        self.edge_proj = Linear(edge_dim, hidden_dim)

        # Transformer层堆叠
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = TransformerConv(
                hidden_dim, hidden_dim,
                heads=num_heads,
                edge_dim=hidden_dim,
                concat=False,
                beta=True
            )
            self.convs.append(conv)

        # 归一化和正则化
        self.norms = torch.nn.ModuleList([LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.dropout = Dropout(0.1)

        # 输出层
        self.out_proj = Linear(2 * hidden_dim, out_dim)  # 2x for concat pooling

    def forward(self, data):
        x, edge_index, edge_attr, node_mask, edge_mask = data.x, data.edge_index, data.edge_attr, data.node_mask, data.edge_mask

        x = x[node_mask, :]
        edge_index = edge_index[:, edge_mask]
        edge_attr = edge_attr[edge_mask, :]

        batch = data.batch

        # 特征投影
        x = F.relu(self.node_proj(x))
        edge_attr = F.relu(self.edge_proj(edge_attr))

        # 消息传递
        for conv, norm in zip(self.convs, self.norms):
            x_res = x
            x = conv(x, edge_index, edge_attr=edge_attr)
            x = norm(x + x_res)  # 残差连接
            x = F.gelu(x)
            x = self.dropout(x)

        # 双池化策略
        global_mean = global_add_pool(x, batch)
        global_max = global_max_pool(x, batch)
        graph_embedding = torch.cat([global_mean, global_max], dim=1)

        # 最终投影
        return self.out_proj(graph_embedding)

def obtain_encoder(data):
    # 初始化模型
    # model = RobotEncoder(
    #     node_dim=18,
    #     edge_dim=8,
    #     hidden_dim=128,
    #     output_dim=256,
    #     num_layers=3,
    #     dropout=0.1
    # )

    # 使用TransformerConv
    model = RobotGraphEncoder(
        node_dim=17,
        edge_dim=8,
        hidden_dim=128,
        num_heads=4,
        num_layers=3,
        out_dim=128
    )
    # 前向传播获取编码
    encoding = model(data)
    print(encoding.shape)

if __name__ == '__main__':
    robot_graph = obtain_robot_graph()
    obtain_encoder(robot_graph)