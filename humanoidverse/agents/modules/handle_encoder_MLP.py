import xml.etree.ElementTree as ET
import numpy as np
import torch
import torch
import torch.nn as nn
import torch.nn.functional as F

class RobustMLLEncoder(nn.Module):
    """
    一个更健壮的MLP编码器，包含了批量归一化（Batch Normalization）和Dropout。
    """
    def __init__(self, input_dim, hidden_dims, output_dim, 
                 activation=nn.ReLU, use_batch_norm=True, dropout_p=0.3):
        """
        初始化一个健壮的MLP编码器。

        参数:
        - input_dim (int): 输入向量的维度。
        - hidden_dims (list of int): 隐藏层神经元数量的列表。例如：[512, 256, 128]。
        - output_dim (int): 输出嵌入向量的维度。
        - activation (torch.nn.Module): 激活函数类。
        - use_batch_norm (bool): 是否在每个隐藏层使用批量归一化。
        - dropout_p (float): Dropout的概率 (0到1之间)。如果为0，则不使用Dropout。
        """
        super(RobustMLLEncoder, self).__init__()
        
        self.use_batch_norm = use_batch_norm
        self.dropout_p = dropout_p
        
        layers = []
        current_dim = input_dim
        
        # 构建隐藏层
        for h_dim in hidden_dims:
            # 线性层
            layers.append(nn.Linear(current_dim, h_dim))
            
            # (可选) 批量归一化层
            if self.use_batch_norm:
                # BatchNorm1d 适用于 (N, C) 或 (N, L) 的输入，正好是我们的情况
                layers.append(nn.BatchNorm1d(h_dim))
            
            # 激活函数层
            layers.append(activation())
            
            # (可选) Dropout层
            if self.dropout_p > 0:
                layers.append(nn.Dropout(p=self.dropout_p))
            
            current_dim = h_dim
            
        # 构建输出层
        # 通常输出层后不跟 归一化、激活 或 Dropout
        layers.append(nn.Linear(current_dim, output_dim))
        
        # 将所有层打包
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """
        定义前向传播。
        
        参数:
        - x (torch.Tensor): 输入的机器人结构向量。
        
        返回:
        - torch.Tensor: 输出的结构嵌入向量。
        """
        return self.network(x)

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

def obtain_data_MLP(xml_path, urdf_path):
    # 解析 XML
    tree = ET.parse(xml_path)
    root = tree.getroot()

    xml_world_body = root.find("worldbody")
    if xml_world_body is None:
        raise ValueError("No read worldbody")

    xml_body_root = xml_world_body.find("body")
    if xml_body_root is None:
        raise ValueError("No read body. XML is incorrect")

    link_inertial = parse_urdf(urdf_path)
    node_edge_node_feats = []
    node_feats = []
    body_name_index = {}
    def _add_xml_node(body_node, parent_index, node_index):
        body_name = body_node.attrib.get('name')
        curr_index = node_index
        body_name_index[body_name] = curr_index

        # 处理body标签
        # if body_node.attrib.get('pos') is None:
        #     raise ValueError(f"This body no pos {body_name}")
        body_pos = [float(x) for x in body_node.attrib.get("pos", "0 0 0").split()]
        body_quat = [float(x) for x in body_node.attrib.get("quat", "1 0 0 0").split()]
        body_feat = body_pos + body_quat


        # 处理inertial标签
        inertial_node = body_node.find("inertial")
        if inertial_node is None:
            raise ValueError(f"This body no inertial {body_name}")
        inertial_pos = [float(x) for x in inertial_node.attrib.get('pos').split()]
        inertial_mass = float(inertial_node.attrib.get('mass', 0))
        inertial_matrix = link_inertial[body_name]

        inertial_feat = inertial_pos + [inertial_mass] + inertial_matrix

        # 合并body_feat和inertial_feat作为一整个节点的特性
        node_feats.append(body_feat + inertial_feat)

        # 处理joint标签并且合并为node_edge_node
        if parent_index != -1:
            joint_node = body_node.find('joint')

            # if joint_node.attrib.get('pos') is None:
            #     raise ValueError(f"This joint no pos {body_name}")
            joint_pos = [float(x) for x in joint_node.attrib.get('pos', "0 0 0").split()]

            if joint_node.attrib.get('axis') is None:
                raise ValueError(f"This joint no quat {body_name}")
            joint_axis = [float(x) for x in joint_node.attrib.get('axis').split()]

            if joint_node.attrib.get('range') is None:
                raise ValueError(f"This joint no range {body_name}")
            joint_range = [float(x) for x in joint_node.attrib.get('range').split()]

            joint_feat = joint_pos + joint_axis + joint_range
            node_edge_node_feats.append(node_feats[parent_index] + joint_feat + body_feat + inertial_feat)

        for next_body_node in body_node.findall("body"):
            _add_xml_node(next_body_node, curr_index, len(node_feats))

    _add_xml_node(xml_body_root, -1, 0)

    node_edge_node_feats = torch.tensor(node_edge_node_feats)
    node_edge_node_feats = padding_zeros(node_edge_node_feats, feat_dim=42, target_dim=35)
    node_edge_node_feats = node_edge_node_feats.unsqueeze(0)

    return node_edge_node_feats.flatten().view(-1).unsqueeze(0)

def padding_zeros(x, feat_dim, target_dim):
    num_nodes = x.size(0)
    padding_size = target_dim - num_nodes

    padding = torch.zeros(padding_size, feat_dim, dtype=x.dtype)

    padded_x = torch.cat([x, padding], dim=0)

    return padded_x



def main(xml_path, urdf_path):
    node_edge_node_feats = obtain_data_MLP(xml_path, urdf_path)
    print(node_edge_node_feats.shape)
    net = RobustMLLEncoder(
                    input_dim=1470,
                    hidden_dims=[512, 256],
                    output_dim=128,
                    use_batch_norm=False, # 启用批量归一化
                    dropout_p=0.05 # 启用Dropout
                )
    
    print(net(node_edge_node_feats).shape)
    
    
if __name__ == "__main__":
    '''
    node_edge_node信息存储顺序:
    pre_body_pos, pre_body_quat, pre_inertial_pos, pre_inertial_mass, pre_inerital_matrix,
    joint_pos, joint_axis, joint_range,
    post_body_pos, post_body_quat, post_inertial_pos, post_inertial_mass, post_inertial_matrix
    '''
    xml_path = r"/root/projects/PBHC_g1_SingleWaistYaw/description/robots/g1/g1_23dof_lock_wrist.xml"
    urdf_path = r"/root/projects/PBHC_g1_SingleWaistYaw/description/robots/g1/g1_23dof_lock_wrist.urdf"
    main(xml_path, urdf_path)