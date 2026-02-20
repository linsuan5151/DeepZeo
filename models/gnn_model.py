import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from utils.config import Config

class Voxel3DCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv3d(2, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(16)
        self.pool1 = nn.MaxPool3d(2)
        
        self.conv2 = nn.Conv3d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(32)
        self.pool2 = nn.MaxPool3d(2)
        
        self.conv3 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm3d(64)
        self.pool3 = nn.MaxPool3d(2)
        
        self.fc = nn.Linear(64 * 8 * 8 * 8, 128)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc(x))
        return x

class DualBranchGNN(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.emb_atom = nn.Embedding(120, Config.ATOM_EMBEDDING_DIM)
        self.emb_degree = nn.Embedding(12, Config.EMB_DIM_DEGREE)
        self.emb_charge = nn.Embedding(15, Config.EMB_DIM_CHARGE)
        self.emb_hyb = nn.Embedding(8, Config.EMB_DIM_HYB)
        self.emb_aromatic = nn.Embedding(2, Config.EMB_DIM_AROMATIC)
        self.emb_chiral = nn.Embedding(4, Config.EMB_DIM_CHIRAL)
        
        total_emb_dim = (Config.ATOM_EMBEDDING_DIM + Config.EMB_DIM_DEGREE + 
                         Config.EMB_DIM_CHARGE + Config.EMB_DIM_HYB + 
                         Config.EMB_DIM_AROMATIC + Config.EMB_DIM_CHIRAL)
        
        self.mol_conv1 = GCNConv(total_emb_dim + 1, Config.HIDDEN_DIM)
        self.mol_conv2 = GCNConv(Config.HIDDEN_DIM, Config.HIDDEN_DIM)
        
        self.zeo_conv1 = GCNConv(total_emb_dim, Config.HIDDEN_DIM)
        self.zeo_conv2 = GCNConv(Config.HIDDEN_DIM, Config.HIDDEN_DIM)
        
        self.voxel_cnn = Voxel3DCNN()
        
        self.global_feat_dim = 17
        self.global_encoder = nn.Sequential(
            nn.Linear(self.global_feat_dim, 64),
            nn.ReLU(),
            nn.Linear(64, Config.HIDDEN_DIM),
            nn.BatchNorm1d(Config.HIDDEN_DIM),
            nn.ReLU()
        )
        
        fusion_dim = Config.HIDDEN_DIM * 4
        self.head = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, len(Config.TARGET_COLS))
        )

    def _embed_features(self, x_idx):
        e1 = self.emb_atom(x_idx[:, 0])
        e2 = self.emb_degree(x_idx[:, 1])
        e3 = self.emb_charge(x_idx[:, 2])
        e4 = self.emb_hyb(x_idx[:, 3])
        e5 = self.emb_aromatic(x_idx[:, 4])
        e6 = self.emb_chiral(x_idx[:, 5])
        return torch.cat([e1, e2, e3, e4, e5, e6], dim=1)

    def forward(self, mol_batch, zeo_batch, voxel_batch):
        # --- molecules GNN ---
        x_m, edge_index_m, batch_m = mol_batch.x, mol_batch.edge_index, mol_batch.batch
        x_m_emb = self._embed_features(x_m)
        x_m_in = torch.cat([x_m_emb, mol_batch.x_charge], dim=1)
        
        x_m_out = F.relu(self.mol_conv1(x_m_in, edge_index_m, edge_weight=mol_batch.edge_weight))
        x_m_out = F.relu(self.mol_conv2(x_m_out, edge_index_m, edge_weight=mol_batch.edge_weight))
        feat_m = global_mean_pool(x_m_out, batch_m)
        
        # --- zeolites GNN ---
        x_z, edge_index_z, batch_z = zeo_batch.x, zeo_batch.edge_index, zeo_batch.batch
        x_z_emb = self._embed_features(x_z)
        
        x_z_out = F.relu(self.zeo_conv1(x_z_emb, edge_index_z))
        x_z_out = F.relu(self.zeo_conv2(x_z_out, edge_index_z))
        feat_z = global_mean_pool(x_z_out, batch_z)
        
        # --- 3D CNN ---
        feat_v = self.voxel_cnn(voxel_batch)
        
        # --- global feature ---
        global_attr = mol_batch.global_attr
        if global_attr.dim() == 3: global_attr = global_attr.squeeze(1)
        feat_global = self.global_encoder(global_attr)
        
        combined = torch.cat([feat_m, feat_z, feat_global, feat_v], dim=1)
        return self.head(combined)