import os
import json
import numpy as np
import multiprocessing
import torch
from glob import glob
from tqdm import tqdm
from joblib import Parallel, delayed
from rdkit import Chem
from rdkit.Chem import AllChem
from pymatgen.core import Structure
from torch_geometric.data import Data, Batch, Dataset
from sklearn.preprocessing import StandardScaler
from utils.config import Config

def get_random_rotation_matrix():
    theta = np.random.uniform(0, 2*np.pi)
    phi = np.random.uniform(0, 2*np.pi)
    z = np.random.uniform(0, 2*np.pi)

    Rx = np.array([[1, 0, 0],
                   [0, np.cos(theta), -np.sin(theta)],
                   [0, np.sin(theta),  np.cos(theta)]])

    Ry = np.array([[np.cos(phi), 0, np.sin(phi)],
                   [0, 1, 0],
                   [-np.sin(phi), 0, np.cos(phi)]])

    Rz = np.array([[np.cos(z), -np.sin(z), 0],
                   [np.sin(z),  np.cos(z), 0],
                   [0, 0, 1]])

    return Rz @ Ry @ Rx

def coords_to_voxel(coords, grid_size=32, res=0.5, sigma=0.5):
    grid = np.zeros((grid_size, grid_size, grid_size), dtype=np.float32)
    limit = (grid_size * res) / 2.0

    mask = (coords[:, 0] > -limit) & (coords[:, 0] < limit) & \
           (coords[:, 1] > -limit) & (coords[:, 1] < limit) & \
           (coords[:, 2] > -limit) & (coords[:, 2] < limit)

    valid_coords = coords[mask]
    if len(valid_coords) == 0:
        return grid

    indices = ((valid_coords + limit) / res).astype(int)
    indices = np.clip(indices, 0, grid_size - 1)

    for idx in indices:
        x, y, z = idx
        x_min, x_max = max(0, x-1), min(grid_size, x+2)
        y_min, y_max = max(0, y-1), min(grid_size, y+2)
        z_min, z_max = max(0, z-1), min(grid_size, z+2)
        grid[x_min:x_max, y_min:y_max, z_min:z_max] += 1.0

    return np.clip(grid, 0, 1.0)

class GraphBuilder:
    def _get_atom_encoding_legacy(self, atomic_num):
        if atomic_num > 100: return 100
        return atomic_num - 1

    def _get_rich_atom_features(self, atom=None, element_symbol=None, is_crystal=False):
        if atom:
            atomic_num = atom.GetAtomicNum()
        elif element_symbol:
            pt = Chem.GetPeriodicTable()
            atomic_num = pt.GetAtomicNumber(element_symbol)
        else:
            atomic_num = 0

        feat_atomic = min(atomic_num, 118)

        if is_crystal or atom is None:
            return [feat_atomic, 0, 5, 0, 0, 0] 

        degree = min(atom.GetDegree(), 10)

        charge = atom.GetFormalCharge()
        charge_idx = charge + 5 
        charge_idx = max(0, min(charge_idx, 14))

        hyb = atom.GetHybridization()
        hyb_map = {
            Chem.rdchem.HybridizationType.S: 0,
            Chem.rdchem.HybridizationType.SP: 1,
            Chem.rdchem.HybridizationType.SP2: 2,
            Chem.rdchem.HybridizationType.SP3: 3,
            Chem.rdchem.HybridizationType.SP3D: 4,
            Chem.rdchem.HybridizationType.SP3D2: 5,
            Chem.rdchem.HybridizationType.UNSPECIFIED: 6
        }
        hyb_idx = hyb_map.get(hyb, 6)

        is_aromatic = 1 if atom.GetIsAromatic() else 0

        chi_map = {
            Chem.rdchem.ChiralType.CHI_UNSPECIFIED: 0,
            Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW: 1,
            Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW: 2,
            Chem.rdchem.ChiralType.CHI_OTHER: 3
        }
        chi_idx = chi_map.get(atom.GetChiralTag(), 0)

        return [feat_atomic, degree, charge_idx, hyb_idx, is_aromatic, chi_idx]

    def _calculate_shape_descriptors(self, coords):
        if coords is None or len(coords) < 2:
            return [0.0] * 10
        coords = coords - np.mean(coords, axis=0)
        cov_matrix = np.cov(coords.T)
        evals, evecs = np.linalg.eigh(cov_matrix)
        idx = evals.argsort()[::-1]
        evals = evals[idx]
        L, W, H = 0.0, 0.0, 0.0
        try:
            aligned_coords = np.dot(coords, evecs[:, idx])
            min_b = np.min(aligned_coords, axis=0)
            max_b = np.max(aligned_coords, axis=0)
            dims = max_b - min_b
            L, W, H = sorted(dims, reverse=True)
        except:
            pass
        rg = np.sqrt(np.mean(np.sum(coords**2, axis=1)))
        pm1 = max(evals[0], 1e-6)
        pm2 = evals[1] if len(evals) > 1 else 0.0
        pm3 = evals[2] if len(evals) > 2 else 0.0
        return [
            rg,
            pm1, pm2, pm3,
            L/(H+1e-6),
            np.sqrt(max(0, 1 - (pm3/pm1))),
            pm3/pm1,
            L, W, H
        ]

    def _extract_global_props(self, props_2d, props_3d, charge_val, coords_3d):
        props_map = {'logp': 0.0, 'tpsa': 0.0, 'rotatable': 0.0, 'h_acceptor': 0.0, 'h_donor': 0.0, 'volume': 0.0}
        for prop in props_2d:
            urn = prop.get('urn', {})
            label, name, val = urn.get('label', ''), urn.get('name', ''), prop.get('value', {})
            if label == 'Log P' and name == 'XLogP3-AA': props_map['logp'] = val.get('fval', 0.0)
            elif label == 'Topological' and name == 'Polar Surface Area': props_map['tpsa'] = val.get('fval', 0.0)
            elif label == 'Count' and name == 'Rotatable Bond': props_map['rotatable'] = float(val.get('ival', 0))
            elif label == 'Count' and name == 'Hydrogen Bond Acceptor': props_map['h_acceptor'] = float(val.get('ival', 0))
            elif label == 'Count' and name == 'Hydrogen Bond Donor': props_map['h_donor'] = float(val.get('ival', 0))
        for prop in props_3d:
            if prop.get('urn', {}).get('name') == 'Volume': props_map['volume'] = prop.get('value', {}).get('fval', 0.0)
        shape_feats = self._calculate_shape_descriptors(coords_3d)
        features_list = [props_map[k] for k in props_map] + [float(charge_val)] + shape_feats
        return torch.tensor(features_list, dtype=torch.float).unsqueeze(0)

    def _extract_partial_charges(self, num_atoms, props_list):
        charges = torch.zeros(num_atoms, 1, dtype=torch.float)
        for prop in props_list:
            if prop.get('urn', {}).get('name') == 'MMFF94 Partial':
                for item in prop.get('value', {}).get('slist', []):
                    try:
                        parts = item.split()
                        charges[int(parts[0])-1] = float(parts[1])
                    except: pass
                break
        return charges

    def build_molecule_graph(self, cid):
        file_3d = os.path.join(Config.CONFORMER_3D_PATH, f"Conformer3D_COMPOUND_CID_{cid}.json")
        file_2d = os.path.join(Config.STRUCTURE_2D_PATH, f"Structure2D_COMPOUND_CID_{cid}.json")
        main_data_2d, props_2d = None, []
        total_charge = 0.0
        smiles_str = None
        
        if os.path.exists(file_2d):
            try:
                with open(file_2d, 'r', encoding='utf-8') as f: d = json.load(f)
                if 'PC_Compounds' in d:
                    main_data_2d = d['PC_Compounds'][0]
                    props_2d = main_data_2d.get('props', [])
                    total_charge = float(main_data_2d.get('charge', 0.0))

                    for prop in props_2d:
                        if prop.get('urn', {}).get('label') == 'SMILES':
                            smiles_str = prop.get('value', {}).get('sval')
                            break
            except: pass
        
        if smiles_str:
            try:
                mol = Chem.MolFromSmiles(smiles_str)
                if mol:
                    mol = Chem.AddHs(mol)
                    params = AllChem.ETKDGv2()
                    params.randomSeed = 42
                    cids_rdkit = AllChem.EmbedMultipleConfs(mol, numConfs=Config.NUM_CONFORMERS, params=params)
                    
                    if len(cids_rdkit) > 0:
                        AllChem.MMFFOptimizeMoleculeConfs(mol, numThreads=0)
                    
                    pos_variants = []
                    if len(cids_rdkit) > 0:
                        for i in range(len(cids_rdkit)):
                            conf = mol.GetConformer(cids_rdkit[i])
                            pos = conf.GetPositions()
                            pos = pos - np.mean(pos, axis=0)
                            pos_variants.append(pos)
                    
                    if len(pos_variants) > 0:
                        while len(pos_variants) < Config.NUM_CONFORMERS:
                            pos_variants.append(pos_variants[0])
                        
                        atom_features = [self._get_rich_atom_features(atom=atom) for atom in mol.GetAtoms()]
                        x = torch.tensor(atom_features, dtype=torch.long)
                        pos_main = torch.tensor(pos_variants[0], dtype=torch.float)
                        pos_variants_tensor = torch.tensor(np.array(pos_variants), dtype=torch.float)
                        
                        edge_indices, edge_weights = [], []
                        for bond in mol.GetBonds():
                            u, v = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                            dist = np.linalg.norm(pos_main[u].numpy() - pos_main[v].numpy())
                            w = 1.0 / (dist + 0.1)
                            edge_indices.extend([[u, v], [v, u]])
                            edge_weights.extend([w, w])
                        
                        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous() if edge_indices else torch.empty((2, 0), dtype=torch.long)
                        edge_weight = torch.tensor(edge_weights, dtype=torch.float) if edge_weights else torch.empty(0)
                        
                        AllChem.ComputeGasteigerCharges(mol)
                        charges = [float(atom.GetProp('_GasteigerCharge')) if atom.HasProp('_GasteigerCharge') else 0.0 for atom in mol.GetAtoms()]
                        x_charge = torch.tensor(charges, dtype=torch.float).unsqueeze(1)
                        
                        return Data(
                            x=x,
                            edge_index=edge_index,
                            edge_weight=edge_weight,
                            x_charge=x_charge,
                            global_attr=self._extract_global_props(props_2d, [], total_charge, pos_main.numpy()),
                            pos=pos_main,
                            pos_variants=pos_variants_tensor
                        )
            except Exception as e:
                pass
        
        return None

    def build_zeolite_graph(self, topology):
        patterns = [os.path.join(Config.CIF_PATH, f"*{topology}*.cif*"), os.path.join(Config.CIF_PATH, topology, "*.cif*")]
        cif_files = []
        for p in patterns: cif_files.extend(glob(p))
        if not cif_files: return None

        try:
            struct = Structure.from_file(cif_files[0])
            
            atom_features = []
            for site in struct:
                feats = self._get_rich_atom_features(element_symbol=site.specie.symbol, is_crystal=True)
                atom_features.append(feats)
            
            x = torch.tensor(atom_features, dtype=torch.long)
            pos = torch.tensor(struct.cart_coords, dtype=torch.float)
            
            nbrs = struct.get_all_neighbors(r=Config.CRYSTAL_RADIUS, include_index=True)
            
            edge_indices, edge_attrs = [], []
            for i, nbr_list in enumerate(nbrs):
                for nbr in sorted(nbr_list, key=lambda x: x[1])[:12]:
                    target_index = nbr[2]
                    distance = nbr[1]
                    edge_indices.append([i, target_index])
                    edge_attrs.append(distance)
            
            edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_attrs, dtype=torch.float).unsqueeze(1)
            
            struct_super = struct.copy()
            struct_super.make_supercell([3, 3, 3]) 
            pos_super = torch.tensor(struct_super.cart_coords, dtype=torch.float)
            pos_super = pos_super - torch.mean(pos_super, dim=0)

            return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, pos=pos, pos_super=pos_super)
        except Exception as e:
            return None

def build_mol_helper(cid):
    return (cid, GraphBuilder().build_molecule_graph(cid))

def build_zeo_helper(topo):
    return (topo, GraphBuilder().build_zeolite_graph(topo))

def prepare_and_cache_data(df):
    if os.path.exists(Config.PROCESSED_CACHE_PATH):
        print(f"Discover cache files: {Config.PROCESSED_CACHE_PATH}")
        try:
            return torch.load(Config.PROCESSED_CACHE_PATH, weights_only=False)
        except Exception as e:
            print(f"failed to load")
    
    print("No cache files found")
    unique_cids = df['CID'].unique()
    unique_topos = df['Topology Code'].unique()

    print(f"  - Number of unique molecules: {len(unique_cids)}")
    print(f"  - Number of unique zeolites: {len(unique_topos)}")
    
    print("Constructing molecular maps")
    mol_results = Parallel(n_jobs=multiprocessing.cpu_count())(
        delayed(build_mol_helper)(cid) for cid in tqdm(unique_cids, desc="Molecules")
    )
    mol_cache = {res[0]: res[1] for res in mol_results if res[1] is not None}
    
    print("Constructing a zeolite diagram")
    zeo_results = Parallel(n_jobs=min(len(unique_topos), multiprocessing.cpu_count()))(
        delayed(build_zeo_helper)(topo) for topo in tqdm(unique_topos, desc="Zeolites")
    )
    zeo_cache = {res[0]: res[1] for res in zeo_results if res[1] is not None}
    
    cache_data = {'mol_cache': mol_cache, 'zeo_cache': zeo_cache}
    print(f"Save the cache to: {Config.PROCESSED_CACHE_PATH}")
    torch.save(cache_data, Config.PROCESSED_CACHE_PATH)
    return cache_data

class ZeoliteDataset(Dataset):
    def __init__(self, df, cache_data, target_scaler=None, props_scaler=None, is_train=False):
        super().__init__()
        self.target_scaler = target_scaler if target_scaler else StandardScaler()
        self.props_scaler = props_scaler if props_scaler else StandardScaler()
        self.is_train = is_train
        
        mol_cache = cache_data['mol_cache']
        zeo_cache = cache_data['zeo_cache']
        
        self.mol_list = []
        self.zeo_list = []
        raw_y_list = []
        
        for idx, row in df.iterrows():
            cid = row['CID']
            topo = row['Topology Code']
            
            if cid in mol_cache and topo in zeo_cache:
                targets = row[Config.TARGET_COLS].values.astype(float)
                if not np.isnan(targets).any():
                    self.mol_list.append(mol_cache[cid])
                    self.zeo_list.append(zeo_cache[topo])
                    raw_y_list.append(targets)
        
        y_all = np.array(raw_y_list)
        if is_train:
            y_norm = self.target_scaler.fit_transform(y_all)
        else:
            y_norm = self.target_scaler.transform(y_all) if hasattr(self.target_scaler, 'mean_') else y_all
            
        self.y_list = [torch.tensor(y, dtype=torch.float) for y in y_norm]
        
        if len(self.mol_list) > 0:
            all_props = torch.cat([m.global_attr for m in self.mol_list], dim=0).numpy()
            if is_train:
                self.props_scaler.fit(all_props)
                
        self.length = len(self.mol_list)
        print(f"Dataset Finish building: {self.length} sample (Train={is_train})")

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        mol_data = self.mol_list[idx].clone()
        zeo_data = self.zeo_list[idx].clone()
        y = self.y_list[idx]

        if hasattr(self.props_scaler, 'mean_'):
            props_raw = mol_data.global_attr.numpy()
            props_norm = self.props_scaler.transform(props_raw)
            mol_data.global_attr = torch.tensor(props_norm, dtype=torch.float)

        mol_coords = mol_data.pos.numpy()
        if hasattr(mol_data, 'pos_variants'):
            variants = mol_data.pos_variants
            if self.is_train:
                conf_idx = np.random.randint(len(variants))
                mol_coords = variants[conf_idx].numpy()
            else:
                mol_coords = variants[0].numpy()
            del mol_data.pos_variants 

        zeo_voxel_coords = zeo_data.pos_super.numpy() if hasattr(zeo_data, 'pos_super') else zeo_data.pos.numpy()
        if hasattr(zeo_data, 'pos_super'): del zeo_data.pos_super 

        if self.is_train:
            rot_matrix = get_random_rotation_matrix()
            mol_coords = np.dot(mol_coords, rot_matrix)
            mol_noise = np.random.normal(0, 0.02, mol_coords.shape)
            zeo_noise = np.random.normal(0, 0.02, zeo_voxel_coords.shape)
            mol_coords += mol_noise
            zeo_voxel_coords += zeo_noise
            
        mol_data.pos = torch.tensor(mol_coords, dtype=torch.float)
        
        grid_mol = coords_to_voxel(mol_coords, Config.VOXEL_SIZE, Config.VOXEL_RES, Config.SIGMA)
        grid_zeo = coords_to_voxel(zeo_voxel_coords, Config.VOXEL_SIZE, Config.VOXEL_RES, Config.SIGMA)
        
        voxel_tensor = torch.tensor(np.stack([grid_mol, grid_zeo], axis=0), dtype=torch.float)
        
        return mol_data, zeo_data, voxel_tensor, y

    @staticmethod
    def gpu_collate(batch):
        mol_list = [item[0] for item in batch]
        zeo_list = [item[1] for item in batch]
        voxel_list = [item[2] for item in batch]
        y_list = [item[3] for item in batch]
        
        return (Batch.from_data_list(mol_list),
                Batch.from_data_list(zeo_list),
                torch.stack(voxel_list),
                torch.stack(y_list))