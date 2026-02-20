import os

class Config:
    BASE_PATH = r"./data"
    DATASET_PATH = os.path.join(BASE_PATH, "Energy_data.xlsx")
    CIF_PATH = os.path.join(BASE_PATH, "molecular sieve")
    CONFORMER_3D_PATH = os.path.join(BASE_PATH, "conformer 3D")
    STRUCTURE_2D_PATH = os.path.join(BASE_PATH, "structure 2D")
    
    PROCESSED_CACHE_PATH = os.path.join(BASE_PATH, "cached_graphs_box64_cleaned.pt")
    
    TARGET_COLS = [
        'Binding Energy (kJ/mol Si)',
        'Directivity Energy (kJ/mol Si)',
        'Competition Energy (kJ/mol Si)',
        'Binding Energy (kJ/mol OSDA)',
        'Competition Energy (kJ/mol OSDA)'
    ]

    BATCH_SIZE = 64
    NUM_WORKERS = 0  
    PIN_MEMORY = True

    ATOM_EMBEDDING_DIM = 64
    HIDDEN_DIM = 128

    EMB_DIM_DEGREE = 8
    EMB_DIM_CHARGE = 8
    EMB_DIM_HYB = 8
    EMB_DIM_AROMATIC = 4
    EMB_DIM_CHIRAL = 4

    LR = 0.0008
    WEIGHT_DECAY = 1e-5
    EPOCHS = 200
    CRYSTAL_RADIUS = 6.0

    VOXEL_SIZE = 64  
    VOXEL_RES = 0.5
    SIGMA = 0.5
    
    MIN_SAMPLES_PER_TOPO = 0

    EARLY_STOPPING_PATIENCE = 20
    EARLY_STOPPING_MIN_DELTA = 0.001

    SCHEDULER_PATIENCE = 5
    SCHEDULER_FACTOR = 0.5
    MIN_LR = 1e-6
    
    NUM_CONFORMERS = 3