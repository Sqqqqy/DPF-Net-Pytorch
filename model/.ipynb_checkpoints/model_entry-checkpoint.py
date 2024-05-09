import torch.nn as nn
from model.pBAE import BAE
from model.pBAE_SGM import BAE_sgm
from model.pBAE_SGM_cate import BAE_sgm_cate
from model.pBAEsine import BAE_sine
from model.tearingAE import PointCloudAutoencoder
# from model.pBAEmanifold import BAE_manifold
from model.pBAE_deform import BAE_deform
from model.pBAE_k_deform import BAE_k_deform
from model.pBAE_k_deform_k_bae import BAE_k_deform_k_bae
from model.pBAE_k_bae import BAE_k_bae
from model.pBAE_k_deform_k_bae_homeo import BAE_k_deform_k_bae_homeo
from model.pBAE_k_deform_k_bae_homeo2 import BAE_k_deform_k_bae_homeo2
from model.pBAE_k_deform_k_bae_homeo3 import BAE_k_deform_k_bae_homeo3
from model.pBAE_k_deform_k_bae_homeo4 import BAE_k_deform_k_bae_homeo4
from model.pBAE_k_deform_k_bae_homeo5 import BAE_k_deform_k_bae_homeo5
from model.pBAE_k_deform_k_bae_corse2fine import BAE_k_deform_k_bae_corse2fine
from model.pBAE_k_deform_k_bae_corse2fine_affine import BAE_k_deform_k_bae_corse2fine_affine
# from model.pBAE_k_deform_split import BAE_k_deform_split
from model.pBAE_k_deform_k_bae_fusion import BAE_k_deform_k_bae_fusion
from model.pBAE_bi_deform import BAE_bi_deform

def select_model(args):
    type2model = {
        # 'bae': BAE(),
        # 'bae_sgm': BAE_sgm(),
        # 'bae_sgm_cate': BAE_sgm_cate(),
        # 'bae_manifold': BAE_manifold(),
        # 'bae_sine': BAE_sine(),
        # 'bae_deform': BAE_deform(),
        # 'bae_k_deform': BAE_k_deform(),
        'bae_k_bae': BAE_k_bae(),
        'bae_k_deform_k_bae': BAE_k_deform_k_bae(),
        'bae_k_deform_k_bae_homeo': BAE_k_deform_k_bae_homeo(),
        'bae_k_deform_k_bae_homeo2': BAE_k_deform_k_bae_homeo2(),
        'bae_k_deform_k_bae_homeo3': BAE_k_deform_k_bae_homeo3(),
        'bae_k_deform_k_bae_homeo4': BAE_k_deform_k_bae_homeo4(),
        'bae_k_deform_k_bae_homeo5': BAE_k_deform_k_bae_homeo5(),
        'bae_k_deform_k_bae_corse2fine': BAE_k_deform_k_bae_corse2fine(),
        'bae_k_deform_k_bae_corse2fine_affine': BAE_k_deform_k_bae_corse2fine_affine(),
        # 'bae_k_deform_k_bae_fusion': BAE_k_deform_k_bae_fusion(),
        # 'bae_k_deform_split': BAE_k_deform_split(),
        # 'bae_bi_deform': BAE_bi_deform(),
        # 'tearing': PointCloudAutoencoder(),
    }
    model = type2model[args.model_type]
    return model

