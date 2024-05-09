import torch.nn as nn

from model.deformed_primitive_field import DPF
from model.deformed_primitive_field import DPF_SVR
from model.deformed_primitive_field import DPF_IM
# from model.deformed_primitive_field_best_chair_table import DPF_best
# from model.deformed_primitive_field_final import DPF_final
def select_model(args):
    type2model = {
        'deformed_primitive_field': DPF(num_template=args.num_part, is_stage2=args.stage2, primitive_type=args.primitive_type),
        # 'deformed_primitive_field_best': DPF_best(num_template=args.num_part, is_stage2=args.stage2, primitive_type=args.primitive_type),
        # 'deformed_primitive_field_final': DPF_final(num_template=args.num_part, is_stage2=args.stage2, primitive_type=args.primitive_type),
        'dpf_svr': DPF_SVR(num_template=args.num_part),
        'dpf_im': DPF_IM(num_template=args.num_part, is_stage2=args.stage2, primitive_type=args.primitive_type),
    }
    model = type2model[args.model_type]
    return model

