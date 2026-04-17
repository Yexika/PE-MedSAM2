import argparse


# cfg_pe.py

def parse_args():
    """
    PE-MedSAM2 2D Training Configuration
    支持动态尺寸: 256, 512, 1024
    """
    parser = argparse.ArgumentParser(description='PE-MedSAM2 2D Training')

    # Basic settings
    parser.add_argument('-net', type=str, default='sam2', help='net type')
    parser.add_argument('-encoder', type=str, default='vit_b', help='encoder type')
    parser.add_argument('-exp_name', default='PE_MedSAM2_2D', type=str, help='experiment name')

    # Visualization
    parser.add_argument('-vis', type=int, default=1, help='visualization frequency')
    parser.add_argument('-train_vis', type=bool, default=True)
    parser.add_argument('-vis_output_dir', type=str, default='./visualization_results')
    parser.add_argument('-vis_freq', type=int, default=10)
    parser.add_argument('-vis_max_samples', type=int, default=5)

    # Prompt
    parser.add_argument('-prompt', type=str, default='click', help='bbox or click')

    # Model weights
    parser.add_argument('-pretrain', type=str,
                        default='/root/autodl-tmp/Medical-SAM2-main/pretrained weight/MedSAM2_pretrain.pth')
    parser.add_argument('-weights', type=str,
                        default='/root/autodl-tmp/Medical-SAM2-main/pretrained weight/MedSAM2_pretrain.pth')

    # Training
    parser.add_argument('-val_freq', type=int, default=1, help='validation frequency')
    parser.add_argument('-gpu', type=bool, default=True)
    parser.add_argument('-gpu_device', type=int, default=0)

    # ★★★ 图像尺寸 - 支持 256, 512, 1024 ★★★
    parser.add_argument('-image_size', type=int, default=1024,
                        help='image size: 256 (fast), 512 (balanced), 1024 (best quality)')
    parser.add_argument('-out_size', type=int, default=1024,
                        help='output size, should match image_size')

    # Distribution
    parser.add_argument('-distributed', default='none', type=str)
    parser.add_argument('--local_rank', type=int, default=0)

    # Dataset
    parser.add_argument('-dataset', default='all', type=str, help='dataset name or "all"')
    parser.add_argument('-data_path', type=str, default='/root/autodl-tmp/datasets/CVC-ClinicDB')

    # SAM2
    parser.add_argument('-sam_ckpt', type=str, default='./checkpoints/sam2.1_hiera_small.pt')
    parser.add_argument('-sam_config', type=str, default='sam2_hiera_s')

    # ★★★ 训练超参数 ★★★
    parser.add_argument('-b', type=int, default=4, help='batch size')
    parser.add_argument('-lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('-epochs', type=int, default=100, help='training epochs')

    # ★★★ 学习率调度和梯度裁剪参数 ★★★
    parser.add_argument('-warmup_epochs', type=int, default=10, help='预热 epochs')
    parser.add_argument('-min_lr_ratio', type=float, default=0.01, help='最小学习率比例')
    parser.add_argument('-max_grad_norm', type=float, default=1.0, help='梯度裁剪阈值')
    parser.add_argument('-weight_decay', type=float, default=0.01, help='权重衰减')

    # Model
    parser.add_argument('-multimask_output', type=int, default=1)
    parser.add_argument('-memory_bank_size', type=int, default=16, help='memory bank size')

    # ★★★ 新增: 禁用记忆库 (消融实验用) ★★★
    parser.add_argument('-disable_memory_bank', action='store_true', default=False,
                        help='完全禁用记忆库，隔离 PE 模块真实贡献')

    # Data loading
    parser.add_argument('-num_workers', type=int, default=4)
    parser.add_argument('-seed', type=int, default=1234, help='random seed')

    # Testing
    parser.add_argument('-model_path', type=str, default='./logs/best_model.pth')
    parser.add_argument('-data_root', type=str, default='/root/autodl-tmp/datasets')
    parser.add_argument('-output_dir', type=str, default='./test_results')

    # ========== ★★★ PE 参数 ★★★ ==========

    # PE 总开关
    parser.add_argument('-use_pe', type=bool, default=True,
                        help='Enable PE modules')

    # 2D 模块开关
    parser.add_argument('-use_lra', type=bool, default=True,
                        help='Use LRA (Low-Rank Adapter)')
    parser.add_argument('-use_pffe', type=bool, default=True,
                        help='Use PFFE (Parameter-Free Feature Enhancement)')
    parser.add_argument('-use_ula', type=bool, default=True,
                        help='Use ULA (Ultra-Lightweight Adapter)')
    parser.add_argument('-use_dsa', type=bool, default=True,
                        help='Use DSA (Dynamic Sparse Attention)')

    # 模块超参数
    parser.add_argument('-lra_rank', type=int, default=4,
                        help='LRA low-rank dimension')
    parser.add_argument('-ula_compression', type=int, default=16,
                        help='ULA compression ratio')
    parser.add_argument('-dsa_sparsity', type=float, default=0.25,
                        help='DSA sparsity ratio (0-1)')
    parser.add_argument('-pffe_scales', type=int, nargs='+', default=[3, 5, 7],
                        help='PFFE Sobel kernel scales')

    # MAL 损失函数参数
    parser.add_argument('-lambda_dice', type=float, default=1.0,
                        help='Weight for Dice loss')
    parser.add_argument('-lambda_bce', type=float, default=0.5,
                        help='Weight for BCE loss')
    parser.add_argument('-lambda_boundary', type=float, default=0.3,
                        help='Weight for boundary loss')
    parser.add_argument('-mal_warmup', type=int, default=10,
                        help='Warmup epochs for boundary loss')

    # ========== ★★★ 消融实验参数 ★★★ ==========
    parser.add_argument('-ablation_name', type=str, default='',
                        help='消融实验名称，非空时权重/日志保存到 ablation 子目录')

    opt = parser.parse_args()

    return opt