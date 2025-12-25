import argparse
from dataset import*

def get_args():
    # Tham số bắt buộc nhập
    parser = argparse.ArgumentParser(description="Train, Pretrain hoặc Evaluate một model AI")
    parser.add_argument("--epoch", type=int, help="Số epoch để train")
    # parser.add_argument("--model", type=str, required=True, help="Đường dẫn đến model")
    parser.add_argument("--mode", type=str, choices=["train", "pretrain", "evaluate"], required=True, help="Chế độ: train hoặc pretrain hoặc evaluate")
    parser.add_argument("--data", type=str, required=True, help="Đường dẫn đến dataset đã giải nén")
    # Tham số trường hợp
    parser.add_argument("--checkpoint", type=str, help="Đường dẫn đến file checkpoint (chỉ dùng cho chế độ pretrain)")
    parser.add_argument("--augment", action='store_true', help="Bật Augmentation cho dữ liệu đầu vào")
    # Tham số mặc định(default)
    parser.add_argument("--saveas", type=str, help="Thư mục lưu checkpoint")
    parser.add_argument("--lr0", type=float, help="learning rate, default = 0.0001")
    parser.add_argument("--batchsize", type=int, help="Batch size, default = 8")

    parser.add_argument("--weight_decay", type=float,  help="weight_decay, default = 1e-6")
    parser.add_argument("--img_size", type=int, nargs=2,  help="Height and width of the image, default = [256, 256]")
    parser.add_argument("--numclass", type=int, help="shape of class, default = 1")
    
    """
    # Với img_size, cách chạy: python script.py --img_size 256 256
    Nếu muốn nhập list dài hơn 3 phần tử, gõ 
    parser.add_argument("--img_size", type=int, nargs='+', default=[256, 256], help="Image dimensions")
    Chạy:
    python script.py --img_size 128 128 3
    """
    parser.add_argument("--loss", type=str, choices=["Dice_loss", "Hybric_loss", "BCEDice_loss", "BCEwDice_loss", "BCEw_loss", "SoftDice_loss", "Combo_loss"], default="Combo_loss", help="Hàm loss sử dụng, default = Combo_loss")
    parser.add_argument("--optimizer", type=str, choices=["Adam", "SGD", "AdamW"], default="AdamW", help="Optimizer sử dụng, default = AdamW")
    args = parser.parse_args()
    
    # Kiểm tra logic tham số
    if args.mode in ["pretrain", "evaluate"] and not args.checkpoint:
        parser.error(f"--checkpoint là bắt buộc khi mode là '{args.mode}'")
        
    return args

def set_seed():
    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main(args):  
    import numpy as np    
    from trainer import Trainer
    # from model import Unet, unet_pyramid_cbam_gate, Swin_unet
    from model import Swin_unet
    import optimizer as optimizer_module
    from dataset import get_dataloaders
    from result import export, export_evaluate
    global trainer
    # from utils import loss_func

    set_seed()
    
    # 1. Khởi tạo Model
    print(f"[INFO] Initializing Model...")
    # model1 = smp.UnetPlusPlus(
    #     encoder_name="efficientnet-b2",
    #     encoder_weights="imagenet",
    #     in_channels=3,
    #     classes=1,
    # )
    model1 = smp.Unet(
        encoder_name="efficientnet-b3", 
        encoder_weights="imagenet",     
        in_channels=3,                  # Ảnh x-quang đầu vào là 1 kênh (grayscale)
        classes=1,                      # Output là 1 kênh (Mask binary)
        decoder_attention_type="scse"   # Module attention không gian & kênh
    )
    # Bạn có thể thay đổi model tùy ý ở đây
    # model1 = Swin_unet.SwinUnet() 
    # # Swin_unet.load_pretrained_encoder(model1)
    # Swin_unet.load_pretrained_encoder(model1, "swinv2_tiny_patch4_window8_256.pth")

    # 2. Khởi tạo Optimizer
    # Lưu ý: Nếu muốn dùng args.lr0 đè lên config, hãy truyền vào optimizer
    opt = optimizer_module.optimizer(model=model1) 

    # 3. Khởi tạo Trainer
    # Lưu ý: Truyền đúng số epoch từ args vào trainer
    trainer = Trainer(model=model1, optimizer=opt)
    # Cập nhật lại num_epochs trong trainer nếu args.epoch khác config
    if args.epoch:
        trainer.num_epochs = args.epoch

    # 4. Lấy Dataloader
    # Giả sử hàm get_dataloaders nhận tham số augment và path data
    # (Bạn cần chắc chắn hàm get_dataloaders trong dataset.py hỗ trợ tham số này)
    trainLoader, validLoader, testLoader = get_dataloaders(augment=args.augment)

    # 5. Xử lý các chế độ (Logic mới gọn hơn)
    if args.mode == "train":
        print("[INFO] Mode: TRAINING FROM SCRATCH")
        trainer.train(trainLoader, validLoader, resume_path=None)
        export(trainer)

    elif args.mode == "pretrain":
        print(f"[INFO] Mode: PRETRAINING (Resume from {args.checkpoint})")
        # Gọi hàm train với resume_path
        trainer.train(trainLoader, validLoader, resume_path=args.checkpoint)
        export(trainer)
    if args.mode == "evaluate":
        print(f"[INFO] Mode: EVALUATING")
        
        # Định nghĩa thư mục lưu ảnh kết quả
        # Tạo folder con bên trong BASE_OUTPUT để gọn gàng
        visual_folder = os.path.join(BASE_OUTPUT, "prediction_images")
        
        # Gọi hàm evaluate với tham số lưu ảnh
        trainer.evaluate(
            test_loader=validLoader, 
            checkpoint_path=args.checkpoint,
            save_visuals=True,          # <--- Bật chế độ lưu ảnh
            output_dir=visual_folder    # <--- Truyền đường dẫn lưu
        )
        
        # Xuất file CSV thống kê
        export_evaluate(trainer)

    # elif args.mode == "evaluate":
    #     print(f"[INFO] Mode: EVALUATING")
    #     trainer.evaluate(test_loader=testLoader, checkpoint_path=args.checkpoint)
    #     export_evaluate(trainer)

if __name__ == "__main__":
    args = get_args()
    main(args)
