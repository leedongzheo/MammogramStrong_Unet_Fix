
from config import*
from train import*
def export(trainer):
    source_file1='last_model.pth'
    source_file2='best_model.pth'
    path=f"output_epoch{trainer.best_epoch_dice}_dice{trainer.best_dice:.4f}"
    output_folder = os.path.join(BASE_OUTPUT,path)
    os.makedirs(output_folder, exist_ok=True)
    # Di chuyển
    exist_file_1=os.path.join(output_folder,source_file1)
    if os.path.exists(exist_file_1):
        os.remove(exist_file_1)
    shutil.move(source_file1, output_folder)
    if os.path.exists(source_file2):
        shutil.move(source_file2, output_folder)
        print(f"Đã di chuyển file tới: {output_folder}")

    # Kiểm tra xem source_file2 có tồn tại không trước khi di chuyển
    if os.path.exists(source_file2):
        shutil.move(source_file2, output_folder)
        print(f"Đã di chuyển file tới: {output_folder}")

    def tensor_to_float(value):
        if isinstance(value, torch.Tensor):
            return value.cpu().item()  # Chuyển tensor về CPU và lấy giá trị float
        elif isinstance(value, list):
            return [tensor_to_float(v) for v in value]  # Xử lý danh sách các tensor
        return value  # Nếu không phải tensor, giữ nguyên
    # Đường dẫn file checkpoint

    checkpoint_path = os.path.join(output_folder,source_file1)
    # source_file3 = "training_history.csv"  # File CSV hiện tại
    # csv_path_full = os.path.join(BASE_OUTPUT,source_file3)
    # Tải checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Đọc các giá trị từ checkpoint
    train_losses = tensor_to_float(checkpoint.get('train_losses', []))
    val_losses = tensor_to_float(checkpoint.get('val_losses', []))
    train_dices = tensor_to_float(checkpoint.get('train_dices', []))
    val_dices = tensor_to_float(checkpoint.get('val_dices', []))
    
    train_ious = tensor_to_float(checkpoint.get('train_ious', []))
    val_ious = tensor_to_float(checkpoint.get('val_ious', []))
    
    best_dice = tensor_to_float(checkpoint.get('best_dice', None))
    best_iou = tensor_to_float(checkpoint.get('best_iou', None))
    best_epoch_dice = tensor_to_float(checkpoint.get('best_epoch_dice', None))
    best_epoch_iou = tensor_to_float(checkpoint.get('best_epoch_iou', None))
    epoch = checkpoint.get('epoch', None)
    # start_epoch=checkpoint.get('start_epoch', None) + 1
    epochs = list(range(1, epoch + 1))
    # epochs = list(range(start_epoch, epoch + 1))
    
    new_data = pd.DataFrame({
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_dices': train_dices,
        'val_dices': val_dices,
        'train_ious': train_ious,
        'val_ious': val_ious,
        'best_dice': [best_dice] * len(epochs),
        'best_iou': [best_iou] * len(epochs),
        'best_epoch_dice': [best_epoch_dice] * len(epochs),
        'best_epoch_iou': [best_epoch_iou] * len(epochs),
        'epoch': epochs
    })
    # Lưu vào file Excel
    output_path = 'training_history_current_1.csv'
    csv_path_currrent = os.path.join(output_folder,output_path)
    # csv_path_currrent = os.path.join(output_folder,output_path)
    new_data.to_csv(csv_path_currrent, index=False)

    print(f"[INFO] Training history saved to {csv_path_currrent}")
    df = pd.read_csv(csv_path_currrent, encoding='ISO-8859-1')  # Hoặc 'latin1', 'windows-1252'
    # df.info()

    # Plot Losses
    plt.figure(figsize=(15, 5))
    # Tick spacing
    max_epoch = df['epoch'].max()
    xticks_range = range(0, max_epoch + 1, 50)
    
    plt.subplot(1, 3, 1)
    plt.plot(df['epoch'], df['train_losses'], label='Train Loss')
    plt.plot(df['epoch'], df['val_losses'], label='Valid Loss')
    plt.title('Training and Validation Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.xticks(xticks_range)
    # Trục x đến max_epoch, trục y đến max loss thật
    plt.xlim(0, df['epoch'].max())
    plt.ylim(0, max(df['train_losses'].max(), df['val_losses'].max()))

    # Tỷ lệ 1:1 nhưng giữ nguyên scale gốc
    plt.gca().set_aspect('auto', adjustable='box')

    # Plot Dice Coefficients
    plt.subplot(1, 3, 2)
    plt.plot(df['epoch'], df['train_dices'], label='Train Dice')
    plt.plot(df['epoch'], df['val_dices'], label='Valid Dice')
    plt.title('Training and Validation Dice Coefficients')
    plt.xlabel('Epoch')
    plt.ylabel('Dice Coefficient')
    plt.legend()
    plt.xticks(xticks_range)
    # plt.gca().set_aspect('equal', adjustable='box')
    plt.xlim(0, max_epoch)
    plt.ylim(0, max(df['train_dices'].max(), df['val_dices'].max()))
    plt.gca().set_aspect('auto', adjustable='box')
    
    # Plot IOU
    plt.subplot(1, 3, 3)
    plt.plot(df['epoch'], df['train_ious'], label='Train Iou')
    plt.plot(df['epoch'], df['val_ious'], label='Valid Iou')
    plt.title('Training and Validation IOU')
    plt.xlabel('Epoch')
    plt.ylabel('IOU')
    plt.legend()
    plt.xticks(xticks_range)
    # plt.gca().set_aspect('equal', adjustable='box')
    # Cân bằng trục x/y
    plt.xlim(0, max_epoch)
    plt.ylim(0, max(df['train_ious'].max(), df['val_ious'].max()))
    plt.gca().set_aspect('auto', adjustable='box')
    
    # Vẽ đồ thị
    plt.tight_layout()
        # Save the plot to a file
    source_file4="metrics_from_excel.png"
    output_metric = os.path.join(output_folder,source_file4)
    plt.savefig(output_metric, dpi=300)  # Tùy chỉnh độ phân giải với tham số dpi
    # source_folder = "/content/output" =>Bỏ

    # destination_folder = "/content/drive/MyDrive/ISIC/output_02-03-2025_PreTrain8With_Dice-CrossELoss_50loop"
    # destination_folder=os.path.join(destination_folder,path)
    # os.makedirs(destination_folder, exist_ok=True)
    # shutil.copytree(output_folder, destination_folder, dirs_exist_ok=True)

    plt.show()
    plt.close()
def export_evaluate(trainer):
    output_folder = BASE_OUTPUT
    os.makedirs(output_folder, exist_ok=True)
    df = pd.DataFrame({
        'ImagePath': trainer.path_list,
        'Dice': trainer.dice_list,
        'IoU': trainer.iou_list
    })
    result_csv = "test_metrics_with_paths.csv"
    output_result = os.path.join(output_folder, result_csv)
    df.to_csv(output_result, index=False)
