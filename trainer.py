from config import *
from utils import *
from optimizer import *

class Trainer:
    def __init__(self, model, optimizer, criterion=loss_func, patience=20, device=DEVICE):
        self.device = device
        self.model = model.to(self.device)
        self.num_epochs = NUM_EPOCHS
        self.criterion = criterion
        self.patience = patience
        self.optimizer = optimizer
        
        # Tracking metrics
        self.early_stop_counter = 0
        self.train_losses, self.val_losses = [], []
        self.train_dices, self.val_dices = [], []
        self.train_ious, self.val_ious = [], []
        
        # Best metrics tracking
        self.best_dice, self.best_epoch_dice = 0.0, 0
        self.best_iou, self.best_epoch_iou = 0.0, 0
        
        # --- THÊM: Theo dõi Best Val Loss cho Early Stopping ---
        self.best_val_loss = float('inf') 
        self.best_epoch_loss = 0
        
        self.log_interval = 1
        self.dice_list = []
        self.iou_list = []
        self.path_list = []
        
        # AMP & Scheduler
        self.use_amp = torch.cuda.is_available()
        self.scaler = GradScaler(enabled=self.use_amp)
        
        # Scheduler
        self.scheduler = CosineAnnealingWarmRestarts(self.optimizer, T_0=10, T_mult=2, eta_min=1e-6)
        
        # Start epoch
        self.start_epoch = 0

    def save_checkpoint(self, epoch, dice, iou, filename):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),   
            'scaler_state_dict': self.scaler.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_dices': self.train_dices,
            'val_dices': self.val_dices,
            'train_ious': self.train_ious,
            'val_ious': self.val_ious,
            'best_dice': self.best_dice,
            'best_iou': self.best_iou,
            'best_epoch_dice': self.best_epoch_dice,
            'best_epoch_iou': self.best_epoch_iou,
            # --- THÊM: Lưu best_val_loss ---
            'best_val_loss': self.best_val_loss,
            'best_epoch_loss': self.best_epoch_loss
        }
        torch.save(checkpoint, filename)

    def load_checkpoint(self, path):
        print(f"[INFO] Loading checkpoint: {path}")
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
            
        self.start_epoch = checkpoint['epoch'] 
        
        # Load history
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.train_dices = checkpoint.get('train_dices', [])
        self.val_dices = checkpoint.get('val_dices', [])
        self.train_ious = checkpoint.get('train_ious', [])
        self.val_ious = checkpoint.get('val_ious', [])
        
        self.best_dice = checkpoint.get('best_dice', 0.0)
        self.best_iou = checkpoint.get('best_iou', 0.0)
        self.best_epoch_dice = checkpoint.get('best_epoch_dice', 0)
        self.best_epoch_iou = checkpoint.get('best_epoch_iou', 0)
        
        # --- THÊM: Load best_val_loss ---
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.best_epoch_loss = checkpoint.get('best_epoch_loss', 0)
        
        print(f"[INFO] Loaded checkpoint from epoch {self.start_epoch}")

    def run_epoch(self, loader, is_train=True):
        """Hàm chung để chạy train hoặc validation cho 1 epoch"""
        self.model.train() if is_train else self.model.eval()
        
        epoch_loss = 0.0
        epoch_dice = 0.0
        epoch_iou = 0.0
        
        desc = "Training" if is_train else "Validation"
        loader_bar = tqdm(enumerate(loader), total=len(loader), desc=desc, leave=False)
        
        for i, (images, masks, _) in loader_bar:
            images, masks = images.to(self.device), masks.to(self.device)
            
            if is_train:
                self.optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=self.use_amp):
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                loss = torch.mean(loss)
                
                with torch.no_grad():
                    # Truyền thẳng logits, hàm utils sẽ tự sigmoid -> threshold
                    batch_dice = torch.mean(dice_coeff_hard(outputs, masks))
                    batch_iou = torch.mean(iou_core_hard(outputs, masks))

            if is_train:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                
                self.scheduler.step(self.current_epoch + i / len(loader)) 

            epoch_loss += loss.item()
            epoch_dice += batch_dice.item()
            epoch_iou += batch_iou.item()

            if (i + 1) % self.log_interval == 0:
                loader_bar.set_postfix({
                    'L': f"{loss.item():.4f}", 
                    'D_Hard': f"{batch_dice.item():.4f}", 
                    'I_Hard': f"{batch_iou.item():.4f}"
                })
        
        avg_loss = epoch_loss / len(loader)
        avg_dice = epoch_dice / len(loader)
        avg_iou = epoch_iou / len(loader)
        
        return avg_loss, avg_dice, avg_iou

    def train(self, train_loader, val_loader, resume_path=None):
        print("-" * 30)
        print(f"Device: {self.device}")
        print(f"Num Epochs: {self.num_epochs}")
        print(f"Early Stopping Monitor: Val Loss (Patience={self.patience})")
        print(f"Best Model Monitor: Val IoU (Hard Metric)")
        print("-" * 30)

        if resume_path:
            self.load_checkpoint(resume_path)
        
        start_time = time.time()
        print(f"[INFO] Starting training from epoch {self.start_epoch + 1}...")

        for epoch in range(self.start_epoch, self.num_epochs):
            self.current_epoch = epoch 
            
            # --- Training ---
            train_loss, train_dice, train_iou = self.run_epoch(train_loader, is_train=True)
            
            # --- Validation ---
            with torch.no_grad():
                val_loss, val_dice, val_iou = self.run_epoch(val_loader, is_train=False)

            # --- Logging ---
            current_lr = self.scheduler.get_last_lr()[0]
            print(f"Epoch {epoch+1}/{self.num_epochs} | LR: {current_lr:.2e}")
            print(f"Train - Loss: {train_loss:.4f}, Dice: {train_dice:.4f}, IoU: {train_iou:.4f}")
            print(f"Val   - Loss: {val_loss:.4f}, Dice: {val_dice:.4f}, IoU: {val_iou:.4f}")

            # Lưu history
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_dices.append(train_dice)
            self.val_dices.append(val_dice)
            self.train_ious.append(train_iou)
            self.val_ious.append(val_iou)

            # --- Checkpoint & Logic tách biệt ---
            
            # 1. Luôn lưu model mới nhất
            self.save_checkpoint(epoch + 1, self.best_dice, self.best_iou, 'last_model.pth')
            if val_dice > self.best_dice:
                self.best_dice = val_dice
                self.best_epoch_dice = epoch + 1
                self.save_checkpoint(epoch + 1, self.best_dice, self.best_iou, 'best_dice_model.pth')
                print(f"[*] New best Dice: {self.best_dice:.4f} at epoch {epoch+1}")
            # 2. Lưu BEST MODEL dựa trên IoU (Theo yêu cầu)
            if val_iou > self.best_iou:
                self.best_iou = val_iou
                self.best_epoch_iou = epoch + 1
                
                self.save_checkpoint(epoch + 1, self.best_dice, self.best_iou, 'best_iou_model.pth')
                print(f"[*] New best IoU: {self.best_iou:.4f} at epoch {epoch+1}")

            # 3. EARLY STOPPING dựa trên Val Loss (Theo yêu cầu)
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_epoch_loss = epoch + 1
                self.early_stop_counter = 0 # Reset counter vì loss giảm (tốt lên)
                # print(f"[*] Best Loss updated: {self.best_val_loss:.4f}") 
            else:
                self.early_stop_counter += 1
                print(f"[!] Loss didn't improve. EarlyStopping counter: {self.early_stop_counter}/{self.patience}")

            if self.early_stop_counter >= self.patience:
                print(f"[INFO] Early stopping triggered at epoch {epoch + 1} due to no improvement in Loss.")
                break
            
            # Cleanup
            gc.collect()
            torch.cuda.empty_cache()

        total_time = time.time() - start_time
        print(f"[INFO] Training completed in {total_time // 60:.0f}m {total_time % 60:.0f}s")
    def evaluate(self, test_loader, checkpoint_path=None, save_visuals=False, output_dir="test_results"):
        """
        Đã cập nhật để hỗ trợ xuất ảnh dự đoán
        """
        if checkpoint_path:
            self.load_checkpoint(checkpoint_path)
        
        self.model.eval()
        self.dice_list, self.iou_list, self.path_list = [], [], []
        
        # Tạo thư mục lưu ảnh nếu cần
        if save_visuals:
            os.makedirs(output_dir, exist_ok=True)
            print(f"[INFO] Saving visualization results to: {output_dir}")

        with torch.no_grad():
            test_bar = tqdm(enumerate(test_loader), total=len(test_loader), desc="Testing")
            for i, (images, masks, image_paths) in test_bar:
                images, masks = images.to(self.device), masks.to(self.device)
                # Forward pass
                logits = self.model(images)
                # 1. Tính Metric (Truyền logits thẳng vào, hàm hard tự lo phần còn lại)
                batch_dices = dice_coeff_hard(logits, masks)
                batch_ious = iou_core_hard(logits, masks)
                if save_visuals:
                    # Tính xác suất để visualize (0 -> 1)
                    probs = torch.sigmoid(logits)
                    # Tạo mask nhị phân (0 hoặc 1) để vẽ
                    preds = (probs > 0.5).float()
                # Lặp từng ảnh trong batch để tính metric và vẽ
                for j in range(images.size(0)):
                    d = batch_dices[j].item()
                    ious = batch_ious[j].item()
                    path = image_paths[j]
                    # # Lấy từng mẫu đơn lẻ
                    # img_single = images[j]
                    # mask_single = masks[j]
                    # pred_single = preds[j]
                    # path = image_paths[j]
                    # logit_single = logits[j] # Lấy logit thô
                    # mask_single = masks[j]
                    # # Cần unsqueeze(0) để khớp dimension tính metric (C, H, W) -> (1, C, H, W)
                    # d = dice_coeff(logit_single.unsqueeze(0), mask_single.unsqueeze(0)).item()
                    # ious = iou_core(logit_single.unsqueeze(0), mask_single.unsqueeze(0)).item()
                    
                    self.dice_list.append(d)
                    self.iou_list.append(ious)
                    self.path_list.append(path)
                    
                    # --- PHẦN BỔ SUNG: VẼ ẢNH ---
                    if save_visuals:
                        # Lấy tên file gốc
                        file_name = os.path.basename(path)
                        # Prefix NORM/MASS
                        is_normal = (masks[j].sum() == 0)
                        prefix = "NORM" if is_normal else "MASS"
                        save_name = f"pred_{prefix}_D{d:.2f}_{file_name}"
                        save_full_path = os.path.join(output_dir, save_name)
                        visualize_prediction(
                            img_tensor=images[j],
                            mask_tensor=masks[j],
                            pred_tensor=preds[j], # Dùng preds đã tính ở trên
                            save_path=save_full_path,
                            iou_score=ious,
                            dice_score=d
                        )
                        # visualize_prediction(
                        #     img_tensor=img_single,
                        #     mask_tensor=mask_single,
                        #     pred_tensor=pred_single,
                        #     save_path=save_full_path,
                        #     iou_score=ious,
                        #     dice_score=d
                        # )
                    # -----------------------------

        avg_dice = sum(self.dice_list) / len(self.dice_list)
        avg_iou = sum(self.iou_list) / len(self.iou_list)
        
        print(f"\n[TEST RESULT] Avg Hard Dice: {avg_dice:.4f}, Avg Hard IoU: {avg_iou:.4f}")
        return avg_dice, avg_iou, self.dice_list, self.iou_list, self.path_list

    # def evaluate(self, test_loader, checkpoint_path=None):
    #     if checkpoint_path:
    #         self.load_checkpoint(checkpoint_path)
        
    #     self.model.eval()
    #     self.dice_list, self.iou_list, self.path_list = [], [], []
        
    #     with torch.no_grad():
    #         test_bar = tqdm(enumerate(test_loader), total=len(test_loader), desc="Testing")
    #         for i, (images, masks, image_paths) in test_bar:
    #             images, masks = images.to(self.device), masks.to(self.device)
                
    #             outputs = self.model(images)

    #             for j in range(images.size(0)):
    #                 output = outputs[j].unsqueeze(0)
    #                 mask = masks[j].unsqueeze(0)
                    
    #                 d = dice_coeff(output, mask).item()
    #                 ious = iou_core(output, mask).item()
                    
    #                 self.dice_list.append(d)
    #                 self.iou_list.append(ious)
    #                 self.path_list.append(image_paths[j])

    #     avg_dice = sum(self.dice_list) / len(self.dice_list)
    #     avg_iou = sum(self.iou_list) / len(self.iou_list)
        
    #     print(f"\n[TEST RESULT] Avg Dice: {avg_dice:.4f}, Avg IoU: {avg_iou:.4f}")
    #     return avg_dice, avg_iou, self.dice_list, self.iou_list, self.path_list

    def get_metrics(self):
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_dices': self.train_dices,
            'val_dices': self.val_dices,
            'train_ious': self.train_ious,
            'val_ious': self.val_ious,
            'best_dice': self.best_dice,
            'best_epoch_dice': self.best_epoch_dice,
            'best_iou': self.best_iou,
            'best_epoch_iou': self.best_epoch_iou,
            'best_val_loss': self.best_val_loss,
            'best_epoch_loss': self.best_epoch_loss,
        }
