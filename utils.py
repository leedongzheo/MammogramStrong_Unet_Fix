from config import*

def to_numpy(tensor):
    # Move tensor to CPU and convert to NumPy array
    return tensor.cpu().detach().item()
# ====== METRICS ======
def dice_coeff_per_image(logits, target, smooth=1e-6):
    
    probs = torch.sigmoid(logits)
    # logits, target: [B, 1, H, W]
    probs = probs.view(probs.size(0), -1)    # [B, N]
    target = target.view(target.size(0), -1) # [B, N]

    intersection = (probs * target).sum(dim=1)  # [B]
    union = probs.sum(dim=1) + target.sum(dim=1)
    dice = (2.0 * intersection + smooth) / (union + smooth)  # [B]
    return dice  # [batch]
# Hàm "dice_coeff_per_image" này tương đương với hàm bên dưới "dice_coeff":
def dice_coeff(logits, target, epsilon=1e-6):
    probs = torch.sigmoid(logits)
    numerator = 2 * torch.sum(target * probs, dim=(1, 2, 3))
    denominator = torch.sum(target + probs, dim=(1, 2, 3))
    dice = (numerator + epsilon) / (denominator + epsilon)
    return dice  # [batch]

def dice_coef_loss_per_image(logits, targets):
    """
    logits : [B, 1, H, W] (raw output từ model, chưa sigmoid)
    targets: [B, 1, H, W] hoặc [B, H, W] (0/1)
    
    """
    # Đảm bảo targets có shape [B, 1, H, W]
    if targets.ndim == 3:
        targets = targets.unsqueeze(1)
    dice = dice_coeff(logits, targets)
    loss = 1.0 - dice
    return loss  # [B]
def binary_focal_loss(logits,
                      targets,
                      alpha: float = 0.8,
                      gamma: float = 2.0,
                      eps: float = 1e-6):
    """
    logits : [B, 1, H, W] (raw logits)
    targets: [B, 1, H, W] hoặc [B, H, W] (0 hoặc 1)
    alpha  : trọng số cho class dương (positive class)
    gamma  : tham số fokal
    return : scalar focal loss
    """
    if targets.ndim == 3:
        targets = targets.unsqueeze(1)

    # BCE từng pixel, không reduce
    bce_loss = F.binary_cross_entropy_with_logits(
        logits, targets, reduction="none"
    )  # [B, 1, H, W]

    # xác suất đúng pt (cho mỗi pixel)
    probs = torch.sigmoid(logits)
    pt = probs * targets + (1 - probs) * (1 - targets)  # [B, 1, H, W]

    # Focal factor
    focal_factor = (1 - pt).clamp(min=eps).pow(gamma)

    # Alpha cho positive / negative
    alpha_factor = targets * alpha + (1 - targets) * (1 - alpha)

    loss = alpha_factor * focal_factor * bce_loss  # [B, 1, H, W]

    return loss.mean()  # scalar
# Detail của binary_focal_loss => Gỡ comment:
def binary_focal_loss_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float | None = None,
    gamma: float = 2.0,
    reduction: str = "mean",
    eps: float = 1e-7,
) -> torch.Tensor:
    """
    Focal loss nhị phân, giống style smp.losses.FocalLoss(mode='binary').

    Args:
        logits:  [B, 1, H, W] (raw model output, chưa sigmoid)
        targets: [B, 1, H, W] hoặc [B, H, W] (0/1)
        alpha:   None (mặc định smp) hoặc float (0..1) nếu muốn weight lớp dương
        gamma:   2.0 (mặc định smp)
        reduction: 'none' | 'mean' | 'sum'
        eps:     số nhỏ chống log(0)

    Returns:
        loss: scalar (nếu reduction != 'none') hoặc tensor cùng shape với logits
    """
    if targets.ndim == 3:
        targets = targets.unsqueeze(1)   # [B, 1, H, W]

    # BCE từng pixel không reduce
    bce = F.binary_cross_entropy_with_logits(
        logits, targets, reduction="none"
    )  # [B, 1, H, W]

    # p_t = p nếu y=1, = 1-p nếu y=0
    probs = torch.sigmoid(logits)
    pt = probs * targets + (1 - probs) * (1 - targets)   # [B, 1, H, W]
    pt = pt.clamp(min=eps, max=1.0 - eps)

    # (1 - p_t)^gamma
    focal_factor = (1.0 - pt) ** gamma

    # alpha-balance: nếu alpha=None => không dùng weighting
    if alpha is not None:
        # cho bài toán binary: weight positive = alpha, negative = 1-alpha
        alpha_t = targets * alpha + (1 - targets) * (1 - alpha)
    else:
        alpha_t = 1.0

    loss = alpha_t * focal_factor * bce  # [B, 1, H, W]

    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    else:  # 'none'
        return loss
class BinaryFocalLoss(torch.nn.Module):
    def __init__(
        self,
        alpha: float | None = None,
        gamma: float = 2.0,
        reduction: str = "mean",
        ignore_index: int | None = None,   # để đó cho giống smp, ta không dùng ở đây
    ):
        """
        Gần giống smp.losses.FocalLoss(mode='binary').

        mode='binary' => chúng ta chỉ xử lý case nhị phân.
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Ở bản smp, ignore_index dùng để bỏ qua pixel có label = ignore_index.
        # Nếu bạn muốn dùng, có thể mask bớt ở đây. Tạm thời ta bỏ qua (giống behavior cơ bản).
        return binary_focal_loss_with_logits(
            logits,
            targets,
            alpha=self.alpha,
            gamma=self.gamma,
            reduction=self.reduction,
        )

def hybric_loss(logits, target):
    dice_loss  = dice_coef_loss_per_image(logits, target)  # như bạn đã có
    focal_loss = BinaryFocalLoss(alpha=None, gamma=2.0, reduction="mean")
    loss = dice_loss + 0.5 * focal_loss(logits, target)
    return loss #default: reduction="mean"

    # Loss: Combo (Dice + Focal)
def hybric_loss_lib(logits, target):
    
    dice_loss = smp.losses.DiceLoss(mode="binary")
    focal_loss = smp.losses.FocalLoss(mode="binary")
    loss = dice_loss(logits, target) + 0.5 * focal_loss(logits, target)
    return loss #default: reduction="mean"

def dice_coeff_global_batch(logits, target, smooth=1e-6):
    probs = torch.sigmoid(logits)
    intersection = (probs * target).sum()
    union = probs.sum() + target.sum()
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return dice  # [batch]

# ====== DICE LOSS (logits -> prob) ======
def dice_coef_loss_global_batch(logits, target):
    dice = dice_coeff_global_batch(logits, target)
    loss = 1.0 - dice
    return loss

def dice_coeff_per_batch(logits, target, smooth=1e-6):
    """
    Trả về Dice loss cho từng mẫu trong batch.
    logits: [B, 1, H, W]  (logits từ model)
    target: [B, 1, H, W]  (mask 0/1)
    return: [B]  (mỗi phần tử là dice_loss của 1 ảnh)
    """
    probs = torch.sigmoid(logits)          # [B, 1, H, W]
    
    # Flatten theo từng ảnh: [B, 1*H*W] -> [B, N]
    probs_flat  = probs.view(probs.size(0), -1)
    target_flat = target.view(target.size(0), -1)

    intersection = (probs_flat * target_flat).sum(dim=1)          # [B]
    union        = probs_flat.sum(dim=1) + target_flat.sum(dim=1) # [B]

    dice = (2.0 * intersection + smooth) / (union + smooth) # [B]
                                    # [B]
    return dice    # không .mean(), để trainer tự xử lý 

def dice_coef_loss_per_batch(logits, target, smooth=1e-6):
    dice  = dice_coeff_per_batch(logits, target)   # [B]
    loss = 1.0 - dice    # không .mean(), để trainer tự xử lý
    return loss

# ====== BCE + Dice (dùng logits) ======
_bce_logits = nn.BCEWithLogitsLoss()  # tạo 1 lần

# def bce_dice_loss(logits, target):
#     bce = _bce_logits(logits, target)
#     dice = dice_coeff(logits, target)
#     loss = bce + dice
#     return loss
def bce_dice_loss(logits, target):
    bce = _bce_logits(logits, target)
    dice = dice_coeff(logits, target).mean()
    loss = bce + (1.0 - dice)
    return loss
# ====== BCE with pos_weight (logits) ======
def bce_weight_loss(logits, target, pos_weight=231.2575):
    pos_w = torch.tensor(pos_weight, device=logits.device)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    loss = bce(logits, target)
    return loss

# ====== BCE(pos_weight) + Dice ======
def bce_dice_weight_loss(logits, target, pos_weight=231.2575):
    bce_w = bce_weight_loss(logits, target, pos_weight)
    dice = dice_coeff(logits, target)
    loss = bce_w + dice
    return loss



def iou_core(logits, target, epsilon=1e-6):
    probs = torch.sigmoid(logits)
    intersection = torch.sum(probs * target, dim=(1, 2, 3))
    union = torch.sum(probs, dim=(1, 2, 3)) + torch.sum(target, dim=(1, 2, 3)) - intersection
    iou = (intersection + epsilon) / (union + epsilon)
    return iou

# ====== Soft Dice Loss (log-dice) ======
def soft_dice_loss(logits, targets, gamma=0.3, eps=1e-6):
    """
    Soft Dice Loss dạng (-log(dice))^gamma, lấy mean trên batch.
    """
    dice = dice_coeff(logits, targets, eps=eps)     # [B]
    dice = dice.clamp(min=eps, max=1.0)            # tránh log(0)
    log_dice = -torch.log(dice)                    # [B]
    loss = log_dice.pow(gamma)                     # [B]
    return loss.mean()                             # scalar
# ====== HARD METRICS (Cho Evaluation - Input là 0 hoặc 1) ======

# --- Cập nhật vào utils.py ---

def dice_coeff_hard(logits, target, threshold=0.5, epsilon=1e-6):
    """
    Tính Dice Score (Hard Metric) cho đánh giá.
    logits: Tensor raw từ model [B, 1, H, W] (CHƯA qua sigmoid)
    target: Tensor Ground Truth [B, 1, H, W] (0 hoặc 1)
    """
    # 1. Tự động chuyển Logits -> Probs -> Binary Preds (0/1) bên trong hàm
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()
    
    # 2. Flatten
    preds_flat = preds.view(preds.size(0), -1)
    target_flat = target.view(target.size(0), -1)
    
    # 3. Tính toán (Có epsilon để xử lý Normal image 0/0 -> 1.0)
    intersection = (preds_flat * target_flat).sum(dim=1)
    dice = (2. * intersection + epsilon) / (preds_flat.sum(dim=1) + target_flat.sum(dim=1) + epsilon)
    
    return dice # Trả về tensor [Batch_size]

def iou_core_hard(logits, target, threshold=0.5, epsilon=1e-6):
    """
    Tính IoU Score (Hard Metric) cho đánh giá.
    """
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()
    
    preds_flat = preds.view(preds.size(0), -1)
    target_flat = target.view(target.size(0), -1)
    
    intersection = (preds_flat * target_flat).sum(dim=1)
    union = preds_flat.sum(dim=1) + target_flat.sum(dim=1) - intersection
    
    iou = (intersection + epsilon) / (union + epsilon)
    
    return iou # Trả về tensor [Batch_size]
    
class ComboLoss(nn.Module):
    def __init__(self, alpha=0.5, ce_ratio=0.5, focal_gamma=2.0):
        """
        alpha (float): Trọng số cho lớp Positive (Mass). 
                       Nếu alpha > 0.5 -> Ưu tiên Mass.
                       Nếu alpha < 0.5 -> Giảm ưu tiên Background (thường dùng trong Focal).
        """
        super().__init__()
        self.alpha = alpha
        self.ce_ratio = ce_ratio
        self.focal_gamma = focal_gamma

    def forward(self, logits, targets):
        # Dice Loss (Không dùng weight, vì bản chất Dice đã tự cân bằng vùng giao thoa)
        dice_loss = dice_coef_loss_per_image(logits, targets).mean()
        
        # Focal Loss (Dùng alpha làm Class Weight)
        focal_loss = binary_focal_loss_with_logits(
            logits, 
            targets, 
            alpha=self.alpha, 
            gamma=self.focal_gamma, 
            reduction="mean"
        )
        
        combo = self.ce_ratio * focal_loss + (1 - self.ce_ratio) * dice_loss
        return combo
def unnormalize(img_tensor):
    """
    Chuyển Tensor (đã normalize ImageNet) về lại ảnh gốc để vẽ
    Mean: [0.485, 0.456, 0.406], Std: [0.229, 0.224, 0.225]
    """
    # Chuyển về CPU và numpy: (C, H, W) -> (H, W, C)
    img = img_tensor.cpu().numpy().transpose(1, 2, 0)
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    # Công thức ngược: x * std + mean
    img = img * std + mean
    
    # Clip giá trị về [0, 1] để tránh lỗi hiển thị
    img = np.clip(img, 0, 1)
    return img

def visualize_prediction(img_tensor, mask_tensor, pred_tensor, save_path, iou_score, dice_score):
    """
    Vẽ và lưu ảnh so sánh: Gốc - Mask thật - Dự đoán
    """
    # 1. Chuẩn bị dữ liệu
    orig_img = unnormalize(img_tensor)
    
    gt_mask = mask_tensor.squeeze().cpu().numpy()
    pred_mask = pred_tensor.squeeze().cpu().numpy()
    
    # 2. Vẽ biểu đồ
    plt.figure(figsize=(12, 4))
    
    # Ảnh gốc
    plt.subplot(1, 3, 1)
    plt.imshow(orig_img)
    plt.title("Ảnh gốc")
    plt.axis('off')
    
    # Ground Truth (Mask thật)
    plt.subplot(1, 3, 2)
    plt.imshow(gt_mask, cmap='gray')
    plt.title("Ground Truth")
    plt.axis('off')
    
    # Kết quả Overlay (Chồng lớp)
    plt.subplot(1, 3, 3)
    plt.imshow(orig_img)
    # Tô màu xanh lá cho Mask thật
    plt.imshow(np.ma.masked_where(gt_mask == 0, gt_mask), cmap='Greens', alpha=0.4)
    # Tô màu đỏ cho Mask dự đoán
    plt.imshow(np.ma.masked_where(pred_mask == 0, pred_mask), cmap='Reds', alpha=0.4)
    plt.title(f"Prediction\nIoU: {iou_score:.2f} | Dice: {dice_score:.2f}")
    plt.axis('off')
    
    # 3. Lưu ảnh
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def loss_func(logits, targets):
    """Dùng CURRENT_LOSS_NAME để quyết định loss nào sẽ được dùng"""
    # QUAN TRỌNG: Phải .mean() để ra scalar
    if loss == "Dice_loss":
        return dice_coef_loss_per_image(logits, targets).mean()
    elif loss == "Hybric_loss":
        return hybric_loss_lib(logits, targets)
    elif loss == "BCEDice_loss":
        return bce_dice_loss(logits, targets)
    elif loss == "BCEw_loss":
        return bce_weight_loss(logits, targets)
    elif loss == "Combo_loss": 
        # Khởi tạo class (nên khởi tạo 1 lần bên ngoài loop train thì tốt hơn, nhưng để đây cho gọn logic)
        criterion = ComboLoss(alpha=0.25, ce_ratio=0.5, focal_gamma=2.0)
        return criterion(logits, targets)
    elif loss == "BCEwDice_loss":
        return bce_dice_weight_loss(logits, targets)
    elif loss == "SoftDice_loss":
        return soft_dice_loss(logits, targets)
    else:
        raise ValueError(f"Unknown loss: {loss}")
    

# -------------------------------------------------------
# def dice_coef_loss(inputs, target, smooth=1e-6):
#     """
#     Dice Loss: Thước đo sự chồng lấn giữa output và ground truth.
#     """
#     inputs = torch.sigmoid(inputs)  # Chuyển logits về xác suất
#     intersection = (inputs * target).sum()
#     union = inputs.sum() + target.sum()
#     dice_score = (2.0 * intersection + smooth) / (union + smooth)
#     return 1 - dice_score  # Dice loss
# def bce_dice_loss(inputs, target):
#     dice_score = dice_coef_loss(inputs, target)
#     bce_loss = nn.BCELoss()
#     bce_score = bce_loss(inputs, target)  # yêu cầu đầu vào signmoid rồi => ko dùng code này do model chưa signmoid
#     return bce_score + dice_score
# def bce_weight_loss(inputs, target, pos_weight = 231.2575):
#     bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]).to(DEVICE)) #=> yêu cầu model chưa signmoid => dùng code này được
#     bce_w_loss = bce(inputs, target)
#     return bce_w_loss
# def bce_dice_weight_loss(inputs, targets):
#     bce_w_loss = bce_weight_loss(inputs, targets) # yêu cầu model chưa signmoid => dùng code này được
#     dice = dice_coef_loss(inputs, targets)
#     return bce_w_loss + dice
    
# def tensor_to_float(value):
#     if isinstance(value, torch.Tensor):
#         return value.cpu().item()  # Chuyển tensor về CPU và lấy giá trị float
#     elif isinstance(value, list):
#         return [tensor_to_float(v) for v in value]  # Xử lý danh sách các tensor
#     return value  # Nếu không phải tensor, giữ nguyên

# # def dice_coeff(pred, target, smooth=1e-5):
# #     pred = torch.sigmoid(pred)  # Chuyển logits về xác suất
# #     intersection = torch.sum(pred * target)
# #     return (2. * intersection + smooth) / (torch.sum(pred) + torch.sum(target) + smooth)
# def dice_coeff(pred, target, epsilon=1e-6):
#     # print("y_pred_shape1: ", pred.shape)
#     # print("target_max:", target.max())
#     y_pred = torch.sigmoid(pred)  # Chuyển logits về xác suất
#     # print("y_pred_max:", y_pred.max())
#     # print("y_pred_shape2: ", y_pred.shape)
#     # print("y_target_shape: ", target.shape)
#     numerator = 2 * torch.sum(target * y_pred, dim=(1, 2, 3))
#     denominator = torch.sum(target + y_pred, dim=(1, 2, 3))
#     dice = (numerator + epsilon) / (denominator + epsilon)
#     # print("shape_dice: ", dice.shape)
#     # return torch.mean(dice)
#     return dice

# # def iou_core(y_pred, y_true, eps=1e-7):
# #     y_pred = torch.sigmoid(y_pred) 
# #     y_true_f = y_true.view(-1)  # flatten
# #     y_pred_f = y_pred.view(-1)  # flatten

# #     intersection = torch.sum(y_true_f * y_pred_f)
# #     union = torch.sum(y_true_f) + torch.sum(y_pred_f) - intersection

# #     return intersection / (union + eps)  # thêm eps để tránh chia 0
# # import torch
# # import torch.nn.functional as F
# def iou_core(pred, target, epsilon=1e-6):
#     pred = torch.sigmoid(pred)  # Chuyển logits về xác suất
#     # Tính intersection và union theo từng ảnh
#     intersection = torch.sum(pred * target, dim=(1, 2, 3))  # Batch_size x 1
#     # print("pred.shape:", pred.shape)
#     # print("target.shape:", target.shape)

#     union = torch.sum(pred, dim=(1, 2, 3)) + torch.sum(target, dim=(1, 2, 3)) - intersection
#     iou = (intersection + epsilon) / (union + epsilon)
#     return iou  # Giữ nguyên theo batch, mỗi ảnh 1 giá trị

# def soft_dice_loss(inputs, targets, gamma=0.3):
#     """
#     Soft Dice Loss dạng Log-Dice, dùng cho segmentation.

#     Args:
#         y_true: Tensor ground truth, shape (batch_size, H, W)
#         y_pred: Tensor prediction, shape (batch_size, H, W)
#         epsilon: Giá trị nhỏ để tránh chia cho 0.
#         gamma: Hệ số mũ cho log-dice.

#     Returns:
#         loss: scalar loss value.
#     """
#     # y_pred = torch.sigmoid(y_pred) 
#     # numerator = 2 * torch.sum(y_true * y_pred, dim=(1, 2))
#     # denominator = torch.sum(y_true + y_pred, dim=(1, 2))
#     # dice = (numerator + epsilon) / (denominator + epsilon)
#     # dice = dice_coeff(y_pred, y_true)
#     # -------------------------------------------
#     # Debug
#     # print("dice_in_step:", dice)
#     # print("dice_in_step_shape:", dice.shape)
#     dice = dice_coef_loss(inputs, targets)
    
#     log_dice = -torch.log(dice)
#     # print("log_dice_in_step", log_dice)
#     loss = torch.pow(log_dice, gamma)
#     # print("loss_in_step", loss)
#     # print("loss_in_step_shape:", loss.shape)
#     loss_mean = torch.mean(loss)
#     # print("loss_mean_in_step", loss_mean)
#     # print("loss_mean_in_step_shape:", loss_mean.shape)
#     return loss_mean

# def inan():
# def loss_func(*kwargs):
#     args = get_args()
#     if args.loss == "Dice_loss":
#         x = dice_coef_loss(*kwargs)
#         return x
#     elif args.loss == "BCEDice_loss":
#         x = bce_dice_loss(*kwargs)
#         return x
#     elif args.loss == "BCEw_loss":
#         x = bce_weight_loss(*kwargs)
#         return x
#     elif args.loss == "BCEwDice_loss":
#         x = bce_dice_weight_loss(*kwargs)
#         return x
#     elif args.loss == "SoftDice_loss":
#         x = soft_dice_loss(*kwargs)
#         return x


    
