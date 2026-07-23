import os
import sys
import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.models.unet_rgbd import LateFusionUNet
from backend.services.preprocessing import preprocess
from backend.services.segmentation import find_wheals

def compute_iou(preds, targets):
    intersection = (preds * targets).sum()
    union = preds.sum() + targets.sum() - intersection
    if union == 0:
        return 1.0
    return intersection / union

def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load the test image
    img_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "Testphotos", "allergy-Testing.jpg"))
    if not os.path.exists(img_path):
        print(f"Test image not found at {img_path}")
        return 100.0

    image = cv2.imread(img_path)
    if image is None:
        print("Failed to read test image")
        return 100.0

    # 2. Preprocess and get ground truth
    prep = preprocess(image)
    H, W = prep["sam_ready_image"].shape[:2]
    sam_mask = np.zeros((H, W), dtype=np.uint8)
    sam_diameters = []
    
    manual_mask_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "Testphotos", "allergy-Testing_mask.png"))
    if os.path.exists(manual_mask_path):
        print(f"Found manual mask at {manual_mask_path}")
        manual_mask = cv2.imread(manual_mask_path, cv2.IMREAD_GRAYSCALE)
        if manual_mask.shape[:2] != (H, W):
            manual_mask = cv2.resize(manual_mask, (W, H), interpolation=cv2.INTER_NEAREST)
        
        sam_mask = (manual_mask > 127).astype(np.uint8)
        
        # Calculate diameter from the largest manual mask contour
        contours, _ = cv2.findContours(sam_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        ppm = 10.0
        if contours:
            # Get the largest contour by area
            largest_cnt = max(contours, key=cv2.contourArea)
            area_px = cv2.contourArea(largest_cnt)
            if area_px > 10:
                (_, _), radius = cv2.minEnclosingCircle(largest_cnt)
                sam_diameters.append(round((radius * 2) / ppm, 2))
    else:
        try:
            wheals = find_wheals(prep, ppm=10.0) # arbitrary ppm for testing
            sam_diameters = [round(w.diameter_mm, 2) for w in wheals]
            for w in wheals:
                cv2.drawContours(sam_mask, [w.contour], -1, 1, -1)
            if len(wheals) == 0:
                print("SAM found no wheals or wasn't loaded, using dummy mask.")
                cv2.circle(sam_mask, (W//2, H//2), 30, 1, -1)
        except FileNotFoundError as e:
            print(f"SAM error: {e}")

    sam_mask = cv2.resize(sam_mask, (256, 256), interpolation=cv2.INTER_NEAREST)

    # 3. Load UNet model
    ckpt_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "checkpoints", "best_rgbd_unet.pth"))
    model = LateFusionUNet(n_classes=1, bilinear=False).to(device)
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded checkpoint from epoch {ckpt.get('epoch', '?')}")
    else:
        print("No checkpoint found. Evaluating untrained model.")

    model.eval()

    # 4. Prepare inputs
    # RGB
    rgb = cv2.resize(cv2.cvtColor(prep["resized"], cv2.COLOR_BGR2RGB), (256, 256))
    rgb = rgb.astype(np.float32) / 255.0
    rgb_tensor = torch.from_numpy(rgb.transpose(2, 0, 1)).float().unsqueeze(0).to(device)

    # Mock Depth (UNet needs depth)
    depth = np.full((256, 256), 0.25, dtype=np.float32) # flat depth
    depth = np.clip((depth - 0.1) / 0.4, 0, 1).astype(np.float32)
    depth_tensor = torch.from_numpy(depth[np.newaxis]).float().unsqueeze(0).to(device)

    # 5. Predict
    with torch.no_grad():
        logits = model(rgb_tensor, depth_tensor)
        preds = (torch.sigmoid(logits) > 0.5).float().cpu().numpy()[0, 0]

    # Calculate UNet wheal measurements mapped back to original size
    preds_u8 = (preds * 255).astype(np.uint8)
    preds_u8_original = cv2.resize(preds_u8, (W, H), interpolation=cv2.INTER_NEAREST)
    contours, _ = cv2.findContours(preds_u8_original, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    unet_diameters = []
    ppm = 10.0
    for cnt in contours:
        area_px = cv2.contourArea(cnt)
        if area_px > 10:
            (_, _), radius = cv2.minEnclosingCircle(cnt)
            diameter_px = radius * 2
            diameter_mm = diameter_px / ppm
            unet_diameters.append(round(diameter_mm, 2))
            
    print(f"SAM (Ground Truth) detected {len(sam_diameters)} wheals with diameters (mm): {sam_diameters}")
    print(f"UNet detected {len(unet_diameters)} wheals with diameters (mm): {unet_diameters}")

    # 6. Calculate Error
    iou = compute_iou(preds, sam_mask)
    
    # Measure error based on diameters if any are detected, else fallback to IoU
    if len(sam_diameters) > 0 and len(unet_diameters) > 0:
        avg_sam = sum(sam_diameters) / len(sam_diameters)
        avg_unet = sum(unet_diameters) / len(unet_diameters)
        diameter_error = abs(avg_sam - avg_unet) / avg_sam * 100.0
        print(f"Average Diameter Error: {diameter_error:.2f}%")
        percent_error = diameter_error
    else:
        percent_error = (1.0 - iou) * 100.0

    print(f"IoU: {iou:.4f}")
    print(f"Percent Error: {percent_error:.2f}%")
    return percent_error

if __name__ == "__main__":
    evaluate()
