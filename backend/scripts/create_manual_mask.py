import cv2
import numpy as np
import os

drawing = False # true if mouse is pressed
brush_size = 15

# Read the original image
img_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "Testphotos", "allergy-Testing.jpg"))
out_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "Testphotos", "allergy-Testing_mask.png"))

if not os.path.exists(img_path):
    print(f"Error: Original image not found at {img_path}")
    exit(1)

img = cv2.imread(img_path)
mask = np.zeros(img.shape[:2], dtype=np.uint8)

# Mouse callback function
def paint(event, x, y, flags, param):
    global drawing, brush_size
    
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        cv2.circle(img, (x, y), brush_size, (0, 0, 255), -1)  # Draw red on preview
        cv2.circle(mask, (x, y), brush_size, 255, -1)         # Draw white on mask
        
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            cv2.circle(img, (x, y), brush_size, (0, 0, 255), -1)
            cv2.circle(mask, (x, y), brush_size, 255, -1)
            
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False

cv2.namedWindow('Draw Mask (Paint the wheal) - Press S to Save, ESC to quit')
cv2.setMouseCallback('Draw Mask (Paint the wheal) - Press S to Save, ESC to quit', paint)

print("Instructions:")
print("- Click and drag to paint the wheal (it will appear red).")
print("- Press 's' to save the mask and exit.")
print("- Press 'ESC' to exit without saving.")

while True:
    cv2.imshow('Draw Mask (Paint the wheal) - Press S to Save, ESC to quit', img)
    k = cv2.waitKey(1) & 0xFF
    if k == 27: # ESC
        print("Cancelled.")
        break
    elif k == ord('s'):
        cv2.imwrite(out_path, mask)
        print(f"Mask saved successfully to {out_path}")
        break

cv2.destroyAllWindows()
