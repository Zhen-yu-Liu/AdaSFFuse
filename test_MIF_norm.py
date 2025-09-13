import os
import numpy as np
import torch
import torch.nn.functional as F
from utils.evaluator import Evaluator
from utils.img_read_save import img_save, image_read_cv2
import warnings
import logging
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.CRITICAL)
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

root_path = '/imagefusion/test'
fuse_path = '/results/MIF'
model_name = 'MIF'
root_path_with_model = os.path.join(root_path, model_name)

metric_result = np.zeros((8))

def read_image_to_tensor(path, color_mode='GRAY'):
    img = image_read_cv2(path, color_mode)
    img = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
    return img.to(device)

for img_name in os.listdir(os.path.join(root_path_with_model, "MRI")):
    ir = read_image_to_tensor(os.path.join(root_path_with_model, "MRI", img_name))
    vi = read_image_to_tensor(os.path.join(root_path_with_model, "CT", img_name))
    fi = read_image_to_tensor(os.path.join(fuse_path, img_name.split('.')[0] + ".png"))

    ir_np = ir.squeeze().cpu().numpy()
    vi_np = vi.squeeze().cpu().numpy()
    fi_np = fi.squeeze().cpu().numpy()

    metric_result += np.array([
        Evaluator.EN(fi_np), 
        Evaluator.SD(fi_np), 
        Evaluator.SF(fi_np), 
        Evaluator.MI(fi_np, ir_np, vi_np), 
        Evaluator.SCD(fi_np, ir_np, vi_np), 
        Evaluator.VIFF(fi_np, ir_np, vi_np), 
        Evaluator.Qabf(fi_np, ir_np, vi_np), 
        Evaluator.SSIM(fi_np, ir_np, vi_np)
    ])

metric_result /= len(os.listdir(fuse_path))

print("\t\t EN\t SD\t SF\t MI\tSCD\tVIF\tQabf\tSSIM")
print(model_name + '\t' + str(np.round(metric_result[0], 2)) + '\t'
      + str(np.round(metric_result[1], 2)) + '\t'
      + str(np.round(metric_result[2], 2)) + '\t'
      + str(np.round(metric_result[3], 2)) + '\t'
      + str(np.round(metric_result[4], 2)) + '\t'
      + str(np.round(metric_result[5], 2)) + '\t'
      + str(np.round(metric_result[6], 2)) + '\t'
      + str(np.round(metric_result[7], 2))
      )
print("=" * 80)
