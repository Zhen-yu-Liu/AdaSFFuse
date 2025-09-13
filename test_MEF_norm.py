from cgi import test
import numpy as np
import os
from utils.img_read_save import img_save,image_read_cv2
from utils.evaluator import Evaluator
import warnings
import logging
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.CRITICAL)
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def main():
    model_name = 'MEF'
    eval_folder='/results/Evaluation'  
    ori_img_folder='/imagefusion/test/MEF'
    metric_result = np.zeros((8))
    for img_name in os.listdir(os.path.join(ori_img_folder,"low_Y")):
            base_name = img_name.rsplit('_', 1)[0]  # 去掉 '_A.png' 部分
            ir = image_read_cv2(os.path.join(ori_img_folder,"low_Y", img_name), 'GRAY')
            # import pdb;pdb.set_trace()
            vi = image_read_cv2(os.path.join(ori_img_folder,"over_Y",  img_name), 'GRAY')

            # vi = image_read_cv2(os.path.join(ori_img_folder,"imageB_Y",  base_name+"_B.png"), 'GRAY')
            # fi = image_read_cv2(os.path.join(eval_folder, img_name.split('.')[0]+".png"), 'GRAY')
            fi = image_read_cv2(os.path.join(eval_folder, img_name), 'GRAY')

            metric_result += np.array([Evaluator.EN(fi), Evaluator.SD(fi)
                                        , Evaluator.SF(fi), Evaluator.MI(fi, ir, vi)
                                        , Evaluator.SCD(fi, ir, vi), Evaluator.VIFF(fi, ir, vi)
                                        , Evaluator.Qabf(fi, ir, vi), Evaluator.SSIM(fi, ir, vi)])

    metric_result /= len(os.listdir(eval_folder))
    print("\t\t EN\t SD\t SF\t MI\tSCD\tVIF\tQabf\tSSIM")
    print(model_name+'\t'+str(np.round(metric_result[0], 2))+'\t'
            +str(np.round(metric_result[1], 2))+'\t'
            +str(np.round(metric_result[2], 2))+'\t'
            +str(np.round(metric_result[3], 2))+'\t'
            +str(np.round(metric_result[4], 2))+'\t'
            +str(np.round(metric_result[5], 2))+'\t'
            +str(np.round(metric_result[6], 2))+'\t'
            +str(np.round(metric_result[7], 2))
            )
    print("="*80)


if __name__ == '__main__':
    main()
