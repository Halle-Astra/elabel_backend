from segment_anything import sam_model_registry, SamPredictor
import os

ckpt_dict = {"vit_h":"sam_vit_h_4b8939.pth"}

# sam_checkpoint = r"D:\Programs\SAM\Layer-Divider-WebUI\models\sam_vit_h_4b8939.pth"
def find_sam_ckpt(model_type):
    this_file = os.path.abspath(__file__)
    this_folder = os.path.split(this_file)[0]
    parent_folder = os.path.split(this_folder)[0]
    ckpt_folder = os.path.join(parent_folder, 'assets', 'models')
    ckpt = os.path.join(ckpt_folder, ckpt_dict[model_type])
    return ckpt

# sam_checkpoint = os.path.abspath(__file__)#r"D:\Programs\SAM\Layer-Divider-WebUI\models\sam_vit_h_4b8939.pth"

class SAM:
    def __init__(self,model_type = "vit_h",device = "cuda"):
        sam_ckpt = find_sam_ckpt(model_type)
        sam = sam_model_registry[model_type](checkpoint=sam_ckpt)
        sam.to(device=device)
        self.predictor = SamPredictor(sam)
        self.model_type = model_type
        self.device = device

    def set_image(self,image):
        self.predictor.set_image(image)

    def predict(self, input_point, input_label):
        masks, _, _ = self.predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            # mask_input=mask_input[None, :, :],
            multimask_output=False,
        )
        mask = masks[0]
        return mask 