# Inference Script --> Given Image, Output Label
import torch
import os
from PIL import Image
from torchvision import transforms
from model.attention_ocr import OCR
from utils.tokenizer import Tokenizer
import json
import matplotlib.pyplot as plt

# Train and infer on same CNN Backbone

# Loading Json File
json_file="config_infer.json"
f = open(json_file, )
data = json.load(f)

cnn_option = data["cnn_option"]
cnn_backbone = data["cnn_backbone_model"][str(cnn_option)]  # list containing model, model_weight
# Tokenizer
tokenizer = Tokenizer(list(data["chars"]))
# Model Architecture
model = OCR(data["img_width"], data["img_height"], data["nh"], tokenizer.n_token,
                data["max_len"] + 1, tokenizer.SOS_token, tokenizer.EOS_token,cnn_backbone).to(device=data["device"])
# Model checkpoint Load
model.load_state_dict(torch.load(data["model_path"]))
# Image Transformation
img_trans = transforms.Compose([
            transforms.Resize((data["img_height"], data["img_width"])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=(0.229, 0.224, 0.225)),
        ])
# Inference Function --> Input: Source Image directory, Text File containing name of Image
def _inference(test_dir,filename):
    count=0
    model.eval()
    with torch.no_grad():
        with open(os.path.join(test_dir, filename), 'r') as fp:
            for img_name in fp.readlines():
                img_name = img_name.strip("\n")
                img_filename=img_name.split(data["data_file_separator"])[0]
                #act_label=img_name.split(data["data_file_separator"])[1]
                image_file = os.path.abspath(os.path.join(test_dir, img_filename))
                img=Image.open(image_file)
                pred = model(img_trans(img).unsqueeze(0).to(device=data["device"]))
                pred_label = tokenizer.translate(pred.squeeze(0).argmax(1))
                print(pred_label)
                plt.imshow(img)
                plt.savefig(data["res_output"] + "/" + pred_label + '.png')
                count=count+1
                print("Saved {} file".format(count))
                # plt.title(pred_label)
                # plt.imshow(img)
                # plt.show()
    print("Complete Saving {} files".format(count))

                #print(pred_label,act_label)

if __name__=="__main__":
    _inference(data["test_dir"],data["inference_file"])