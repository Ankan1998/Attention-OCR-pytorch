# Attention-OCR-pytorch
Using Attention to improve performance of OCR

### Steps to run with custom dataset
1. Check sample directory for data format structure
2. Download inceptionv3, or inception_resnet_v2 weights<br>
 inceptionv3 : 'https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth' <br>
 inception_resnet_v2 : 'http://data.lip6.fr/cadene/pretrainedmodels/inceptionresnetv2-520b38e4.pth'
3. Configure train config file
4. Run train.py after configuring config file
5. For inference, configure config_infer file
6. Run inference.py

#### CITATION
This work is based on paper
Attention-based Extraction of Structured Information from Street View Imagery <br>
Zbigniew Wojna∗ Alex Gorban† Dar-Shyang Lee† Kevin Murphy† Qian Yu† Yeqing Li† Julian Ibarz†
∗ University College London † Google Inc.


#### Github Help and reference
1. https://github.com/wptoux/attention-ocr (Main repository)
2. https://github.com/emedvedev/attention-ocr
3. https://github.com/chenjun2hao/Attention_ocr.pytorch
4. https://github.com/da03/Attention-OCR
