# Exploring the Design Space of Autoregressive Models for Efficient and Scalable Image Generation


## Repository Structure
Here's an overview of the repository structure:

      ├ MaskGIL/
      |    ├── Metrics/                               <- evaluation tool
      |    |      ├── inception_metrics.py                  
      |    |      └── sample_and_eval.py
      |    |    
      |    ├── Network/                             
      |    |      ├── Taming/                         <- VQGAN architecture   
      |    |      ├── tokenizer/                      <- VQGAN architecture  
      |    |      ├── gpt.py                          <- Bi-Direction LLaMA architecture      
      |    |      └── transformer.py                  <- Bi-Transformer architecture  
      |    |
      |    ├── Trainer/                               <- Main class for training
      |    |      ├── trainer.py                      <- Abstract trainer     
      |    |      └── vit.py                          <- Trainer of MaskGIL
      |    ├── images/                                <- Image samples         
      |    |
      |    ├── requirements.yaml                      <- help to install env 
      |    ├── FID_sample.py                          <- sample 50K images for FID
      |    └── main.py                                <- Main
      

## 🔥 Update
[2024.10.07] Code and c2i checkpoints are released !

## 🌿 Introduction


## 🦄 Class-conditional image generation on ImageNet

Method | params| tokens | FID (256x256) | weight 
--- |:---:|:---:|:---:|:---:|
MaskGIL-B   | 111M | 16x16 | 5.46 | [c2i_B_256.pt]()
MaskGIL-L   | 343M | 16x16 | 3.80 | [c2i_L_256.pt]()
MaskGIL-XL  | 775M | 24x24 | 2.62 | [c2i_X_256.pt]()
MaskGIL-XXL | 1.4B | 24x24 | 2.34 | [c2i_XXL_256.pt]()
