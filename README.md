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
      

