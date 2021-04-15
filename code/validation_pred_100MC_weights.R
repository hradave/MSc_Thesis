library(pROC)
library(ggplot2)

validation_pred_100MC_weights = read.csv("D:\\LiU\\Semester 4\\Thesis\\TCGA_DX\\saved_models\\small\\validation_pred_100MC_weights.csv",
                                         sep = ";")

roca = roc(validation_pred_100MC_weights$label,validation_pred_100MC_weights$resnet_block4_random_weights)
rocb = roc(validation_pred_100MC_weights$label,validation_pred_100MC_weights$resnet_block4_IN_weights)
rocc = roc(validation_pred_100MC_weights$label,validation_pred_100MC_weights$resnet_block4_IN_finetuned)

graph = ggroc(list("Random weights initialization (AUC = 0.9963)" = roca, 
                   "ImageNet initialization (AUC = 0.9973)" = rocb, 
                   "ImageNet initialization + freezing (AUC = 0.9469)" = rocc), 
              aes = c("colour"),
              size =1) 

graph + theme(legend.title=element_blank()) + 
  ggtitle("Weight initialization comparison (patch level, validation set)") +
  scale_colour_manual(values = c("blue", "orange", "green"))
                        
roc.test(roca, rocb,method="delong")
roc.test(roca, rocc,method="delong")
roc.test(rocb, rocc,method="delong")
roc.test(rocb, roca,method="delong")


########################


validation_pred_30MC_weights3 = read.csv("D:\\LiU\\Semester 4\\Thesis\\TCGA_DX\\saved_models\\small\\validation_pred_30MC_weights3.csv",
                                         sep = ";")

roca = roc(validation_pred_30MC_weights3$label,validation_pred_30MC_weights3$resnet_block4_random_weights3_40_epochs_lr)
rocb = roc(validation_pred_30MC_weights3$label,validation_pred_30MC_weights3$resnet_block4_IN_weights3)
rocc = roc(validation_pred_30MC_weights3$label,validation_pred_30MC_weights3$resnet_block4_IN_finetuned3)

graph = ggroc(list("Random weights initialization (AUC = 0.9309)" = roca, 
                   "ImageNet initialization (AUC = 0.9463)" = rocb, 
                   "ImageNet initialization + freezing (AUC = 0.9358)" = rocc), 
              aes = c("colour"),
              size =1) 

graph + theme(legend.title=element_blank()) + 
  ggtitle("Weight initialization comparison (patch level, validation set)") +
  scale_colour_manual(values = c("blue", "orange", "green"))

roc.test(roca, rocb,method="delong")
roc.test(roca, rocc,method="delong")
roc.test(rocb, rocc,method="delong")
roc.test(rocb, roca,method="delong")


########################


validation_pred_30MC_weights3_aug = read.csv("D:\\LiU\\Semester 4\\Thesis\\TCGA_DX\\saved_models\\small\\validation_pred_30MC_weights3_aug.csv",
                                         sep = ";")

roca = roc(validation_pred_30MC_weights3_aug$label,validation_pred_30MC_weights3_aug$resnet_block4_IN_weights3_full_aug)
rocb = roc(validation_pred_30MC_weights3_aug$label,validation_pred_30MC_weights3_aug$resnet_block4_IN_weights3_no_aug)
rocc = roc(validation_pred_30MC_weights3_aug$label,validation_pred_30MC_weights3_aug$resnet_block4_IN_weights3)

graph = ggroc(list("Flip + rotation augmentation (AUC = 0.8735)" = roca, 
                   "No augmenation (AUC = 0.9454)" = rocb, 
                   "Flip augmentation (AUC = 0.9463)" = rocc), 
              aes = c("colour"),
              size =1) 

graph + theme(legend.title=element_blank()) + 
  ggtitle("Augmentation comparison (patch level, validation set)") +
  scale_colour_manual(values = c("blue", "orange", "green"))

roc.test(roca, rocb,method="delong")
roc.test(roca, rocc,method="delong")
roc.test(rocb, rocc,method="delong")
roc.test(rocb, roca,method="delong")
