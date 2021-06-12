library(pROC)
library(ggplot2)
library(caret)
library(ModelMetrics)

#######################################################################################
#                                          SMALL
#######################################################################################

############################################### WEIGHTS ###############################################

#--------------------------------------------------------------------------------------



validation_pred_30MC_weights3 = read.csv("D:\\LiU\\Semester 4\\Thesis\\TCGA_DX\\saved_models\\small\\validation_pred_30MC_weights3.csv",
                                         sep = ";")

roca = roc(validation_pred_30MC_weights3$label,validation_pred_30MC_weights3$resnet_block4_random_weights3_40_epochs_lr)
rocb = roc(validation_pred_30MC_weights3$label,validation_pred_30MC_weights3$resnet_block4_IN_weights3)
rocc = roc(validation_pred_30MC_weights3$label,validation_pred_30MC_weights3$resnet_block4_IN_finetuned3)

colours = c("#1F77B4","#FF7F0E", "#2CA02C")
graph = ggroc(list("Pre-trained" = rocb, 
                   "Pre-trained and fine-tuned" = rocc,
                   "Random weights initialization" = roca),
              aes = c("colour"),
              size =1) + 
  theme(legend.title=element_blank()) +
  scale_color_manual(values = colours)+ 
  theme(legend.position='none')

graph

name = "small_weights_roc"
graph + ggsave(paste0(name,".png"), type = "cairo")

roc.test(roca, rocb,method="delong")
roc.test(roca, rocc,method="delong")
roc.test(rocb, rocc,method="delong")
roc.test(rocb, roca,method="delong")

############################################### AUGMENTATION ###############################################

#--------------------------------------------------------------------------------------

validation_pred_30MC_weights3_aug = read.csv("D:\\LiU\\Semester 4\\Thesis\\TCGA_DX\\saved_models\\small\\validation_pred_30MC_weights3_aug.csv",
                                             sep = ";")

roca = roc(validation_pred_30MC_weights3_aug$label,validation_pred_30MC_weights3_aug$resnet_block4_IN_weights3_full_aug)
rocb = roc(validation_pred_30MC_weights3_aug$label,validation_pred_30MC_weights3_aug$resnet_block4_IN_weights3_no_aug)
rocc = roc(validation_pred_30MC_weights3_aug$label,validation_pred_30MC_weights3_aug$resnet_block4_IN_weights3)

colours = c("#1F77B4","#D62728", "#FFD700")
graph = ggroc(list("Mirroring" = rocc, 
                   "Mirroring and rotating" = roca,
                   "No augmentation" = rocb),
              aes = c("colour"),
              size =1) + 
  theme(legend.title=element_blank()) +
  scale_color_manual(values = colours)+ 
  theme(legend.position='none')

graph

name = "small_augm_roc"
graph + ggsave(paste0(name,".png"), type = "cairo")

roc.test(roca, rocb,method="delong")
roc.test(roca, rocc,method="delong")
roc.test(rocb, rocc,method="delong")
roc.test(rocb, roca,method="delong")

#######################################################################################
#                                          MEDIUM
#######################################################################################

############################################### WEIGHTS ###############################################

#--------------------------------------------------------------------------------------

validation_pred_30MC_weights3_medium = read.csv("D:\\LiU\\Semester 4\\Thesis\\TCGA_DX\\saved_models\\medium\\validation_pred_30MC_weights3_medium.csv",
                                                sep = ";")

roca = roc(validation_pred_30MC_weights3_medium$label,validation_pred_30MC_weights3_medium$resnet_block4_random_weights3_100_epochs_lr)
rocb = roc(validation_pred_30MC_weights3_medium$label,validation_pred_30MC_weights3_medium$resnet_block4_IN_weights3_2)
rocc = roc(validation_pred_30MC_weights3_medium$label,validation_pred_30MC_weights3_medium$resnet_block4_IN_finetuned3)

colours = c("#1F77B4","#FF7F0E", "#2CA02C")
graph = ggroc(list("Pre-trained" = rocb, 
                   "Pre-trained and fine-tuned" = rocc,
                   "Random weights initialization" = roca),
              aes = c("colour"),
              size =1) + 
  theme(legend.title=element_blank()) +
  scale_color_manual(values = colours)+ 
  theme(legend.position='none')

graph

name = "medium_weights_roc"
graph + ggsave(paste0(name,".png"), type = "cairo")

roc.test(roca, rocb,method="delong")
roc.test(roca, rocc,method="delong")
roc.test(rocb, rocc,method="delong")
roc.test(rocb, roca,method="delong")
############################################### AUGMENTATION ###############################################

#--------------------------------------------------------------------------------------

validation_pred_30MC_weights3_medium_aug = read.csv("D:\\LiU\\Semester 4\\Thesis\\TCGA_DX\\saved_models\\medium\\validation_pred_30MC_weights3_medium_aug.csv",
                                                    sep = ";")

roca = roc(validation_pred_30MC_weights3_medium_aug$label,validation_pred_30MC_weights3_medium_aug$resnet_block4_IN_weights3_no_aug)
rocb = roc(validation_pred_30MC_weights3_medium_aug$label,validation_pred_30MC_weights3_medium_aug$resnet_block4_IN_weights3_2)
rocc = roc(validation_pred_30MC_weights3_medium_aug$label,validation_pred_30MC_weights3_medium_aug$resnet_block4_IN_weights3_full_aug)

colours = c("#1F77B4","#D62728", "#FFD700")
graph = ggroc(list("Mirroring" = rocb, 
                   "Mirroring and rotating" = rocc,
                   "No augmentation" = roca),
              aes = c("colour"),
              size =1) + 
  theme(legend.title=element_blank()) +
  scale_color_manual(values = colours)+ 
  theme(legend.position='none')

graph

name = "medium_augm_roc"
graph + ggsave(paste0(name,".png"), type = "cairo")

roc.test(roca, rocb,method="delong")
roc.test(roca, rocc,method="delong")
roc.test(rocb, rocc,method="delong")
roc.test(rocb, roca,method="delong")

