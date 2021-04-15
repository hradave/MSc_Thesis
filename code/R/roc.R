library(pROC)
library(readxl)
library(ggplot2)

# read slide aggregation excel file
slide_agg = read_xlsx("D:\\LiU\\Semester 4\\Thesis\\TCGA_DX\\saved_models\\slide_aggregation.xlsx", sheet = 'resnet_block4_finetuned')

roca = roc(slide_agg$test_label,slide_agg$small_majvote)
rocb = roc(slide_agg$test_label,slide_agg$small_weightedMIL)
rocc = roc(slide_agg$test_label,slide_agg$small_logreg)
rocd = roc(slide_agg$test_label,slide_agg$small_standardMIL)
roce = roc(slide_agg$test_label,slide_agg$small_spatial)

roca = roc(slide_agg$test_label,slide_agg$medium_majvote)
rocb = roc(slide_agg$test_label,slide_agg$medium_weightedMIL)
rocc = roc(slide_agg$test_label,slide_agg$medium_logreg)
rocd = roc(slide_agg$test_label,slide_agg$medium_standardMIL)
roce = roc(slide_agg$test_label,slide_agg$medium_spatial)

plot(roca)
plot(rocb)

graph = ggroc(list("Majority vote" = roca, "Weighted MIL" = rocb, "Logistic regression" = rocc, 
                   "Standard MIL" = rocd, "Spatial smoothing" =  roce), aes = c("colour"),
              size =1) 

graph + theme(legend.title=element_blank()) + ggtitle("672 x 672 patch size")


roc.test(roca, rocb,method="delong")
roc.test(roca, rocc,method="delong")
roc.test(roca, rocd,method="delong")
roc.test(roca, roce,method="delong")
roc.test(rocb, rocc,method="delong")
roc.test(rocb, rocd,method="delong")
roc.test(rocb, roce,method="delong")
roc.test(rocc, rocd,method="delong")
roc.test(rocc, roce,method="delong")
roc.test(rocd, roce,method="delong")

