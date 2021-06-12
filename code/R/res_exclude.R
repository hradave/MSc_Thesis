library(ggplot2)
res_exclude = read.csv("D:\\LiU\\Semester 4\\Thesis\\Sources\\figures\\res_exclude.csv")

colours = c("#FF7F0E", "#1F77B4")
p = ggplot(data = res_exclude, aes(x = x, y = y, col=class)) +
  geom_point(size = 2) + 
  scale_color_manual(values = colours) + 
  xlab("X pixels") + ylab("Y pixels") + 
  theme(legend.title = element_blank()) +
  #theme(legend.position = c(0.8, 0.5)) +
  theme(axis.title=element_text(size=10)) + 
  #theme(legend.position='none') + 
  theme(legend.text = element_text(size=10)) +
  scale_y_continuous(labels=function(x) format(x, big.mark = ",", scientific = FALSE)) +
  scale_x_continuous(labels=function(x) format(x, big.mark = ",", scientific = FALSE))
p

name = "res_exclude"
p + ggsave(paste0(name,".png"), type = "cairo")
