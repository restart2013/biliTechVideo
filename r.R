data <- read.csv('data_reg.csv',encoding = 'UTF-8')
data
data <- subset(data, select = -time_month_12 )
data <- subset(data, select = -time_date_31 )
data <- subset(data, select = -time_hour_23 )
data <- subset(data, select = -author_sex_男 )
data <- subset(data, select = -duration_.1hour )

xnam <- paste0(colnames(data)[3:258])
f <- as.formula(paste("log(play) ~ ", paste(xnam, collapse = "+")))
f
fit <- lm(f,data=data)
options(max.print=10000)
summary(fit)

library(car)
vif(fit)
