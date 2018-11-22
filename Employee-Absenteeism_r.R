rm(list=ls(all=T))
setwd("D:/Project")
x=c("ggplot2", "corrgram", "DMwR", "caret", "randomForest", "unbalanced", "dummies", "e1071", "Information", "MASS", "rpart", "gbm", "ROSE", "xlsx", "DataCombine", "xlsx")
lapply(x, require, character.only=TRUE)
df_train = read.xlsx('Absenteeism_at_work_Project.xls', sheetIndex = 1)


#Exploratory Data Analysis
# Shape of the data
dim(df_train)
# Structure of the data
str(df_train)
# Lets see the colum names of the data
colnames(df_train)
#splitting columns into "continuous" and "catagorical"
num_list = c('Distance.from.Residence.to.Work', 'Service.time', 'Age',
                    'Work.load.Average.day.', 'Transportation.expense',
                    'Hit.target', 'Weight', 'Height', 
                    'Body.mass.index', 'Absenteeism.time.in.hours')

cat_list = c('ID','Reason.for.absence','Month.of.absence','Day.of.the.week',
                     'Seasons','Disciplinary.failure', 'Education', 'Social.drinker',
                     'Social.smoker', 'Son', 'Pet')

----------------------------------------------------------------------------
#Missing Values Analysis
#Create dataframe with missing percentage
missing_val = data.frame(apply(df_train,2,function(x){sum(is.na(x))}))
#convert rownames into proper column
missing_val$Columns = row.names(missing_val)
#removing rownames which were previously the index
row.names(missing_val) = NULL
#Renaming first column as "Missing_percentage"
names(missing_val)[1] =  "Missing_percentage"
#calclulating missing value percentage for each column
missing_val$Missing_percentage = (missing_val$Missing_percentage/nrow(df_train)) * 100
#Arrange in descending order
missing_val = missing_val[order(-missing_val$Missing_percentage),]
missing_val = missing_val[,c(2,1)]# Reordering columns

#df_train[7, 7]
#Actual Value = 52
#df_train[7, 7]=NA

#Mean Method
#df_train$Distance.from.Residence.to.Work[is.na(df_train$Distance.from.Residence.to.Work)] = mean(df_train$Distance.from.Residence.to.Work, na.rm = T)
# Mean = 29.63735

#df_train[7, 7]=NA
#Median Method
#df_train$Distance.from.Residence.to.Work[is.na(df_train$Distance.from.Residence.to.Work)] = median(df_train$Distance.from.Residence.to.Work, na.rm = T)
# Median = 26

# kNN Imputation
# KNN = 52
df_train = knnImputation(df_train, k = 3)

# Checking for missing value
sum(is.na(df_train))
----------------------------------------------------------------------------------


#Outlier Analysis
#Boxplot for continuous variables
for (i in 1:length(num_list))
{
  assign(paste0("gn",i), ggplot(aes_string(y = (num_list[i]), x = "Absenteeism.time.in.hours"), data = subset(df_train))+ 
           stat_boxplot(geom = "errorbar", width = 0.5) +
           geom_boxplot(outlier.colour="red", fill = "grey" ,outlier.shape=18,
                        outlier.size=1, notch=FALSE) +
           theme(legend.position="bottom")+
           labs(y=num_list[i],x="Absenteeism.time.in.hours")+
           ggtitle(paste("Box plot of absenteeism for",num_list[i])))
}

#Plotting plots together
gridExtra::grid.arrange(gn1,gn2, gn3, ncol=3)
gridExtra::grid.arrange(gn4,gn5,gn6, ncol=3)
gridExtra::grid.arrange(gn7,gn8,ncol=2)
gridExtra::grid.arrange(gn9,gn10,ncol=2)

#loop to remove outliers from all variables
for(i in num_list)
{
  print(i)
  #Extract outliers
  val = df_train[,i][df_train[,i] %in% boxplot.stats(df_train[,i])$out]
  #Remove outliers
  df_train = df_train[which(!df_train[,i] %in% val),]
}

#Replace all outliers with NA and impute using KNN
for(i in num_list)
{
  val = df_train[,i][df_train[,i] %in% boxplot.stats(df_train[,i])$out]
  df_train[,i][df_train[,i] %in% val] = NA
}

# Imputing missing values
df_train = knnImputation(df_train,k=3)
-------------------------------------------------------------------------------------

#Feature Selection
# Correlation Plot 
corrgram(df_train[,num_list], order = F,
         upper.panel=panel.pie, text.panel=panel.txt, main = "Correlation Plot")


## Dimension Reduction
df_train = subset(df_train, select = -c(Weight))
-----------------------------------------------------------------------------------


#Feature Scaling
#Normality check
hist(df$Absenteeism.time.in.hours)

#Updating the continuous and catagorical variable
num_list = c('Distance.from.Residence.to.Work', 'Service.time', 'Age',
                    'Work.load.Average.day.', 'Transportation.expense',
                    'Hit.target', 'Height', 
                    'Body.mass.index')

cat_list = c('ID','Reason.for.absence','Disciplinary.failure', 
                     'Social.drinker', 'Son', 'Pet', 'Month.of.absence', 'Day.of.the.week', 'Seasons',
                     'Education', 'Social.smoker')


# Normalization
for(i in num_list)
{
  print(i)
  df_train[,i] = (df_train[,i] - min(df_train[,i]))/(max(df_train[,i])-min(df_train[,i]))
}

# Creating dummy variables for categorical variables
df_train = dummy.data.frame(df_train, cat_list)
----------------------------------------------------------------------------------


#Model Development
#Cleaning the environment
rmExcept("df_train")

#Divide data into train and test using stratified sampling method
set.seed(123)
train.index = sample(1:nrow(df_train), 0.8 * nrow(df_train))
features_train = df_train[ train.index,]
features_test  = df_train[-train.index,]

-------------------------------------------------------------------------------

#Decision tree 

model_dt = rpart(Absenteeism.time.in.hours ~., data = features_train, method = "anova")

#Summary of model
summary(model_dt)

#write rules into disk
write(capture.output(summary(model_dt)), "Rules.txt")

predict_dt = predict(model_dt,features_test[,names(features_test) != "Absenteeism.time.in.hours"])

# For test data 
print(postResample(pred = predict_dt, obs = features_test[,107]))
#RMSE: 2.230267
#Rsquared: 0.415035  
------------------------------------------------------------------------------------

#Linear Regression
set.seed(123)

#Develop Model on training data
model_lr = lm(Absenteeism.time.in.hours ~ ., data = features_train)

#Lets predict for test data
predict_lr = predict(model_lr,features_test[,names(features_test) != "Absenteeism.time.in.hours"])


# For test data 
print(postResample(pred = predict_lr, obs = features_test[,107]))

#RMSE: 2.3214801
#Rsquared: 0.3642748
---------------------------------------------------------------------------------

#Random Forest

set.seed(123)

#Develop Model on training data
model_RF = randomForest(Absenteeism.time.in.hours~., data = features_train)

#Lets predict for test data
predict_rf = predict(model_RF,features_test[,names(features_test) != "Absenteeism.time.in.hours"])

#For test data 
print(postResample(pred = predict_rf, obs = features_test[,107]))

#RMSE: 2.1910645
#Rsquared: 0.4060271