## Project: Wifi Locationing
## Script purpose: use the signal intensity recorded from multiple wifi hotspots within the building 
#to determine users location using Machine Learning Models.
## Date: 6 Feb 2019
## Author: Maria Farrés



########################## SET ENVIRONMENT #######################################

#Load required libraries 
pacman:: p_load("readr","dplyr", "tidyr", "ggplot2", "plotly", 
                "data.table", "reshape2","ggridges", "party",
                "esquisse", "caret", "randomForest", 
                "hablar")

#Set working directory
setwd("C:/Users/usuario/Desktop/UBIQUM/Project 8 - Wifi locationing/Wifi-locationing")


#Read initial data sets
original_train <- read_csv("./DataSets/trainingData.csv")
original_test <- read_csv("./DataSets/validationData.csv")
new_test <- read_csv("./DataSets/testData.csv")


############################## PRE-PROCESSING ########################################

# wide format df
wide_train <- original_train # train in wide format
wide_test <- original_test # test in wide format



# long format df
# melt original_train to change [ , 1:520] attributes to variables (WAPs as ID)
vars_not_waps <- colnames(original_train[,521:529])
long_train <- melt(original_train, id.vars = vars_not_waps)
rm(vars_not_waps)
names(long_train)[10]<- paste("WAPid")
names(long_train)[11]<- paste("WAPrecord")


# melt original_test 
vars_not_waps <- colnames(original_test[,521:529])
long_test <- melt(original_test, id.vars = vars_not_waps)
rm(vars_not_waps)
names(long_test)[10]<- paste("WAPid")
names(long_test)[11]<- paste("WAPrecord")


rm(original_train)
rm(original_test)


# MISSING VALUES
# Set NAs to low intensity RSSI 
# NAs mean that the WAPs have not recorded signal in a determinate user location
# instead of deleting those signals we set them to -105 in a new data frame 
# to indicate that the signal is really low in that location

vars_waps <- colnames(wide_train[, 1:520])
wide_train[, vars_waps][wide_train[, vars_waps] == 100] <- -105 # place them as low signal 
wide_test[, vars_waps][wide_test[, vars_waps] == 100] <- -105 
rm(vars_waps)


# Set NAs to low intensity RSSI in long format #####
long_train[,11][long_train[, 11] == 100] <- -105 # in long_train WAPid is the only
long_test[,11][long_test[, 11] == 100] <- -105  # attribute changed as it contains the RSSI



# create another long df which does not contain -105 dbm observations
# this helps reduce the sample size for an initial exploration
long_train <- filter(long_train, long_train$WAPrecord != -105)
long_test <- filter(long_test, long_test$WAPrecord != -105)


# DATA TYPES & CLASS TREATMENT

# LONG FORMAT (TRAIN&TEST)
long_train <- long_train %>% convert(num(LONGITUDE, LATITUDE, WAPrecord),   # check -> sapply(long_train, class)
                                     fct(BUILDINGID, FLOOR, USERID, PHONEID,
                                         RELATIVEPOSITION, SPACEID, WAPid),
                                     dtm(TIMESTAMP))

levels(long_train$BUILDINGID) <- c("TI", "TD","TC")
levels(long_train$RELATIVEPOSITION) <- c("Inside", "Outside")



long_test <- long_test %>% convert(num(LONGITUDE, LATITUDE, WAPrecord),   # check -> sapply(long_test, class)
                                   fct(BUILDINGID, FLOOR, USERID, PHONEID,
                                       RELATIVEPOSITION, SPACEID, WAPid),
                                   dtm(TIMESTAMP))
levels(long_test$BUILDINGID) <- c("TI", "TD","TC")
levels(long_test$RELATIVEPOSITION) <- c("Inside", "Outside")


# WIDE FORMAT
wide_train <- wide_train %>% convert(num(LONGITUDE, LATITUDE),   # check -> sapply(wide_train, class)
                                     fct(BUILDINGID, FLOOR, USERID, PHONEID,
                                         RELATIVEPOSITION, SPACEID),
                                     dtm(TIMESTAMP))
levels(wide_train$BUILDINGID) <- c("TI", "TD","TC")
levels(wide_train$RELATIVEPOSITION) <- c("Inside", "Outside")



wide_test <- wide_test %>% convert(num(LONGITUDE, LATITUDE),     # check -> sapply(wide_test, class)
                                   fct(BUILDINGID, FLOOR, USERID, PHONEID,
                                       RELATIVEPOSITION, SPACEID),
                                   dtm(TIMESTAMP))
levels(wide_test$BUILDINGID) <- c("TI", "TD","TC")
levels(wide_test$RELATIVEPOSITION) <- c("Inside", "Outside")


# NEW TEST 

new_test <- new_test %>% convert(num(LONGITUDE, LATITUDE),     # check -> sapply(new_test, class)
                                   fct(BUILDINGID, FLOOR, USERID, PHONEID,
                                       RELATIVEPOSITION, SPACEID),
                                   dtm(TIMESTAMP))
levels(new_test$BUILDINGID) <- c("TI", "TD","TC")
levels(new_test$RELATIVEPOSITION) <- c("Inside", "Outside")

# ZERO VARIANCE & DUPLICATES 

# check if there are WAPs that have no variance in all their records 

ZeroVar_check_train <- nearZeroVar(wide_train[1:520], saveMetrics = TRUE) # there are 55 WAPs with 0 variance
wide_train <- wide_train[-which(ZeroVar_check_train$zeroVar == TRUE)] # we remove them as they might ditort our model


ZeroVar_check_test <- nearZeroVar(wide_test[1:520], saveMetrics = TRUE) #
wide_test <- wide_test[-which(ZeroVar_check_test$zeroVar == TRUE)]


    # ZeroVar_check_newtest <- nearZeroVar(new_test[1:520], saveMetrics = TRUE) #
    # new_test <- new_test[-which(ZeroVar_check_newtest$zeroVar == TRUE)]

rm(ZeroVar_check_test, ZeroVar_check_train)

vars_waps_tr <- grep("WAP", names(wide_train), value = TRUE) # grep 465 WAPs remaining after applying zeroVar
vars_waps_tst <- grep("WAP", names(wide_test), value = TRUE) # grep 367 WAPs " "
vars_waps_newtst <- grep("WAP", names(new_test), value = TRUE) # grep 270 WAPs " "
common_waps <- intersect(vars_waps_tst, vars_waps_tr) # 312 common waps


common_waps_tr<- select_at(wide_train[vars_waps_tr], common_waps)
common_waps_tst <- select_at(wide_test[vars_waps_tst], common_waps)


wide_train <- cbind(common_waps_tr, wide_train[466:474])
wide_test <- cbind(common_waps_tst, wide_test[368:376])


rm(common_waps_tr, common_waps_tst)

# real duplicates
wide_train <- unique(wide_train)
long_train <- unique(long_train)
        # new_test <- unique(new_test) # 5179 to 5172 obs
        # new_wide_train <- unique(new_wide_train) # 19937 to 19288



# FEATURE ENGINEERING  
# Create new attribute BUILDING-FLOOR
long_train$BuildingFloor <-  paste(long_train$BUILDINGID, long_train$FLOOR, sep = "-")
long_test$BuildingFloor <- paste(long_test$BUILDINGID, long_test$FLOOR, sep = "-")

############################# EXPLORATORY / DESCRIPTIVE ANALYSIS #############################3

# Records distributed by Building and floor in TRAIN
# the vast majority of records in train happen in Building TC
# this might have an impact when applying the prediction in test set
ggplot(data = long_train) +
  aes(x = WAPrecord, fill = FLOOR) +
  geom_histogram(bins = 30) +
  theme_minimal() +
  facet_wrap(vars(BUILDINGID))



# In TEST 
# the vast majority of records in test happen in Building TI
# this might have an impact when applying the prediction in test set
ggplot(data = long_test) +
  aes(x = WAPrecord, fill = FLOOR) +
  geom_histogram(bins = 30) +
  theme_minimal() +
  facet_wrap(vars(BUILDINGID))


# records taken distributed by location and floor in TRAIN
# this helps us see the buildings shape and distribution
ggplot(data = long_train) +
  aes(x = LONGITUDE, y = LATITUDE, color = FLOOR) +
  geom_point() +
  theme_minimal()


ggplot(data = long_test) +                            
  aes(x = LONGITUDE, y = LATITUDE, color = FLOOR) +  # the same plot has been applied to test
  geom_point() +          # and we can see that the samples are taken in a more arbitrary way
  theme_minimal()         # as the test set locations were chosen by the users randomly




# ANALYSE APPARENTLY TOO GOOD SIGNALS 
# Signals greater than -30dBm are extremely rare to happen in normal conditions
# we need to analyse if there is something wrong in this records, therefore we isolate 
# signals greater than -30 to further study them
tooGood_signals <- long_train %>% filter(WAPrecord > -30)


# tooGood_signals boxplot by WAPS in each building
ggplot(data = tooGood_signals) +                      # we detect that almost all extreme signals come from 
  aes(x = WAPid, y = WAPrecord, fill = BUILDINGID) +  # the same building (TC)
  geom_boxplot() +                                    # exept from one WAP in building TD
  theme_minimal()+ 
  theme(axis.text.x = element_text(angle = 90, hjust = 1))


# histogram tooGood_signals by bulding&floor to detect if these are concentrated in a specific floor 
ggplot(data = tooGood_signals) +        # Indeed, these RSSI happen in floor 3&4 of building TC
  aes(x = WAPrecord, fill = FLOOR) +
  geom_histogram(bins = 30) +
  theme_minimal() +
  facet_wrap(vars(BUILDINGID))

# histogram to understand if tooGood_signals come from one specific phone model or user
ggplot(data = tooGood_signals) +      
  aes(x = PHONEID, fill = FLOOR) +
  geom_bar() +
  theme_minimal() +
  facet_wrap(vars(BUILDINGID))     # phone 19 is responsible for the vast majority of them

ggplot(data = tooGood_signals) +
  aes(x = USERID, fill = FLOOR) +
  geom_bar() +
  theme_minimal() +
  facet_wrap(vars(BUILDINGID))     # phone 19 is used by user 6 (max responsible for >-30dbm RSSI)


# BEHAVIOURAL ANALYSIS OF USER 6
# We need to determine if the rest of his records are also affected or not
# this, will lead us to decide whether to use his recordings or not
USER6 <- long_train %>% filter(USERID == 6)

# it gives bad results but it also records reasonable RSSI 
ggplot(data = USER6) +
  aes(x = WAPrecord) +
  geom_histogram(bins = 30, fill = "#0c4c8a") +
  theme_minimal() # although 84% of his records are affected

rm(USER6)

# Can we remove this user completely though? 
# would it leave us with too few data in building TC floor 3&4? -> ANALYSIS OF RECORDS IN TC FLOOR 3 & 4
TC3_exploration <- long_train %>% filter(BuildingFloor == "TC-3")
ggplot(data = TC3_exploration) +    
  aes(x = WAPrecord, fill = USERID) +
  geom_histogram(bins = 30) +
  theme_minimal()

TC4_exploration <- long_train %>% filter(BuildingFloor == "TC-4")
ggplot(data = TC4_exploration) +
  aes(x = WAPrecord, fill = USERID) +
  geom_histogram(bins = 30) +
  theme_minimal()



# User 6 captured a big proportion of data in TC3 and TC4
# what would happen in terms of data amount if we removed user 6 records?
TC_exploration <- long_train %>% filter(BUILDINGID == "TC")
ggplot(data = TC_exploration) +
  aes(x = FLOOR, fill = USERID, weight = WAPrecord) +
  geom_bar() +                  # although we could still predict T3 
  theme_minimal()               # TC4 would end up with really few data 
rm(TC3_exploration, TC4_exploration, TC_exploration)


# as user 6 represents a big part of the records of TC3&4, 
# we only remove the data >-30dbm recorded; but we do not remove all user 6 records
long_train <- long_train %>% filter(WAPrecord <= -30)

vars_waps_tr <- grep("WAP", names(wide_train), value = TRUE) # now 312 columns
wide_train <- wide_train %>% filter_at(vars_waps_tr, any_vars(. < -30)) # filter rows that do not contain values above -30dbm in WAP columns

vars_waps_tst <- grep("WAP", names(wide_test), value = TRUE) # now 312 columns
  # wide_test does not contain any >-30 dbm value

vars_waps_newtst <- grep("WAP", names(new_test), value = TRUE) # now 246 columns
  # new_test does not contain any >-30 dbm value

        # vars_waps_trnew <- grep("WAP", names(new_wide_train), value = TRUE) # now 246 columns


# ANALYSIS OF VALUABLE SIGNALS' INTENSITY
# this will help us group the signals by intensity, and consequently help us estimate
# the location of the WAPs in relation to the users (useful for building prediction)
long_train$signalQuality <- ifelse(long_train$WAPrecord >=-66, "1. Top signal",
                                   ifelse(long_train$WAPrecord >=-69, "2. Very Good signal",
                                          ifelse(long_train$WAPrecord >=-79, "3. Good signal",
                                                 ifelse(long_train$WAPrecord >=-89, "4. Bad signal",
                                                        "5. Very bad signal"))))

# Signal intensity distribution by building 
ggplot(data = long_train) +
  aes(x = WAPrecord, fill = signalQuality) +
  geom_histogram(bins = 30) +
  theme_minimal() +            # the majority of the signals in each building are bad 
  facet_wrap(vars(BUILDINGID)) # we should try to only keep the top signals  

# do top signal WAPs cover all buildings? 
top_signals_df <- filter(long_train, 
                         long_train$signalQuality ==
                           "1. Top signal")

# Compare all waps records...
ggplot(data = long_train) +       
  aes(x = LONGITUDE, y = LATITUDE) +
  geom_point(color = "#0c4c8a") +
  theme_minimal()
# to only TOP signal waps <-  # the TOP signals do not cover buildings completely!
ggplot(data = top_signals_df) +
  aes(x = LONGITUDE, y = LATITUDE) +
  geom_point(color = "#0c4c8a") +
  theme_minimal()
rm(top_signals_df)

# and what if we consider "top & Very Good signals"?
vg_top_signals <- filter(long_train, long_train$signalQuality =="1. Top signal" | long_train$signalQuality =="2. Very Good signal") 
vg_top_signals_wide <- wide_train %>% filter_at(vars_waps_tr, any_vars(. > -69)) # filter WAPs with values >-69 to take them for modelling


# only TOP AND VG signal waps 
ggplot(data = vg_top_signals) +
  aes(x = LONGITUDE, y = LATITUDE, fill = signalQuality) +
  geom_point(color = "#0c4c8a") +
  theme_minimal()       # although precision would increase, there are tiny areas 
# (specially in building TD) that stay uncovered

rm(vg_top_signals)

# let's analyse this including FLOOR attribute to see the distribution of signals by quality 
# in each building and floor

# signals per floor 3D plot in building TI  
buildingTI <- filter(long_train, long_train$BUILDINGID =="TI")
ggplot(data = buildingTI) +
  aes(x = LONGITUDE, y = LATITUDE, color = signalQuality) +
  geom_point() +
  theme_minimal() +
  facet_wrap(vars(FLOOR))

# signals per floor 3D plot in building TD
buildingTD <- filter(long_train, long_train$BUILDINGID =="TD")
ggplot(data = buildingTD) +
  aes(x = LONGITUDE, y = LATITUDE, color = signalQuality) +
  geom_point() +
  theme_minimal() +
  facet_wrap(vars(FLOOR))

# signals per floor 3D plot in building TC
buildingTC <- filter(long_train, long_train$BUILDINGID =="TC")
ggplot(data = buildingTC) +
  aes(x = LONGITUDE, y = LATITUDE, color = signalQuality) +
  geom_point() +
  theme_minimal() +
  facet_wrap(vars(FLOOR))




######################### ATTRIBUTE SELECTION & ENGINEERING ##########################

# in order to reduce dimensions but keep important information given by the waps
# we will run a PCA to create a new data set containing components with the highest variance

# Preprocess to standarize the WAPs attributes
compress <- preProcess(wide_train[ ,vars_waps_tr], 
                       method = c("center", "scale", "pca"), 
                       thresh = 0.80) # PCA needed 77 components to capture 80 percent of the variance

  # compress_newtr <- preProcess(new_wide_train[ ,vars_waps_trnew], 
  #                               method = c("center", "scale", "pca"), 
  #                               thresh = 0.80) # PCA needed 63 components to capture 80 percent of the variance
  # 


# PCA visualization
pca_train <- prcomp(wide_train[, vars_waps_tr], 
                    center= TRUE, scale.= TRUE, rank. = 77) # dim(pca_train$x) the matrix x has 
#the principal component score vectors in a 19300 - 143 dimension
PCs_sd <- pca_train$sdev #compute standard deviation of each principal component
PCs_var <- PCs_sd^2 # compute variance
# PCs_var[1:77] # find the PCs with max variance (we check variance  
# of first 77 components) to retain as much information as possible 
# pca_train$rotation[1:5,1:4] # gets PC weights (look at first 4 principal components and first 5 rows)


var_explained <- PCs_var/sum(PCs_var) # proportion of variance explained by each component
plot(var_explained, xlab = "Principal Component", # var_explained Results:  1st component explains 6.8% variance. 
     ylab = "Proportion of Variance Explained",          # 2nd 5.4% variance
     type = "b")                                         # 3rd 4.2% variance...


# cumulative proportion of variance explained plot  
plot(cumsum(var_explained), xlab = "Principal Component", # shows that taking 77 components 
     ylab = "Cumulative Proportion of Variance Explained", # results in variance close to ~ 80%
     type = "b") 



# Preparing the TRAINING set for PCA
training_PCA <- predict(compress, wide_train[,vars_waps_tr])
vars_not_waps_tr <- wide_train[ ,c("BUILDINGID","FLOOR","LONGITUDE", "LATITUDE",
                                   "PHONEID", "USERID","SPACEID", "RELATIVEPOSITION")]
training_PCA <- cbind(training_PCA, vars_not_waps_tr)

# Preparing the VALIDATION set for PCA
testing_PCA <- predict(compress, wide_test[,vars_waps_tst])
vars_not_waps_tst <- wide_test[ ,c("BUILDINGID","FLOOR","LONGITUDE", "LATITUDE",
                                   "PHONEID", "USERID","SPACEID", "RELATIVEPOSITION")]
testing_PCA <- cbind(testing_PCA, vars_not_waps_tst)

# Preparing the NEW TRAIN and TEST set for PCA

      # new_training_PCA <-  predict(compress_newtr, new_wide_train[,vars_waps_trnew])
      # vars_not_waps_newtr <- new_wide_train[ ,c("BUILDINGID","FLOOR","LONGITUDE", "LATITUDE",
      #                                    "PHONEID", "USERID","SPACEID", "RELATIVEPOSITION")]
      # new_training_PCA <- cbind(new_training_PCA, vars_not_waps_newtr)
 

testing_newPCA <- predict(compress, new_test[,vars_waps_newtst])
vars_not_waps_newtst <- new_test[ ,c("BUILDINGID","FLOOR","LONGITUDE", "LATITUDE",
                                   "PHONEID", "USERID","SPACEID", "RELATIVEPOSITION")]
testing_newPCA <- cbind(testing_newPCA, vars_not_waps_newtst)

      # new_testing_PCA <- predict(compress_newtr, new_test[,vars_waps_newtst])
      # vars_not_waps_newtst <- new_test[ ,c("BUILDINGID","FLOOR","LONGITUDE", "LATITUDE",
      #                                    "PHONEID", "USERID","SPACEID", "RELATIVEPOSITION")]
      # new_testing_PCA <- cbind(new_testing_PCA, vars_not_waps_newtst)


rm(PCs_sd, PCs_var, var_explained,
   vars_not_waps_tr, vars_not_waps_tst, vars_not_waps_newtst,
   compress)
#################################### SAMPLING & CROSS-VALIDATION #####################################

# Smart sampling
  # for training dfs available: wide_train / training_PCA 

sample_train <- wide_train %>% group_by(BUILDINGID, FLOOR) %>% sample_n(727) # it takes 727 samples from each building & floor

sample_PCA <- training_PCA %>% group_by(BUILDINGID, FLOOR) %>% sample_n(727) # it takes 727 samples from each building & floor

      # sample_newtr <- new_wide_train %>% group_by(BUILDINGID, FLOOR) %>% sample_n(727) # it takes 727 samples from each building & floor
      # 
      # sample_newPCA <- new_training_PCA %>% group_by(BUILDINGID, FLOOR) %>% sample_n(727) # it takes 727 samples from each building & floor

  

# Cross-Validation
fitControl <- trainControl(
  method = "repeatedcv",
  predictionBounds = c(0,NA),
  number = 10,
  repeats = 3)



########################################### MODELLING ###########################################


#### PREDICTING BUILDINGID ####

options(digits = 3)
# MODELS TRIED: RF   |   TO TRY: C5.0, SVM/SVR, KNN, LM, Model Trees, RandomForest 

# RANDOM FOREST (DECISION BASED MODEL)
set.seed(123)
# Train a random forest using waps as independent variable (sample_train)
# best mtry search for sample_train 
#(wide format df without zerovar & different waps & duplicates & >-30dbm)

# bestmtry_waps_build <- tuneRF(sample_train[vars_waps_tr],      # look for the best mtry
#                       sample_train$BUILDINGID,
#                       ntreeTry=100,
#                       stepFactor=2,
#                       improve=0.05,
#                       trace=TRUE,
#                       plot=T) # Result: 5

# system.time(buildingRF_waps <-randomForest(y=sample_train$BUILDINGID,    #      TI   TD   TC class.error
#                                  x=sample_train[vars_waps_tr],          # TI 2907    1    0    0.000344
#                                   importance=T,                         # TD    1 2904    3    0.001376
#                                   method="rf",                          # TC    0   16 3619    0.004402
#                                   ntree=100,
#                                   mtry=5)) # best mtry

# saveRDS(buildingRF_waps, "./Models/buildingRF_waps.rds")
buildingRF_waps <- readRDS("./Models/buildingRF_waps.rds")
confusionMatrix(buildingRF_waps$predicted, sample_train$BUILDINGID) # accuracy = 99.8%
# kappa = 99.7%


# Train a random forest using pcs instead of waps (sample_PCA)
pcs <- grep("PC", names(sample_PCA), value = TRUE) 

set.seed(123)
# model & confusion matrix
# system.time(buildingRF_pcs <-randomForest(y=sample_PCA$BUILDINGID,             #      TI   TD   TC class.error
#                                  x=sample_PCA[pcs],                             # TI 2908    0    0    0.000000
#                                  importance=T,                                  # TD    1 2907    0    0.000344
#                                  method="rf",                                   # TC    0   15 3620    0.004127
#                                  ntree=100))
# 
# saveRDS(buildingRF_pcs, "./Models/buildingRF_pcs.rds")

buildingRF_pcs <- readRDS("./Models/buildingRF_pcs.rds") # better results checking confusion matrix! 
confusionMatrix(buildingRF_pcs$predicted, sample_PCA$BUILDINGID) # accuracy = 99.8%  # kappa = 99.8%



#### PREDICTING BUILDING IN VALIDATION ####
predB_RFwaps <- predict(buildingRF_waps, newdata = wide_test)
confusionMatrix(predB_RFwaps, wide_test$BUILDINGID)  # Accuracy in test 99,8%

predB_RFpcs <- predict(buildingRF_pcs, newdata = testing_PCA)
confusionMatrix(predB_RFpcs, wide_test$BUILDINGID)  # Accuracy in test 100%

# Reference
# Prediction  TI  TD  TC
# TI 536   0   0
# TD   0 307   0
# TC   0   0 268



# Create df with real and predicted results (by all models with best results)
b_predictions <- as.data.frame(wide_test$BUILDINGID)
b_predictions$predictionRFwaps <- predB_RFwaps
b_predictions$predictionRFpcs <- predB_RFpcs


# add final predictions to sample_train and wide_test
sample_train$pred_building <- buildingRF_pcs$predicted
sample_PCA$pred_building <- buildingRF_pcs$predicted
wide_test$pred_building <- predB_RFpcs
testing_PCA$pred_building <- predB_RFpcs


###### PREDICTING BUILDING IN NEW TEST <- NEW TEST PREDICTION OUTPUT

new_predB_RFwaps <- predict(buildingRF_waps, newdata = new_test) # BETTER ACCURACY IN TRAIN&VALIDATION
  # new_predB_RFpcs <- predict(buildingRF_pcs, newdata = testing_newPCA) 

new_test$pred_building <- new_predB_RFwaps
testing_newPCA$pred_building <- new_predB_RFwaps

############################################# FLOOR ##################################################

##### TRAINING MODELS FOR FLOOR ####

# RANDOM FOREST (DECISION BASED MODEL)
set.seed(123)

# Train a random forest using waps as independent variable to predict floor
# bestmtry_waps_floor <- tuneRF(sample_train[vars_waps_tr],      # look for the best mtry
#                       sample_train$FLOOR,
#                       ntreeTry=100,
#                       stepFactor=2,
#                       improve=0.05,
#                       trace=TRUE,
#                       plot=T) # Result: 34

# model & confusion matrix
# system.time(floorRF_waps <- randomForest(y= sample_train$FLOOR,           # 0    1    2    3   4  class.error
#                                         x= sample_train[vars_waps_tr], # 0 2158    9    0   14   0  0.01055
#                                         importance=T,               # 1    3 2175    2    1   0     0.00275
#                                         method="rf",                # 2    0    3 2172    6   0     0.00413
#                                         ntree=100,                  # 3    0    0    2 2178   1     0.00138
#                                         mtry=34)) # best mtry       # 4    0    0    0    2 725     0.00275

# ONLY WAP VARS USED (a bit worst result using pred_building)

# saveRDS(floorRF_waps, "./Models/floorRF_waps.rds")
floorRF_waps <- readRDS("./Models/floorRF_waps.rds")  # better results! 
confusionMatrix(floorRF_waps$predicted, sample_train$FLOOR)  # Accuracy 99.5%
# kappa 99.4%




# Train a random forest using pcs instead of waps (sample_PCA)
set.seed(123)
# best mtry search for sample_PCA (wide format df without duplicates & >-30dbm & PCA applied)
# bestmtry_pcs_floor <- tuneRF(sample_PCA[pcs],      # look for the best mtry
#                       sample_PCA$FLOOR,
#                       ntreeTry=100,
#                       stepFactor=2,
#                       improve=0.05,
#                       trace=TRUE,
#                       plot=T) # Result: 11
# model & confusion matrix
# system.time(floorRF_pcs <-randomForest(y=sample_PCA$FLOOR,                # 0    1    2    3   4 class.error
#                                  x=sample_PCA[pcs],                # 0   2155   11    0   15   0     0.01192
#                                  importance=T,                     # 1    7   2169    5    0   0     0.00550
#                                  method="rf",                      # 2    2     10 2152   17   0     0.01330
#                                  ntree=100,                        # 3    0     0     18 2161   2     0.00917
#                                  mtry=11)) # best mtry             # 4    0     0      0   8   719    0.01100

# saveRDS(floorRF_pcs, "./Models/floorRF_pcs.rds")
floorRF_pcs <- readRDS("./Models/floorRF_pcs.rds")
# confusionMatrix(floorRF_pcs$predicted, sample_PCA$FLOOR)



#### TESTING MODELS FOR FLOOR ####

predF_RFwaps <- predict(floorRF_waps, newdata = wide_test) # Accuracy in test 90%
confusionMatrix(predF_RFwaps, wide_test$FLOOR)             # kappa 87.3%
# Reference
# Prediction   0   1   2   3   4
# 0  117   2   1   0   1
# 1   10 410   7   0   0
# 2    4  37 287   4   0
# 3    1  13  11 165   7
# 4    0   0   0   3  31


predF_RFpcs <- predict(floorRF_pcs, newdata = testing_PCA) # Accuracy in test 90%
confusionMatrix(predF_RFpcs, testing_PCA$FLOOR) # Accuracy in test 86%; not included in final df


# Create df with real and predicted results (by all models with best results)
f_predictions <- as.data.frame(wide_test$FLOOR)
f_predictions$predictionRFwaps <- predF_RFwaps


# add final predictions to sample_train and wide_test
sample_train$pred_floor <- floorRF_waps$predicted
sample_PCA$pred_floor <- floorRF_waps$predicted
wide_test$pred_floor <- predF_RFwaps
testing_PCA$pred_floor <- predF_RFwaps


#### PREDICTING FLOOR IN NEW TEST <- FINAL OUTPUT

# new_predF_RFwaps <- predict(floorRF_waps, newdata = new_test) 
new_predF_RFpcs <- predict(floorRF_pcs, newdata = testing_newPCA) # BETTER ACCURACY IN TRAIN&VALIDATION

new_test$pred_floor <- new_predF_RFpcs
testing_newPCA$pred_floor <- new_predF_RFpcs

########################################### LONGITUDE ##################################################

##### TRAINING MODELS FOR LONGITUDE ####

# Train a random forest using WAPS & BUILDING independent variable to predict floor
vars_longwaps <- grep("WAP|pred_building" , 
                      names(sample_train), value= TRUE)

vars_longpcs <- grep("PC|pred_building" , 
                     names(sample_PCA), value = TRUE) 

set.seed(123)
# system.time(longRF_waps <- randomForest(y= sample_train$LONGITUDE,
#                                         x= sample_train[vars_longwaps],
#                                         importance=T,
#                                         method="rf",
#                                         ntree=100))
# 
# saveRDS(longRF_waps, "./Models/longRF_waps.rds")
longRF_waps <- readRDS("./Models/longRF_waps.rds")


# system.time(longRF_pcs <- randomForest(y= sample_PCA$LONGITUDE,
#                                         x= sample_PCA[vars_longpcs],
#                                         importance=T,
#                                         method="rf",
#                                         ntree=100))
# saveRDS(longRF_pcs, "./Models/longRF_pcs.rds")
longRF_pcs <- readRDS("./Models/longRF_pcs.rds")


#### TESTING MODELS FOR LONGITUDE ####
set.seed(123)
predLG_RFwaps <- predict(longRF_waps, newdata = wide_test) # Accuracy in test 99.5% | MAE 6.08
postResample(predLG_RFwaps, wide_test$LONGITUDE)  

set.seed(123)
predLG_RFpcs <- predict(longRF_pcs, newdata = testing_PCA) # Accuracy in test 99.5% | MAE 6.28 
postResample(predLG_RFpcs, testing_PCA$LONGITUDE)



# add final predictions to sample_train and wide_test
sample_train$pred_long <- longRF_pcs$predicted
sample_PCA$pred_long <- longRF_pcs$predicted
wide_test$pred_long <- predLG_RFpcs
testing_PCA$pred_long <- predLG_RFpcs


# Create df with real and predicted results (by all models with best results)
long_predictions <- as.data.frame(wide_test$LONGITUDE)
long_predictions$predictionRFwaps <- predLG_RFwaps
long_predictions$predictionRFpcs <- predLG_RFpcs
names(long_predictions)[1] <- paste("real.in.validation")

# error analysis for longitude predictions
long_predictions$RF_MAE<- abs(long_predictions$real.in.validation - 
                                long_predictions$predictionRFwaps)
long_predictions$RF_MRE <- long_predictions$RF_MAE/long_predictions$real.in.validation


# Error analysis for longitude (Random Forest model)
# MAE
ggplot(data = long_predictions,
       aes(x= long_predictions$real.in.validation,
           y = long_predictions$RF_MAE))+
  geom_smooth()+geom_point()+ggtitle("Longitude Random Forest MAE Analysis")


# MRE
ggplot(data = long_predictions,
       aes(x= long_predictions$real.in.validation,
           y = long_predictions$RF_MRE))+
  geom_smooth()+geom_point()+ggtitle("Longitude Random Forest MRE Analysis")


#### PREDICT LONGITUDE IN NEW TEST <- FINAL OUTPUT
        # common_waps2 <- intersect(vars_waps_newtst, vars_waps_tr) # 238 common waps
        # common_waps_tr2 <- select_at(wide_train[vars_waps_tr], common_waps2)
        # # common_waps_newtst <- select_at(new_test[vars_waps_newtst], common_waps2)
        # 
        # # new_wide_train <- cbind(common_waps_tr2, wide_train[466:474])
        # # new_test <- cbind(common_waps_newtst, new_test[271:279])

new_predLG_RFwaps <- predict(longRF_waps, newdata = new_test) 
new_predLG_RFpcs <- predict(longRF_pcs, newdata = testing_newPCA) 


new_test$pred_long <- new_predLG_RFwaps # BETTER ACCURACY IN TRAIN&VALIDATION
testing_newPCA$pred_long <- new_predLG_RFwaps

#################################### LATITUDE ######################################

vars_latwaps <- grep("WAP|pred_building|pred_long", names(sample_train), value = TRUE)

vars_latpcs <- grep("PC|pred_building|pred_long", names(sample_PCA), value = TRUE) 

# Train a random forest using WAPS & BUILDING independent variable to predict floor
set.seed(123)
# system.time(latRF_waps <- randomForest(y= sample_train$LATITUDE,
#                                         x= sample_train[vars_latwaps],
#                                         importance=T,
#                                         method="rf",
#                                         ntree=100))
# saveRDS(latRF_waps, "./Models/latRF_waps.rds")
latRF_waps <- readRDS("./Models/latRF_waps.rds")


# Train a random forest using pcs instead of waps (sample_PCA)
set.seed(123)
# model with PCs instead of waps
# system.time(latRF_pcs <- randomForest(y=sample_PCA$LATITUDE,
#                                  x=sample_PCA[vars_latpcs],
#                                  importance=T,
#                                  method="rf",
#                                  ntree=100))
saveRDS(latRF_waps, "./Models/latRF_pcs.rds")
latRF_pcs <- readRDS("./Models/latRF_pcs.rds")

#### TESTING MODELS FOR LATITUDE ####
predLAT_RFwaps <- predict(latRF_waps, newdata = wide_test) # Accuracy in test 98.5% | MAE 5.96 
postResample(predLAT_RFwaps, wide_test$LATITUDE)

predLAT_RFpcs <- predict(latRF_pcs, newdata = testing_PCA) # Accuracy in test 98.6% | MAE 5.73 
postResample(predLAT_RFpcs, testing_PCA$LATITUDE)


# add final predictions to sample_train and wide_test
sample_train$pred_lat <- latRF_waps$predicted
wide_test$pred_lat <- predLAT_RFwaps

# Create df with real and predicted results (by all models with best results)
lat_predictions <- as.data.frame(wide_test$LATITUDE)
lat_predictions$predictionRFpcs <- predLAT_RFpcs
names(lat_predictions)[1] <- paste("real.in.validation")

# error analysis for latitude predictions
lat_predictions$RF_MAE<- abs(lat_predictions$real.in.validation - lat_predictions$predictionRFwaps)
lat_predictions$RF_MRE <- lat_predictions$RF_MAE/lat_predictions$real.in.validation


# Error analysis for latitude (Random Forest model)
# MAE
ggplot(data = lat_predictions,
       aes(x= lat_predictions$real.in.validation,
           y = lat_predictions$RF_MAE))+
  geom_smooth()+geom_point()+ggtitle("Latitude Random Forest MAE Analysis")


# MRE
ggplot(data = lat_predictions,
       aes(x= lat_predictions$real.in.validation,
           y = lat_predictions$RF_MRE))+
  geom_smooth()+geom_point()+ggtitle("Latitude Random Forest MRE Analysis")


#### PREDICT LATITUDE IN NEW TEST <- FINAL OUTPUT

x <- new_test[!new_test %in% sample_train]

x <- sample_train[!sample_train %in% new_test]

data2[data1$char1 ! %in% 
        >> > c("string1","string2"),1]

new_predLAT_RFwaps <- predict(latRF_waps, newdata = new_test) 
new_predLAT_RFpcs <- predict(latRF_pcs, newdata = testing_newPCA) # BETTER ACCURACY IN TRAIN&VALIDATION




##################################### FINAL PREDICTIONS ####################################

final_predictions <- as.data.frame(cbind(b_predictions$predictionRFpcs, 
                           f_predictions$predictionRFwaps, 
                           long_predictions$predictionRFwaps,
                           lat_predictions$predictionRFpcs))

names(final_predictions) <- c("pred_building", "pred_floor", "pred_longitude", "pred_latitude")


# esquisser()

