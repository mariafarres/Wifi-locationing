

## Project: Wifi Locationing
## Script purpose: use the signal intensity recorded from multiple wifi hotspots within the building 
  #to determine users location using Machine Learning Models.
## Date: 6 Feb 2019
## Author: Maria Farrés



########################## SET ENVIRONMENT #######################################

#Load required libraries 
pacman:: p_load("readr","dplyr", "tidyr", "ggplot2", "plotly", 
                "data.table", "reshape2","ggridges", "party",
                "esquisse", "caret", "randomForest", "grDevices")

#Set working directory
setwd("C:/Users/usuario/Desktop/UBIQUM/Project 8 - Wifi locationing/Wifi-locationing")


#Read initial data sets
original_train <- read_csv("./DataSets/trainingData.csv")
original_test <- read_csv("./DataSets/validationData.csv")



######################## PRE-PROCESSING ##########################################

# wide format df
wide_train <- original_train # train in wide format
wide_test <- original_test # test in wide format



# long format df
# melt original_train to change [ , 1:520] attributes to variables (WAPs as ID)
varnames <- colnames(original_train[,521:529])
long_train <- melt(original_train, id.vars = varnames)
names(long_train)[10]<- paste("WAPid")
names(long_train)[11]<- paste("WAPrecord")

# melt original_test 
varnames.test <- colnames(original_test[,521:529])
long_test <- melt(original_test, id.vars = varnames.test)
names(long_test)[10]<- paste("WAPid")
names(long_test)[11]<- paste("WAPrecord")

rm(original_train)
rm(original_test)

# MISSING VALUES
# Set NAs to low intensity RSSI 
  # NAs mean that the WAPs have not recorded signal in a determinate user location
  # instead of deleting those signals we set them to -105 in a new data frame 
  # to indicate that the signal is really low in that location

wide_train[, 1:520][wide_train[, 1:520] == 100] <- -105 # place them as low signal 
wide_test[, 1:520][wide_test[, 1:520] == 100] <- -105 


# Set NAs to low intensity RSSI in long format #####
long_train[,11][long_train[, 11] == 100] <- -105 
long_test[,11][long_test[, 11] == 100] <- -105 



# create another long df which does not contain -105 dbm observations
  # this helps reduce the sample size for an initial exploration
long_train <- filter(long_train, long_train$WAPrecord != -105)
long_test <- filter(long_test, long_test$WAPrecord != -105)


# DATA TYPES & CLASS TREATMENT

# TRAIN
long_train$FLOOR <- as.factor(long_train$FLOOR)
long_train$BUILDINGID <- as.factor(long_train$BUILDINGID)
levels(long_train$BUILDINGID) <- c("TI",
                                  "TD",
                                  "TC")
long_train$SPACEID <- as.factor(long_train$SPACEID)
long_train$RELATIVEPOSITION <- as.factor(long_train$RELATIVEPOSITION)
levels(long_train$RELATIVEPOSITION) <- c("Inside",
                                     "Outside")
long_train$USERID <- as.factor(long_train$USERID)
long_train$PHONEID <- as.factor(long_train$PHONEID)
long_train$WAPid <- as.factor(long_train$WAPid)
long_train$TIMESTAMP <- as.POSIXct(long_train$TIMESTAMP, origin="1970-01-01")



wide_train$FLOOR <- as.factor(wide_train$FLOOR)
wide_train$BUILDINGID <- as.factor(wide_train$BUILDINGID)
levels(wide_train$BUILDINGID) <- c("TI",
                                  "TD",
                                  "TC")
wide_train$SPACEID <- as.factor(wide_train$SPACEID)
wide_train$RELATIVEPOSITION <- as.factor(wide_train$RELATIVEPOSITION)
levels(wide_train$RELATIVEPOSITION) <- c("Inside",
                                     "Outside")
wide_train$USERID <- as.factor(wide_train$USERID)
wide_train$PHONEID <- as.factor(wide_train$PHONEID)
wide_train$TIMESTAMP <- as.POSIXct(wide_train$TIMESTAMP, origin="1970-01-01")



# TEST

long_test$FLOOR <- as.factor(long_test$FLOOR)
long_test$BUILDINGID <- as.factor(long_test$BUILDINGID)
levels(long_test$BUILDINGID) <- c("TI",
                                  "TD",
                                  "TC")
long_test$SPACEID <- as.factor(long_test$SPACEID)
long_test$RELATIVEPOSITION <- as.factor(long_test$RELATIVEPOSITION)
levels(long_test$RELATIVEPOSITION) <- c("Inside",
                                     "Outside")
long_test$USERID <- as.factor(long_test$USERID)
long_test$PHONEID <- as.factor(long_test$PHONEID)
long_test$WAPid <- as.factor(long_test$WAPid)
long_test$TIMESTAMP <- as.POSIXct(long_test$TIMESTAMP, origin="1970-01-01")




wide_test$FLOOR <- as.factor(wide_test$FLOOR)
wide_test$BUILDINGID <- as.factor(wide_test$BUILDINGID)
levels(wide_test$BUILDINGID) <- c("TI",
                                  "TD",
                                  "TC")
wide_test$SPACEID <- as.factor(wide_test$SPACEID)
wide_test$RELATIVEPOSITION <- as.factor(wide_test$RELATIVEPOSITION)
levels(wide_test$RELATIVEPOSITION) <- c("Inside",
                                     "Outside")
wide_test$USERID <- as.factor(wide_test$USERID)
wide_test$PHONEID <- as.factor(wide_test$PHONEID)
wide_train$TIMESTAMP <- as.POSIXct(wide_train$TIMESTAMP, origin="1970-01-01")



# ZERO VARIANCE & DUPLICATES 

# check if there are WAPs that have no variance in all their records 

ZeroVar_check_train <- nearZeroVar(wide_train[1:520], saveMetrics = TRUE) # there are 55 WAPs with 0 variance
wide_train <- wide_train[-which(ZeroVar_check_train$zeroVar == TRUE)] # we remove them as they might ditort our model
rm(ZeroVar_check_train)

ZeroVar_check_test <- nearZeroVar(wide_test[1:520], saveMetrics = TRUE) #
wide_test <- wide_test[-which(ZeroVar_check_test$zeroVar == TRUE)]
rm(ZeroVar_check_test)

waps_wtr <- grep("WAP", names(wide_train), value = TRUE) 
waps_wtst <- grep("WAP", names(wide_test), value = TRUE)
common_waps <- intersect(waps_wtst, waps_wtr)

common_waps_tr<- select_at(wide_train[waps_wtr], common_waps)
common_waps_tst <- select_at(wide_test[waps_wtst], common_waps)

wide_train <- cbind(common_waps_tr, wide_train[466:474])
wide_test <- cbind(common_waps_tst, wide_test[368:376])


# real duplicates
wide_train <- unique(wide_train)
long_train <- unique(long_train)




# FEATURE ENGINEERING  
# Create new attribute BUILDING-FLOOR
long_train$BuildingFloor <- paste(long_train$BUILDINGID, long_train$FLOOR, sep = "-")
long_test$BuildingFloor <- paste(long_test$BUILDINGID, long_test$FLOOR, sep = "-")

wide_train$BuildingFloor <- paste(wide_train$BUILDINGID, wide_train$FLOOR, sep = "-")
wide_test$BuildingFloor <- paste(wide_test$BUILDINGID, wide_test$FLOOR, sep = "-")



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


# Can we remove this user completely though? 
# would it leave us with too few data in building TC floor 3&4? -> ANALYSIS OF RECORDS IN TC FLOOR 3 & 4
TC3_exploration <- long_train %>% filter(BuildingFloor == "TC-3")
ggplot(data = TC3_exploration) +    
  aes(x = WAPrecord, fill = USERID) +
  geom_histogram(bins = 30) +
  theme_minimal()
rm(TC3_exploration)


TC4_exploration <- long_train %>% filter(BuildingFloor == "TC-4")
ggplot(data = TC4_exploration) +
  aes(x = WAPrecord, fill = USERID) +
  geom_histogram(bins = 30) +
  theme_minimal()
rm(TC4_exploration)

# User 6 captured a big proportion of data in TC3 and TC4
# what would happen in terms of data amount if we removed user 6 records?
TC_exploration <- long_train %>% filter(BUILDINGID == "TC")
ggplot(data = TC_exploration) +
  aes(x = FLOOR, fill = USERID, weight = WAPrecord) +
  geom_bar() +                  # although we could still predict T3 
  theme_minimal()               # TC4 would end up with really few data 
rm(TC_exploration)


# as user 6 represents a big part of the records of TC3&4, 
# we only remove the data >-30dbm recorded; but we do not remove all user 6 records
long_train <- long_train %>% filter(WAPrecord <= -30)

waps_wtr2 <- grep("WAP", names(wide_train), value = TRUE) 
wide_train <- wide_train %>% filter_at(waps_wtr2, any_vars(. < -30)) # filter rows that do not contain values above -30dbm in WAP columns




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

# and what if we consider "top & Very Good signals"?
vg_top_signals <- filter(long_train, long_train$signalQuality =="1. Top signal" | long_train$signalQuality =="2. Very Good signal") 
vg_top_signals_wide <- wide_train %>% filter_at(waps, any_vars(. > -69)) # filter WAPs with values >-69 to take them for modelling



  # only TOP AND VG signal waps 
ggplot(data = vg_top_signals) +
  aes(x = LONGITUDE, y = LATITUDE, fill = signalQuality) +
  geom_point(color = "#0c4c8a") +
  theme_minimal()       # although precision would increase, there are tiny areas 
                        # (specially in building TD) that stay uncovered



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
compress <- preProcess(wide_train[ ,waps_wtr2], 
                       method = c("center", "scale", "pca"), 
                       thresh = 0.80) # PCA needed 77 components to capture 80 percent of the variance


# PCA visualization
pca_train <- prcomp(wide_train[,waps_wtr2], center= TRUE, scale.= TRUE, rank. = 77) # dim(pca_train$x) the matrix x has the principal component score vectors in a 19300 × 143 dimension
      # pca_train$rotation[1:5,1:4] #  rotation gets the components weights (look at first 4 principal components and first 5 rows)
PCs_sd <- pca_train$sdev #compute standard deviation of each principal component
PCs_var <- PCs_sd^2 # compute variance
      # PCs_var[1:77] # find the PCs with max variance (we check variance  
         # of first 77 components) to retain as much information as possible 

var_explained <- PCs_var/sum(PCs_var) # proportion of variance explained by each component
plot(var_explained, xlab = "Principal Component", # var_explained Results:  1st component explains 6.8% variance. 
     ylab = "Proportion of Variance Explained",          # 2nd 5.4% variance
     type = "b")                                         # 3rd 4.2% variance...


# cumulative proportion of variance explained plot  
plot(cumsum(var_explained), xlab = "Principal Component", # shows that taking 77 components 
     ylab = "Cumulative Proportion of Variance Explained", # results in variance close to ~ 80%
     type = "b") 



# Preparing the TRAINING set for PCA
training_PCA <- predict(compress, wide_train[,waps_wtr2])
not_waps <- wide_train[ ,c("BUILDINGID","FLOOR","LONGITUDE", "LATITUDE",
                           "PHONEID", "USERID","SPACEID", "RELATIVEPOSITION")]
training_PCA <- cbind(training_PCA, not_waps)


# Preparing the TESTING set for PCA
testing_PCA <- predict(compress, wide_test[,waps_wtst2])
not_waps_tst <- wide_test[ ,c("BUILDINGID","FLOOR","LONGITUDE", "LATITUDE",
                           "PHONEID", "USERID","SPACEID", "RELATIVEPOSITION")]
testing_PCA <- cbind(testing_PCA, not_waps_tst)




#################################### SAMPLING & CROSS-VALIDATION #####################################
# Smart sampling

sample_wide <- wide_train %>% group_by(BUILDINGID, FLOOR) %>% sample_n(727) # it takes x samples from each building & floor

sample_PCA <- training_PCA %>% group_by(BUILDINGID, FLOOR) %>% sample_n(727) # it takes x samples from each building & floor


# Data partition 


# Cross-Validation
fitControl <- trainControl(
  method = "repeatedcv",
  predictionBounds = c(0,NA),
  number = 10,
  repeats = 3)


# PREPARE TRAIN 





########################################### MODELLING ###########################################


#### PREDICTING BUILDINGID ####

options(digits = 3)
# MODELS TRIED: RF   |   TO TRY: C5.0, SVM/SVR, KNN, LM, Model Trees, RandomForest 

# RANDOM FOREST (DECISION BASED MODEL)
set.seed(123)
# Train a random forest using waps as independent variable 
  # best mtry search for sample_wide 
    #(wide format df without zerovar & different waps & duplicates & >-30dbm)

    # bestmtry_waps_build <- tuneRF(sample_wide[waps_wtr2],      # look for the best mtry
    #                       sample_wide$BUILDINGID,
    #                       ntreeTry=100,
    #                       stepFactor=2,
    #                       improve=0.05,
    #                       trace=TRUE,
    #                       plot=T) # Result: 5

      # system.time(buildingRF_waps <-randomForest(y=sample_wide$BUILDINGID,    #      TI   TD   TC class.error
      #                                  x=sample_wide[waps_wtr2],              # TI 2907    1    0    0.000344
      #                                   importance=T,                         # TD    1 2904    3    0.001376
      #                                   method="rf",                          # TC    0   16 3619    0.004402
      #                                   ntree=100,
      #                                   mtry=5)) # best mtry

      
# saveRDS(buildingRF_waps, "./Models/buildingRF_waps.rds")
# buildingRF_waps <- readRDS("./Models/buildingRF_waps.rds")
# confusionMatrix(buildingRF_waps$predicted, sample_wide$BUILDINGID) # accuracy = 99.8%
                                                                   # kappa = 99.7%



# Train a random forest using pcs instead of waps (sample_PCA)
pcs <- grep("PC", names(sample_PCA), value = TRUE) 
set.seed(123)
  # best mtry search for sample_PCA (wide format df without duplicates & >-30dbm & PCA applied)
    # bestmtry_pcs_build <- tuneRF(sample_PCA[pcs],      # look for the best mtry
    #                       sample_PCA$BUILDINGID,
    #                       ntreeTry=100,
    #                       stepFactor=2,
    #                       improve=0.05,
    #                       trace=TRUE,
    #                       plot=T) # Result: 4 & 16
      # model & confusion matrix
      # system.time(buildingRF_pcs <-randomForest(y=sample_PCA$BUILDINGID,             #      TI   TD   TC class.error
      #                                  x=sample_PCA[pcs],                             # TI 2907    1    0    0.000344
      #                                  importance=T,                                  # TD    0 2908    0    0.000000
      #                                  method="rf",                                   # TC    0   16 3619    0.004402
      #                                  ntree=100,
      #                                  mtry=4)) # best mtry


# saveRDS(buildingRF_pcs, "./Models/buildingRF_pcs.rds")
buildingRF_pcs <- readRDS("./Models/buildingRF_pcs.rds") # better results checking confusion matrix! 
confusionMatrix(buildingRF_pcs$predicted, sample_PCA$BUILDINGID) # accuracy = 99.8%
                                                                   # kappa = 99.7%


#### PREDICTING BUILDING IN TEST & RUN ERROR ANALYSIS ####
predB_RFwaps <- predict(buildingRF_waps, newdata = wide_test)
predB_RFpcs <- predict(buildingRF_pcs, newdata = testing_PCA)



# Create df with real and predicted results (by all models with best results)
b_predictions <- as.data.frame(b_predictions)
b_predictions$real <- wide_test$BUILDINGID
b_predictions$predictionRFwaps <- predB_RFwaps




############################################# FLOOR ##################################################

##### TRAINING MODELS FOR FLOOR ####

# RANDOM FOREST (DECISION BASED MODEL)
set.seed(123)

# Train a random forest using waps as independent variable to predict floor
  # bestmtry_waps_floor <- tuneRF(sample_wide[waps],      # look for the best mtry
  #                       sample_wide$FLOOR,
  #                       ntreeTry=100,
  #                       stepFactor=2,
  #                       improve=0.05,
  #                       trace=TRUE,
  #                       plot=T) # Result: 42 

    # model & confusion matrix
    # system.time(floorRF_waps <-randomForest(y= sample_wide$FLOOR,        # 0    1    2    3   4  class.error
    #                                         x= sample_wide[waps],   # 0 2156    4    0   21   0 0.0114626318
    #                                         importance=T,           # 1    2 2177    0    2   0 0.0018340211
    #                                         method="rf",            # 2    0    7 2165    9   0 0.0073360844
    #                                         ntree=100,              # 3    0    0    1 2179   1 0.0009170105
    #                                         mtry=42)) # best mtry   # 4    0    0    0    2 725 0.0027510316                     
# saveRDS(floorRF_waps, "./Models/floorRF_waps.rds")
floorRF_waps <- readRDS("./Models/floorRF_waps.rds")  # better results! 
confusionMatrix(floorRF_waps$predicted, sample_wide$FLOOR)

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
    # system.time(floorRF_pcs <-randomForest(y=sample_PCA$FLOOR,            # 0    1    2    3   4 class.error
    #                                  x=sample_PCA[pcs],              # 0 2160    6    0   15   0 0.009628611
    #                                  importance=T,                     # 1    8 2161   10    2   0 0.009170105
    #                                  method="rf",                      # 2    0   10 2150   21   0 0.014213663
    #                                  ntree=100,                        # 3    0    1   17 2162   1 0.008711600
    #                                  mtry=11)) # best mtry             # 4    0    0    2    4 721 0.008253095

# saveRDS(floorRF_pcs, "./Models/floorRF_pcs.rds")
# floorRF_pcs <- readRDS("./Models/floorRF_pcs.rds")
# confusionMatrix(floorRF_pcs$predicted, sample_PCA$FLOOR)



#### TESTING MODELS FOR FLOOR ####

  # ISSUE: variables in the training data missing in newdata

predict_floorRF_waps <- predict(floorRF_waps, newdata = wide_test)

predict_floorRF_pcs <- predict(floorRF_pcs, newdata = wide_test)


wide_test$lmpredictions <- applymodel1
# wide_test$absolute.errorlm <- abs(testing$Volume - 
#                                   testing$lmpredictions)
# testing$relative.errorlm <- testing$absolute.errorlm/testing$Volume
# 
# Errors.LM <- ggplot(data = testing, aes(x =Volume, y = absolute.errorlm))+
#   geom_smooth()+geom_point()+ggtitle("Absolute Errors in LM")
# Errors.LM
# 
# Metrics.LM <- postResample(pred = testing$lmpredictions, obs = testing$Volume)
# 






# # Preprocess to dummify building and incorporate it to floor prediction
# train_regression <- dummyVars("~BUILDINGID", data = train_classification) 
# <- data.frame(predict(existing.dummified, 
#                       newdata = existing)) # new df with dummified Product Type




# floor_predictions <- c()
# long_predictions <- c()
# lat_predictions <- c()
# 

# models_resamples <- resamples(list(RF = modelRF_class, RF = modelRF_wide))



#### PREDICT LONGITUDE ####




#### PREDICT LATITUDE ####


# GRADIENT BOOSTING TREES
set.seed(123)
modelGBT <- caret::train(BUILDINGID~ .,
                         data = sample_wide, 
                         trControl= fitControl, 
                         method = "gbm")

modelGBT$results # shrinkage 0.1; interaction depth 3; ntrees 150;  R^2 0.9972912; MAE;0.01499280 
postResample




# estandarizar phone tendria sentido


#how do we decide how many components should we select for modeling stage


# DISTANCE BASED MODELLING
# Before running distance based models we need to standarize the attributes 
# for them to be in the same scale
# Attributes tandardization to check if it helps the model perform better 
# DUMMIFY <- + range 




#LINEAR MODEL ()
set.seed(123)
model1_lm <- lm(BUILDINGID ~ .,
           data = sample_wide)
model1_lm


# DISTANCE BASED MODELS

sapply(wide_train[, waps],var) # check variance
range(sapply(wide_train[, waps],var)) # variance seems strong in this context
widetrain_standardized <- as.data.frame(scale(wide_train[, waps]))
sapply(widetrain_standardized, sd) # effect of standarization sd = 1


################################### PREDICT LONGITUDE ############################

# DECISION BASED
# RANDOM FOREST

# DISTANCE BASED






################################# OTHER STUFF ############################


# Create and set list to contain all the predictions

# predictions_list <- c()
# 
# for (model in building_models) {
#   building_predictions[[model]] <- as.data.frame()
# }




# COUNT WAPs per BUILDING & FLOOR
# # filter by building and floor to examine signals further
# building_floor_df <- c()
# WAPid_count <- c()
# 
# building_floor <- unique(long_train$BuildingFloor)
# 
# for (bf in building_floor) {
#   building_floor_df[[bf]] <- as.data.frame(filter(long_train, BuildingFloor == bf))
# 
#   WAPid_count[[bf]] <- distinct(building_floor_df[[bf]], WAPid, .keep_all= TRUE) # Find the WAPs in each floor and building
# 
#   detection <- (WAPid_count[[bf]]$WAPid %in% WAPid_count[[bf]]$WAPid ==TRUE) # 146 WAPs  appear in two df at the same time
# }