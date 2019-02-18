

## Project: Wifi Locationing
## Script purpose: use the signal intensity recorded from multiple wifi hotspots within the building 
  #to determine users location using Machine Learning Models.
## Date: 6 Feb 2019
## Author: Maria Farrés



########################## SET ENVIRONMENT #######################################

#Load required libraries 
pacman:: p_load("readr","dplyr", "tidyr", "ggplot2", "plotly", 
                "data.table", "reshape2","ggridges", "party",
                "esquisse", "caret", "randomForest")

#Set working directory
setwd("C:/Users/usuario/Desktop/UBIQUM/Project 8 - Wifi locationing")

#Read initial data sets
original_train <- read_csv("./trainingData.csv")
original_test <- read_csv("./validationData.csv")



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




# DUPLICATES 
long_train <- unique(long_train)



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



# as user 6 represents a big part of the records of TC3&4, 
# we only remove the data >-30dbm that he recorded
long_train <- long_train %>% filter(WAPrecord <= -30)




# CHECK RSSI WEIGHT
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
buildingTD<- filter(long_train, long_train$BUILDINGID =="TD")
ggplot(data = buildingTD) +
  aes(x = LONGITUDE, y = LATITUDE, color = signalQuality) +
  geom_point() +
  theme_minimal() +
  facet_wrap(vars(FLOOR))

  # signals per floor 3D plot in building TC
buildingTC<- filter(long_train, long_train$BUILDINGID =="TC")
ggplot(data = buildingTC) +
  aes(x = LONGITUDE, y = LATITUDE, color = signalQuality) +
  geom_point() +
  theme_minimal() +
  facet_wrap(vars(FLOOR))





















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




####trying things####

# TC_vg_top_signals <- filter(long_train, long_train$BUILDINGID =="TC", long_train$signalQuality =="1. Top signal" | long_train$signalQuality =="2. Very Good signal") 
# 
# ggplot(data = TC_vg_top_signals) +
#   aes(x = LONGITUDE, y = LATITUDE, color = WAPid) +
#   geom_point() +
#   theme_minimal()+
#   facet_wrap(vars(FLOOR))


# ####TO DO!#####
# decisonTree <- ctree(brand ~ salary + age, data = cr)
# plot(ct, tp_args = list(text = TRUE))
