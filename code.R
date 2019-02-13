## WIFI LOCATIONING ##

####SET ENVIRONMENT####

pacman:: p_load("readr","dplyr", "tidyr", "ggplot2", "plotly", 
                "data.table", "reshape2","ggridges", "party",
                "esquisse", "caret", "randomForest")

setwd("C:/Users/usuario/Desktop/UBIQUM/Project 8 - Wifi locationing")
original_train <- read_csv("./trainingData.csv")
original_test <- read_csv("./validationData.csv")




#### PRE-PROCESSING ####

#### SET NAs to low intensity RSSI in wide format #####
wide_train <- original_train
wide_test <- original_test

wide_train[, 1:520][wide_train[, 1:520] == 100] <- -105 # place them as low signal 
wide_test[, 1:520][wide_test[, 1:520] == 100] <- -105 


#### SET NAs to low intensity RSSI in long format #####

# melt original_train to change [ , 1:520] attributes to variables (WAPs as ID)
varnames <- colnames(original_train[,521:529])
long_train <- melt(original_train, id.vars = varnames)
names(long_train)[10]<- paste("WAPid")
names(long_train)[11]<- paste("WAPrecord")
long_train[,11][long_train[, 11] == 100] <- -105 



# melt original_test 
varnames.test <- colnames(original_test[,521:529])
long_test <- melt(original_test, id.vars = varnames.test)
names(long_test)[10]<- paste("WAPid")
names(long_test)[11]<- paste("WAPrecord")
long_test[,11][long_test[, 11] == 100] <- -105 





##### SET NAs Remove -105 dbm to reduce sample size ####
long_train <- filter(long_train, long_train$WAPrecord != -105)
long_test <- filter(long_test, long_test$WAPrecord != -105)


#### DUPLICATES ####
long_train <- unique(long_train)


#### DATA TYPES ####

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




#### FEATURE SELECTION & ENGINEERING ####
## Create new attribute BUILDING-FLOOR

long_train$BuildingFloor <- paste(long_train$BUILDINGID, long_train$FLOOR, sep = "-")
long_test$BuildingFloor <- paste(long_test$BUILDINGID, long_test$FLOOR, sep = "-")

wide_train$BuildingFloor <- paste(wide_train$BUILDINGID, wide_train$FLOOR, sep = "-")
wide_test$BuildingFloor <- paste(wide_test$BUILDINGID, wide_test$FLOOR, sep = "-")



#### ANALYSE DISTRIBUTION ####

# Records distributed by Building and floor in TRAIN
ggplot(data = long_train) +
  aes(x = WAPrecord, fill = FLOOR) +
  geom_histogram(bins = 30) +
  theme_minimal() +
  facet_wrap(vars(BUILDINGID))

# In TEST 
ggplot(data = long_test) +
  aes(x = WAPrecord, fill = FLOOR) +
  geom_histogram(bins = 30) +
  theme_minimal() +
  facet_wrap(vars(BUILDINGID))



# Records distributed by location and floor in TRAIN
ggplot(data = long_train) +
  aes(x = LONGITUDE, y = LATITUDE, color = FLOOR) +
  geom_point() +
  theme_minimal()


# In TEST
ggplot(data = long_test) +
  aes(x = LONGITUDE, y = LATITUDE, color = FLOOR) +
  geom_point() +
  theme_minimal()


#### ANALYSE TOP SIGNALS ####
tooGood_signals <- long_train %>% filter(WAPrecord > -30)

# boxplot by WAPS in each building
ggplot(data = tooGood_signals) +
  aes(x = WAPid, y = WAPrecord, fill = BUILDINGID) +
  geom_boxplot() +
  theme_minimal()+ 
  theme(axis.text.x = element_text(angle = 90, hjust = 1))


# histogram of TOP WAP records by bulding&floor 
ggplot(data = tooGood_signals) +
  aes(x = WAPrecord, fill = FLOOR) +
  geom_histogram(bins = 30) +
  theme_minimal() +
  facet_wrap(vars(BUILDINGID))

# histograms phone 19-user 6 giving TOO GOOD SIGNALS
ggplot(data = tooGood_signals) +
  aes(x = PHONEID, fill = FLOOR) +
  geom_bar() +
  theme_minimal() +
  facet_wrap(vars(BUILDINGID))

ggplot(data = tooGood_signals) +
  aes(x = USERID, fill = FLOOR) +
  geom_bar() +
  theme_minimal() +
  facet_wrap(vars(BUILDINGID))


##### user 6 behaviour analysis ####
USER6 <- long_train %>% filter(USERID == 6)

# it gives bad results but it also records reasonable RSSI
ggplot(data = USER6) +
  aes(x = WAPrecord) +
  geom_histogram(bins = 30, fill = "#0c4c8a") +
  theme_minimal()

## TC Floor 3-4 analysis 
TC3_exploration <- long_train %>% filter(BuildingFloor == "TC-3")

# User 6 captured a big proportion of data in TC3 and TC4
ggplot(data = TC3_exploration) +
  aes(x = WAPrecord, fill = USERID) +
  geom_histogram(bins = 30) +
  theme_minimal()


TC4_exploration <- long_train %>% filter(BuildingFloor == "TC-4")

ggplot(data = TC4_exploration) +
  aes(x = WAPrecord, fill = USERID) +
  geom_histogram(bins = 30) +
  theme_minimal()


# what would happen in terms of data amount if we removed user 6 records?

TC_exploration <- long_train %>% filter(BUILDINGID == "TC")

ggplot(data = TC_exploration) +
  aes(x = FLOOR, weight = WAPrecord) +
  geom_bar(fill = "#0c4c8a") +
  theme_minimal()



# as user 6 represents a big part of the records of TC3&4, we only remove the mistaken data >-30
long_train <- long_train %>% filter(WAPrecord <= -30)


#### CHECK RSSI WEIGHT  ####

long_train$signalQuality <- ifelse(long_train$WAPrecord >=-66, "1. Top signal",
                                  ifelse(long_train$WAPrecord >=-69, "2. Very Good signal",
                                         ifelse(long_train$WAPrecord >=-79, "3. Good signal",
                                                ifelse(long_train$WAPrecord >=-89, "4. Bad signal",
                                                       "5. Very bad signal"))))
                                  
# Signal intensity distribution
ggplot(data = long_train) +
  aes(x = WAPrecord, fill = signalQuality) +
  geom_histogram(bins = 30) +
  theme_minimal() +
  facet_wrap(vars(BUILDINGID))



# Do Top signal WAPs cover all buildings? not completely!
top_signals_df <- filter(long_train, long_train$signalQuality =="1. Top signal")

  # all waps
ggplot(data = long_train) +
  aes(x = LONGITUDE, y = LATITUDE) +
  geom_point(color = "#0c4c8a") +
  theme_minimal()

  # only TOP signal waps 
ggplot(data = top_signals_df) +
  aes(x = LONGITUDE, y = LATITUDE) +
  geom_point(color = "#0c4c8a") +
  theme_minimal()



# What about top & VG signals?
vg_top_signals <- filter(long_train, long_train$signalQuality =="1. Top signal" | long_train$signalQuality =="2. Very Good signal") 

  # only TOP AND VG signal waps 
ggplot(data = vg_top_signals) +
  aes(x = LONGITUDE, y = LATITUDE, fill = signalQuality) +
  geom_point(color = "#0c4c8a") +
  theme_minimal()



# filter by building and floor to examine signals further

building_floor_df <- c()
WAPid_count <- c()

building_floor <- unique(long_train$BuildingFloor)

for (bf in building_floor) {
  building_floor_df[[bf]] <- as.data.frame(filter(long_train, BuildingFloor == bf))
  
  WAPid_count[[bf]] <- distinct(building_floor_df[[bf]], WAPid, .keep_all= TRUE) # Find the WAPs in each floor and building
  
  detection <- (WAPid_count[[bf]]$WAPid %in% WAPid_count[[bf]]$WAPid ==TRUE) # 146 WAPs  appear in two df at the same time
}


#3D plot per building and floor to check RSSI 
  #TI
buildingTI <- filter(long_train, long_train$BUILDINGID =="TI")
ggplot(data = buildingTI) +
  aes(x = LONGITUDE, y = LATITUDE, color = signalQuality) +
  geom_point() +
  theme_minimal() +
  facet_wrap(vars(FLOOR))

  #TD

buildingTD<- filter(long_train, long_train$BUILDINGID =="TD")
ggplot(data = buildingTD) +
  aes(x = LONGITUDE, y = LATITUDE, color = signalQuality) +
  geom_point() +
  theme_minimal() +
  facet_wrap(vars(FLOOR))

  #TC
buildingTC<- filter(long_train, long_train$BUILDINGID =="TC")
ggplot(data = buildingTC) +
  aes(x = LONGITUDE, y = LATITUDE, color = signalQuality) +
  geom_point() +
  theme_minimal() +
  facet_wrap(vars(FLOOR))




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
