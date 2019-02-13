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
widetrain <- original_train
widetest <- original_test

widetrain[, 1:520][widetrain[, 1:520] == 100] <- -105 # place them as low signal 
widetest[, 1:520][widetest[, 1:520] == 100] <- -105 


#### SET NAs to low intensity RSSI in long format #####

# melt original_train to change [ , 1:520] attributes to variables (WAPs as ID)
varnames <- colnames(original_train[,521:529])
longtrain <- melt(original_train, id.vars = varnames)
names(longtrain)[10]<- paste("WAPid")
names(longtrain)[11]<- paste("WAPrecord")
longtrain[,11][longtrain[, 11] == 100] <- -105 



# melt original_test 
varnames.test <- colnames(original_test[,521:529])
longtest <- melt(original_test, id.vars = varnames.test)
names(longtest)[10]<- paste("WAPid")
names(longtest)[11]<- paste("WAPrecord")
longtest[,11][longtest[, 11] == 100] <- -105 





##### SET NAs Remove -105 dbm to reduce sample size ####
longtrain <- filter(longtrain, longtrain$WAPrecord != -105)
longtest <- filter(longtest, longtest$WAPrecord != -105)


#### DUPLICATES ####
longtrain <- unique(longtrain)


#### DATA TYPES ####

# TRAIN
longtrain$FLOOR <- as.factor(longtrain$FLOOR)
longtrain$BUILDINGID <- as.factor(longtrain$BUILDINGID)
levels(longtrain$BUILDINGID) <- c("TI",
                                  "TD",
                                  "TC")
longtrain$SPACEID <- as.factor(longtrain$SPACEID)
longtrain$RELATIVEPOSITION <- as.factor(longtrain$RELATIVEPOSITION)
levels(longtrain$RELATIVEPOSITION) <- c("Inside",
                                     "Outside")
longtrain$USERID <- as.factor(longtrain$USERID)
longtrain$PHONEID <- as.factor(longtrain$PHONEID)
longtrain$WAPid <- as.factor(longtrain$WAPid)
longtrain$TIMESTAMP <- as.POSIXct(longtrain$TIMESTAMP, origin="1970-01-01")



widetrain$FLOOR <- as.factor(widetrain$FLOOR)
widetrain$BUILDINGID <- as.factor(widetrain$BUILDINGID)
levels(widetrain$BUILDINGID) <- c("TI",
                                  "TD",
                                  "TC")
widetrain$SPACEID <- as.factor(widetrain$SPACEID)
widetrain$RELATIVEPOSITION <- as.factor(widetrain$RELATIVEPOSITION)
levels(widetrain$RELATIVEPOSITION) <- c("Inside",
                                     "Outside")
widetrain$USERID <- as.factor(widetrain$USERID)
widetrain$PHONEID <- as.factor(widetrain$PHONEID)
widetrain$TIMESTAMP <- as.POSIXct(widetrain$TIMESTAMP, origin="1970-01-01")



# TEST

longtest$FLOOR <- as.factor(longtest$FLOOR)
longtest$BUILDINGID <- as.factor(longtest$BUILDINGID)
levels(longtest$BUILDINGID) <- c("TI",
                                  "TD",
                                  "TC")
longtest$SPACEID <- as.factor(longtest$SPACEID)
longtest$RELATIVEPOSITION <- as.factor(longtest$RELATIVEPOSITION)
levels(longtest$RELATIVEPOSITION) <- c("Inside",
                                     "Outside")
longtest$USERID <- as.factor(longtest$USERID)
longtest$PHONEID <- as.factor(longtest$PHONEID)
longtest$WAPid <- as.factor(longtest$WAPid)
longtest$TIMESTAMP <- as.POSIXct(longtest$TIMESTAMP, origin="1970-01-01")




widetest$FLOOR <- as.factor(widetest$FLOOR)
widetest$BUILDINGID <- as.factor(widetest$BUILDINGID)
levels(widetest$BUILDINGID) <- c("TI",
                                  "TD",
                                  "TC")
widetest$SPACEID <- as.factor(widetest$SPACEID)
widetest$RELATIVEPOSITION <- as.factor(widetest$RELATIVEPOSITION)
levels(widetest$RELATIVEPOSITION) <- c("Inside",
                                     "Outside")
widetest$USERID <- as.factor(widetest$USERID)
widetest$PHONEID <- as.factor(widetest$PHONEID)
widetrain$TIMESTAMP <- as.POSIXct(widetrain$TIMESTAMP, origin="1970-01-01")




#### FEATURE SELECTION & ENGINEERING ####
## Create new attribute BUILDING-FLOOR

longtrain$BuildingFloor <- paste(longtrain$BUILDINGID, longtrain$FLOOR, sep = "-")
longtest$BuildingFloor <- paste(longtest$BUILDINGID, longtest$FLOOR, sep = "-")

widetrain$BuildingFloor <- paste(widetrain$BUILDINGID, widetrain$FLOOR, sep = "-")
widetest$BuildingFloor <- paste(widetest$BUILDINGID, widetest$FLOOR, sep = "-")



#### ANALYSE DISTRIBUTION ####

# Records distributed by Building and floor in TRAIN
ggplot(data = longtrain) +
  aes(x = WAPrecord, fill = FLOOR) +
  geom_histogram(bins = 30) +
  theme_minimal() +
  facet_wrap(vars(BUILDINGID))

# In TEST 
ggplot(data = longtest) +
  aes(x = WAPrecord, fill = FLOOR) +
  geom_histogram(bins = 30) +
  theme_minimal() +
  facet_wrap(vars(BUILDINGID))



# Records distributed by location and floor in TRAIN
ggplot(data = longtrain) +
  aes(x = LONGITUDE, y = LATITUDE, color = FLOOR) +
  geom_point() +
  theme_minimal()


# In TEST
ggplot(data = longtest) +
  aes(x = LONGITUDE, y = LATITUDE, color = FLOOR) +
  geom_point() +
  theme_minimal()


#### ANALYSE TOP SIGNALS ####
tooGood_signals <- longtrain %>% filter(WAPrecord > -30)

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
USER6 <- longtrain %>% filter(USERID == 6)

# it gives bad results but it also records reasonable RSSI
ggplot(data = USER6) +
  aes(x = WAPrecord) +
  geom_histogram(bins = 30, fill = "#0c4c8a") +
  theme_minimal()

## TC Floor 3-4 analysis 
TC3_exploration <- longtrain %>% filter(BuildingFloor == "TC-3")

# User 6 captured a big proportion of data in TC3 and TC4
ggplot(data = TC3_exploration) +
  aes(x = WAPrecord, fill = USERID) +
  geom_histogram(bins = 30) +
  theme_minimal()


TC4_exploration <- longtrain %>% filter(BuildingFloor == "TC-4")

ggplot(data = TC4_exploration) +
  aes(x = WAPrecord, fill = USERID) +
  geom_histogram(bins = 30) +
  theme_minimal()


# what would happen in terms of data amount if we removed user 6 records?

TC_exploration <- longtrain %>% filter(BUILDINGID == "TC")

ggplot(data = TC_exploration) +
  aes(x = FLOOR, weight = WAPrecord) +
  geom_bar(fill = "#0c4c8a") +
  theme_minimal()



# as user 6 represents a big part of the records of TC3&4, we only remove the mistaken data >-30
longtrain <- longtrain %>% filter(WAPrecord <= -30)


#### CHECK RSSI WEIGHT  ####

longtrain$signalQuality <- ifelse(longtrain$WAPrecord >=-66, "Top signal",
                                  ifelse(longtrain$WAPrecord >=-69, "Very Good signal",
                                         ifelse(longtrain$WAPrecord >=-79, "Good signal",
                                                ifelse(longtrain$WAPrecord >=-89, "Bad signal",
                                                       "Very bad signal"))))
                                  
# Signal intensity distribution
ggplot(data = longtrain) +
  aes(x = WAPrecord, fill = signalQuality) +
  geom_histogram(bins = 30) +
  theme_minimal() +
  facet_wrap(vars(BUILDINGID))



# Do Top signal WAPs cover all buildings? not completely!
TOPsignals_df <- filter(longtrain, longtrain$signalQuality =="Top signal")

  # all waps
ggplot(data = longtrain) +
  aes(x = LONGITUDE, y = LATITUDE) +
  geom_point(color = "#0c4c8a") +
  theme_minimal()

  # only TOP signal waps 
ggplot(data = TOPsignals_df) +
  aes(x = LONGITUDE, y = LATITUDE) +
  geom_point(color = "#0c4c8a") +
  theme_minimal()


# What about top & VG signals
VGandTOPsignals_df <- filter(longtrain, longtrain$signalQuality =="Top signal" | longtrain$signalQuality =="Veri Good signal") 





# filter by building and floor

building_floor_df <- c()
WAPid_count <- c()

building_floor <- unique(longtrain$BuildingFloor)

for (bf in building_floor) {
  building_floor_df[[bf]] <- filter(longtrain, BuildingFloor == bf)
  
  WAPid_count[[bf]] <- distinct(building_floor_df[[bf]], WAPid, .keep_all= TRUE) # Find the WAPs in each floor and building
  
  detection <- (WAPid_count[[bf]]$WAPid %in% WAPid_count[[bf]]$WAPid ==TRUE) # 146 WAPs  appear in two df at the same time
}





# ####TO DO!#####
# decisonTree <- ctree(brand ~ salary + age, data = cr)
# plot(ct, tp_args = list(text = TRUE))
