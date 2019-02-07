## WIFI LOCATIONING ##

####SET ENVIRONMENT####

pacman:: p_load("dplyr", "tidyr", "ggplot2", "plotly", "data.table", "ggridges")


trainingData <- read_csv("C:/Users/usuario/Desktop/UBIQUM/Project 8 - Wifi locationing/trainingData.csv")
validationData <- read_csv("C:/Users/usuario/Desktop/UBIQUM/Project 8 - Wifi locationing/validationData.csv")



#### PRE-PROCESSING ####
#### "NA 100dbm" TREATMENT #####

# melt trainingData to change [ , 1:520] attributes to variables (WAPs as ID)
varnames <- colnames(trainingData[,521:529])
long.train <- melt(trainingData, id.vars = varnames)
names(long.train)[10]<- paste("WAPid")
names(long.train)[11]<- paste("WAPrecord")


# filter df to ONLY keep actual records (remove 100dbm records)
long.train <- filter(long.train, long.train$WAPrecord != 100)

  # longtable$TIMESTAMP <- NULL
write.csv(long.train, file= "longtrain.csv")



# convert ValidationData to longtable too
varnames.test <- colnames(validationData[,521:529])
long.test <- melt(validationData, id.vars = varnames.test)
names(long.test)[10]<- paste("WAPid")
names(long.test)[11]<- paste("WAPrecord")

long.test <- filter(long.test, long.test$WAPrecord != 100)
write.csv(long.test, file= "longtest.csv")


# compare test WAPs to train WAPs

  # new train ONLY with attributes in test 

detection <- (long.train$WAPid %in% long.test$WAPid)
long.train$detection <- detection
newdf.train <- long.train %>% filter(long.train$detection == TRUE) # newdf containing ONLY common WAPs in train & test


  # new test ONLY with attributes in train

detection1 <- (long.test$WAPid %in% long.train$WAPid)
long.test$detection1 <- detection1
newdf.test <- long.test %>% filter(long.test$detection1 == TRUE) # newdf containing ONLY common WAPs in train & test






# ggplot(data= longtable, aes(longtable$WAPid,longtable$WAPrecord)) + 
#   geom_density_ridges(stat = "density_ridges", scale = 1)


## OUTLIERS ##



## VAR IMP ##

