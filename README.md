## Wifi-locationing (IoT analytics) - Project Description


**Project Goal:** investigate the feasibility of using "wifi fingerprinting" to determine a person's location in indoor spaces by evaluating multiple machine learning models. Therefore, signal intensities recorded from multiple wifi hotspots within the building are used.

**Data characteristics:** we have been provided by 3 data bases: **1- TRAIN** a large database of wifi fingerprints for a multi-building industrial campus with a location (building, floor, and location ID) associated with each fingerprint. **2- VALIDATION** containing the expected results already; allows to check the model performance. **3- TEST** no results provided; to re-validate the model's fit independently of the data.


## Technical Approach
**Language used:** R programming


**1. PRE-PROCESSING (DATA QUALITY)**
- Missing values treatment 
- Data classes conversion = hablar:: convert()
- Zero variance exploration & treatment
- Duplicates
- Descriptive analysis (long format df)
  * VISUALIZATION TOOL: GGPLOT
- Outliers treatment


**2. FEATURE SELECTION & ENGINEERING**
- Combining attributes (BuildingFloor)
- Creating Principal Components for PCA

**3. PRE-MODELLING**
- Smart sampling
- Cross-validation

**4. MODELLING**
- Random Forest = RandomForest()
  * applying PCA & not applying it

**5. ACCURACY AND CONFIDENCE ANALYSIS**
  * VISUALIZATION TOOL: GGPLOT

**6. PREDICTION**
  * VALIDATION SET
  * TEST SET
