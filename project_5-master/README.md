# Predicting Gun Deaths in the United States for 2019
**By: Brianna Rivera, Eddie Olmos, James Sobrino, Monday Oshoikpor**

## Table of Contents

[Executive Summary](#executive-summary) <br>
[Problem Statement](#problem-statement) <br>
[Background Information](#background-information) <br>
[Software Requirements](#software-requirements) <br>
    [Project Workflow](#project-workflow) <br>
[The Data Science Process](#the-data-science-process) <br>
    [ATF Data](#atf) <br>
    [CDC Data](#cdc) <br>
    [USCB Data](#uscb) <br>
[Data Dictionary](#data-dictionary) <br>
[Data Cleaning, EDA, and Preprocessing](#data-cleaning-eda-and-preprocessing) <br>
[Modeling](#modeling) <br>
[Evaluation and Conceptual Understanding](#evaluation-and-conceptual-understanding) <br>
    [Linear Regression](#linear-regression) <br>
    [Decision Tree Regressor](#decision-tree-regressor) <br>
    [Random Forest Regressor](#random-forest-regressor) <br>
    [Voting Regressor](#voting-regressor) <br>

### Executive Summary

Our best model is the voting classifier model, with an RMSE of 93.  The voting classifier uses the predictions of our linear regression, decision tree, and random forest model to predict new estimates.  By using the data compiled from the ATF, CDC, and USCB, we were successfully able to predict the number of gun deaths per state in 2019 (give or take 93 deaths on average) from the data collected from 2014 through 2018.  We predicted about 38,000 gun deaths in the United States for the year of 2019.

Our recommendations to reduce gun violence:

> Reduce easy access to guns. <br>
> Eliminate funding restrictions on this type of research. <br>
> End Legal Immunity for gun manufacturers. <br>
> Invest in communities to promote wellbeing and mental health support. <br>

Next Steps:

> Compare our predictions to the actual number of deaths to see which states are doing better than our predictions and which states are doing worse than our predictions.  Then take a deeper dive into any gun reform measures those states are enforcing to see how effective they are according to the data.

### Problem Statement

There has been series of gun violence across the 50 states in the USA. Records have shown that gun deaths score among the highest cause of deaths in the USA. As part of solution to the menace, The Department of Homeland Security decided to conduct a research to predict gun deaths in the United States for the year 2019 using data from 2014 through 2018.


### Background Information

The United States (US) has the 28th highest rate of deaths from gun violence in the world.  Active shooter incidents have become more common in the US. - Every day, more than **100** Americans are killed with guns and more than **200** are shot and wounded.  Firearms are the leading cause of death for American children and teens.  American taxpayers pay a daily average of $34.8 million for medical care, first responders, ambulances, police, and criminal justice services related to gun violence.  There are three times as many drivers as gun owners, yet yearly gun deaths (40k) surpass deaths from car accidents.  Internationally, the US stands alone as the leader in gun-related killings as a percentage of all homicides:
> US - 73% <br>
> Canada - 39% <br>
> Australia - 22% <br>
> England & Wales - 4% <br>

### Software Requirements
> Jupyter Notebooks/Labs <br>
> Python packages: <br>
>> Sklearn <br>
>>> train_test_split <br>
>>> cross_val_score <br>
>>> mean_squared_error <br>
>>> StandardScaler <br>
>>> DecisionTreeRegressor <br>
>>> RandomForestRegressor <br>
>>> VotingRegressor <br>
>>> GridSearchCV <br>
>>> LinearSVR <br>

#### Project Workflow
> pip install any required packages above not already installed on your local machine. <br>
> Run the "gun_death_analysis_2019.ipynb" in the project repo to conduct the analysis yourself. <br>
> The required CSV files will be contained in the "datasets" subfolder. <br>

### The Data Science Process

**Data Acquisition, Ingestion, and Cleaning**
For this project, we needed to compile data from three main data sources.  The Bureau of Alcohol, Tobacco, and Firearms and Explosives (ATF) for the number and types of registered weapons in each state by year.  The Centers for Disease Control and Prevention (CDC) for the number of firearm deaths.  And finally The United States Census Bureau (USCB) for population estimates for each state by year.

#### ATF

Each year the ATF releases a report called the Firearms Commerce Report in the United States.  This report presents data drawn from a number of ATF reports and records in one comprehensive document. It also provides comparative data from as far back as 1975 for context, analyses of trends over the years, and a fuller picture of the state of firearms commerce in the United States today.  In these reports, the ATF shares a table titled Exhibit 8. National Firearms Act Registered Weapons by State which denotes the following data points broken down by state:
> Any Other Weapon<sup>1</sup> <br>
> Destructive Devices<sup>2</sup> <br>
> Machinegun<sup>3</sup> <br>
> Silencer<sup>4</sup> <br>
> Short Barreled Rifle<sup>5</sup> <br>
> Short Barreled Shotgun<sup>6</sup> <br> <br>

<sup>1</sup> The term “any other weapon” means any weapon or device capable of being concealed on the person from which a shot can be discharged through the energy of an explosive, a pistol or revolver having a barrel with a smooth bore designed or redesigned to fire a fixed shotgun shell, weapons with combination shotgun and rifle barrels 12 inches or more, less than 18 inches in length, from which only a single discharge can be made from either barrel without manual reloading, and shall include any such weapon which may be readily restored to fire. Such term shall not include a pistol or a revolver having a rifled bore, or rifled bores, or weapons designed, made, or intended to be fired from the shoulder and not capable of firing fixed ammunition.

<sup>2</sup> Destructive device generally is defined as (a) Any explosive, incendiary, or poison gas (1) bomb, (2) grenade, (3) rocket having a propellant charge of more than 4 ounces, (4) missile having an explosive or incendiary charge of more than one-quarter ounce, (5) mine, or (6) device similar to any of the devices described in the preceding paragraphs of this definition; (b) any type of weapon (other than a shotgun or a shotgun shell which the Director finds is generally recognized as particularly suitable for sporting purposes) by whatever name known which will, or which may be readily converted to, expel a projectile by the action of an explosive or other propellant, and which has any barrel with a bore of more than one-half inch in diameter; and (c) any combination of parts either designed or intended for use in converting any device into any destructive device described in paragraph (a) or (b) of this section and from which a destructive device may be readily assembled. The term shall not include any device which is neither designed nor redesigned for use as a weapon; any device, although originally designed for use as a weapon, which is redesigned for use as a signaling, pyrotechnic, line throwing, safety, or similar device; surplus ordnance sold, loaned, or given by the Secretary of the Army pursuant to the provisions of section 4684(2), 4685, or 4686 of title 10, United States Code; or any other device which the Director finds is not likely to be used as a weapon, is an antique, or is a rifle which the owner intends to use solely for sporting, recreational, or cultural purposes.

<sup>3</sup> Machinegun is defined as any weapon which shoots, is designed to shoot, or can be readily restored to shoot, automatically more than one shot, without manual reloading, by a single function of the trigger. The term shall also include the frame or receiver of any such weapon, any part designed and intended solely and exclusively, or combination of parts designed and intended, for use in converting a weapon into a machinegun, and any combination of parts from which a machinegun can be assembled if such parts are in the possession or under the control of a person.

<sup>4</sup> Silencer is defined as any device for silencing, muffling, or diminishing the report of a portable firearm, including any combination of parts, designed or redesigned, and intended for the use in assembling or fabricating a firearm silencer or firearm muffler, and any part intended only for use in such assembly or fabrication.

<sup>5</sup> Short-barreled rifle is defined as a rifle having one or more barrels less than 16 inches in length, and any weapon made from a rifle, whether by alteration, modification, or otherwise, if such weapon, as modified, has an overall length of less than 26 inches.

<sup>6</sup> Short-barreled shotgun is defined as a shotgun having one or more barrels less than 18 inches in length, and any weapon made from a shotgun, whether by alteration, modification, or otherwise, if such weapon as modified has an overall length of less than 26 inches.

We cleaned and properly compiled each report from 2014 through 2019 into a CSV file called "reg_weapons.csv".

#### CDC

Each year the CDC gathers data on gun deaths broken down by state.  It's important to note the Center for Disease Control and Prevention tracks this information, as it doesn't fall under the classical definition of a disease and institutions such as the National Rifle Association have lobbied to keep them from tracking this data.  Regardless, these reports have been collected from 2014 through 2019 and they track:
> Death Rate<sup>1</sup> <br>
> Deaths

<sup>1</sup> The number of deaths per 100,000 total population.

We downloaded and then cleaned both the death rate and number of deaths by state from 2014 through 2019 into a CSV file called "targets.csv" since the number of deaths will the the target variable for our machine learning models.

#### USCB

The U.S. Census Bureau is the leading source of statistical information about the nation’s people. Our population statistics come from decennial censuses, which count the entire U.S. population every ten years, along with several other surveys.  They also provide estimates for off census years which is what we used for our annual population data by state.

We downloaded and cleaned the population estimates from 2014 through 2019 into a CSV file called "population.csv".

### Data Dictionary

| year | state | gun_death_rate | gun_deaths | any_other_weapon | destructive_device | machinegun | silencer | short_barreled_rifle | short_barreled_shotgun | total_weapons | population | state_id | region | elect_res_2020 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| int | object | float | int | int | int | int | int | int | int | int | int | int | object | object |
| Year of observation | State observed | Gun death rate of state | Gun deaths of state  | Number of any other weapons | Number of Destructive Devices | Number of machineguns | Number of silencers | Number of short barreled rifles | Number of short barreled shotguns | Total of all weapon columns  | Population estimate of state | Numerically labeled state codes | US region which state resides | Political leaning for 2020 election (Red: Republican, Blue: Democratic) |

### Data Cleaning, EDA, and Preprocessing

There was no way to download the ATF reports directly, so we were forced to copy and paste the tables and manicure the data to align to our schema.

The CDC data was clean, but wasn't in the right format so we had to transpose each year into its own rows and stack the years together to compile the master dataset.  They also had the states represented by abbreviation so it was necessary to convert them into full state names so it was able to be merged later with the other datasets.  
The same issue arose with the USCB population estimates, along with some string manipulation to ensure it was able to merge with the other datasets too. 

Once all CSV files were created and put into the datasets subfolder, we were able to import all CSVs into a iPython Notebook and merge them into one master dataset which we then were able to model. 

### Modeling

We manually split our data into train and test groups by filtering the data.  For the train set, we used all years less than 2019, and for the test set we used all rows that were equal to 2019.  Thus, we used the 2014 through 2018 data to predict gun deaths by state for 2019.  We also noticed our target column was heavily right skewed, so we conducted a log transformation of the gun deaths to help it trend closer towards normal distribution.

We were able to run the following regression models:
> Linear Regression <br>
> Decision Tree Regressor <br>
> Random Forest Regressor <br>
> Voting Regressor <br>

### Evaluation and Conceptual Understanding

#### Linear Regression

| Training Score | Testing Score | Test RMSE |
| --- | --- | --- |
| 0.995 | 0.990 | 108 |

#### Decision Tree Regressor

| Training Score | Testing Score | Test RMSE |
| --- | --- | --- |
| 0.99 | 0.97 | 135 |

Best Parameters:
> ccp_alpha = 0 <br>
> max_depth = 9 <br>
> min_samples_leaf = 4 <br>
> min_samples_split = 8 <br>

#### Random Forest Regressor

| Training Score | Testing Score | Test RMSE |
| --- | --- | --- |
| 0.98 | 0.99 | 107 |

Best Parameters:
> ccp_alpha = 0 <br>
> max_depth = 9 <br>
> min_samples_leaf = 4 <br>
> min_samples_split = 9 <br>
> n_estimators = 75 <br>

#### Voting Regressor

| Training Score | Testing Score | Test RMSE |
| --- | --- | --- |
| 0.99 | 0.98 | 93 |

Estimators:
> Linear Regression <br>
> Decision Tree Regressor <br>
> Random Forest Regressor <br>