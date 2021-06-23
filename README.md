# SIR-mobility

The project uses the public transportation data, specifically NYC subway turnstile data, to derive the mobility of NYC residents, which was then fed to the extended SIR model we developed to simulate and predict the daily variation of COVID-19 cases in NYC.

### Data

In **data/covid_19_data** folder, *cases-by-day.csv* contains the daily confirmed cases, 7-day average confirmed cases, as well as the daily case count and 7-day average case count for each borough in NYC. The prefix *BX* stands for Bronx, *BK* for Brooklyn, *MN* for Manhattan, *QN* for Queen, *SI* for Staten Island. *deaths-by-day.csv* contains the daily death among confirmed cases. *now-data-by-day.csv* truncated the data table and only displays the data for the last 90 days. More datasets can be found in https://github.com/nychealth/coronavirus-data

#### Columns in the Dataset

| Column name                  | Description                                                |
| -----------                  | -----------                                                |
| date_of_interest             | Record date                                                |
| CASE_COUNT                   | Count of confirmed cases in the date_of_interest           |
| CASE_COUNT_7DAY_AVG	         | Average count of confirmed cases in a 7-day window         |
| BX_CASE_COUNT                | Count of confirmed cases for Bronx borough                 |
| BX_PROBABLE_CASE_COUNT       | Count of probable cases for Bronx borough                  |
| BX_CASE_COUNT_7DAY_AVG       | Average count of cases in a 7-day window for Brons borough |

### Mobility Data

In **data/mobility** folder, each file contains the number of subway riders from one borough to another, as well as leaving the city, for each date of interest. The files seperate the riders in the morning time and night time. Under **data/mobility/fraction** folder, each file contains the fractions of subway riders to the local population from borough to borough.

### Turnstile_data

In **data/turnstile_data/** folder, each file contains the detailed daily count of arraival and departures from borough to other boroughs, at each time stamp of the day. The original datasets can be accessed from: http://web.mta.info/developers/turnstile.html. Each file contains one-week turnstile data for all of the subway stations in NYC.

### example.py

### SIR_transport_functions.py
