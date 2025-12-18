# Project Overview / blueprint

## About the Raw Data
The raw data is sourced from `skuld/data/data_long.csv`. This data is in a long format with the following columns:
- **timestamp**: Unix timestamp (integer) representing the time of observation.
- **ticker**: A string (e.g., `FCT.NZ`) identifying where the data belongs. If empty, it represents macro data (e.g., GDP).
- **feature**: The name of the data point (e.g., `Close`).
- **value**: The observed value.

### Example Data
Sample rows from the dataset:
```
1.60946E+12	New Registrations - Goods road motor vehicles	3986
1.75672E+12	AUS.NZ	Close	3.9149999618530273
```

### Data Granularity
- **Price information**: Observations are daily.
- **Macro observations**: Typically quarterly or annual.

Given this granularity, the model is best suited for medium- to long-term predictions.


---

## Project Directory Structure
files subject to change, however the directory structure should remain constant. 
```
skuld/
├── README.md
├── data/
│   ├── data_long.csv
│   └── legacy/
│       ├── evaluation_metrics.csv
│       └── trade_simulation.csv
├── java/
│   ├── norn.iml
│   ├── pom.xml
│   ├── src/
│   │   ├── main/
│   │   │   ├── java/
│   │   │   │   └── lazic/
│   │   │   │       ├── Main.java
│   │   │   │       └── sources/
│   │   │   │           ├── config/
│   │   │   │           │   └── Tickers.java
│   │   │   │           ├── examples/
│   │   │   │           │   └── SourceTemplate.java
│   │   │   │           ├── NzBusinessConfidence.java
│   │   │   │           ├── NzGdp.java
│   │   │   │           ├── NzLaborStats.java
│   │   │   │           ├── NzRatesFx.java
│   │   │   │           ├── NzVehicleRegistrations.java
│   │   │   │           ├── YfFinances.java
│   │   │   │           └── YfPrices.java
│   │   │   │       └── utils/
│   │   │   │           ├── db/
│   │   │   │           └── ingest/
│   │   │   │               ├── CsvLongParser.java
│   │   │   │               ├── DataPoint.java
│   │   │   │               ├── DataSourceBase.java
│   │   │   │               ├── IngestManager.java
│   │   │   │               └── WebHtmlGetter.java
│   │   │   └── resources/
│   │   └── test/
│   └── target/
│       ├── classes/
│       ├── data/
│       ├── dependency/
│       ├── generated-sources/
│       ├── generated-test-sources/
│       ├── maven-archiver/
│       ├── maven-status/
│       ├── test-classes/
│       └── tools/
├── python/
│   └── ml-pipeline/
│       ├── requirements.txt
│       ├── config/
│       │   ├── column_names.py
│       │   └── file_paths.py
│       ├── data-preparation/
│       │   ├── data-splitting/
│       │   │   ├── chunking/
│       │   │   └── train-test/
│       │   ├── labeling/
│       │   ├── long-to-wide/
│       │   └── transformations/
│       ├── documentation/
│       │   └── README.md
│       ├── evaluation/
│       │   ├── model-evaluation/
│       │   ├── trade-simulation/
│       │   └── trade-simulation-evaluation/
│       ├── learner/
│       ├── runnables/
│       ├── tests/
│       │   ├── data-preparation/
│       │   │   ├── data-splitting/
│       │   │   ├── labeling/
│       │   │   ├── long-to-wide/
│       │   │   └── transformations/
│       │   ├── evaluation/
│       │   ├── learner/
│       │   ├── utils/
│       │   └── validation/
│       ├── utils/
│       │   └── file_utils.py
│       └── validation/
```

## Project Philosophy
The project adheres to the following principles:
- **Modularity**: Each step is treated as "data in, data out" without state.
- **Leakage Prevention**: Ensuring no data leakage.
- **Efficiency**: Code is optimized for speed and parallelization where appropriate.
- **Simplicity**: Complex code is abstracted away, maintaining a clean project structure.
- **Centralized config** Using py files for config is fine, all constants should be defined in config files, making sure to adhere to separation of concerns. 
- **Documentation**: For each module, a readme should exist in the root of the modules directory explaining the module, eg. ml-pipeline/evaluation/README.md
- **Test driven development**: Before the writing or modification of a module, write tests to define its behaviour, and then create the module, ensuring the tests pass. 

---

## Project Goal
The ultimate goal is to deploy the project with a web GUI, supporting the loading of multiple data files. Initially, the project will use **XGBoost** or **dask-xgboost**, with plans to explore deep learning in the future. The project aims to assist in making informed investment decisions for the **New Zealand Exchange (NZX)**. The project will start off small scale, and will be ran from the command line manually by myself, however room to grow should be left, by keeping things maintainable and simple. 

---

## Quantifying Ticker Performance
The model will classify whether a ticker's price will chain at least **X%** within **Y days**. Prediction probabilities will rank investment decisions. For example:
- The model is 80% confident that `ANZ` will gain at least 10% in 365 days.
To clarify this is simple price difference between buy and sell time. If a ticker gains 100% in a day, but then loses 100% the next day, this should not matter as long as the price after the lookahead time exceeds x% or not. we are just measuring % change up and down here. 
---

## Dependency Management
The project uses **UV**. To run scripts, use:
```
uv run script.py
```

---

## Data Pipeline
The data pipeline consists of the following steps:
1. **Convert Data**: Transform data from long to wide format.
2. **Train/Test Split**: Split the data into training and testing sets.
3. **Feature Engineering**: Apply feature engineering techniques.
4. **Scaling**: Scale the feature data per ticker. Macro data should be scaled globally. 
   - **Ticker Identification**: Any feature column that does NOT start with the prefix `MACRO_` 
     should be treated as ticker-specific data and scaled per ticker.
   - **Macro Identification**: Any feature column that starts with `MACRO_` should be treated 
     as macro data and scaled globally across all tickers.
   - **Scaler Type**: Use `StandardScaler` from scikit-learn for both ticker and macro features.
   - **Fit Strategy**: Fit all scalers exclusively on training data to prevent leakage. 
     For rolling windows, fit a new scaler for each window's training set.
   - **Scaler Persistence**: Save fitted scalers with naming convention: 
     `{ticker}_window{N}_scaler.pkl` for ticker scalers and `macro_window{N}_scaler.pkl` 
     for macro scalers.
5. **Model Training**: Train the model.
6. **Predictions**: Use the trained model to make predictions on the test data.
7. **Evaluation**: Evaluate the predictions.

### Additional Considerations
- **Modularity**: Each step in the pipeline is implemented as an independent module with clear input/output interfaces.
- **Scalability**: The pipeline supports distributed data processing using tools like `dask` for handling large datasets.
- **Data Validation**: A data validation step ensures the integrity of input data, checking for missing values, incorrect data types, and column mismatches.

---

## Runnables
The project includes the following runnable components:
- **Rolling Window Runner**: A configurable runner that splits data based on time, producing a collection of prediction files.
- **Predict**: The end product, making predictions about the future.
- **Evaluation**: Loads predictions from the rolling window runner and performs:
  - **Backtests**: Based on raw data.
  - **Analytics**: Includes model accuracy (classification metrics) and trading simulations (e.g., Sharpe ratio).
- **Tests**: Comprehensive tests for each module.

---

## Testing Strategy
The project uses **pytest** for testing. The testing strategy includes:
- **Unit Tests**: For individual modules.
- **Integration Tests**: For the entire pipeline.
- **Performance Tests**: To ensure scalability and efficiency.

---

## Test-Driven Development
Test-driven development (TDD) should be employed to ensure code correctness and maintainability.

## Directory structure
Directory structure is deep, to avoid overly wide and overwhelming configurations. Test directory structure mirrors that of its parent directory. 

## Model config
365 day lookahead (time in the future to predict for)
2%+ gain for class 1 
5 rolling window iterations
test rolling window iteration size = 1 day 
train rolling window iteration size = as big as possible
rolling window movement (how far to move the window back in time) = 1.3333 years
default xgboost config

# Data at different points
The end goal is to end up with a proovably accurate model, where the predictions are listed in a format similar to the following:
timestamp, ticker, prediction probability
160000, ANZ.NZ, 0.88

However to do this we need to prove the model is accurate by testing it on a variety of different time periods using a rolling window. 

And to ensure the model actual results in monetary gain, simulate using the predictions to buy and sell different tickers. 

# Metrics
The reported metrics should at very least be: those covered in sk learns classification summary
and for the trading sim should be, median return, lqr return, uqr return, stdev, sharpe ratio.

# Processes
Long to wide: this converts the data from a long to a wide format, taking the timestamp, ticker, feature, value columns and translating them to: timestamp, ticker, feature1, feature2, feature3... feautreN. To do this, use the 'close' observations timestamp as a target, for example note the following data in a long format. 
13 ANZ, revenue, 1000
12, ANZ, close, 15 
11, , Vehicle registrations, 900

The wide version of this data would look like:
timestamp, ticker, close, Vehicle registrations, revenue
12, ANZ, 15, 900, null (leave empty, including this creates leakage)
**Edge Cases**:
- If a ticker's first Close timestamp is before any macro data: macro columns will be null
- If a ticker has OHLV but no Close: skip that timestamp (Close is required anchor)
- Multiple values at same timestamp for same ticker-feature: take the last value (shouldn't happen in clean data)

For even more examples here is more sample data pulled straight from data_long.csv
timestamp	ticker	feature	value
1.53493E+12	ASP.NZ	Low	1.493999958
1.76216E+12	ARG.NZ	Low	1.2799999713897705
1.47678E+12	ASF.NZ	Close	7.259359836578369
1.68907E+12	SKL.NZ	High	4.670000076293945
1.15754E+12	STU.NZ	Volume	22704
7.04678E+11		OECD_CCICP	99.57451
1.66185E+12	KFL.NZ	High	1.6200000047683716
1.20246E+12	SPY.NZ	Open	1.159999966621399
1.54884E+12	SPY.NZ	Open	0.20499999821186066
1.70497E+12	FBU.NZ	Open	4.849999904632568
1.63775E+12	RTO.NZ	High	0.3681289851665497
1.19588E+12	HG=F	Open	2.9600000381469727
1.59671E+12	ERD.NZ	Low	3.267328977584839

Because scaling is done separately for ticker and macro data, the MACRO_ prefix should be added as a process just before converting to a wide format, this way avoiding coupling that would occur if it were done during the long to wide process.

# Dependencies
UV is a dependency manager developed by astral. Use uv sync after including new dependencies to install them, making sure to add the new dependencies to the relevant dependency lists. use uv run to run any code, be it tests or runnables. eg. uv run pytest.

# Feature engineering
Starting out we can keep this simple, add additional cyclical time columns, eg time of year. More can be added later. 

# Missing data handling
Use a hybrid aproach of indicator columns, and imputation, though be careful with not using data forward in time of the current observation.

# Target variable (labels)
### Target Variable (Label) Construction

**Definition**: Binary classification of whether a ticker gains ≥X% within Y days.

**Formula**:
```python
# For observation at timestamp t with close price P_t
# Look ahead Y days to timestamp t+Y with close price P_t+Y

price_change_pct = ((P_t+Y - P_t) / P_t) * 100

target = 1 if price_change_pct >= X else 0
```

**Configuration** (from Model Config):
- X = 2% (threshold gain)
- Y = 365 days (lookahead period)

**Implementation Details**:
1. For each ticker-timestamp observation, calculate the close price Y days in the future
2. If future timestamp doesn't exist (e.g., end of dataset), drop that observation from training
3. Only use the **exact** price at t+Y days, not the maximum within the window
4. Handle splits/dividends: use adjusted close prices if available

**Example**:
```
timestamp: 2020-01-01, Close: $100
timestamp: 2020-12-31 (365 days later), Close: $103
price_change_pct = (103 - 100) / 100 * 100 = 3%
target = 1 (since 3% >= 2%)
```

**Edge Cases**:
- If ticker has gap in data > Y days: cannot calculate target, drop observation
- If ticker delisted before t+Y: drop observation (cannot know future price)
- For production predictions: target is unknown (predicting the future)

### Trading Simulation

**Objective**: Simulate using model predictions to make buy/sell decisions and evaluate 
monetary performance.

**Strategy**:
- **Buy signal**: When model probability ≥ threshold (start with 0.7)
- **Position size**: Equal-weighted across all buy signals on that day
- **Sell signal**: Sell after Y days (365) regardless of performance (time-based exit)
- **Capital**: Start with $100,000 virtual capital
- **Transaction costs**: Assume 0.1% per trade (buy and sell)

**Simulation Process**:
1. For each day in test set:
   - Identify all tickers with probability ≥ threshold
   - Calculate position size: available_capital / number_of_buy_signals
   - Execute buys at Close price + 0.1% transaction cost
2. Track open positions: each has buy_date, buy_price, ticker
3. For each open position where current_date = buy_date + Y days:
   - Execute sell at Close price - 0.1% transaction cost
   - Calculate return: (sell_price - buy_price) / buy_price * 100
   - Add proceeds back to available_capital

**Output Metrics** (for each rolling window):
- Total return: (final_capital - initial_capital) / initial_capital * 100
- Median return per trade
- Lower quartile return (LQR)
- Upper quartile return (UQR)
- Standard deviation of returns
- Sharpe ratio: (mean_return - risk_free_rate) / std_return
  - Assume risk_free_rate = 0 for simplicity

**Aggregation Across Windows**:
Report mean and std of each metric across the 5 rolling windows.

# Code style 
Avoid code duplication, use standard python convetions, avoid lengthy doc strings. Follow the conventions of the rest of the codebase. Think thoughtfully about architecture before implementing, could this be done a better way?

# Data process path/journey
Load data long csv, 
prefix macro data, identified by having an empty ticker value, 
convert data from long to wide,
perform test train split. The from to time of which depends on what operation is being performed.,
Perform imputation and add identifier columns that state if data has been imputed or is missing or is present for a target column,
add feature engineered features such as cyclical time based features (it is implied that train and test are done separetly for all post split steps), 
identify and scale macro data,
identify and scale ticker data,
Train a model,
Make predictions using the test data,
Run evaluation steps including trading simulation and model clasificiation summary.

This process with vary slightly depending on what runner is run, for example there is no point performing evaluation on predictions abot the future, since the future is not known. For the sliding window runner, it would wrap all of this logic in many itterations, and instead evaluating on aggregate predictions rather than the individual predictions themselves. 

# Extra notes
Regarding the trade simulation, provide a baseline, results buying every single ticker, comparing results against the model. 

