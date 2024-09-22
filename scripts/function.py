import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Function to load the data
def load_data(train_path, test_path):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    return train, test

# Function to inspect the data for missing values and structure
def inspect_data(df):
    print(df.info())
    print(df.isnull().sum())
    print(df.describe())
# Function to check the distribution of promotions in the training and test sets
def check_promo_distribution(train, test):
    plt.figure(figsize=(10,6))
    sns.countplot(x='Promo', data=train)
    plt.title('Distribution of Promo in Train Set')
    plt.show()

    plt.figure(figsize=(10,6))
    sns.countplot(x='Promo', data=test)
    plt.title('Distribution of Promo in Test Set')
    plt.show()

    promo_train_dist = train['Promo'].value_counts(normalize=True)
    promo_test_dist = test['Promo'].value_counts(normalize=True)
    print("Promo distribution in Train Set:\n", promo_train_dist)
    print("Promo distribution in Test Set:\n", promo_test_dist)
# Function to visualize sales before, during, and after holidays
def check_holiday_sales(train):
    train['StateHoliday'] = train['StateHoliday'].replace(0, '0')  # Handle numeric '0' as string
    plt.figure(figsize=(10,6))
    sns.boxplot(x='StateHoliday', y='Sales', data=train)
    plt.title('Sales Before, During, and After Holidays')
    plt.show()

    # Analyzing average sales before, during, and after holidays
    sales_by_holiday = train.groupby('StateHoliday')['Sales'].mean()
    print("Average Sales based on Holidays:\n", sales_by_holiday)
# Function to analyze seasonal purchase behavior
def check_seasonal_behavior(train):
    train['Month'] = pd.to_datetime(train['Date']).dt.month
    plt.figure(figsize=(12,6))
    sns.boxplot(x='Month', y='Sales', data=train)
    plt.title('Sales Distribution by Month (Seasonality)')
    plt.xticks(np.arange(1,13), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    plt.show()

    christmas_sales = train[train['Month'] == 12]['Sales'].mean()
    easter_sales = train[(train['Month'] == 3) | (train['Month'] == 4)]['Sales'].mean()
    print("Average sales in December (Christmas):", christmas_sales)
    print("Average sales during Easter (March/April):", easter_sales)
# Function to analyze the correlation between sales and number of customers
def check_correlation_sales_customers(train):
    correlation = train[['Sales', 'Customers']].corr()
    print("Correlation between Sales and Customers:\n", correlation)

    plt.figure(figsize=(10,6))
    sns.scatterplot(x='Customers', y='Sales', data=train)
    plt.title('Correlation Between Number of Customers and Sales')
    plt.show()
# Function to analyze how promotions affect sales and customer numbers
def check_promo_impact(train):
    plt.figure(figsize=(10,6))
    sns.boxplot(x='Promo', y='Sales', data=train)
    plt.title('Effect of Promotions on Sales')
    plt.show()

    plt.figure(figsize=(10,6))
    sns.boxplot(x='Promo', y='Customers', data=train)
    plt.title('Effect of Promotions on Number of Customers')
    plt.show()
# Function to identify stores where promos are most effective
def check_promo_deployment(train):
    promo_sales_by_store = train.groupby(['Store', 'Promo'])['Sales'].mean().unstack()
    promo_sales_by_store['Promo Impact'] = promo_sales_by_store[1] - promo_sales_by_store[0]
    promo_sales_by_store = promo_sales_by_store.sort_values('Promo Impact', ascending=False)

    print("Stores where promos have the highest impact:\n", promo_sales_by_store.head(10))
# Function to analyze customer behavior during store opening and closing times
def check_store_open_sales(train):
    plt.figure(figsize=(10,6))
    sns.boxplot(x='Open', y='Sales', data=train)
    plt.title('Sales When Store is Open vs Closed')
    plt.show()

    weekday_open = train.groupby(['Store', 'DayOfWeek'])['Open'].mean().unstack()
    always_open_stores = weekday_open[(weekday_open == 1).all(axis=1)].index.tolist()
    print("Stores open on all weekdays:", always_open_stores)

    weekend_sales = train[train['DayOfWeek'].isin([6, 7])].groupby('Store')['Sales'].mean()
    always_open_sales = weekend_sales[weekend_sales.index.isin(always_open_stores)]
    other_stores_sales = weekend_sales[~weekend_sales.index.isin(always_open_stores)]

    print("Average weekend sales for stores always open:", always_open_sales.mean())
    print("Average weekend sales for other stores:", other_stores_sales.mean())
# Function to analyze the impact of assortment type on sales
def check_assortment_impact(train):
    plt.figure(figsize=(10,6))
    sns.boxplot(x='Assortment', y='Sales', data=train)
    plt.title('Effect of Assortment Type on Sales')
    plt.show()

    avg_sales_by_assortment = train.groupby('Assortment')['Sales'].mean()
    print("Average sales by assortment type:\n", avg_sales_by_assortment)
# Function to analyze the impact of competition distance on sales
def check_competition_distance(train):
    plt.figure(figsize=(10,6))
    sns.scatterplot(x='CompetitionDistance', y='Sales', data=train)
    plt.title('Effect of Competition Distance on Sales')
    plt.show()

    no_competition_sales = train[train['CompetitionDistance'].isna()]['Sales'].mean()
    competition_sales = train[train['CompetitionDistance'].notna()]['Sales'].mean()
    print("Average sales for stores without competition data (NA):", no_competition_sales)
    print("Average sales for stores with competition data:", competition_sales)
# Function to analyze the impact of opening or reopening of competitors
def check_competitor_reopening(train):
    competition_na = train[train['CompetitionDistance'].isna()]['Store'].unique()
    competition_reopen_sales = train[train['Store'].isin(competition_na)].groupby('Store')['Sales'].mean()

    print("Sales impact for stores with later competition reopening:", competition_reopen_sales.head())
