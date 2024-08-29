import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns


def visualizing_time_series_data(data, column_name):
    # Use seaborn style defaults and set the default figure size
    sns.set(rc={'figure.figsize': (11, 4)})
    axes = data[column_name].plot(
        marker='.', alpha=0.5, linestyle='None', figsize=(11, 9), subplots=True)
    for ax in axes:
        ax.set_ylabel('Daily Totals (GWh)')
    plt.ylabel(column_name)
    # plt.show()


if __name__ == '__main__':

    dataset_path = './opsd_germany_daily.csv'
    # Read data from .csv file
    opsd_daily = pd.read_csv(dataset_path)
    print(f'The shape of data is {opsd_daily.shape}')
    print(f'Data type : {opsd_daily.dtypes}')
    print(opsd_daily.head(5))

    opsd_daily = opsd_daily.set_index('Date')
    print(opsd_daily.head(5))
    opsd_daily = pd.read_csv(dataset_path, index_col=0, parse_dates=True)
    opsd_daily['Year'] = opsd_daily.index.year
    opsd_daily['Month'] = opsd_daily.index.month
    opsd_daily['Weekday Name'] = opsd_daily.index.day_name()

    # Display a random sampling of 5 rows
    print(opsd_daily.sample(7, random_state=0))
    # Function for Time-based indexing:
    print(opsd_daily.loc['2014-01-20':'2014-01-22'])
    # Function for Partial string indexing:
    print(opsd_daily.loc['2012-02'])

    # Visualizing time series data
    columns = ['Consumption', 'Wind', 'Solar']
    visualizing_time_series_data(opsd_daily, columns)

    # Seasonal patterns
    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
    for name, ax in zip(['Consumption', 'Solar', 'Wind'], axes):
        sns.boxplot(data=opsd_daily, x='Month', y=name, ax=ax)
        ax.set_ylabel('GWh')
        ax.set_title(name)
        # Remove the automatic x-axis label from all but the bottom subplot
        if ax != axes[-1]:
            ax.set_xlabel('')
    # plt.show()

    # Frequency of observations
    print(pd.date_range('1998-03-10', '1998-03-15', freq='D'))
    print(pd.date_range('2004-09-20', periods=8, freq='h'))
    times_sample = pd.to_datetime(['2013-02-03', '2013-02-06', '2013-02-08'])
    # Select the specified dates and just the Consumption column
    consum_sample = opsd_daily.loc[times_sample, ['Consumption']].copy()
    consum_sample
    # Convert the data to daily frequency, without filling any missings
    consum_freq = consum_sample.asfreq('D')
    # Create a column with missings forward filled
    consum_freq['Consumption - Forward Fill'] = consum_sample.asfreq(
        'D', method='ffill')
    print(consum_freq)

    # Resampling
    data_columns = ['Consumption', 'Wind', 'Solar', 'Wind+Solar']
    opsd_weekly_mean = opsd_daily[data_columns].resample('W').mean()
    print(opsd_weekly_mean.head(3))
    print(opsd_daily.shape[0])
    print(opsd_weekly_mean.shape[0])

    # Start and end of the date range to extract
    start, end = '2017-01', '2017-06'
    # Plot daily and weekly resampled time series together
    fig, ax = plt.subplots()
    ax.plot(opsd_daily.loc[start:end, 'Solar'],
            marker='.', linestyle='-', linewidth=0.5, label='Daily')
    ax.plot(opsd_weekly_mean.loc[start:end, 'Solar'],
            marker='o', markersize=8, linestyle='-', label='Weekly Mean Resample')
    ax.set_ylabel('Solar Production (GWh)')
    ax.legend()
    # plt.show()

    # Compute the annual sums, setting the value to NaN for any year which has
    # fewer than 360 days of data
    opsd_annual = opsd_daily[data_columns].resample('YE').sum(min_count=360)
    opsd_annual = opsd_annual.set_index(opsd_annual.index.year)
    opsd_annual.index.name = 'Year'
    # Compute the ratio of Wind+Solar to Consumption
    opsd_annual['Wind+Solar/Consumption'] = opsd_annual['Wind+Solar'] / \
        opsd_annual['Consumption']
    print(opsd_annual.tail(3))

    # Plot from 2012 onwards, because there is no solar production data in earlier years
    ax = opsd_annual.loc[2012:, 'Wind+Solar/Consumption'].plot.bar(color='C0')
    ax.set_ylabel('Fraction')
    ax.set_ylim(0, 0.3)
    ax.set_title('Wind + Solar Share of Annual Electricity Consumption')
    plt.xticks(rotation=0)
    # plt.show()

    # Rolling windows
    print(opsd_daily[data_columns].head(8))
    opsd_7d = opsd_daily[data_columns].rolling(7, center=True).mean()
    opsd_7d.head(8)

# The min_periods=360 argument accounts for a few isolated missing days in the
# wind and solar production time series
opsd_365d = opsd_daily[data_columns].rolling(
    window=365, 
    center=True, 
    min_periods=360
).mean()

# Plot daily, 7-day rolling mean, and 365-day rolling mean time series
fig, ax = plt.subplots()
ax.plot(opsd_daily['Consumption'], marker='.', markersize=2, color='0.6',
linestyle='None', label='Daily')
ax.plot(opsd_7d['Consumption'], linewidth=2, label='7-d Rolling Mean')
ax.plot(opsd_365d['Consumption'], color='0.2', linewidth=3,
label='Trend (365-d Rolling Mean)')
# Set x-ticks to yearly interval and add legend and labels
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.legend()
ax.set_xlabel('Year')
ax.set_ylabel('Consumption (GWh)')
ax.set_title('Trends in Electricity Consumption')
plt.show()

# Plot 365-day rolling mean time series of wind and solar power
fig, ax = plt.subplots()
for nm in ['Wind', 'Solar', 'Wind+Solar']:
    ax.plot(opsd_365d[nm], label=nm)
    # Set x-ticks to yearly interval, adjust y-axis limits, add legend and labels
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.set_ylim(0, 400)
    ax.legend()
    ax.set_ylabel('Production (GWh)')
    ax.set_title('Trends in Electricity Production (365-d Rolling Means)')
plt.show()