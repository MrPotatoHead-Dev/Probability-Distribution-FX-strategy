import pandas as pd
import matplotlib.pyplot as plt
import warnings
import numpy as np
from structure import StructureData
import pandas_ta as ta
import plotly.graph_objects as go
import seaborn as sns
warnings.simplefilter("ignore")





def plot_ochl(df: pd.DataFrame, title: str) -> None:
    df_copy = df.copy()
    candlestick_trace  = data=go.Candlestick(x=df_copy.index,
                        open=df_copy['open'],
                        high=df_copy['high'],
                        low=df_copy['low'],
                        close=df_copy['close'])
    signals = df_copy[df_copy['signal'].isin([1, 2])]
    signals_trace = go.Scatter(x=signals.index,
                               y=signals['close'],
                               mode='markers',
                               marker=dict(symbol='circle',
                                           size=9,
                                           color=['red' if sig == 1 else 'green' for sig in signals['signal']],
                                           line=dict(color='black', width=1)),
                               name='Signal Points')
    exits = df_copy[df_copy['exit'].isin([1, 2])]
    exit_trace = go.Scatter(x=exits.index,
                               y=exits['close'],
                               mode='markers',
                               marker=dict(symbol='cross',
                                           size=12,
                                           color=['red' if sig == 1 else 'green' for sig in exits['exit']],
                                           line=dict(color='black', width=1)),
                               name='Signal Points')
    vwap_trace = go.Scatter(x=df_copy.index,
                            y=df_copy['vwap'],
                            mode='lines',
                            line=dict(color='blue', width=1),
                            name='VWAP')

    # Create the figure
    fig = go.Figure(data=[candlestick_trace, signals_trace, vwap_trace,  exit_trace]) # add: , ,
    
    # Update layout
    fig.update_layout(title=title)
    fig.update_layout(xaxis=dict(type='category', categoryorder='category ascending', showticklabels=False))
    fig.update(layout_xaxis_rangeslider_visible=False)
    fig.update_layout(width=1920, height=1080)
    # fig.write_image(f'results/round2/{filename}.png')
    fig.show()
    # Show the plot

def get_df() -> pd.DataFrame:
    data = pd.read_csv('data/GBPUSD_Daily.csv', sep='\t')
    df = StructureData(df=data, tf="htf")
    df['ret'] = df.close.pct_change().dropna()
    return df

def distribution_plot(df:pd.DataFrame, bins:int):
    ret = df['ret']
    d_mean = ret.mean()
    var = ret.var()
    d_std = np.sqrt(var)
    lower_q = round(ret.describe()[4],5)
    upper_q = round(ret.describe()[6],5)
    fig, ax = plt.subplots(figsize=(10, 6))
    # Plot KDE plot
    sns.kdeplot(ret, ax=ax, color='blue', label='KDE')
    # Plot histogram
    sns.histplot(ret, ax=ax, color='orange', bins=bins, label='Histogram')
    vertical_lines = [lower_q, upper_q]
    for line in vertical_lines:
        plt.axvline(line, color='red', linestyle='--', linewidth=1)
    plt.axvline(0, color='black', linestyle='--', linewidth=0.5)
    ax.legend()
    plt.xlabel('Return as percentage change')
    plt.show()

def kde_plot(df:pd.DataFrame):
    bws = [0.005, 0.009, 0.05]
    for w in bws:
        sns.kdeplot(data=df, x='ret', bw_adjust=w)
    plt.show()

def add_signals_to_df(df:pd.DataFrame, pd_factor:float = 0, rsi_factor: int = 0, rsi_length: int=14 ) -> pd.DataFrame:
    df_copy = df.copy()
    lower_q = (round(df_copy.ret.describe()[4],5) - pd_factor)
    upper_q = (round(df_copy.ret.describe()[6],5) + pd_factor)
    df_copy['signal'] =  None
    df_copy['exit'] = None
    df_copy['rsi'] = ta.rsi(df_copy.close, length=rsi_length)
    df_copy['vwap'] = ta.vwap(df_copy.high, df_copy.low, df_copy.close, df_copy.volume, anchor='W')

    for i in range(len(df_copy)):
        p_ret = df_copy['ret'].iloc[i-1]
        cur_ret = df_copy['ret'].iloc[i]
        # add entry signals
        if p_ret <lower_q and p_ret < 2* cur_ret :
            df_copy.at[df.index[i],'signal'] = 2
        elif p_ret > upper_q and p_ret > cur_ret*2:
            df_copy.at[df.index[i],'signal'] = 1
        
        # add exit signal
        # if below then above
        rsi_value_longs = 50 + rsi_factor
        rsi_value_shorts = 50 - rsi_factor
        if df_copy['rsi'].iloc[i-1] < rsi_value_longs and df_copy['rsi'].iloc[i] > rsi_value_longs:
            df_copy.at[df_copy.index[i],'exit'] = 1
        # if above then below
        elif df_copy['rsi'].iloc[i-1] > rsi_value_shorts and df_copy['rsi'].iloc[i] < rsi_value_shorts:
            df_copy.at[df_copy.index[i],'exit'] = 2
    
    return df_copy


def backtesting(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Longs: below vwap -> PD signal
    Shorts: above vwawp ->  PD signal
    Exit contions are either:
        1. price retraces to the mid point of the RSI
        2. close long if there is a bearish close below the vwap (in DD). Close short if there is a bullish close above vwap (in DD)'''
    

    results = pd.DataFrame(index=df.index, columns=['entry_price', 'exit_price', 'type', 'exit_date'])
    position_open = False
    position_index = None
    for i in range(len(df)):
        close_price = df['close'].iloc[i]
                # Close position
        if position_open :
                
                #Close short
                if results['type'].loc[position_index] == "short":
                    
                    # stoploss being triggered
                    if (close_price > results['entry_price'].loc[position_index] and 
                        df['trend'].iloc[i] =='up_candle' and
                        df['vwap'].iloc[i] < close_price ) :
                            position_open = False
                            results['exit_price'].loc[position_index] = close_price
                            results['exit_date'].loc[position_index] = df.index[i]
                            position_index = None
                            
                    elif df['exit'].iloc[i] == 2:
                            # add data to the results df
                            position_open = False
                            results['exit_price'].loc[position_index] = close_price
                            results['exit_date'].loc[position_index] = df.index[i]
                            position_index = None
    
                
                #close long
                elif results['type'].loc[position_index] == "long":
                    # stoploss being triggered
                    if (close_price < results['entry_price'].loc[position_index] and 
                        df['trend'].iloc[i] =='down_candle' and
                        df['vwap'].iloc[i] > close_price ):
                            # add data to the results df
                            position_open = False
                            results['exit_price'].loc[position_index] = close_price  
                            results['exit_date'].loc[position_index] = df.index[i]
                            position_index = None 

                    # hits tp  
                    elif df['exit'].iloc[i] == 1:
                            # add data to the results df
                            position_open = False
                            results['exit_price'].loc[position_index] =close_price 
                            results['exit_date'].loc[position_index] = df.index[i]
                            position_index = None
        # open position
        if close_price > df['vwap'].iloc[i]:
            
            if  not position_open and df['signal'].iloc[i] == 1:
                #open short position
                
                position_open=True
                results['entry_price'].iloc[i]= close_price
                results['type'].iloc[i] = "short"
                position_index = df.index[i]
                

        elif close_price <  df['vwap'].iloc[i]:
            if  not position_open and df['signal'].iloc[i] == 2:
                #open long position
                position_open=True
                results['entry_price'].iloc[i] = close_price
                results['type'].iloc[i] = "long"
                position_index = df.index[i]
                


    results = results.dropna()
    results['exit_date'] =pd.to_datetime(results['exit_date'])
    results['pips_per_trade'] = results.apply(pips_per_trade, axis=1)
   
    return results

def plot_cumulative_gains(balance: list):

    plt.figure(figsize=(10, 6))
    plt.plot(balance)
    plt.xlabel('Trade Index')
    plt.ylabel('Account Balance')
    plt.title('Cumulative Account Balance Over Trades')
    plt.grid(True)
    plt.show()



def pips_per_trade(row):
     
     if row['type'] == 'long':
        pip_values = row['exit_price'] -  row['entry_price']
        return pip_values*10000
     else:
        pip_values = row['entry_price'] -  row['exit_price']
        return pip_values*10000


def trade_stats(results: pd.DataFrame, starting_balance:int = 10000):
   

    total_trades = len(results)
    long_trades = results[results['type'] == 'long']
    long_wins = len(long_trades[long_trades['pips_per_trade'] > 0])
    short_trades = results[results['type'] == 'short']
    short_wins = len(short_trades[short_trades['pips_per_trade'] > 0])
    pip_sum = results['pips_per_trade'].sum()
    risk_per_pip = 10 # 10$ per pip
    results['risk_per_pip'] = results['pips_per_trade'] *risk_per_pip
    wins = len(results[results['pips_per_trade'] >0 ])


    # calculate statistics     
    win_rate = round((wins / total_trades) * 100, 2)
    long_win_rate = round((long_wins / len(long_trades)) * 100, 2)
    short_win_rate = round((short_wins / len(short_trades)) * 100, 2)
    average_win = pip_sum / wins if wins > 0 else 0
    average_loss = pip_sum / (total_trades - wins) if wins < total_trades else 0
    average_return = pip_sum / total_trades
    largest_winner = max(results['risk_per_pip'])
    largest_loser = min(results['risk_per_pip'])
    gains = results['risk_per_pip'].cumsum()  
    balance = gains + 10000
    

    # Calculate Sharpe Ratio
    excess_returns = average_return - 0.0054
    std_dev = np.std(results['risk_per_pip'])
    sharpe_ratio = excess_returns / std_dev 


    stats = {
        "Total Trades:": total_trades,
        "Total Wins:": wins,
        "Win Rate [%]:": str(win_rate) + "%",
        "Total Pips [pips]:": round(pip_sum,2),
        "Long Win Rate [%]:": str(long_win_rate) + "%",
        "Short Win Rate [%]:": str(short_win_rate) + "%",
        "Average Win Size [pips]:": round(average_win,2),
        "Average Loss Size [pips]:": round(average_loss,2),
        "Largest Winner [pips]:": round(largest_winner,2),
        "Largest Loser [pips]:": round(largest_loser,2),
        "Sharp ratio:": round(sharpe_ratio,2),
        "Closing balance [$]:": round(balance[-1],2),
        "Total gains [%]:":str( round(((balance[-1]-10000)/10000 )*100,2)) + "%",
        "gains_matrix": round(((balance[-1]-10000)/10000 )*100,2),
        "wr_matrix": win_rate
    }
    return stats, balance

    





if __name__ == "__main__":
     
    df = get_df()
    # distribution_plot(df=df, bins=100)
    df = add_signals_to_df(df=df)
    year = 2023

    # seperate 2 months of data to check signals
    # month = 10
    # n_month = month+1
    # data = df[(df['year'] == year) & ((df['month'] == month) | (df['month'] == n_month))]
    # res = backtesting(data) 
    # plot_ochl(data, 'testing')


    # reduce dataframe to one year
    data = df[df['year'] == year]
    # res = backtesting(data)
    # stats, balance = trade_stats(res)
    # for key, value in stats.items():
    #     print(key, value)
    # plot_cumulative_gains(balance=balance)

    # strategy optimisation
    rsi_factors = [0, 5, 10, 15, 20]
    pd_factors = [0, 0.001, 0.002,0.003,0.004]
    rsi_factor_label = [50 + factor for factor in rsi_factors]
    pb_factor_label = [round(0.0033 + factor,4) for factor in pd_factors] 

    total_gains_matrix = [[0 for _ in pd_factors] for _ in rsi_factors]
    wr_matrix = [[0 for _ in pd_factors] for _ in rsi_factors]
    sr_matrix = [[0 for _ in pd_factors] for _ in rsi_factors]
    tt_matrix = [[0 for _ in pd_factors] for _ in rsi_factors]
    
    # grid search
    for i, rsi_factor in enumerate(rsi_factors):
        for j, pd_factor in enumerate(pd_factors):
            df = add_signals_to_df(df=data, pd_factor=pd_factor, rsi_factor=rsi_factor, rsi_length=14)
            res = backtesting(df=df)
            stats, balance = trade_stats(res)
            total_gains = stats['gains_matrix']
            wr = stats["wr_matrix"]
            sr = stats['Sharp ratio:']
            tt = stats['Total Trades:']
            total_gains_matrix[i][j] = total_gains
            wr_matrix[i][j] = wr
            sr_matrix[i][j] = sr
            tt_matrix[i][j] = tt

    fig, axes = plt.subplots(2, 2, figsize=(15, 6))


    # Plot the first heatmap (Total Gains)
    sns.heatmap(total_gains_matrix, annot=True, cmap="YlGnBu", fmt='.0f', xticklabels=pb_factor_label, yticklabels=rsi_factor_label, ax=axes[0, 0])
    axes[0, 0].set_xlabel('Probability Distribution Factors')
    axes[0, 0].set_ylabel('RSI Factors')
    axes[0, 0].set_title('Total Gains Heatmap [%]')

    # Plot the second heatmap (Winrate)
    sns.heatmap(wr_matrix, annot=True, cmap="YlGnBu", fmt='.0f', xticklabels=pb_factor_label, yticklabels=rsi_factor_label, ax=axes[0, 1])
    axes[0, 1].set_xlabel('Probability Distribution Factors')
    axes[0, 1].set_ylabel('RSI Factors')
    axes[0, 1].set_title('Winrate Heatmap [%]')

    # Plot the third heatmap (Sharp ratio)
    sns.heatmap(sr_matrix, annot=True, cmap="YlGnBu", fmt='.2f', xticklabels=pb_factor_label, yticklabels=rsi_factor_label, ax=axes[1, 0])
    axes[1, 0].set_xlabel('Probability Distribution Factors')
    axes[1, 0].set_ylabel('RSI Factors')
    axes[1, 0].set_title('Sharp ratio Heatmap')

    # Plot the fourth heatmap (Total trades)
    sns.heatmap(tt_matrix, annot=True, cmap="YlGnBu", fmt='.0f', xticklabels=pb_factor_label, yticklabels=rsi_factor_label, ax=axes[1, 1])
    axes[1, 1].set_xlabel('Probability Distribution Factors')
    axes[1, 1].set_ylabel('RSI Factors')
    axes[1, 1].set_title('Total Trades Heatmap')

    # Adjust layout
    plt.tight_layout()

    # Show the plot
    plt.show()
