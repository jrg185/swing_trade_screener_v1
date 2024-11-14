import pandas as pd
import yfinance as yf
from datetime import datetime
import os
import numpy as np
from sklearn.metrics import r2_score

# Import local modules
from stock_screener import StockScreener
from risk_manager import RiskManager

def get_stock_universe():
    """Get universe of stocks to analyze"""
    print("Fetching stock universe...")
    
    try:
        # Get S&P 500 components
        sp500 = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
        sp500_tickers = sp500['Symbol'].tolist()
        print(f"Found {len(sp500_tickers)} S&P500 stocks")
        
        # Clean tickers
        all_tickers = [ticker for ticker in sp500_tickers 
                      if isinstance(ticker, str) and 
                      '.' not in ticker and 
                      '^' not in ticker and
                      len(ticker) < 5]
        
        print(f"Found {len(all_tickers)} clean tickers before filtering")
        
        # Initial filter for price and volume with relaxed criteria
        filtered_tickers = []
        print("Starting price and volume filtering...")
        
        for i, ticker in enumerate(all_tickers):
            print(f"Filtering stock {i+1}/{len(all_tickers)}: {ticker}", end='\r')
            try:
                stock = yf.Ticker(ticker)
                hist = stock.history(period='1mo')
                if len(hist) == 0:
                    continue
                    
                current_price = hist['Close'].iloc[-1]
                avg_volume = hist['Volume'].mean()
                
                # Relaxed criteria
                if (current_price <= 100 and     # Increased from 50
                    current_price >= 1.0 and     # Decreased from 1.5
                    avg_volume >= 200000):       # Decreased from 500000
                    filtered_tickers.append(ticker)
                    
            except Exception as e:
                continue
        
        print(f"\nFound {len(filtered_tickers)} stocks meeting price and volume criteria")
        
        if len(filtered_tickers) < 10:
            print("Warning: Very few stocks found, using all filtered tickers")
        
        return filtered_tickers
        
    except Exception as e:
        print(f"Error in stock universe creation: {str(e)}")
        return all_tickers[:100]  # Return first 100 tickers if filtering fails
        
    except Exception as e:
        print(f"Error in stock universe creation: {str(e)}")
        # Fallback to a curated list of S&P 500 stocks under $200
        fallback_tickers = [
            'AMD', 'CSCO', 'INTC', 'ORCL', 'CRM',
            'BAC', 'WFC', 'MS', 'GS', 'USB',
            'F', 'GE', 'PG', 'KO', 'DIS',
            'XOM', 'CVX', 'COP', 'EOG', 'SLB'
        ]
        print("Using fallback stock list due to fetch error")
        return fallback_tickers

def get_stock_info(ticker):
    """Get basic stock information including sector, market cap, and volume"""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        return {
            'sector': info.get('sector', 'Unknown'),
            'market_cap': info.get('marketCap', 0),
            'avg_volume': info.get('averageVolume', 0)
        }
    except Exception as e:
        print(f"Error getting info for {ticker}: {str(e)}")
        return {
            'sector': 'Unknown',
            'market_cap': 0,
            'avg_volume': 0
        }

def create_metrics_explanation():
    """Create DataFrame explaining each metric"""
    explanations = {
        'Metric': [
            'Rank',
            'Ticker',
            'Current Price',
            'Entry Price',
            'Target Price',
            'Stop Loss',
            'Confidence',
            'R-Squared',
            'Prediction Accuracy',
            'Volatility',
            'Risk/Reward',
            'Expected Value',
            'Position Size',
            'Position Value',
            'Risk Amount',
            'Score',
            'Avg Volume',
            'Market Cap',
            'Sector'
        ],
        'Explanation': [
            'Overall ranking based on combined score',
            'Stock ticker symbol',
            'Current market price of the stock',
            'Suggested entry price for the trade',
            'Predicted target price based on model',
            'Suggested stop loss price',
            'Model\'s confidence in the prediction',
            'Model\'s goodness of fit',
            'Historical accuracy of model predictions',
            'Stock\'s historical volatility',
            'Ratio of potential profit to potential loss',
            'Risk-adjusted expected value of trade',
            'Recommended number of shares to trade',
            'Total dollar value of position',
            'Maximum dollar risk for position',
            'Combined rating incorporating all metrics',
            'Average daily trading volume',
            'Company\'s market capitalization',
            'Company\'s business sector'
        ],
        'Interpretation': [
            'Lower number = better opportunity',
            'N/A',
            'Current market value',
            'Price to enter the trade',
            'Expected price target',
            'Exit trade if price falls below this',
            'Higher = more confident prediction',
            'Closer to 1.0 = better fit',
            'Higher = more historically accurate',
            'Lower = more stable price action',
            '>2.0 generally considered good',
            'Higher = better risk-adjusted return',
            'Based on portfolio risk management',
            'Should not exceed position limits',
            'Maximum loss if stop loss hit',
            'Higher score = better overall opportunity',
            'Higher = more liquid',
            'Size of the company',
            'Industry classification'
        ]
    }
    return pd.DataFrame(explanations)

import pandas as pd
import yfinance as yf
from stock_screener import StockScreener
from risk_manager import RiskManager
from datetime import datetime
import os
import numpy as np
from sklearn.metrics import r2_score

def rank_trades(portfolio_value=100000):
    """Generate ranked list of swing trade opportunities"""
    output_dir = r"C:\Users\josep\Downloads"
    
    screener = StockScreener(
        max_price=100,
        min_volume=200000
    )
    
    tickers = get_stock_universe()
    trades = []
    print("\nAnalyzing stocks...")
    
    debug_stats = {
        'no_result': 0,
        'low_r2': 0,
        'low_confidence': 0,
        'low_conviction': 0,
        'low_winrate': 0,
        'low_precision': 0
    }
    
    for i, ticker in enumerate(tickers, 1):
        try:
            result = screener.analyze_stock(
                ticker=ticker,
                forward_days=5,
                required_return=0.005
            )
            
            if not result:
                debug_stats['no_result'] += 1
                continue
                
            # Enhanced debug info with clear separators
            print(f"\n{'='*50}")
            print(f"Analyzing {ticker} ({i}/{len(tickers)}):")
            print(f"{'='*50}")
            print("\nModel Quality Metrics:")
            print(f"R²: {result.get('r2_score', 0):.3f}")
            print(f"Total Trades: {result.get('total_trades', 0)}")
            print(f"Profitable Trades: {result.get('profitable_trades', 0)}")
            print(f"Trade Success Ratio: {result.get('trade_ratio', 0):.3f}")
            
            print("\nPrediction Metrics:")
            print(f"Direction: {result.get('predicted_direction', 'None')}")
            print(f"Confidence: {result.get('prediction_confidence', 0):.3f}")
            print(f"Win Rate (Overall): {result.get('win_rate', 0):.3f}")
            print(f"Precision ({result.get('predicted_direction', 'None')}): {result.get('precision', 0):.3f}")
            
            print("\nRisk Metrics:")
            print(f"Avg Win: {result.get('avg_win', 0):.3f}")
            print(f"Avg Loss: {result.get('avg_loss', 0):.3f}")
            print(f"{'='*50}\n")
            
            # Modified filtering criteria
            if result['r2_score'] < 0.4:
                debug_stats['low_r2'] += 1
                continue
            
            # Check for strong conviction in either direction
            prediction_conf = result['prediction_confidence']
            strong_up = prediction_conf >= 0.6
            strong_down = prediction_conf <= 0.4
            
            if not (strong_up or strong_down):
                debug_stats['low_conviction'] += 1
                continue
                
            if result['win_rate'] < 0.6:
                debug_stats['low_winrate'] += 1
                continue
                
            if result['precision'] < 0.6:
                debug_stats['low_precision'] += 1
                continue
            
            info = get_stock_info(ticker)
            
            # Inside rank_trades function, where we append to trades list
            trades.append({
                'Ticker': ticker,
                'Sector': info.get('sector', 'Unknown'),
                'Current Price': result['current_price'],
                'Target Price': result['target_price'],
                'Stop Loss': result['stop_loss'],
                'Support': result['support_level'],
                'Resistance': result['resistance_level'],
                'Expected Move': f"{result['expected_move_pct']:.1f}%",
                'Direction': result['predicted_direction'],
                'Conviction': abs(prediction_conf - 0.5) * 2,
                'Win Rate': result['win_rate'],
                'Precision': result['precision'],
                'Avg Win': result['avg_win'],
                'Avg Loss': result['avg_loss'],
                'Trade Ratio': result['trade_ratio'],
                'R-Squared': result['r2_score'],
                'Technical Score': result.get('technical_scores', {}).get('total_score', 0),
                'ATR': result['atr'],
                'Market Cap': info.get('market_cap', 0),
                'Avg Volume': result.get('avg_volume', 0)
            })
                
        except Exception as e:
            print(f"\nError analyzing {ticker}: {str(e)}")
            continue
    
    # Print debug statistics
    print("\nDebug Statistics:")
    print(f"Total stocks analyzed: {len(tickers)}")
    print(f"Failed to get result: {debug_stats['no_result']}")
    print(f"Failed R² threshold: {debug_stats['low_r2']}")
    print(f"Failed conviction threshold: {debug_stats['low_conviction']}")
    print(f"Failed win rate threshold: {debug_stats['low_winrate']}")
    print(f"Failed precision threshold: {debug_stats['low_precision']}")
    
    if not trades:
        print("No valid trades found!")
        return None
    
    # Create DataFrame and calculate score
    df = pd.DataFrame(trades)
    
    # Updated scoring system that works for both directions
    df['Score'] = (
        df['Conviction'] * 
        df['Win Rate'] * 
        df['R-Squared'] * 
        (df.get('Technical Score', 100) / 100) *  # Normalize technical score
        (1 + df['Trade Ratio']) *  # Boost score for higher trade success ratio
        (1 + df['Avg Win'].abs() / (1 + df['Avg Loss'].abs()))  # Consider risk/reward
    )
    
    df = df.sort_values('Score', ascending=False)
    df.insert(0, 'Rank', range(1, len(df) + 1))
    
    # Format numeric columns
    df['Market Cap'] = df['Market Cap'].apply(lambda x: f"${x/1e9:.1f}B" if x >= 1e9 else f"${x/1e6:.1f}M")
    df['Avg Volume'] = df['Avg Volume'].apply(lambda x: f"{x/1e6:.1f}M")
    
    numeric_cols = [
        'Current Price', 'Conviction', 'Win Rate', 'Precision',
        'Avg Win', 'Avg Loss', 'Trade Ratio', 'R-Squared', 
        'Technical Score', 'Score'
    ]
    df[numeric_cols] = df[numeric_cols].round(6)
    
    # Save results
    filename = f'trade_opportunities_{datetime.now().strftime("%Y%m%d_%H%M")}.xlsx'
    filepath = os.path.join(output_dir, filename)
    
    with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
        # Main opportunities sheet
        df.to_excel(writer, sheet_name='Trade Opportunities', index=False)
        
        # Separate sheets for longs and shorts
        longs = df[df['Direction'] == 'Up']
        shorts = df[df['Direction'] == 'Down']
        
        if len(longs) > 0:
            longs.to_excel(writer, sheet_name='Long Opportunities', index=False)
        if len(shorts) > 0:
            shorts.to_excel(writer, sheet_name='Short Opportunities', index=False)
        
        # Sector Analysis
        sector_analysis = df.groupby(['Sector', 'Direction']).agg({
            'Score': ['mean', 'count'],
            'Win Rate': 'mean',
            'Precision': 'mean',
            'R-Squared': 'mean',
            'Trade Ratio': 'mean'
        }).round(4)
        sector_analysis.to_excel(writer, sheet_name='Sector Analysis')
    
    print(f"\nAnalysis complete. Found {len(trades)} valid trades.")
    print(f"Results saved to: {filepath}")
    
    # Display top opportunities in both directions
    if len(longs) > 0:
        print("\nTop Long Opportunities:")
        display_cols = ['Rank', 'Ticker', 'Direction', 'Conviction', 'Win Rate', 
                       'Precision', 'Trade Ratio', 'Score']
        print(longs[display_cols].head().to_string(index=False))
    
    if len(shorts) > 0:
        print("\nTop Short Opportunities:")
        print(shorts[display_cols].head().to_string(index=False))
    
    return filepath
        
         
    df = pd.DataFrame(trades)
    
    # Calculate combined score with risk metrics
    df['Score'] = (
        df['Prediction Accuracy'] * 
        df['Confidence'] * 
        df['R-Squared'] * 
        df['Risk/Reward']
    ) / df['Volatility']
    
    df = df.sort_values('Score', ascending=False)
    df.insert(0, 'Rank', range(1, len(df) + 1))
    
    # Format market cap and volume
    df['Market Cap'] = df['Market Cap'].apply(lambda x: f"${x/1e9:.1f}B" if x >= 1e9 else f"${x/1e6:.1f}M")
    df['Avg Volume'] = df['Avg Volume'].apply(lambda x: f"{x/1e6:.1f}M")
    
    # Round numeric columns
    numeric_cols = [
        'Current Price', 'Entry Price', 'Target Price', 'Stop Loss', 
        'Confidence', 'R-Squared', 'Prediction Accuracy', 'Volatility', 
        'Risk/Reward', 'Score', 'Expected Value', 'Position Value', 
        'Risk Amount'
    ]
    df[numeric_cols] = df[numeric_cols].round(2)
    
    # Create filename with timestamp
    filename = f'trade_opportunities_{datetime.now().strftime("%Y%m%d_%H%M")}.xlsx'
    filepath = os.path.join(output_dir, filename)
    
    # Save to Excel with multiple sheets
    with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Trade Opportunities', index=False)
        create_metrics_explanation().to_excel(writer, sheet_name='Metrics Explanation', index=False)
        
        # Add risk analysis sheet
        risk_analysis = pd.DataFrame({
            'Metric': ['Total Position Value', 'Total Risk Amount', 'Average Position Size', 'Average Risk/Reward'],
            'Value': [
                f"${df['Position Value'].sum():,.2f}",
                f"${df['Risk Amount'].sum():,.2f}",
                f"${df['Position Value'].mean():,.2f}",
                f"{df['Risk/Reward'].mean():.2f}"
            ]
        })
        risk_analysis.to_excel(writer, sheet_name='Risk Analysis', index=False)
        
        # Add sector analysis
        sector_analysis = df.groupby('Sector')[['Score', 'Position Value']].agg({
            'Score': 'mean',
            'Position Value': 'sum'
        }).round(2)
        sector_analysis = sector_analysis.sort_values('Score', ascending=False)
        sector_analysis.to_excel(writer, sheet_name='Sector Analysis')
    
    print(f"\nAnalysis complete. Found {len(trades)} valid trades.")
    print(f"Results saved to: {filepath}")
    
    # Display top 10 trades
    display_cols = [
        'Rank', 'Ticker', 'Sector', 'Current Price', 'Target Price', 
        'Stop Loss', 'Confidence', 'R-Squared', 'Risk/Reward', 
        'Position Size', 'Risk Amount', 'Score'
    ]
    print("\nTop 10 Trading Opportunities:")
    print(df[display_cols].head(10).to_string(index=False))
    
    return filepath

if __name__ == "__main__":
    output_file = rank_trades()