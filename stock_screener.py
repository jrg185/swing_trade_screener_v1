import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    r2_score, 
    accuracy_score, 
    precision_score, 
    recall_score
)
import warnings
warnings.filterwarnings('ignore')

class StockScreener:
    def __init__(self, max_price=100, min_volume=200000):
        self.max_price = max_price
        self.min_volume = min_volume
        self.scaler = StandardScaler()
    
    def calculate_technical_score(self, df):
        """Calculate comprehensive technical score with weighted components"""
        scores = {}
        
        # 1. Trend Analysis (0-100 points)
        try:
            # MA Positioning
            ma20 = df['Close'] > df['SMA_20']
            ma50 = df['Close'] > df['SMA_50']
            ma_alignment = df['SMA_20'] > df['SMA_50']  # Golden cross condition
            
            # Trend strength
            trend_strength = abs(df['Close'] - df['SMA_50']) / df['SMA_50']
            
            trend_score = (
                (ma20.iloc[-1] * 30) +  # 30 points for being above 20MA
                (ma50.iloc[-1] * 30) +  # 30 points for being above 50MA
                (ma_alignment.iloc[-1] * 20) +  # 20 points for golden cross
                (min(trend_strength.iloc[-1] * 100, 20))  # Up to 20 points for trend strength
            )
            scores['Trend'] = trend_score
            
        except Exception as e:
            print(f"Error calculating trend score: {str(e)}")
            scores['Trend'] = 0
        
        # 2. Momentum Analysis (0-100 points)
        try:
            # RSI conditions (30-70 optimal range)
            rsi = df['RSI'].iloc[-1]
            rsi_score = (
                50 +  # Base score
                (30 if 40 <= rsi <= 60 else  # Optimal range
                 20 if (35 <= rsi < 40) or (60 < rsi <= 65) else  # Good range
                 0)  # Overbought/oversold
            )
            
            # MACD conditions
            macd_positive = df['MACD'].iloc[-1] > df['MACD_Signal'].iloc[-1]
            macd_increasing = df['MACD_Hist'].iloc[-1] > df['MACD_Hist'].iloc[-2]
            
            macd_score = (
                (macd_positive * 25) +  # 25 points for positive MACD
                (macd_increasing * 25)   # 25 points for increasing histogram
            )
            
            scores['Momentum'] = (rsi_score + macd_score) / 2
            
        except Exception as e:
            print(f"Error calculating momentum score: {str(e)}")
            scores['Momentum'] = 0
        
        # 3. Volume Analysis (0-100 points)
        try:
            # Recent volume trend
            vol_ma_ratio = df['Volume'].iloc[-1] / df['Volume_MA'].iloc[-1]
            vol_increasing = df['Volume'].iloc[-1] > df['Volume'].iloc[-2]
            
            volume_score = (
                (min(vol_ma_ratio * 50, 50) if vol_ma_ratio > 1 else 0) +  # Up to 50 points for volume above average
                (vol_increasing * 30) +  # 30 points for increasing volume
                (20 if df['Volume'].iloc[-1] > df['Volume'].mean() else 0)  # 20 points for above average volume
            )
            scores['Volume'] = volume_score
            
        except Exception as e:
            print(f"Error calculating volume score: {str(e)}")
            scores['Volume'] = 0
        
        # 4. Support/Resistance Analysis (0-100 points)
        try:
            # Calculate price percentile
            price_range = df['High'].max() - df['Low'].min()
            current_price = df['Close'].iloc[-1]
            price_position = (current_price - df['Low'].min()) / price_range
            
            # Distance from recent highs/lows
            dist_from_high = (df['High'].rolling(20).max().iloc[-1] - current_price) / current_price
            dist_from_low = (current_price - df['Low'].rolling(20).min().iloc[-1]) / current_price
            
            sr_score = (
                (50 if 0.3 <= price_position <= 0.7 else  # Optimal range
                 30 if 0.2 <= price_position <= 0.8 else  # Acceptable range
                 0) +  # Too extended
                (50 if dist_from_high > 0.02 and dist_from_low > 0.02 else 25)  # Not too close to extremes
            )
            scores['SR'] = sr_score
            
        except Exception as e:
            print(f"Error calculating S/R score: {str(e)}")
            scores['SR'] = 0
        
        # Calculate final weighted score
        final_score = (
            scores['Trend'] * 0.35 +      # 35% weight on trend
            scores['Momentum'] * 0.25 +   # 25% weight on momentum
            scores['Volume'] * 0.20 +     # 20% weight on volume
            scores['SR'] * 0.20           # 20% weight on support/resistance
        )
        
        # Return both final score and components
        return {
            'total_score': final_score,
            'trend_score': scores['Trend'],
            'momentum_score': scores['Momentum'],
            'volume_score': scores['Volume'],
            'sr_score': scores['SR']
        }
    
    def analyze_stock(self, ticker, forward_days=5, required_return=0.005):
        """Analyze a single stock with price targets"""
        try:
            # Get stock data
            stock = yf.Ticker(ticker)
            df = stock.history(period='3mo')
            
            if len(df) < 30:
                return None
                
            # Calculate technical indicators
            df['Returns'] = df['Close'].pct_change()
            df['SMA_20'] = df['Close'].rolling(window=20).mean()
            df['SMA_50'] = df['Close'].rolling(window=50).mean()
            df['Daily_Range'] = (df['High'] - df['Low']) / df['Close']
            df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
            
            # Calculate RSI
            df['RSI'] = self.calculate_rsi(df)
            
            # Calculate MACD
            df['MACD'], df['MACD_Signal'], df['MACD_Hist'] = self.calculate_macd(df['Close'])
            
            # Calculate ATR
            df['ATR'] = self.calculate_atr(df)
            
            # Create feature matrix for ML model
            X = pd.DataFrame(index=df.index)
            X['Price_to_MA20'] = df['Close'] / df['SMA_20'] - 1
            X['Price_to_MA50'] = df['Close'] / df['SMA_50'] - 1
            X['Daily_Range'] = df['Daily_Range']
            X['Volume_Ratio'] = df['Volume'] / df['Volume_MA']
            X['RSI'] = df['RSI'] / 100
            X['MACD_Hist'] = df['MACD_Hist']
            
            # Create target variable
            future_returns = df['Close'].shift(-forward_days).pct_change(forward_days)
            y = pd.Series((future_returns > required_return).astype(int))
            
            # Handle missing data
            mask = ~y.isna()
            X = X[mask]
            y = y[mask]
            
            if len(X) < 20:
                return None
            
            # Calculate model metrics
            model_metrics = self.calculate_model_metrics(X, y)
            
            # Get current price and technical scores
            current_price = df['Close'].iloc[-1]
            technical_scores = self.calculate_technical_score(df)
            
            # Calculate price targets
            volatility = df['Returns'].std() * np.sqrt(252)  # Annualized volatility
            atr = df['ATR'].iloc[-1]
            
            # Calculate expected move based on confidence and volatility
            confidence = model_metrics['prediction_confidence']
            direction = model_metrics['predicted_direction']
            expected_move_pct = volatility * confidence / np.sqrt(252) * 5  # 5-day forecast
            
            if direction == 'Up':
                target_price = current_price * (1 + expected_move_pct)
                stop_loss = current_price * (1 - (expected_move_pct * 0.5))  # Tighter stop for longs
            else:
                target_price = current_price * (1 - expected_move_pct)
                stop_loss = current_price * (1 + (expected_move_pct * 0.5))  # Tighter stop for shorts
            
            # Support and resistance levels
            support = current_price - (1.5 * atr)
            resistance = current_price + (1.5 * atr)
            
            # Combine all results
            result = {
                'ticker': ticker,
                'current_price': current_price,
                'target_price': target_price,
                'stop_loss': stop_loss,
                'support_level': support,
                'resistance_level': resistance,
                'expected_move_pct': expected_move_pct * 100,
                'atr': atr,
                'volatility': volatility,
                'technical_scores': technical_scores,
                **model_metrics  # Unpack model metrics
            }
            
            return result
            
        except Exception as e:
            print(f"Error analyzing {ticker}: {str(e)}")
            return None

    def calculate_model_metrics(self, X, y):
        """Calculate comprehensive model performance metrics using walk-forward validation"""
        try:
            tscv = TimeSeriesSplit(n_splits=5)
            
            metrics = {
                'precision_up': [],    # Precision for upward predictions
                'precision_down': [],  # Precision for downward predictions
                'win_rate': [],       # Overall prediction accuracy
                'avg_win': [],        # Average gain on correct predictions
                'avg_loss': [],       # Average loss on incorrect predictions
                'total_trades': 0,    # Total number of trades
                'profitable_trades': 0 # Number of profitable trades
            }
            
            # Train final model for current prediction
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=3,
                min_samples_split=5,
                random_state=42
            )
            
            # Walk-forward validation
            for train_idx, test_idx in tscv.split(X):
                X_train = X.iloc[train_idx]
                X_test = X.iloc[test_idx]
                y_train = y.iloc[train_idx]
                y_test = y.iloc[test_idx]
                
                # Scale features
                X_train_scaled = self.scaler.fit_transform(X_train)
                X_test_scaled = self.scaler.transform(X_test)
                
                # Train model
                model.fit(X_train_scaled, y_train)
                
                # Get predictions
                y_pred = model.predict(X_test_scaled)
                y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
                
                # Calculate precision for up and down predictions separately
                up_mask = y_pred == 1
                down_mask = y_pred == 0
                
                if np.any(up_mask):
                    precision_up = np.mean(y_test[up_mask] == y_pred[up_mask])
                    metrics['precision_up'].append(precision_up)
                
                if np.any(down_mask):
                    precision_down = np.mean(y_test[down_mask] == y_pred[down_mask])
                    metrics['precision_down'].append(precision_down)
                
                # Calculate overall win rate (accuracy across all trades)
                win_rate = np.mean(y_test == y_pred)
                metrics['win_rate'].append(win_rate)
                
                # Track total and profitable trades
                metrics['total_trades'] += len(y_pred)
                metrics['profitable_trades'] += np.sum(y_test == y_pred)
                
                # Calculate average wins and losses
                correct_trades = y_test == y_pred
                if np.any(correct_trades):
                    metrics['avg_win'].append(y_test[correct_trades].mean())
                if np.any(~correct_trades):
                    metrics['avg_loss'].append(y_test[~correct_trades].mean())
            
            # Get current prediction
            X_scaled = self.scaler.fit_transform(X)
            model.fit(X_scaled, y)
            current_features = X_scaled[-1:] 
            current_pred = model.predict_proba(current_features)[0]
            direction_pred = model.predict(current_features)[0]
            
            # Calculate R² score
            y_pred_full = model.predict_proba(X_scaled)[:, 1]
            r2 = r2_score(y, y_pred_full)
            
            # Determine precision based on predicted direction
            if direction_pred == 1:  # Up prediction
                precision = np.mean(metrics['precision_up']) if metrics['precision_up'] else 0
            else:  # Down prediction
                precision = np.mean(metrics['precision_down']) if metrics['precision_down'] else 0
            
            return {
                'prediction_confidence': current_pred[1],
                'predicted_direction': 'Up' if direction_pred == 1 else 'Down',
                'win_rate': np.mean(metrics['win_rate']),  # Overall accuracy
                'precision': precision,  # Direction-specific accuracy
                'avg_win': np.mean(metrics['avg_win']) if metrics['avg_win'] else 0,
                'avg_loss': np.mean(metrics['avg_loss']) if metrics['avg_loss'] else 0,
                'total_trades': metrics['total_trades'],
                'profitable_trades': metrics['profitable_trades'],
                'trade_ratio': metrics['profitable_trades'] / metrics['total_trades'] if metrics['total_trades'] > 0 else 0,
                'r2_score': r2
            }
            
        except Exception as e:
            print(f"Error in model metrics calculation: {str(e)}")
            return {
                'prediction_confidence': 0,
                'predicted_direction': 'Down',
                'win_rate': 0,
                'precision': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'total_trades': 0,
                'profitable_trades': 0,
                'trade_ratio': 0,
                'r2_score': 0
            }
        
    def calculate_technical_indicators(self, df):
        """Calculate comprehensive technical indicators"""
        # Price Action Indicators
        df['Returns'] = df['Close'].pct_change()
        df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
        
        # Moving Averages & Trends
        for period in [5, 10, 20, 50, 200]:
            df[f'MA_{period}'] = df['Close'].rolling(window=period).mean()
            df[f'MA_{period}_Slope'] = df[f'MA_{period}'].diff(5) / 5
        
        # Trend Strength
        df['Trend_Strength'] = abs(df['MA_50'] - df['MA_200']) / df['MA_50']
        
        # Volatility Indicators
        df['Daily_Range'] = (df['High'] - df['Low']) / df['Close']
        df['ATR'] = self.calculate_atr(df)
        df['Regime_Volatility'] = df['ATR'] / df['Close']
        
        # Enhanced Momentum
        df['RSI'] = self.calculate_rsi(df)
        df['ADX'] = self.calculate_adx(df)
        df['MFI'] = self.calculate_mfi(df)
        macd, signal, hist = self.calculate_macd(df['Close'])
        df['MACD'] = macd
        df['MACD_Signal'] = signal
        df['MACD_Hist'] = hist
        
        # Volume Analysis
        df['VWAP'] = self.calculate_vwap(df)
        df['Volume_Force'] = self.calculate_force_index(df)
        df['Volume_Returns'] = df['Volume'].pct_change()
        df['Volume_MA20'] = df['Volume'].rolling(window=20).mean()
        df['Relative_Volume'] = df['Volume'] / df['Volume_MA20']
        df['OBV'] = self.calculate_obv(df)
        
        # Risk Metrics
        df['Volatility_20d'] = df['Returns'].rolling(20).std() * np.sqrt(252)
        df['Downside_Risk'] = df['Returns'].clip(upper=0).rolling(20).std() * np.sqrt(252)
        df['Rolling_Sharpe'] = self.calculate_rolling_sharpe(df)
        
        return df
    
    def calculate_adx(self, df, period=14):
        """Calculate Average Directional Index"""
        df = df.copy()
        # Calculate True Range
        df['TR'] = pd.DataFrame({
            'HL': df['High'] - df['Low'],
            'HPC': abs(df['High'] - df['Close'].shift()),
            'LPC': abs(df['Low'] - df['Close'].shift())
        }).max(axis=1)
        
        # Calculate directional movement
        df['DMplus'] = np.where((df['High'] - df['High'].shift()) > (df['Low'].shift() - df['Low']),
                               np.maximum(df['High'] - df['High'].shift(), 0), 0)
        df['DMminus'] = np.where((df['Low'].shift() - df['Low']) > (df['High'] - df['High'].shift()),
                                np.maximum(df['Low'].shift() - df['Low'], 0), 0)
        
        # Calculate smoothed values
        for col in ['TR', 'DMplus', 'DMminus']:
            df[f'{col}_Smooth'] = df[col].rolling(period).mean()
        
        # Calculate DI values
        df['DIplus'] = 100 * df['DMplus_Smooth'] / df['TR_Smooth']
        df['DIminus'] = 100 * df['DMminus_Smooth'] / df['TR_Smooth']
        
        # Calculate ADX
        df['DX'] = 100 * abs(df['DIplus'] - df['DIminus']) / (df['DIplus'] + df['DIminus'])
        ADX = df['DX'].rolling(period).mean()
        
        return ADX
    
    def calculate_mfi(self, df, period=14):
        """Calculate Money Flow Index"""
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        money_flow = typical_price * df['Volume']
        
        positive_flow = pd.Series(0, index=df.index)
        negative_flow = pd.Series(0, index=df.index)
        
        # Calculate positive and negative money flow
        flow_mask = typical_price > typical_price.shift(1)
        positive_flow[flow_mask] = money_flow[flow_mask]
        negative_flow[~flow_mask] = money_flow[~flow_mask]
        
        positive_mf = positive_flow.rolling(period).sum()
        negative_mf = negative_flow.rolling(period).sum()
        
        mfi = 100 - (100 / (1 + positive_mf / negative_mf))
        return mfi
    
    def calculate_vwap(self, df):
        """Calculate Volume Weighted Average Price"""
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        vwap = (typical_price * df['Volume']).cumsum() / df['Volume'].cumsum()
        return vwap
    
    def calculate_force_index(self, df, period=13):
        """Calculate Force Index"""
        force = df['Close'].diff(1) * df['Volume']
        return force.ewm(span=period, adjust=False).mean()
    
    def calculate_rolling_sharpe(self, df, risk_free_rate=0.02, period=252):
        """Calculate rolling Sharpe ratio"""
        excess_returns = df['Returns'] - risk_free_rate/252
        return (excess_returns.rolling(window=period).mean() * 252) / \
               (df['Returns'].rolling(window=period).std() * np.sqrt(252))
    
    def create_features(self, df):
        """Create enhanced feature set for ML model"""
        features = pd.DataFrame(index=df.index)
        
        def clean_feature(series, weight=1.0):
            series = series.replace([np.inf, -np.inf], np.nan)
            if series.isna().any():
                series = series.fillna(series.mean() if series.mean() != 0 else 0)
            return series.clip(lower=series.mean() - 3*series.std(),
                             upper=series.mean() + 3*series.std()) * weight
        
        # Trend Features
        features['Trend_Strength'] = clean_feature(df['Trend_Strength'])
        features['ADX'] = clean_feature(df['ADX'] / 100)
        
        # Momentum Features
        features['RSI'] = clean_feature(df['RSI'] / 100)
        features['MFI'] = clean_feature(df['MFI'] / 100)
        features['MACD_Hist'] = clean_feature(df['MACD_Hist'])
        
        # Volume Features
        features['Relative_Volume'] = clean_feature(df['Relative_Volume'])
        features['Volume_Force'] = clean_feature(df['Volume_Force'])
        
        # Volatility Features
        features['ATR_Ratio'] = clean_feature(df['ATR'] / df['Close'])
        features['Volatility_20d'] = clean_feature(df['Volatility_20d'])
        
        # Risk Metrics
        features['Rolling_Sharpe'] = clean_feature(df['Rolling_Sharpe'])
        features['Downside_Risk'] = clean_feature(df['Downside_Risk'])
        
        return features.fillna(0)
    
    def calculate_atr(self, df, period=14):
        """Calculate Average True Range"""
        df = df.copy()
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        return true_range.rolling(period).mean()
    
    def calculate_rsi(self, df, period=14):
        """Calculate Relative Strength Index"""
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def calculate_macd(self, prices, slow=26, fast=12, signal=9):
        """Calculate MACD, Signal line, and Histogram"""
        exp1 = prices.ewm(span=fast, adjust=False).mean()
        exp2 = prices.ewm(span=slow, adjust=False).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        histogram = macd - signal_line
        return macd, signal_line, histogram
    
    def calculate_obv(self, df):
        """Calculate On-Balance Volume"""
        obv = pd.Series(index=df.index, dtype='float64')
        obv.iloc[0] = 0
        
        for i in range(1, len(df)):
            if df['Close'].iloc[i] > df['Close'].iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] + df['Volume'].iloc[i]
            elif df['Close'].iloc[i] < df['Close'].iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] - df['Volume'].iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i-1]
        
        return obv
    
    def prepare_training_data(self, df, forward_days=7, required_return=0.02):
        """Prepare features and labels for ML model"""
        X = self.create_features(df)
        
        # Create target variable using forward returns
        future_returns = df['Close'].shift(-forward_days).pct_change(forward_days)
        y = (future_returns > required_return).astype(int)
        
        # Remove samples where we don't have future data
        mask = ~y.isna()
        X = X[mask]
        y = y[mask]
        
        return X, y, future_returns[mask]
    
    def train_model(self, X, y):
        """Train model with simpler ensemble approach"""
        try:
            # Split data with standard train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, shuffle=False  # Using shuffle=False for time series
            )
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Create and train base models
            rf1 = RandomForestClassifier(
                n_estimators=100,
                max_depth=3,
                min_samples_split=5,
                random_state=42
            )
            
            rf2 = RandomForestClassifier(
                n_estimators=100,
                max_depth=5,
                min_samples_split=5,
                random_state=43
            )
            
            # Train models
            rf1.fit(X_train_scaled, y_train)
            rf2.fit(X_train_scaled, y_train)
            
            # Get predictions from both models
            pred1 = rf1.predict_proba(X_test_scaled)[:, 1]
            pred2 = rf2.predict_proba(X_test_scaled)[:, 1]
            
            # Average predictions (ensemble)
            y_pred = (pred1 + pred2) / 2
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred > 0.5)
            r2 = r2_score(y_test, y_pred)
            
            # Create final model by training on full dataset
            final_model = RandomForestClassifier(
                n_estimators=100,
                max_depth=4,
                min_samples_split=5,
                random_state=42
            )
            
            final_model.fit(self.scaler.transform(X), y)
            
            return final_model, accuracy, r2, y_test, y_pred
            
        except Exception as e:
            print(f"Error in model training: {str(e)}")
            return None, 0.0, 0.0, None, None
            
     