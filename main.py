from neuralintents import BasicAssistant
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as web
import mplfinance as mpf
import pickle
import sys
import datetime as dt
import os
import nltk

# Download required NLTK data if not already downloaded
def ensure_nltk_data():
    required_packages = ['punkt', 'punkt_tab', 'averaged_perceptron_tagger', 'universal_tagset']
    for package in required_packages:
        try:
            nltk.data.find(f'tokenizers/{package}')
        except LookupError:
            print(f"Downloading {package}...")
            nltk.download(package, quiet=True)

# Initial portfolio
portfolio = {'AAPL': 20, 'TSLA': 5, 'GS': 10}

# Load portfolio if exists
try:
    with open('portfolio.pkl', 'rb') as f:
        portfolio = pickle.load(f)
except FileNotFoundError:
    with open('portfolio.pkl', 'wb') as f:
        pickle.dump(portfolio, f)

# Save portfolio
def save_portfolio():
    with open('portfolio.pkl', 'wb') as f:
        pickle.dump(portfolio, f)

# Add to portfolio
def add_portfolio():
    ticker = input("Which stock do you want to add: ").upper()
    try:
        amount = int(input("How many shares do you want to add: "))
        if ticker in portfolio.keys():
            portfolio[ticker] += amount
        else:
            portfolio[ticker] = amount
        save_portfolio()
        print(f"{amount} shares of {ticker} added to your portfolio.")
    except ValueError:
        print("Invalid input. Please enter a valid number.")

# Remove from portfolio
def remove_portfolio():
    ticker = input("Which stock do you want to sell: ").upper()
    try:
        amount = int(input("How many shares do you want to sell: "))
        if ticker in portfolio.keys():
            if amount <= portfolio[ticker]:
                portfolio[ticker] -= amount
                if portfolio[ticker] == 0:
                    del portfolio[ticker]
                save_portfolio()
                print(f"{amount} shares of {ticker} sold.")
            else:
                print("You don't have enough shares!")
        else:
            print(f"You don't own any shares of {ticker}.")
    except ValueError:
        print("Invalid input. Please enter a valid number.")

# Show portfolio
def show_portfolio():
    print("Your portfolio:")
    for ticker, shares in portfolio.items():
        print(f"You own {shares} shares of {ticker}.")

# Portfolio worth
def portfolio_worth():
    total = 0
    for ticker, shares in portfolio.items():
        try:
            data = web.DataReader(ticker, 'yahoo')
            price = data['Close'].iloc[-1]
            total += price * shares
        except Exception:
            print(f"Could not retrieve data for {ticker}.")
    print(f"Your portfolio is worth ${total:.2f} USD.")

# Portfolio gains
def portfolio_gains():
    starting_date = input("Enter the starting date for comparison (YYYY-MM-DD): ")
    sum_now = 0
    sum_then = 0

    try:
        for ticker, shares in portfolio.items():
            data = web.DataReader(ticker, 'yahoo')
            price_now = data['Close'].iloc[-1]
            price_then = data.loc[data.index == starting_date]['Close'].values[0]
            sum_now += price_now * shares
            sum_then += price_then * shares

        relative_gains = ((sum_now - sum_then) / sum_then) * 100
        absolute_gains = sum_now - sum_then
        print(f"Relative Gains: {relative_gains:.2f}%")
        print(f"Absolute Gains: ${absolute_gains:.2f} USD")
    except IndexError:
        print("There was no trading on the specified day.")
    except Exception:
        print("An error occurred while calculating gains.")

# Plot chart
def plot_chart():
    ticker = input("Choose a ticker symbol: ").upper()
    starting_string = input("Choose a starting date (DD/MM/YYYY): ")

    plt.style.use('dark_background')
    try:
        start = dt.datetime.strptime(starting_string, "%d/%m/%Y")
        end = dt.datetime.now()

        data = web.DataReader(ticker, 'yahoo', start, end)

        colors = mpf.make_marketcolors(up='#00ff00', down='#ff0000', wick='inherit', edge='inherit', volume='in')
        mpf_style = mpf.make_mpf_style(base_mpf_style='nightclouds', marketcolors=colors)
        mpf.plot(data, type='candle', style=mpf_style, volume=True)
    except Exception as e:
        print(f"Could not plot data for {ticker}: {e}")

# Exit program
def bye():
    print("Goodbye!")
    sys.exit(0)

# Make sure NLTK data is downloaded
ensure_nltk_data()

# Assistant setup
mappings = {
    'plot_chart': plot_chart,
    'add_portfolio': add_portfolio,
    'remove_portfolio': remove_portfolio,
    'show_portfolio': show_portfolio,
    'portfolio_worth': portfolio_worth,
    'portfolio_gains': portfolio_gains,
    'bye': bye
}

# Create assistant instance
assistant = BasicAssistant('intents.json', mappings, model_name="financial_assistant_model")

# Train and save the model if it doesn't exist
if not os.path.exists("financial_assistant_model.keras"):
    print("Training new model...")
    assistant.fit_model()
    assistant.save_model()
else:
    print("Loading existing model...")
    assistant.load_model()

# Main loop
print("\nFinancial Assistant is ready! You can:")
print("- Show your portfolio")
print("- Add or remove stocks")
print("- Check portfolio worth")
print("- View portfolio gains")
print("- Plot stock charts")
print("- Say 'bye' to exit")

while True:
    message = input("\nHow can I assist you? ")
    assistant.request(message)


