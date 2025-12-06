import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.offline as pyo
from plotly.subplots import make_subplots
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import datetime
from datetime import date
from pandas_datareader import data
import json

# Configuration
exclude_recent = True
years = 10

# Equity sector ETFs as predictors
equity_sector_etfs = ["XLE", "XLU", "XLI", "XLV", "XLY", "XLP", "XLK", "XLF", "XLB", "SPY", "XRT", "XLRE", "XLC"]
equity_sector_etfs = [ticker + ".US" for ticker in equity_sector_etfs]

equity_sector_focus = [
    "Energy", "Utilities", "Industrials", "Healthcare", "Consumer Discretionary",
    "Consumer Staples", "Technology", "Financials", "Materials", "SP500",
    "Retail", "Real Estate", "Communications"
]

if exclude_recent:
    equity_sector_etfs = equity_sector_etfs[:-3]
    equity_sector_focus = equity_sector_focus[:-3]

# Date range
start_date = datetime.datetime.now() - datetime.timedelta(days=365*years)
end_date = date.today()

print("Fetching equity sector data from Stooq...")
# Get equity sector data
try:
    etf_data = data.DataReader(equity_sector_etfs, 'stooq', start_date, end_date)
    final_df = etf_data['Close'].sort_index(ascending=True)
    indexed_df = final_df.apply(lambda col: col / col.dropna().iloc[0] * 100)
    indexed_df = indexed_df.bfill()

    # Create sector ratios vs SPY
    for sector in equity_sector_etfs:
        indexed_df[f"{sector} / SPY"] = indexed_df[sector] / indexed_df["SPY.US"] * 100

    ratios = [f"{sector} / SPY" for sector in equity_sector_etfs[:-1]]
    print(f"Successfully loaded {len(equity_sector_etfs)} equity sector ETFs")

except Exception as e:
    print(f"Error fetching equity data: {e}")
    raise

def load_fx_data():
    """Load FX data from the EMFX_risk_diffusion CSV"""
    try:
        # Load from GitHub raw URL
        fx_raw_url = 'https://raw.githubusercontent.com/DataVizHonduran/EMFX_risk_diffusion/main/fx_data_raw.csv'
        print(f"Loading FX data from {fx_raw_url}")

        df_fx = pd.read_csv(fx_raw_url, index_col=0, parse_dates=True)
        df_fx = df_fx.bfill().ffill()

        print(f"Successfully loaded FX data: {df_fx.shape}")
        print(f"Available currencies: {list(df_fx.columns)}")
        return df_fx

    except Exception as e:
        print(f"Error loading FX data: {e}")
        raise

def calculate_fair_value(currency, indexed_df, fx_df, ratios):
    """Calculate fair value for a single currency using equity sector ratios"""
    try:
        # Merge equity and FX data
        chart_df = indexed_df[ratios].merge(fx_df[[currency]], left_index=True, right_index=True, how="inner")
        chart_df = chart_df.bfill()
        chart_df = chart_df.loc[:, chart_df.isnull().sum() <= 10]

        # Prepare data for model
        y = chart_df[currency]
        valid_ratios = [col for col in ratios if col in chart_df.columns]
        X = chart_df[valid_ratios].bfill()

        if len(X) < 100:  # Need sufficient data
            print(f"Insufficient data for {currency}: only {len(X)} rows")
            return None

        # Train model with random train/test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        lasso = Lasso(max_iter=10000, alpha=0.9)
        lasso.fit(X_train, y_train)

        # Calculate predictions and residuals
        chart_df["ypred"] = lasso.predict(chart_df[valid_ratios].bfill())
        chart_df["resids"] = (chart_df[currency] - chart_df["ypred"]) / chart_df[currency] * 100

        # Model metrics
        y_pred = lasso.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r_squared = r2_score(y_test, y_pred)

        # Get model coefficients info
        coefficients = lasso.coef_
        coef_with_names = [(valid_ratios[i], coefficients[i]) for i in range(len(coefficients)) if coefficients[i] != 0]
        coef_with_names.sort(key=lambda x: abs(x[1]), reverse=True)

        non_zero_coefficients = [round(x[1], 3) for x in coef_with_names]
        selected_vars = [x[0] for x in coef_with_names]

        # Current fair value metrics
        current_actual = chart_df[currency].iloc[-1]
        current_predicted = chart_df["ypred"].iloc[-1]
        current_residual = chart_df["resids"].iloc[-1]

        print(f"  {currency}: R²={r_squared:.3f}, Residual={current_residual:.1f}%, Predictors={len(selected_vars)}")

        return {
            'currency': currency,
            'current_actual': current_actual,
            'current_predicted': current_predicted,
            'current_residual': current_residual,
            'r_squared': r_squared,
            'mse': mse,
            'chart_data': chart_df,
            'model': lasso,
            'valid_ratios': valid_ratios,
            'coefficients': non_zero_coefficients,
            'selected_vars': selected_vars,
            'coef_with_names': coef_with_names
        }

    except Exception as e:
        print(f"Error calculating fair value for {currency}: {e}")
        return None

def create_main_dashboard(summary_df):
    """Create the main overview dashboard with clickable bar chart"""

    # Main bar chart showing current fair values
    colors = ['red' if x < 0 else 'green' for x in summary_df['Residual_%']]

    fig = go.Figure()

    # Create clickable bar chart with links
    hover_text = []

    for i, row in summary_df.iterrows():
        currency = row['Currency']
        residual = row['Residual_%']
        r_squared = row['R_Squared']
        hover_text.append(
            f"<b>{currency}</b><br>"
            f"Current Rate: {row['Current_Rate']:.4f}<br>"
            f"Fair Value: {row['Fair_Value']:.4f}<br>"
            f"Residual: {residual:.1f}%<br>"
            f"R²: {r_squared:.2f}<br>"
            f"<i>Click for detailed analysis</i>"
        )

    # Main horizontal bar chart
    fig.add_trace(go.Bar(
        y=summary_df['Currency'],
        x=summary_df['Residual_%'],
        orientation='h',
        marker_color=colors,
        name='Cheap/Rich (%)',
        text=[f"{x:.1f}%" for x in summary_df['Residual_%']],
        textposition='outside',
        hovertemplate='%{customdata}<extra></extra>',
        customdata=hover_text
    ))

    # Add vertical line at zero
    fig.add_vline(x=0, line_dash="dash", line_color="gray")

    # Add currency links as annotations
    currency_links_html = "<br>".join([
        f'<a href="{currency.lower()}_analysis.html" style="text-decoration:none; color:#1f77b4;">📊 {currency}</a>'
        for currency in summary_df['Currency']
    ])

    last_updated = datetime.datetime.now().strftime('%Y-%m-%d %H:%M UTC')

    fig.update_layout(
        title={
            'text': "FX Fair Value: Equity Sector Model",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 28}
        },
        xaxis_title="Residual (% Deviation from Fair Value)",
        yaxis_title="Currency Pair",
        height=700,
        template="plotly_white",
        hovermode='closest',
        showlegend=False,
        margin=dict(r=250, l=100)
    )

    # Add methodology note
    methodology_text = (
        "<b>Methodology:</b><br>"
        "FX fair values estimated using LASSO regression<br>"
        "with equity sector performance ratios as predictors.<br>"
        "Green = Undervalued, Red = Overvalued<br>"
        "<br>"
        "<b>Sectors Used:</b> Energy, Utilities, Industrials,<br>"
        "Healthcare, Consumer Discretionary, Staples,<br>"
        "Technology, Financials, Materials, Retail"
    )

    fig.add_annotation(
        text=methodology_text,
        xref="paper", yref="paper",
        x=-0.15, y=0.5,
        xanchor='left', yanchor='middle',
        showarrow=False,
        font=dict(size=10),
        align="left",
        bgcolor="rgba(255,255,255,0.9)",
        bordercolor="gray",
        borderwidth=1
    )

    # Add navigation links
    fig.add_annotation(
        text="<b>Currency Details:</b><br>" + currency_links_html,
        xref="paper", yref="paper",
        x=1.02, y=1,
        xanchor='left', yanchor='top',
        showarrow=False,
        font=dict(size=11),
        align="left",
        bgcolor="rgba(255,255,255,0.8)",
        bordercolor="gray",
        borderwidth=1
    )

    # Add "Last Updated" annotation
    fig.add_annotation(
        text=f"Last Updated: {last_updated}",
        xref="paper", yref="paper",
        x=1, y=-0.08,
        xanchor='right', yanchor='top',
        showarrow=False,
        font=dict(size=12, color="gray")
    )

    return fig

def create_individual_currency_page(currency, result):
    """Create individual currency analysis page with coefficient table"""

    chart_data = result['chart_data']

    # Add white space at right side of charts
    new_dates = pd.date_range(start=chart_data.index[-1], periods=int(len(chart_data.index) * .05), freq='D')
    empty_rows = pd.DataFrame(index=new_dates)
    chart_data = pd.concat([chart_data, empty_rows])

    # Create three-panel chart
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            f'{currency} vs Fair Value Model',
            'Residual (% Deviation)',
            'Top Sector Predictors'
        ),
        specs=[[{"type": "scatter"}, {"type": "scatter"}],
               [{"type": "table", "colspan": 2}, None]],
        row_heights=[0.6, 0.4],
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )

    # Left panel: Actual vs Model
    fig.add_scatter(
        x=chart_data.index,
        y=chart_data[currency],
        mode='lines',
        name=currency,
        line=dict(color='blue', width=2),
        row=1, col=1
    )

    fig.add_scatter(
        x=chart_data.index,
        y=chart_data["ypred"],
        mode='lines',
        name="Fair Value Model",
        line=dict(color='red', width=2, dash='dash'),
        row=1, col=1
    )

    # Right panel: Residuals
    fig.add_scatter(
        x=chart_data.index,
        y=chart_data["resids"],
        mode='lines',
        name="Residual (%)",
        line=dict(color='green', width=2),
        row=1, col=2
    )

    # Add zero line to residuals
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=2)

    # Bottom panel: Coefficient table
    coef_data = result['coef_with_names'][:10]  # Top 10 predictors

    if coef_data:
        sector_names = [x[0].replace('.US / SPY', '').replace('.US', '') for x in coef_data]
        coefficients = [f"{x[1]:.4f}" for x in coef_data]

        fig.add_trace(go.Table(
            header=dict(
                values=['<b>Equity Sector</b>', '<b>Coefficient</b>'],
                fill_color='lightgray',
                align='left',
                font=dict(size=12)
            ),
            cells=dict(
                values=[sector_names, coefficients],
                fill_color='white',
                align='left',
                font=dict(size=11)
            )
        ), row=2, col=1)

    # Model info
    r_squared = result['r_squared']
    mse = result['mse']
    current_residual = result['current_residual']
    current_actual = result['current_actual']
    current_predicted = result['current_predicted']

    # Update layout
    last_updated = datetime.datetime.now().strftime('%Y-%m-%d %H:%M UTC')

    interpretation = "UNDERVALUED" if current_residual > 0 else "OVERVALUED"
    interpretation_color = "green" if current_residual > 0 else "red"

    fig.update_layout(
        title={
            'text': f"{currency} Fair Value Analysis - R²: {r_squared:.3f}",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20}
        },
        width=1400,
        height=900,
        template="plotly_white",
        showlegend=True
    )

    # Add model details
    model_info = f"""
    <b>Current Status:</b><br>
    Actual Rate: {current_actual:.4f}<br>
    Fair Value: {current_predicted:.4f}<br>
    Residual: <span style="color:{interpretation_color}"><b>{current_residual:.1f}%</b></span><br>
    Interpretation: <span style="color:{interpretation_color}"><b>{interpretation}</b></span><br>
    <br>
    <b>Model Performance:</b><br>
    R-squared: {r_squared:.3f}<br>
    MSE: {mse:.2f}<br>
    Active Predictors: {len(result['selected_vars'])}/10
    """

    fig.add_annotation(
        text=model_info,
        xref="paper", yref="paper",
        x=0.02, y=0.98,
        xanchor='left', yanchor='top',
        showarrow=False,
        font=dict(size=11),
        align="left",
        bgcolor="rgba(255,255,255,0.9)",
        bordercolor="gray",
        borderwidth=1
    )

    # Back to main link
    fig.add_annotation(
        text='<a href="fx_equity_sector_fair_value.html" style="color:#1f77b4; font-size:14px;">← Back to Overview</a>',
        xref="paper", yref="paper",
        x=1, y=1.05,
        xanchor='right', yanchor='bottom',
        showarrow=False,
        font=dict(size=12)
    )

    # Last updated
    fig.add_annotation(
        text=f"Last Updated: {last_updated}",
        xref="paper", yref="paper",
        x=1, y=-0.05,
        xanchor='right', yanchor='top',
        showarrow=False,
        font=dict(size=11, color="gray")
    )

    return fig

# Main execution
print("\n" + "="*60)
print("FX FAIR VALUE ANALYSIS - EQUITY SECTOR MODEL")
print("="*60)

# Load FX data
df_fx = load_fx_data()

# Define currencies to analyze (all available in the FX dataset)
currencies_to_analyze = [col for col in df_fx.columns if col not in ['Date']]
print(f"\nAnalyzing {len(currencies_to_analyze)} currencies...")

# Calculate fair values for all currencies
results = {}
fair_value_summary = []

for currency in currencies_to_analyze:
    print(f"Processing {currency}...")
    result = calculate_fair_value(currency, indexed_df, df_fx, ratios)

    if result:
        results[currency] = result
        fair_value_summary.append({
            'Currency': currency,
            'Current_Rate': result['current_actual'],
            'Fair_Value': result['current_predicted'],
            'Residual_%': result['current_residual'],
            'R_Squared': result['r_squared'],
            'Num_Predictors': len(result['selected_vars'])
        })

# Create summary DataFrame
summary_df = pd.DataFrame(fair_value_summary)
summary_df = summary_df.sort_values('Residual_%', ascending=True)

print(f"\n{'='*60}")
print(f"Successfully processed {len(summary_df)} currencies")
print(f"{'='*60}\n")
print("Fair Value Summary (sorted by residual):")
print(summary_df.to_string(index=False))
print()

# Generate all dashboard files
print("\n" + "="*60)
print("GENERATING DASHBOARD FILES")
print("="*60)

# Main overview dashboard
main_fig = create_main_dashboard(summary_df)
config = {'displayModeBar': False, 'responsive': True}

# Save main dashboard
pyo.plot(main_fig, filename="fx_equity_sector_fair_value.html", auto_open=False, config=config)
print("✓ Main dashboard saved: fx_equity_sector_fair_value.html")

# Create individual currency pages
for currency, result in results.items():
    individual_fig = create_individual_currency_page(currency, result)
    filename = f"{currency.lower()}_analysis.html"
    pyo.plot(individual_fig, filename=filename, auto_open=False, config=config)
    print(f"✓ {currency} analysis saved: {filename}")

# Save summary data
summary_data = {
    'last_updated': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'currencies_analyzed': len(summary_df),
    'fair_value_summary': summary_df.to_dict('records'),
    'individual_pages': [f"{currency.lower()}_analysis.html" for currency in results.keys()],
    'methodology': {
        'model': 'LASSO Regression',
        'predictors': 'Equity Sector ETF Performance Ratios',
        'sectors': equity_sector_focus,
        'alpha': 0.9,
        'train_test_split': '70/30'
    }
}

with open('fx_equity_sector_data.json', 'w') as f:
    json.dump(summary_data, f, indent=2)

print(f"\n✓ Summary data saved: fx_equity_sector_data.json")
print(f"\n{'='*60}")
print("DASHBOARD GENERATION COMPLETE")
print(f"{'='*60}")
print(f"\nMain Dashboard: fx_equity_sector_fair_value.html")
print(f"Currency Pages: {len(results)} individual analysis pages")
print(f"Data Export: fx_equity_sector_data.json")
print()
