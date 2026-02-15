"""
Multi-Channel Advertising Performance Dashboard
FINAL VERSION - Platform-Specific Insights

Author: Sathvik
Date: February 2025
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from google.cloud import bigquery
from google.oauth2 import service_account
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Multi-Channel Advertising Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

PLATFORM_COLORS = {
    'Facebook': '#6A176E',
    'Google': '#DD513A',
    'TikTok': '#FCA50A'
}

BOX_COLOR = '#1a1a2e'

st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
        background-color: #0f0f1e !important;
        color: white !important;
    }
    
    [data-testid="stSidebar"] {
        background-color: #16213e !important;
    }
    
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    
    .stMetric {
        background-color: #1a1a2e !important;
        color: white !important;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.3);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .stMetric label {
        color: white !important;
        font-size: 14px !important;
        font-weight: 600 !important;
    }
    
    .stMetric [data-testid="stMetricValue"] {
        color: white !important;
        font-size: 28px !important;
        font-weight: bold !important;
    }
    
    .stMetric [data-testid="stMetricDelta"] {
        color: #FCA50A !important;
    }
    
    h1, h2, h3, h4, h5, h6, p, span, div, label, li, td, th {
        color: white !important;
    }
    
    h1, h2, h3 {
        color: white !important;
        font-weight: 600 !important;
    }
    
    .stButton button {
        background-color: #1a1a2e;
        color: white !important;
        border: 1px solid rgba(255, 255, 255, 0.2);
        padding: 10px 24px;
        border-radius: 5px;
        font-weight: 600;
    }
    
    .dataframe {
        background-color: rgba(26, 26, 46, 0.8) !important;
        color: white !important;
    }
    
    .dataframe th {
        background-color: #1a1a2e !important;
        color: white !important;
        font-weight: 600 !important;
    }
    
    .dataframe td {
        color: white !important;
        background-color: rgba(26, 26, 46, 0.5) !important;
    }
    
    .streamlit-expanderHeader {
        background-color: #1a1a2e !important;
        color: white !important;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    hr {
        border-color: rgba(255, 255, 255, 0.2) !important;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("Multi-Channel Advertising Performance Dashboard")
st.markdown("### Unified Analytics for Facebook, Google & TikTok")
st.markdown("---")

# ============================================================================
# BIGQUERY
# ============================================================================

@st.cache_resource
def get_bigquery_client():
    try:
        # For Streamlit Cloud deployment - uses secrets
        credentials = service_account.Credentials.from_service_account_info(
            st.secrets["gcp_service_account"]
        )
        client = bigquery.Client(credentials=credentials)
        return client
    except Exception as e:
        st.error(f"Authentication failed: {e}")
        st.error("Make sure you've added GCP credentials to Streamlit secrets")
        st.stop()

@st.cache_data(ttl=600)
def load_data():
    client = get_bigquery_client()
    
    # Get project ID dynamically from secrets
    project_id = st.secrets["gcp_service_account"]["project_id"]
    
    query = f"""
    SELECT 1 for test
    """
    try:
        df = client.query(query).to_dataframe()
        df['date'] = pd.to_datetime(df['date'])
        return df
    except Exception as e:
        st.error(f"Error: {e}")
        st.stop()

def calculate_metrics(df):
    df = df.copy()
    df['ctr'] = np.where(df['impressions'] > 0, (df['clicks'] / df['impressions']) * 100, 0)
    df['cpc'] = np.where(df['clicks'] > 0, df['cost'] / df['clicks'], 0)
    df['cpm'] = np.where(df['impressions'] > 0, (df['cost'] / df['impressions']) * 1000, 0)
    df['cpa'] = np.where(df['conversions'] > 0, df['cost'] / df['conversions'], 0)
    df['conversion_rate'] = np.where(df['clicks'] > 0, (df['conversions'] / df['clicks']) * 100, 0)
    df['roas'] = np.where(df['cost'] > 0, df['conversion_value'] / df['cost'], 0)
    return df

def aggregate_platform_metrics(df):
    platform_agg = df.groupby('platform').agg({
        'impressions': 'sum', 'clicks': 'sum', 'cost': 'sum', 'conversions': 'sum',
        'video_views': 'sum', 'reach': 'sum', 'conversion_value': 'sum'
    }).reset_index()
    platform_agg['ctr'] = (platform_agg['clicks'] / platform_agg['impressions'] * 100)
    platform_agg['cpc'] = (platform_agg['cost'] / platform_agg['clicks'])
    platform_agg['cpm'] = (platform_agg['cost'] / platform_agg['impressions'] * 1000)
    platform_agg['cpa'] = (platform_agg['cost'] / platform_agg['conversions'])
    platform_agg['conversion_rate'] = (platform_agg['conversions'] / platform_agg['clicks'] * 100)
    platform_agg['roas'] = (platform_agg['conversion_value'] / platform_agg['cost'])
    return platform_agg

# ============================================================================
# LOAD DATA
# ============================================================================

with st.spinner('Loading data from BigQuery...'):
    df_raw = load_data()
    df = calculate_metrics(df_raw)
    st.success(f"Data loaded: {len(df):,} records | {df['platform'].nunique()} platforms | {df['date'].nunique()} days")

# ============================================================================
# SIDEBAR
# ============================================================================

st.sidebar.header("Filters & Controls")
st.sidebar.markdown("---")

min_date, max_date = df['date'].min().date(), df['date'].max().date()

col1, col2 = st.sidebar.columns(2)
with col1:
    start_date = st.date_input("Start Date", value=min_date, min_value=min_date, max_value=max_date)
with col2:
    end_date = st.date_input("End Date", value=max_date, min_value=min_date, max_value=max_date)

platforms = st.sidebar.multiselect("Platforms", options=sorted(df['platform'].unique()), default=sorted(df['platform'].unique()))
campaigns = st.sidebar.multiselect("Campaigns", options=sorted(df['campaign_name'].unique()), default=[])


# APPLY FILTERS
df_filtered = df[(df['date'].dt.date >= start_date) & (df['date'].dt.date <= end_date)].copy()
if platforms:
    df_filtered = df_filtered[df_filtered['platform'].isin(platforms)]
if campaigns:
    df_filtered = df_filtered[df_filtered['campaign_name'].isin(campaigns)]
if len(df_filtered) == 0:
    st.warning("No data")
    st.stop()

platform_metrics = aggregate_platform_metrics(df_filtered)

magma_layout = dict(
    paper_bgcolor='rgba(26, 26, 46, 0.3)',
    plot_bgcolor='rgba(15, 15, 30, 0.5)',
    font=dict(color='white', size=12, family='Arial'),
    title_font=dict(color='white', size=16),
    xaxis=dict(gridcolor='rgba(255, 255, 255, 0.1)', color='white'),
    yaxis=dict(gridcolor='rgba(255, 255, 255, 0.1)', color='white'),
    legend=dict(font=dict(color='white'))
)

# ============================================================================
# KPIs
# ============================================================================

st.markdown("## Key Performance Indicators")

total_impressions = df_filtered['impressions'].sum()
total_clicks = df_filtered['clicks'].sum()
total_cost = df_filtered['cost'].sum()
total_conversions = df_filtered['conversions'].sum()
total_revenue = df_filtered['conversion_value'].sum()

overall_ctr = (total_clicks / total_impressions * 100) if total_impressions > 0 else 0
overall_cpc = (total_cost / total_clicks) if total_clicks > 0 else 0
overall_cpa = (total_cost / total_conversions) if total_conversions > 0 else 0
overall_conversion_rate = (total_conversions / total_clicks * 100) if total_clicks > 0 else 0
overall_roas = (total_revenue / total_cost) if total_cost > 0 else 0

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("Total Spend", f"${total_cost:,.2f}", f"CPC: ${overall_cpc:.2f}")
with col2:
    st.metric("Impressions", f"{total_impressions:,.0f}", f"CTR: {overall_ctr:.2f}%")
with col3:
    st.metric("Clicks", f"{total_clicks:,.0f}", f"Conv: {overall_conversion_rate:.2f}%")
with col4:
    st.metric("Conversions", f"{total_conversions:,.0f}", f"CPA: ${overall_cpa:.2f}")
with col5:
    st.metric("ROAS", f"{overall_roas:.2f}x", f"Rev: ${total_revenue:,.0f}")

# ============================================================================
# PLATFORM PERFORMANCE
# ============================================================================

st.markdown("---")
st.markdown("## Platform Performance Comparison")

col1, col2, col3 = st.columns(3)

with col1:
    fig_spend = px.pie(platform_metrics, values='cost', names='platform',
        title='<b>Spend Distribution</b>', color='platform', color_discrete_map=PLATFORM_COLORS, hole=0.4)
    fig_spend.update_traces(textposition='inside', textinfo='percent+label', textfont=dict(size=14, color='white'))
    fig_spend.update_layout(**magma_layout)
    st.plotly_chart(fig_spend, use_container_width=True)

with col2:
    fig_conv = px.pie(platform_metrics, values='conversions', names='platform',
        title='<b>Conversions Distribution</b>', color='platform', color_discrete_map=PLATFORM_COLORS, hole=0.4)
    fig_conv.update_traces(textposition='inside', textinfo='percent+label', textfont=dict(size=14, color='white'))
    fig_conv.update_layout(**magma_layout)
    st.plotly_chart(fig_conv, use_container_width=True)

with col3:
    fig_clicks = px.pie(platform_metrics, values='clicks', names='platform',
        title='<b>Clicks Distribution</b>', color='platform', color_discrete_map=PLATFORM_COLORS, hole=0.4)
    fig_clicks.update_traces(textposition='inside', textinfo='percent+label', textfont=dict(size=14, color='white'))
    fig_clicks.update_layout(**magma_layout)
    st.plotly_chart(fig_clicks, use_container_width=True)

col1, col2 = st.columns(2)

with col1:
    fig_cpa = px.bar(platform_metrics.sort_values('cpa'), x='platform', y='cpa',
        title='<b>Cost Per Acquisition (CPA)</b>', color='platform', color_discrete_map=PLATFORM_COLORS, text='cpa')
    fig_cpa.update_traces(texttemplate='$%{text:.2f}', textposition='inside', 
        textfont=dict(size=16, color='white'), insidetextanchor='middle')
    fig_cpa.update_layout(**magma_layout, showlegend=False, yaxis_title='CPA ($)')
    st.plotly_chart(fig_cpa, use_container_width=True)

with col2:
    fig_ctr = px.bar(platform_metrics.sort_values('ctr', ascending=False), x='platform', y='ctr',
        title='<b>Click-Through Rate (CTR)</b>', color='platform', color_discrete_map=PLATFORM_COLORS, text='ctr')
    fig_ctr.update_traces(texttemplate='%{text:.2f}%', textposition='inside',
        textfont=dict(size=16, color='white'), insidetextanchor='middle')
    fig_ctr.update_layout(**magma_layout, showlegend=False, yaxis_title='CTR (%)')
    st.plotly_chart(fig_ctr, use_container_width=True)

st.markdown("### Platform Metrics Summary")
platform_display = platform_metrics.copy().round(2)
platform_display = platform_display[['platform', 'impressions', 'clicks', 'cost', 'conversions', 'ctr', 'cpc', 'cpm', 'cpa', 'conversion_rate', 'roas']]

platform_display['impressions'] = platform_display['impressions'].apply(lambda x: f"{x:,.0f}")
platform_display['clicks'] = platform_display['clicks'].apply(lambda x: f"{x:,.0f}")
platform_display['cost'] = platform_display['cost'].apply(lambda x: f"${x:,.2f}")
platform_display['conversions'] = platform_display['conversions'].apply(lambda x: f"{x:,.0f}")
platform_display['ctr'] = platform_display['ctr'].apply(lambda x: f"{x:.2f}%")
platform_display['cpc'] = platform_display['cpc'].apply(lambda x: f"${x:.2f}")
platform_display['cpm'] = platform_display['cpm'].apply(lambda x: f"${x:.2f}")
platform_display['cpa'] = platform_display['cpa'].apply(lambda x: f"${x:.2f}")
platform_display['conversion_rate'] = platform_display['conversion_rate'].apply(lambda x: f"{x:.2f}%")
platform_display['roas'] = platform_display['roas'].apply(lambda x: f"{x:.2f}x")

platform_display.columns = ['Platform', 'Impressions', 'Clicks', 'Total Spend', 'Conversions', 'CTR', 'CPC', 'CPM', 'CPA', 'Conv. Rate', 'ROAS']
st.dataframe(platform_display, use_container_width=True, hide_index=True)

# ============================================================================
# TIME SERIES
# ============================================================================

st.markdown("---")
st.markdown("## Performance Trends Over Time")

daily_metrics = df_filtered.groupby(['date', 'platform']).agg({
    'impressions': 'sum', 'clicks': 'sum', 'cost': 'sum', 'conversions': 'sum'
}).reset_index()

col1, col2 = st.columns([3, 1])
with col1:
    time_metric = st.selectbox("Select Metric", options=['cost', 'conversions', 'clicks', 'impressions'],
        format_func=lambda x: {'cost': 'Spend ($)', 'conversions': 'Conversions', 'clicks': 'Clicks', 'impressions': 'Impressions'}[x])
with col2:
    chart_type = st.radio("Chart Type", ["Line", "Area"], horizontal=True)

if chart_type == "Line":
    fig_time = px.line(daily_metrics, x='date', y=time_metric, color='platform', color_discrete_map=PLATFORM_COLORS,
        title=f'<b>{time_metric.capitalize()} Trend by Platform</b>', markers=True)
else:
    fig_time = px.area(daily_metrics, x='date', y=time_metric, color='platform', color_discrete_map=PLATFORM_COLORS,
        title=f'<b>{time_metric.capitalize()} Trend by Platform</b>')

fig_time.update_layout(**magma_layout, xaxis_title='Date', yaxis_title=time_metric.capitalize(), hovermode='x unified')
st.plotly_chart(fig_time, use_container_width=True)

# ============================================================================
# CONVERSION FUNNEL
# ============================================================================

st.markdown("---")
st.markdown("## Conversion Funnel Analysis")

col1, col2 = st.columns([2, 1])

with col1:
    funnel_data = pd.DataFrame({
        'Stage': ['Impressions', 'Clicks', 'Conversions'],
        'Count': [total_impressions, total_clicks, total_conversions]
    })
    
    fig_funnel = go.Figure(go.Funnel(
        y=funnel_data['Stage'], x=funnel_data['Count'],
        textposition="inside", textinfo="value+percent initial",
        marker=dict(color=['#6A176E', '#DD513A', '#FCA50A']),
        textfont=dict(size=16, color='white')
    ))
    fig_funnel.update_layout(**magma_layout, title='<b>Overall Conversion Funnel</b>', height=400)
    st.plotly_chart(fig_funnel, use_container_width=True)

with col2:
    st.markdown("### Funnel Metrics")
    
    drop_off_1 = (1 - total_clicks/total_impressions)*100 if total_impressions > 0 else 0
    drop_off_2 = (1 - total_conversions/total_clicks)*100 if total_clicks > 0 else 0
    overall_conv = (total_conversions/total_impressions*100) if total_impressions > 0 else 0
    
    for label, value, desc in [
        ("Drop-off Rate", f"{drop_off_1:.1f}%", "Impressions → Clicks"),
        ("Drop-off Rate", f"{drop_off_2:.1f}%", "Clicks → Conversions"),
        ("Overall Rate", f"{overall_conv:.3f}%", "End-to-End Conversion")
    ]:
        st.markdown(f"""
        <div style='background-color: {BOX_COLOR}; padding: 15px; border-radius: 10px; margin-bottom: 10px; 
                    border: 1px solid rgba(255, 255, 255, 0.2);'>
            <h4 style='color: white; margin: 0;'>{label}</h4>
            <p style='font-size: 24px; font-weight: bold; color: #FCA50A; margin: 5px 0;'>{value}</p>
            <p style='color: rgba(255, 255, 255, 0.8); margin: 0; font-size: 12px;'>{desc}</p>
        </div>
        """, unsafe_allow_html=True)

# ============================================================================
# CAMPAIGN PERFORMANCE
# ============================================================================

st.markdown("---")
st.markdown("## Campaign Performance Analysis")

campaign_metrics = df_filtered.groupby(['platform', 'campaign_name']).agg({
    'impressions': 'sum', 'clicks': 'sum', 'cost': 'sum', 'conversions': 'sum'
}).reset_index()

campaign_metrics['ctr'] = (campaign_metrics['clicks'] / campaign_metrics['impressions'] * 100)
campaign_metrics['cpa'] = campaign_metrics['cost'] / campaign_metrics['conversions']
campaign_metrics['conversion_rate'] = (campaign_metrics['conversions'] / campaign_metrics['clicks'] * 100)

campaign_metrics = campaign_metrics[campaign_metrics['conversions'] > 0].sort_values('conversions', ascending=False)

num_campaigns = st.slider("Number of Campaigns to Display", 5, 20, 10)
top_campaigns = campaign_metrics.head(num_campaigns)

fig_campaigns = px.bar(top_campaigns, x='campaign_name', y='conversions', color='platform',
    color_discrete_map=PLATFORM_COLORS, title=f'<b>Top {num_campaigns} Campaigns by Conversions</b>',
    text='conversions', hover_data=['cost', 'cpa', 'ctr'])
fig_campaigns.update_traces(texttemplate='%{text:,.0f}', textposition='inside',
    textfont=dict(size=13, color='white'), insidetextanchor='middle')
fig_campaigns.update_layout(**magma_layout, xaxis_title='Campaign', yaxis_title='Total Conversions', xaxis_tickangle=-45, height=500)
st.plotly_chart(fig_campaigns, use_container_width=True)

st.markdown("### Campaign Efficiency Analysis")

fig_scatter = px.scatter(campaign_metrics, x='cost', y='conversions', size='clicks',
    color='platform', color_discrete_map=PLATFORM_COLORS, hover_name='campaign_name',
    title='<b>Campaign Efficiency: Cost vs Conversions</b>')
fig_scatter.update_layout(**magma_layout, height=500)
st.plotly_chart(fig_scatter, use_container_width=True)

st.markdown(f"### Top {num_campaigns} Campaign Details")
campaign_display = top_campaigns.copy().round(2)

campaign_display['impressions'] = campaign_display['impressions'].apply(lambda x: f"{x:,.0f}")
campaign_display['clicks'] = campaign_display['clicks'].apply(lambda x: f"{x:,.0f}")
campaign_display['cost'] = campaign_display['cost'].apply(lambda x: f"${x:,.2f}")
campaign_display['conversions'] = campaign_display['conversions'].apply(lambda x: f"{x:,.0f}")
campaign_display['ctr'] = campaign_display['ctr'].apply(lambda x: f"{x:.2f}%")
campaign_display['cpa'] = campaign_display['cpa'].apply(lambda x: f"${x:.2f}")
campaign_display['conversion_rate'] = campaign_display['conversion_rate'].apply(lambda x: f"{x:.2f}%")

campaign_display.columns = ['Platform', 'Campaign', 'Impressions', 'Clicks', 'Total Spend', 'Conversions', 'CTR', 'CPA', 'Conv. Rate']
st.dataframe(campaign_display, use_container_width=True, hide_index=True)

# ============================================================================
# DAY OF WEEK
# ============================================================================

st.markdown("---")
st.markdown("## Day of Week Performance")

df_filtered_dow = df_filtered.copy()
df_filtered_dow['day_of_week'] = df_filtered_dow['date'].dt.day_name()
df_filtered_dow['day_num'] = df_filtered_dow['date'].dt.dayofweek

dow_metrics = df_filtered_dow.groupby(['day_of_week', 'day_num', 'platform']).agg({
    'conversions': 'sum', 'cost': 'sum'
}).reset_index()

dow_metrics = dow_metrics.sort_values('day_num')
dow_metrics['cpa'] = dow_metrics['cost'] / dow_metrics['conversions']

col1, col2 = st.columns(2)

with col1:
    fig_dow_conv = px.bar(dow_metrics, x='day_of_week', y='conversions', color='platform',
        color_discrete_map=PLATFORM_COLORS, title='<b>Conversions by Day of Week</b>', 
        barmode='group', text='conversions')
    fig_dow_conv.update_traces(texttemplate='%{text:,.0f}', textposition='inside',
        textfont=dict(size=11, color='white'))
    fig_dow_conv.update_layout(**magma_layout, xaxis_title='Day of Week', yaxis_title='Conversions')
    st.plotly_chart(fig_dow_conv, use_container_width=True)

with col2:
    fig_dow_cpa = px.line(dow_metrics, x='day_of_week', y='cpa', color='platform',
        color_discrete_map=PLATFORM_COLORS, title='<b>CPA by Day of Week</b>', markers=True)
    fig_dow_cpa.update_layout(**magma_layout, xaxis_title='Day of Week', yaxis_title='CPA ($)')
    st.plotly_chart(fig_dow_cpa, use_container_width=True)

# ============================================================================
# EFFICIENCY METRICS
# ============================================================================

st.markdown("---")
st.markdown("## Efficiency Metrics Dashboard")

col1, col2 = st.columns(2)

with col1:
    fig_cpc = px.bar(platform_metrics.sort_values('cpc'), x='platform', y='cpc',
        title='<b>Cost Per Click (CPC)</b>', color='platform', color_discrete_map=PLATFORM_COLORS, text='cpc')
    fig_cpc.update_traces(texttemplate='$%{text:.2f}', textposition='inside',
        textfont=dict(size=16, color='white'), insidetextanchor='middle')
    fig_cpc.update_layout(**magma_layout, showlegend=False, yaxis_title='CPC ($)')
    st.plotly_chart(fig_cpc, use_container_width=True)
    
    fig_cpm = px.bar(platform_metrics.sort_values('cpm'), x='platform', y='cpm',
        title='<b>Cost Per Thousand Impressions (CPM)</b>', color='platform', color_discrete_map=PLATFORM_COLORS, text='cpm')
    fig_cpm.update_traces(texttemplate='$%{text:.2f}', textposition='inside',
        textfont=dict(size=16, color='white'), insidetextanchor='middle')
    fig_cpm.update_layout(**magma_layout, showlegend=False, yaxis_title='CPM ($)')
    st.plotly_chart(fig_cpm, use_container_width=True)

with col2:
    fig_conv_rate = px.bar(platform_metrics.sort_values('conversion_rate', ascending=False), x='platform', y='conversion_rate',
        title='<b>Conversion Rate</b>', color='platform', color_discrete_map=PLATFORM_COLORS, text='conversion_rate')
    fig_conv_rate.update_traces(texttemplate='%{text:.2f}%', textposition='inside',
        textfont=dict(size=16, color='white'), insidetextanchor='middle')
    fig_conv_rate.update_layout(**magma_layout, showlegend=False, yaxis_title='Conversion Rate (%)')
    st.plotly_chart(fig_conv_rate, use_container_width=True)
    
    fig_roas = px.bar(platform_metrics.sort_values('roas', ascending=False), x='platform', y='roas',
        title='<b>Return on Ad Spend (ROAS)</b>', color='platform', color_discrete_map=PLATFORM_COLORS, text='roas')
    fig_roas.update_traces(texttemplate='%{text:.2f}x', textposition='inside',
        textfont=dict(size=16, color='white'), insidetextanchor='middle')
    fig_roas.update_layout(**magma_layout, showlegend=False, yaxis_title='ROAS')
    st.plotly_chart(fig_roas, use_container_width=True)

# ============================================================================
# PLATFORM-SPECIFIC INSIGHTS
# ============================================================================

st.markdown("---")
st.markdown("## Platform-Specific Insights")

# Prepare data for insights
df_filtered['week_start'] = df_filtered['date'].dt.to_period('W').apply(lambda r: r.start_time)

# Daily by platform
daily_by_platform = df_filtered.groupby(['platform', 'date']).agg({'conversions': 'sum', 'cost': 'sum'}).reset_index()

# Weekly by platform
weekly_by_platform = df_filtered.groupby(['platform', 'week_start']).agg({'conversions': 'sum', 'cost': 'sum'}).reset_index()

col1, col2, col3 = st.columns(3)

# FACEBOOK
with col1:
    st.markdown("### 📘 Facebook")
    
    fb_daily = daily_by_platform[daily_by_platform['platform'] == 'Facebook']
    fb_weekly = weekly_by_platform[weekly_by_platform['platform'] == 'Facebook']
    
    if len(fb_daily) > 0:
        best_day = fb_daily.nlargest(1, 'conversions')
        st.markdown(f"""
        <div style='background-color: #6A176E; padding: 15px; border-radius: 10px; margin-bottom: 10px; border: 2px solid rgba(255, 255, 255, 0.3);'>
            <p style='color: white; margin: 0; font-weight: 600; font-size: 13px;'>📅 Best Day</p>
            <p style='font-size: 20px; font-weight: bold; color: white; margin: 5px 0;'>{pd.Timestamp(best_day['date'].values[0]).strftime('%A, %b %d')}</p>
            <p style='color: rgba(255, 255, 255, 0.9); margin: 0; font-size: 13px;'>{best_day['conversions'].values[0]:,.0f} conversions</p>
        </div>
        """, unsafe_allow_html=True)
        
        worst_day = fb_daily.nsmallest(1, 'conversions')
        st.markdown(f"""
        <div style='background-color: #6A176E; padding: 15px; border-radius: 10px; margin-bottom: 10px; border: 2px solid rgba(255, 255, 255, 0.3);'>
            <p style='color: white; margin: 0; font-weight: 600; font-size: 13px;'>📉 Lowest Day</p>
            <p style='font-size: 20px; font-weight: bold; color: white; margin: 5px 0;'>{pd.Timestamp(worst_day['date'].values[0]).strftime('%A, %b %d')}</p>
            <p style='color: rgba(255, 255, 255, 0.9); margin: 0; font-size: 13px;'>{worst_day['conversions'].values[0]:,.0f} conversions</p>
        </div>
        """, unsafe_allow_html=True)
        
        best_week = fb_weekly.nlargest(1, 'conversions')
        st.markdown(f"""
        <div style='background-color: #6A176E; padding: 15px; border-radius: 10px; margin-bottom: 10px; border: 2px solid rgba(255, 255, 255, 0.3);'>
            <p style='color: white; margin: 0; font-weight: 600; font-size: 13px;'>📈 Best Week</p>
            <p style='font-size: 20px; font-weight: bold; color: white; margin: 5px 0;'>Week of {pd.Timestamp(best_week['week_start'].values[0]).strftime('%b %d')}</p>
            <p style='color: rgba(255, 255, 255, 0.9); margin: 0; font-size: 13px;'>{best_week['conversions'].values[0]:,.0f} conversions</p>
        </div>
        """, unsafe_allow_html=True)
        
        worst_week = fb_weekly.nsmallest(1, 'conversions')
        st.markdown(f"""
        <div style='background-color: #6A176E; padding: 15px; border-radius: 10px; border: 2px solid rgba(255, 255, 255, 0.3);'>
            <p style='color: white; margin: 0; font-weight: 600; font-size: 13px;'>📉 Lowest Week</p>
            <p style='font-size: 20px; font-weight: bold; color: white; margin: 5px 0;'>Week of {pd.Timestamp(worst_week['week_start'].values[0]).strftime('%b %d')}</p>
            <p style='color: rgba(255, 255, 255, 0.9); margin: 0; font-size: 13px;'>{worst_week['conversions'].values[0]:,.0f} conversions</p>
        </div>
        """, unsafe_allow_html=True)

# GOOGLE - RED/ORANGE-RED THEME
with col2:
    st.markdown("### 🔴 Google")
    
    g_daily = daily_by_platform[daily_by_platform['platform'] == 'Google']
    g_weekly = weekly_by_platform[weekly_by_platform['platform'] == 'Google']
    
    if len(g_daily) > 0:
        best_day = g_daily.nlargest(1, 'conversions')
        st.markdown(f"""
        <div style='background-color: #DD513A; padding: 15px; border-radius: 10px; margin-bottom: 10px; border: 2px solid rgba(255, 255, 255, 0.3);'>
            <p style='color: white; margin: 0; font-weight: 600; font-size: 13px;'>📅 Best Day</p>
            <p style='font-size: 20px; font-weight: bold; color: white; margin: 5px 0;'>{pd.Timestamp(best_day['date'].values[0]).strftime('%A, %b %d')}</p>
            <p style='color: rgba(255, 255, 255, 0.9); margin: 0; font-size: 13px;'>{best_day['conversions'].values[0]:,.0f} conversions</p>
        </div>
        """, unsafe_allow_html=True)
        
        worst_day = g_daily.nsmallest(1, 'conversions')
        st.markdown(f"""
        <div style='background-color: #DD513A; padding: 15px; border-radius: 10px; margin-bottom: 10px; border: 2px solid rgba(255, 255, 255, 0.3);'>
            <p style='color: white; margin: 0; font-weight: 600; font-size: 13px;'>📉 Lowest Day</p>
            <p style='font-size: 20px; font-weight: bold; color: white; margin: 5px 0;'>{pd.Timestamp(worst_day['date'].values[0]).strftime('%A, %b %d')}</p>
            <p style='color: rgba(255, 255, 255, 0.9); margin: 0; font-size: 13px;'>{worst_day['conversions'].values[0]:,.0f} conversions</p>
        </div>
        """, unsafe_allow_html=True)
        
        best_week = g_weekly.nlargest(1, 'conversions')
        st.markdown(f"""
        <div style='background-color: #DD513A; padding: 15px; border-radius: 10px; margin-bottom: 10px; border: 2px solid rgba(255, 255, 255, 0.3);'>
            <p style='color: white; margin: 0; font-weight: 600; font-size: 13px;'>📈 Best Week</p>
            <p style='font-size: 20px; font-weight: bold; color: white; margin: 5px 0;'>Week of {pd.Timestamp(best_week['week_start'].values[0]).strftime('%b %d')}</p>
            <p style='color: rgba(255, 255, 255, 0.9); margin: 0; font-size: 13px;'>{best_week['conversions'].values[0]:,.0f} conversions</p>
        </div>
        """, unsafe_allow_html=True)
        
        worst_week = g_weekly.nsmallest(1, 'conversions')
        st.markdown(f"""
        <div style='background-color: #DD513A; padding: 15px; border-radius: 10px; border: 2px solid rgba(255, 255, 255, 0.3);'>
            <p style='color: white; margin: 0; font-weight: 600; font-size: 13px;'>📉 Lowest Week</p>
            <p style='font-size: 20px; font-weight: bold; color: white; margin: 5px 0;'>Week of {pd.Timestamp(worst_week['week_start'].values[0]).strftime('%b %d')}</p>
            <p style='color: rgba(255, 255, 255, 0.9); margin: 0; font-size: 13px;'>{worst_week['conversions'].values[0]:,.0f} conversions</p>
        </div>
        """, unsafe_allow_html=True)

# TIKTOK - YELLOW/ORANGE THEME
with col3:
    st.markdown("### 🎵 TikTok")
    
    tt_daily = daily_by_platform[daily_by_platform['platform'] == 'TikTok']
    tt_weekly = weekly_by_platform[weekly_by_platform['platform'] == 'TikTok']
    
    if len(tt_daily) > 0:
        best_day = tt_daily.nlargest(1, 'conversions')
        st.markdown(f"""
        <div style='background-color: #FCA50A; padding: 15px; border-radius: 10px; margin-bottom: 10px; border: 2px solid rgba(255, 255, 255, 0.3);'>
            <p style='color: #1a1a2e; margin: 0; font-weight: 600; font-size: 13px;'>📅 Best Day</p>
            <p style='font-size: 20px; font-weight: bold; color: #1a1a2e; margin: 5px 0;'>{pd.Timestamp(best_day['date'].values[0]).strftime('%A, %b %d')}</p>
            <p style='color: #1a1a2e; margin: 0; font-size: 13px; opacity: 0.9;'>{best_day['conversions'].values[0]:,.0f} conversions</p>
        </div>
        """, unsafe_allow_html=True)
        
        worst_day = tt_daily.nsmallest(1, 'conversions')
        st.markdown(f"""
        <div style='background-color: #FCA50A; padding: 15px; border-radius: 10px; margin-bottom: 10px; border: 2px solid rgba(255, 255, 255, 0.3);'>
            <p style='color: #1a1a2e; margin: 0; font-weight: 600; font-size: 13px;'>📉 Lowest Day</p>
            <p style='font-size: 20px; font-weight: bold; color: #1a1a2e; margin: 5px 0;'>{pd.Timestamp(worst_day['date'].values[0]).strftime('%A, %b %d')}</p>
            <p style='color: #1a1a2e; margin: 0; font-size: 13px; opacity: 0.9;'>{worst_day['conversions'].values[0]:,.0f} conversions</p>
        </div>
        """, unsafe_allow_html=True)
        
        best_week = tt_weekly.nlargest(1, 'conversions')
        st.markdown(f"""
        <div style='background-color: #FCA50A; padding: 15px; border-radius: 10px; margin-bottom: 10px; border: 2px solid rgba(255, 255, 255, 0.3);'>
            <p style='color: #1a1a2e; margin: 0; font-weight: 600; font-size: 13px;'>📈 Best Week</p>
            <p style='font-size: 20px; font-weight: bold; color: #1a1a2e; margin: 5px 0;'>Week of {pd.Timestamp(best_week['week_start'].values[0]).strftime('%b %d')}</p>
            <p style='color: #1a1a2e; margin: 0; font-size: 13px; opacity: 0.9;'>{best_week['conversions'].values[0]:,.0f} conversions</p>
        </div>
        """, unsafe_allow_html=True)
        
        worst_week = tt_weekly.nsmallest(1, 'conversions')
        st.markdown(f"""
        <div style='background-color: #FCA50A; padding: 15px; border-radius: 10px; border: 2px solid rgba(255, 255, 255, 0.3);'>
            <p style='color: #1a1a2e; margin: 0; font-weight: 600; font-size: 13px;'>📉 Lowest Week</p>
            <p style='font-size: 20px; font-weight: bold; color: #1a1a2e; margin: 5px 0;'>Week of {pd.Timestamp(worst_week['week_start'].values[0]).strftime('%b %d')}</p>
            <p style='color: #1a1a2e; margin: 0; font-size: 13px; opacity: 0.9;'>{worst_week['conversions'].values[0]:,.0f} conversions</p>
        </div>
        """, unsafe_allow_html=True)


# ============================================================================
# OVERALL INSIGHTS
# ============================================================================

st.markdown("---")
st.markdown("## Overall Insights (All Platforms Combined)")

# Add day_of_week column if not already present
if 'day_of_week' not in df_filtered.columns:
    df_filtered['day_of_week'] = df_filtered['date'].dt.day_name()

# Overall day of week
overall_dow = df_filtered.groupby('day_of_week').agg({'conversions': 'sum', 'cost': 'sum'}).reset_index()
overall_dow['cpa'] = overall_dow['cost'] / overall_dow['conversions']

col1, col2, col3 = st.columns(3)

with col1:
    best_day_overall = overall_dow.nlargest(1, 'conversions')
    st.markdown(f"""
    <div style='background-color: {BOX_COLOR}; padding: 15px; border-radius: 10px; border: 1px solid rgba(255, 255, 255, 0.2);'>
        <p style='color: #FCA50A; margin: 0; font-weight: 600; font-size: 13px;'>📅 Highest Volume</p>
        <p style='font-size: 22px; font-weight: bold; color: white; margin: 5px 0;'>{best_day_overall['day_of_week'].values[0]}</p>
        <p style='color: rgba(255, 255, 255, 0.9); margin: 0; font-size: 13px;'>{best_day_overall['conversions'].values[0]:,.0f} conversions</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    most_efficient = overall_dow.nsmallest(1, 'cpa')
    st.markdown(f"""
    <div style='background-color: {BOX_COLOR}; padding: 15px; border-radius: 10px; border: 1px solid rgba(255, 255, 255, 0.2);'>
        <p style='color: #FCA50A; margin: 0; font-weight: 600; font-size: 13px;'>💰 Most Cost-Effective</p>
        <p style='font-size: 22px; font-weight: bold; color: white; margin: 5px 0;'>{most_efficient['day_of_week'].values[0]}</p>
        <p style='color: rgba(255, 255, 255, 0.9); margin: 0; font-size: 13px;'>${most_efficient['cpa'].values[0]:.2f} CPA</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    worst_day_overall = overall_dow.nsmallest(1, 'conversions')
    st.markdown(f"""
    <div style='background-color: {BOX_COLOR}; padding: 15px; border-radius: 10px; border: 1px solid rgba(255, 255, 255, 0.2);'>
        <p style='color: #FCA50A; margin: 0; font-weight: 600; font-size: 13px;'>⚠️ Lowest Volume</p>
        <p style='font-size: 22px; font-weight: bold; color: white; margin: 5px 0;'>{worst_day_overall['day_of_week'].values[0]}</p>
        <p style='color: rgba(255, 255, 255, 0.9); margin: 0; font-size: 13px;'>{worst_day_overall['conversions'].values[0]:,.0f} conversions</p>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# DATA EXPLORER
# ============================================================================

st.markdown("---")
st.markdown("## Raw Data Explorer")

with st.expander("View and Download Data"):
    available_columns = ['date', 'platform', 'campaign_name', 'impressions', 'clicks', 'cost', 
                        'conversions', 'ctr', 'cpc', 'cpm', 'cpa', 'conversion_rate', 'roas']
    
    default_columns = ['date', 'platform', 'campaign_name', 'impressions', 'clicks', 'cost', 'conversions', 'ctr', 'cpa']
    
    selected_columns = st.multiselect("Select Columns", options=available_columns, default=default_columns)
    
    if selected_columns:
        export_df = df_filtered[selected_columns].copy()
        numeric_cols = export_df.select_dtypes(include=[np.number]).columns
        export_df[numeric_cols] = export_df[numeric_cols].round(2)
        export_df = export_df.sort_values('date', ascending=False)
        
        st.dataframe(export_df, use_container_width=True, hide_index=True, height=400)
        
        col1, col2 = st.columns(2)
        with col1:
            csv = export_df.to_csv(index=False)
            st.download_button("Download as CSV", data=csv, 
                file_name=f"advertising_data_{start_date}_{end_date}.csv", mime="text/csv")
        with col2:
            st.metric("Total Rows", f"{len(export_df):,}")


# SIDEBAR INFO
st.sidebar.markdown("---")
st.sidebar.info(f"""
**Data Summary:**
- Records: {len(df_filtered):,}
- Days: {(end_date - start_date).days + 1}
- Platforms: {len(platforms)}
- Campaigns: {df_filtered['campaign_name'].nunique()}
""")

if st.sidebar.button("Reload from BigQuery"):
    st.cache_data.clear()

    st.rerun()


