#!/usr/bin/env python3
"""
VLM Analysis Dashboard - Streamlit UI
A standalone dashboard for model comparison and cost analysis
Reads from MLflow without making any changes
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import os
from datetime import datetime, timedelta
import time
from pathlib import Path

# Configure Streamlit page
st.set_page_config(
    page_title="VLM Analysis Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .success-metric {
        border-left-color: #28a745;
    }
    .warning-metric {
        border-left-color: #ffc107;
    }
    .danger-metric {
        border-left-color: #dc3545;
    }
    .sidebar-section {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_mlflow_connection():
    """Setup MLflow connection (read-only)"""
    try:
        import mlflow
        
        # Get credentials from environment or Streamlit secrets
        tracking_uri = os.environ.get("MLFLOW_TRACKING_URI") or st.secrets.get("MLFLOW_TRACKING_URI")
        username = os.environ.get("MLFLOW_TRACKING_USERNAME") or st.secrets.get("MLFLOW_TRACKING_USERNAME")
        password = os.environ.get("MLFLOW_TRACKING_PASSWORD") or st.secrets.get("MLFLOW_TRACKING_PASSWORD")
        
        if not all([tracking_uri, username, password]):
            return False, "Missing MLflow credentials"
        
        # Setup connection
        mlflow.set_tracking_uri(tracking_uri)
        os.environ["MLFLOW_TRACKING_USERNAME"] = username
        os.environ["MLFLOW_TRACKING_PASSWORD"] = password
        
        # Test connection
        experiments = mlflow.search_experiments()
        return True, f"Connected to MLflow ({len(experiments)} experiments found)"
        
    except Exception as e:
        return False, f"MLflow connection failed: {str(e)}"

@st.cache_data
def get_experiments():
    """Get all MLflow experiments"""
    try:
        import mlflow
        experiments = mlflow.search_experiments()
        
        exp_data = []
        for exp in experiments:
            exp_data.append({
                "name": exp.name,
                "experiment_id": exp.experiment_id,
                "creation_time": pd.to_datetime(exp.creation_time, unit='ms'),
                "last_update_time": pd.to_datetime(exp.last_update_time, unit='ms'),
                "lifecycle_stage": exp.lifecycle_stage
            })
        
        return pd.DataFrame(exp_data)
    except Exception as e:
        st.error(f"Error loading experiments: {e}")
        return pd.DataFrame()

@st.cache_data
def get_experiment_runs(experiment_id):
    """Get all runs for a specific experiment"""
    try:
        import mlflow
        runs = mlflow.search_runs(experiment_ids=[experiment_id])
        return runs
    except Exception as e:
        st.error(f"Error loading runs: {e}")
        return pd.DataFrame()

def simulate_model_comparison_data():
    """Generate realistic model comparison data for demonstration"""
    np.random.seed(42)
    
    # Simulate test samples
    n_samples = 50
    categories = ['Electronics', 'Clothing', 'Home & Garden', 'Sports', 'Books']
    
    data = {
        'sample_id': range(n_samples),
        'category': np.random.choice(categories, n_samples),
        'base_similarity': np.random.normal(0.65, 0.15, n_samples).clip(0, 1),
        'finetuned_similarity': np.random.normal(0.78, 0.12, n_samples).clip(0, 1),
        'ground_truth_length': np.random.normal(45, 15, n_samples).astype(int),
        'base_length': np.random.normal(38, 12, n_samples).astype(int),
        'finetuned_length': np.random.normal(47, 10, n_samples).astype(int)
    }
    
    df = pd.DataFrame(data)
    df['improvement'] = df['finetuned_similarity'] - df['base_similarity']
    df['improvement_pct'] = (df['improvement'] / df['base_similarity']) * 100
    
    return df

def simulate_cost_analysis_data():
    """Generate realistic cost analysis data for demonstration"""
    # Infrastructure costs
    training_hours = 4
    costs = {
        "compute": {
            "gpu_cost": 8 * 4.00 * training_hours,  # 8 H100s * $4/hour * 4 hours
            "cpu_cost": 128 * 0.05 * training_hours,
            "memory_cost": 1600 * 0.01 * training_hours
        },
        "storage": {
            "models_storage": 2000 * 0.10 / 30 / 24 * training_hours,
            "datasets_storage": 1000 * 0.10 / 30 / 24 * training_hours,
            "logs_storage": 100 * 0.10 / 30 / 24 * training_hours
        },
        "network": {
            "model_download": 50 * 0.05,
            "dataset_download": 20 * 0.05,
            "monitoring_data": 5 * 0.05
        }
    }
    
    # Calculate totals
    compute_total = sum(costs["compute"].values())
    storage_total = sum(costs["storage"].values())
    network_total = sum(costs["network"].values())
    grand_total = compute_total + storage_total + network_total
    
    # ROI scenarios
    roi_scenarios = {
        "E-commerce Automation": {
            "monthly_savings": 19900,  # (2.00 - 0.01) * 10000
            "payback_months": grand_total / 19900,
            "3_year_roi": (19900 * 12 * 3 - grand_total) / grand_total * 100
        },
        "Content Scaling": {
            "monthly_savings": 45000,  # 50000 - 5000
            "payback_months": grand_total / 45000,
            "3_year_roi": (45000 * 12 * 3 - grand_total) / grand_total * 100
        },
        "Personalization": {
            "monthly_revenue": 22500000,  # 1M visitors * 15% * $150
            "payback_months": grand_total / 22500000,
            "3_year_roi": (22500000 * 12 * 3 - grand_total) / grand_total * 100
        }
    }
    
    return {
        "costs": costs,
        "totals": {
            "compute_total": compute_total,
            "storage_total": storage_total,
            "network_total": network_total,
            "grand_total": grand_total
        },
        "roi_scenarios": roi_scenarios,
        "training_hours": training_hours
    }

def create_model_comparison_dashboard():
    """Create the model comparison dashboard"""
    st.markdown('<h2 style="color: #1f77b4;">🔍 Model Comparison Analysis</h2>', unsafe_allow_html=True)
    
    # Load comparison data
    comparison_data = simulate_model_comparison_data()
    
    # Key Metrics Row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_improvement = comparison_data['improvement'].mean()
        st.metric(
            "Avg Similarity Improvement",
            f"{avg_improvement:.3f}",
            f"{avg_improvement*100:.1f}% better"
        )
    
    with col2:
        improvement_rate = (comparison_data['improvement'] > 0).sum() / len(comparison_data) * 100
        st.metric(
            "Samples Improved",
            f"{improvement_rate:.0f}%",
            f"{(comparison_data['improvement'] > 0).sum()}/{len(comparison_data)} samples"
        )
    
    with col3:
        base_avg = comparison_data['base_similarity'].mean()
        finetuned_avg = comparison_data['finetuned_similarity'].mean()
        st.metric(
            "Base Model Avg",
            f"{base_avg:.3f}",
            f"vs {finetuned_avg:.3f} fine-tuned"
        )
    
    with col4:
        max_improvement = comparison_data['improvement'].max()
        st.metric(
            "Best Improvement",
            f"{max_improvement:.3f}",
            f"{max_improvement*100:.1f}% gain"
        )
    
    # Visualization Row 1
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Similarity Distribution Comparison")
        
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=comparison_data['base_similarity'],
            name='Base Model',
            opacity=0.7,
            nbinsx=20
        ))
        fig.add_trace(go.Histogram(
            x=comparison_data['finetuned_similarity'],
            name='Fine-tuned Model',
            opacity=0.7,
            nbinsx=20
        ))
        
        fig.update_layout(
            barmode='overlay',
            xaxis_title='Semantic Similarity to Ground Truth',
            yaxis_title='Frequency',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Per-Sample Improvement")
        
        # Create improvement chart
        improvement_colors = ['green' if x > 0 else 'red' for x in comparison_data['improvement']]
        
        fig = go.Figure(data=go.Bar(
            x=comparison_data['sample_id'],
            y=comparison_data['improvement'],
            marker_color=improvement_colors,
            name='Improvement'
        ))
        
        fig.update_layout(
            xaxis_title='Test Sample',
            yaxis_title='Similarity Improvement',
            height=400,
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Visualization Row 2
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Performance by Category")
        
        category_stats = comparison_data.groupby('category').agg({
            'base_similarity': 'mean',
            'finetuned_similarity': 'mean',
            'improvement': 'mean'
        }).round(3)
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            name='Base Model',
            x=category_stats.index,
            y=category_stats['base_similarity'],
            marker_color='lightblue'
        ))
        fig.add_trace(go.Bar(
            name='Fine-tuned Model',
            x=category_stats.index,
            y=category_stats['finetuned_similarity'],
            marker_color='darkblue'
        ))
        
        fig.update_layout(
            barmode='group',
            xaxis_title='Product Category',
            yaxis_title='Average Similarity',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Text Length Analysis")
        
        length_data = pd.DataFrame({
            'Base Model': comparison_data['base_length'],
            'Fine-tuned Model': comparison_data['finetuned_length'],
            'Ground Truth': comparison_data['ground_truth_length']
        })
        
        fig = go.Figure()
        for column in length_data.columns:
            fig.add_trace(go.Box(
                y=length_data[column],
                name=column,
                boxpoints='outliers'
            ))
        
        fig.update_layout(
            yaxis_title='Description Length (words)',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Detailed Data Table
    st.subheader("Detailed Results")
    
    # Add filters
    col1, col2 = st.columns(2)
    with col1:
        category_filter = st.multiselect(
            "Filter by Category",
            options=comparison_data['category'].unique(),
            default=comparison_data['category'].unique()
        )
    
    with col2:
        improvement_threshold = st.slider(
            "Min Improvement Threshold",
            min_value=float(comparison_data['improvement'].min()),
            max_value=float(comparison_data['improvement'].max()),
            value=0.0,
            step=0.001,
            format="%.3f"
        )
    
    # Filter data
    filtered_data = comparison_data[
        (comparison_data['category'].isin(category_filter)) &
        (comparison_data['improvement'] >= improvement_threshold)
    ]
    
    # Display filtered data
    display_columns = ['sample_id', 'category', 'base_similarity', 'finetuned_similarity', 
                      'improvement', 'improvement_pct']
    st.dataframe(
        filtered_data[display_columns].round(3),
        use_container_width=True,
        height=300
    )
    
    # Export option
    if st.button("Export Results to CSV"):
        csv = filtered_data.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name=f"model_comparison_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

def create_cost_analysis_dashboard():
    """Create the cost analysis dashboard"""
    st.markdown('<h2 style="color: #28a745;">💰 Cost Analysis & ROI</h2>', unsafe_allow_html=True)
    
    # Load cost data
    cost_data = simulate_cost_analysis_data()
    
    # Key Metrics Row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Training Cost",
            f"${cost_data['totals']['grand_total']:.2f}",
            f"${cost_data['totals']['grand_total']/cost_data['training_hours']:.2f}/hour"
        )
    
    with col2:
        best_scenario = min(cost_data['roi_scenarios'].items(), 
                           key=lambda x: x[1]['payback_months'])
        st.metric(
            "Fastest Payback",
            f"{best_scenario[1]['payback_months']:.1f} months",
            f"{best_scenario[0]}"
        )
    
    with col3:
        best_roi = max(cost_data['roi_scenarios'].items(), 
                      key=lambda x: x[1]['3_year_roi'])
        st.metric(
            "Best 3-Year ROI",
            f"{best_roi[1]['3_year_roi']:.0f}%",
            f"{best_roi[0]}"
        )
    
    with col4:
        compute_percentage = (cost_data['totals']['compute_total'] / 
                            cost_data['totals']['grand_total']) * 100
        st.metric(
            "Compute Cost Share",
            f"{compute_percentage:.0f}%",
            f"${cost_data['totals']['compute_total']:.2f}"
        )
    
    # Visualization Row 1
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Cost Breakdown")
        
        cost_categories = ['Compute', 'Storage', 'Network']
        cost_values = [
            cost_data['totals']['compute_total'],
            cost_data['totals']['storage_total'],
            cost_data['totals']['network_total']
        ]
        
        fig = go.Figure(data=go.Pie(
            labels=cost_categories,
            values=cost_values,
            hole=0.3,
            textinfo='label+percent+value',
            texttemplate='%{label}<br>%{percent}<br>$%{value:.2f}'
        ))
        
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("ROI Comparison by Scenario")
        
        scenarios = list(cost_data['roi_scenarios'].keys())
        payback_periods = [cost_data['roi_scenarios'][s]['payback_months'] for s in scenarios]
        three_year_rois = [cost_data['roi_scenarios'][s]['3_year_roi'] for s in scenarios]
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Payback Period (Months)', '3-Year ROI (%)'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        fig.add_trace(
            go.Bar(x=scenarios, y=payback_periods, name='Payback Months',
                  marker_color='orange'),
            row=1, col=1
        )
        
        colors = ['green' if roi > 100 else 'orange' if roi > 50 else 'red' 
                 for roi in three_year_rois]
        fig.add_trace(
            go.Bar(x=scenarios, y=three_year_rois, name='3-Year ROI %',
                  marker_color=colors),
            row=1, col=2
        )
        
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    # Detailed Cost Breakdown
    st.subheader("Detailed Cost Analysis")
    
    tab1, tab2, tab3 = st.tabs(["💻 Compute Costs", "�� Storage Costs", "🌐 Network Costs"])
    
    with tab1:
        compute_df = pd.DataFrame([
            {"Component": "GPU (8x H100)", "Cost per Hour": "$32.00", 
             "Total Hours": cost_data['training_hours'], 
             "Total Cost": f"${cost_data['costs']['compute']['gpu_cost']:.2f}"},
            {"Component": "CPU (128 vCPUs)", "Cost per Hour": "$6.40", 
             "Total Hours": cost_data['training_hours'], 
             "Total Cost": f"${cost_data['costs']['compute']['cpu_cost']:.2f}"},
            {"Component": "Memory (1600 GB)", "Cost per Hour": "$16.00", 
             "Total Hours": cost_data['training_hours'], 
             "Total Cost": f"${cost_data['costs']['compute']['memory_cost']:.2f}"}
        ])
        st.dataframe(compute_df, use_container_width=True)
    
    with tab2:
        storage_df = pd.DataFrame([
            {"Component": "Model Storage (2TB)", "Monthly Cost": "$200.00", 
             "Usage Hours": cost_data['training_hours'], 
             "Total Cost": f"${cost_data['costs']['storage']['models_storage']:.2f}"},
            {"Component": "Dataset Storage (1TB)", "Monthly Cost": "$100.00", 
             "Usage Hours": cost_data['training_hours'], 
             "Total Cost": f"${cost_data['costs']['storage']['datasets_storage']:.2f}"},
            {"Component": "Logs Storage (100GB)", "Monthly Cost": "$10.00", 
             "Usage Hours": cost_data['training_hours'], 
             "Total Cost": f"${cost_data['costs']['storage']['logs_storage']:.2f}"}
        ])
        st.dataframe(storage_df, use_container_width=True)
    
    with tab3:
        network_df = pd.DataFrame([
            {"Component": "Model Download", "Data Size": "50 GB", 
             "Rate": "$0.05/GB", "Total Cost": f"${cost_data['costs']['network']['model_download']:.2f}"},
            {"Component": "Dataset Download", "Data Size": "20 GB", 
             "Rate": "$0.05/GB", "Total Cost": f"${cost_data['costs']['network']['dataset_download']:.2f}"},
            {"Component": "Monitoring Data", "Data Size": "5 GB", 
             "Rate": "$0.05/GB", "Total Cost": f"${cost_data['costs']['network']['monitoring_data']:.2f}"}
        ])
        st.dataframe(network_df, use_container_width=True)
    
    # ROI Timeline Analysis
    st.subheader("ROI Timeline Analysis")
    
    months = list(range(1, 37))  # 3 years
    initial_investment = cost_data['totals']['grand_total']
    
    fig = go.Figure()
    
    for scenario_name, scenario_data in cost_data['roi_scenarios'].items():
        if 'monthly_savings' in scenario_data:
            monthly_benefit = scenario_data['monthly_savings']
        else:
            monthly_benefit = scenario_data['monthly_revenue']
        
        cumulative_savings = [monthly_benefit * m for m in months]
        net_value = [savings - initial_investment for savings in cumulative_savings]
        
        fig.add_trace(go.Scatter(
            x=months,
            y=net_value,
            mode='lines',
            name=scenario_name,
            line=dict(width=3)
        ))
    
    fig.add_hline(y=0, line_dash="dash", line_color="red", 
                  annotation_text="Break-even")
    
    fig.update_layout(
        xaxis_title='Months',
        yaxis_title='Net Value ($)',
        height=400,
        hovermode='x unified'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Cost Sensitivity Analysis
    st.subheader("Cost Sensitivity Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**GPU Price Sensitivity**")
        multipliers = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
        base_gpu_cost = cost_data['costs']['compute']['gpu_cost']
        
        adjusted_costs = []
        for mult in multipliers:
            adjusted_gpu = base_gpu_cost * mult
            adjusted_total = (cost_data['totals']['grand_total'] - base_gpu_cost + adjusted_gpu)
            adjusted_costs.append(adjusted_total)
        
        fig = go.Figure(data=go.Scatter(
            x=multipliers,
            y=adjusted_costs,
            mode='lines+markers',
            line=dict(color='red', width=3),
            marker=dict(size=8)
        ))
        
        fig.update_layout(
            xaxis_title='GPU Price Multiplier',
            yaxis_title='Total Training Cost ($)',
            height=300
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.write("**Training Duration Sensitivity**")
        duration_hours = [1, 2, 4, 6, 8, 12, 16, 24]
        duration_costs = []
        
        for hours in duration_hours:
            # Recalculate costs for different durations
            compute_cost = (8 * 4.00 + 128 * 0.05 + 1600 * 0.01) * hours
            storage_cost = (2000 + 1000 + 100) * 0.10 / 30 / 24 * hours
            network_cost = 75 * 0.05  # Fixed network cost
            total_cost = compute_cost + storage_cost + network_cost
            duration_costs.append(total_cost)
        
        fig = go.Figure(data=go.Scatter(
            x=duration_hours,
            y=duration_costs,
            mode='lines+markers',
            line=dict(color='blue', width=3),
            marker=dict(size=8)
        ))
        
        fig.update_layout(
            xaxis_title='Training Duration (Hours)',
            yaxis_title='Total Cost ($)',
            height=300
        )
        st.plotly_chart(fig, use_container_width=True)

def create_mlflow_browser():
    """Create MLflow experiment browser"""
    st.markdown('<h2 style="color: #6f42c1;">📊 MLflow Experiment Browser</h2>', unsafe_allow_html=True)
    
    # MLflow connection status
    mlflow_connected, mlflow_status = load_mlflow_connection()
    
    if mlflow_connected:
        st.success(f"✅ {mlflow_status}")
        
        # Load experiments
        experiments_df = get_experiments()
        
        if not experiments_df.empty:
            st.subheader("Available Experiments")
            
            # Filter experiments
            col1, col2 = st.columns(2)
            with col1:
                search_term = st.text_input("Search experiments", placeholder="Enter search term...")
            with col2:
                stage_filter = st.selectbox("Filter by stage", 
                                          ["All"] + experiments_df['lifecycle_stage'].unique().tolist())
            
            # Apply filters
            filtered_experiments = experiments_df.copy()
            if search_term:
                filtered_experiments = filtered_experiments[
                    filtered_experiments['name'].str.contains(search_term, case=False)
                ]
            if stage_filter != "All":
                filtered_experiments = filtered_experiments[
                    filtered_experiments['lifecycle_stage'] == stage_filter
                ]
            
            # Display experiments
            for _, exp in filtered_experiments.iterrows():
                with st.expander(f"🧪 {exp['name']} (ID: {exp['experiment_id']})"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**Created:** {exp['creation_time'].strftime('%Y-%m-%d %H:%M:%S')}")
                        st.write(f"**Stage:** {exp['lifecycle_stage']}")
                    with col2:
                        st.write(f"**Last Update:** {exp['last_update_time'].strftime('%Y-%m-%d %H:%M:%S')}")
                    
                    # Load runs for this experiment
                    if st.button(f"Load Runs", key=f"load_{exp['experiment_id']}"):
                        runs_df = get_experiment_runs(exp['experiment_id'])
                        
                        if not runs_df.empty:
                            st.write(f"**{len(runs_df)} runs found:**")
                            
                            # Display run summary
                            run_summary = runs_df[['run_id', 'status', 'start_time']].copy()
                            if 'tags.mlflow.runName' in runs_df.columns:
                                run_summary['run_name'] = runs_df['tags.mlflow.runName']
                            run_summary['run_id_short'] = run_summary['run_id'].str[:8] + "..."
                            run_summary['start_time'] = pd.to_datetime(run_summary['start_time']).dt.strftime('%Y-%m-%d %H:%M')
                            
                            st.dataframe(
                                run_summary[['run_name', 'run_id_short', 'status', 'start_time']] if 'run_name' in run_summary.columns 
                                else run_summary[['run_id_short', 'status', 'start_time']],
                                use_container_width=True
                            )
                        else:
                            st.info("No runs found in this experiment")
        else:
            st.info("No experiments found")
    else:
        st.error(f"❌ {mlflow_status}")
        st.info("""
        To connect to MLflow, set these environment variables or add them to your Streamlit secrets:
        - `MLFLOW_TRACKING_URI`
        - `MLFLOW_TRACKING_USERNAME` 
        - `MLFLOW_TRACKING_PASSWORD`
        """)

def main():
    """Main Streamlit application"""
    # Header
    st.markdown('<div class="main-header">🤖 VLM Analysis Dashboard</div>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.markdown("### 🎛️ Dashboard Controls")
        
        # Navigation
        page = st.selectbox(
            "Select Dashboard",
            ["🔍 Model Comparison", "💰 Cost Analysis", "📊 MLflow Browser"],
            index=0
        )
        
        st.markdown("### ℹ️ About")
        st.info("""
        This dashboard provides comprehensive analysis of your VLM fine-tuning results:
        
        • **Model Comparison**: Evaluate base vs fine-tuned model performance
        • **Cost Analysis**: Calculate infrastructure costs and ROI
        • **MLflow Browser**: View your experiments (read-only)
        
        All data is read-only - no changes are made to your MLflow experiments.
        """)
        
        st.markdown("### 🔧 Technical Info")
        st.code(f"""
Dashboard Version: 1.0
Streamlit: {st.__version__}
Last Updated: {datetime.now().strftime('%Y-%m-%d')}
        """)
        
        # Data refresh button
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.cache_data.clear()
            st.experimental_rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Main content area
    if page == "🔍 Model Comparison":
        create_model_comparison_dashboard()
    elif page == "💰 Cost Analysis":
        create_cost_analysis_dashboard()
    elif page == "📊 MLflow Browser":
        create_mlflow_browser()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 1rem;'>
        VLM Analysis Dashboard | Read-only MLflow Integration | 
        <a href='#' style='color: #1f77b4;'>Documentation</a> | 
        <a href='#' style='color: #1f77b4;'>Support</a>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
