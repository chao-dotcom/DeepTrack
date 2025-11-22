"""
Create interactive web dashboard using Streamlit
Implements Priority 4 from docs/5.md
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import json
import cv2
import numpy as np
from PIL import Image


def load_results():
    """Load benchmark results"""
    summary_path = Path("outputs/mot_results/summary.json")
    if summary_path.exists():
        with open(summary_path, 'r') as f:
            return json.load(f)
    return None


def main_dashboard():
    """Main dashboard page"""
    st.set_page_config(
        page_title="People Tracking System - Dashboard",
        page_icon="👥",
        layout="wide"
    )
    
    st.title("👥 People Tracking System - Performance Dashboard")
    st.markdown("---")
    
    # Load results
    results = load_results()
    
    if results is None:
        st.warning("⚠️ No benchmark results found. Please run benchmark first.")
        st.info("Run: `python scripts/run_full_benchmark.py`")
        return
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page",
        ["Overview", "Per-Sequence Analysis", "Comparison", "Visualizations"]
    )
    
    if page == "Overview":
        show_overview(results)
    elif page == "Per-Sequence Analysis":
        show_sequence_analysis(results)
    elif page == "Comparison":
        show_comparison(results)
    elif page == "Visualizations":
        show_visualizations(results)


def show_overview(results):
    """Show overview metrics"""
    st.header("📊 Overview Metrics")
    
    if results.get('average'):
        avg = results['average']
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("MOTA", f"{avg.get('mota', 0):.3f}", delta=None)
        with col2:
            st.metric("MOTP", f"{avg.get('motp', 0):.3f}", delta=None)
        with col3:
            st.metric("IDF1", f"{avg.get('idf1', 0):.3f}", delta=None)
        with col4:
            st.metric("Precision", f"{avg.get('precision', 0):.3f}", delta=None)
        with col5:
            st.metric("Recall", f"{avg.get('recall', 0):.3f}", delta=None)
        
        # Performance gauge
        st.subheader("Performance Gauge")
        mota = avg.get('mota', 0)
        
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = mota * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "MOTA Score (%)"},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 50], 'color': "lightgray"},
                    {'range': [50, 70], 'color': "gray"},
                    {'range': [70, 100], 'color': "lightgreen"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 80
                }
            }
        ))
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)


def show_sequence_analysis(results):
    """Show per-sequence analysis"""
    st.header("📈 Per-Sequence Analysis")
    
    sequences = []
    metrics = []
    
    for seq_name, seq_results in results.get('per_sequence', {}).items():
        sequences.append(seq_name)
        metrics.append({
            'Sequence': seq_name,
            'MOTA': seq_results.get('mota', 0),
            'MOTP': seq_results.get('motp', 0),
            'IDF1': seq_results.get('idf1', 0),
            'Precision': seq_results.get('precision', 0),
            'Recall': seq_results.get('recall', 0),
            'ID_Switches': seq_results.get('id_switches', 0)
        })
    
    df = pd.DataFrame(metrics)
    
    # Interactive table
    st.dataframe(df, use_container_width=True)
    
    # Bar chart
    st.subheader("MOTA by Sequence")
    fig = px.bar(df, x='Sequence', y='MOTA', 
                 title='MOTA Performance Across Sequences',
                 color='MOTA', color_continuous_scale='Viridis')
    st.plotly_chart(fig, use_container_width=True)


def show_comparison(results):
    """Show comparison with baselines"""
    st.header("🔍 Comparison with Baselines")
    
    # Baseline data
    baselines = {
        'Simple Tracker': {'MOTA': 0.45, 'MOTP': 0.70, 'IDF1': 0.50},
        'DeepSORT': {'MOTA': 0.65, 'MOTP': 0.75, 'IDF1': 0.65},
        'Our System': results.get('average', {}),
        'SOTA': {'MOTA': 0.80, 'MOTP': 0.80, 'IDF1': 0.75}
    }
    
    comparison_data = []
    for method, metrics in baselines.items():
        comparison_data.append({
            'Method': method,
            'MOTA': metrics.get('mota', metrics.get('MOTA', 0)),
            'MOTP': metrics.get('motp', metrics.get('MOTP', 0)),
            'IDF1': metrics.get('idf1', metrics.get('IDF1', 0))
        })
    
    df = pd.DataFrame(comparison_data)
    
    # Comparison chart
    fig = go.Figure()
    
    for metric in ['MOTA', 'MOTP', 'IDF1']:
        fig.add_trace(go.Bar(
            name=metric,
            x=df['Method'],
            y=df[metric],
            text=df[metric].round(3),
            textposition='auto'
        ))
    
    fig.update_layout(
        barmode='group',
        title='Performance Comparison with Baselines',
        xaxis_title='Method',
        yaxis_title='Score',
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(df, use_container_width=True)


def show_visualizations(results):
    """Show visualizations"""
    st.header("📊 Visualizations")
    
    # Radar chart
    if results.get('average'):
        avg = results['average']
        categories = ['MOTA', 'MOTP', 'IDF1', 'Precision', 'Recall']
        values = [
            avg.get('mota', 0),
            avg.get('motp', 0),
            avg.get('idf1', 0),
            avg.get('precision', 0),
            avg.get('recall', 0)
        ]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name='Our System'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=True,
            title="Performance Radar Chart"
        )
        
        st.plotly_chart(fig, use_container_width=True)


if __name__ == '__main__':
    main_dashboard()

