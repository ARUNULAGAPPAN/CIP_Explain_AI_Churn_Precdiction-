import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import plotly.express as px
from streamlit_lottie import st_lottie
import shap
import matplotlib.pyplot as plt
import requests
import os
import time

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="ExplainAI - Churn Prediction", layout="wide", initial_sidebar_state="collapsed")

# --- SESSION STATE ---
if 'page' not in st.session_state:
    st.session_state.page = 'landing'
if 'loading' not in st.session_state:
    st.session_state.loading = False

# --- PREMIUM DARK THEME CSS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .stApp {
        background: linear-gradient(180deg, #0a0a0a 0%, #0f0f0f 50%, #0a0a0a 100%);
        color: #ffffff;
    }
    
    /* Reduce default top padding */
    .main .block-container {
        padding-top: 1rem !important;
    }
    
    /* Hide scrollbar */
    ::-webkit-scrollbar { display: none; }
    * { -ms-overflow-style: none; scrollbar-width: none; }
    
    /* Hero Section Centering */
    .hero-section {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        text-align: center;
        width: 100%;
        padding: 20px 0;
    }
    
    /* Hero Title - Extra Large */
    .hero-title {
        font-size: 9rem;
        font-weight: 900;
        background: linear-gradient(135deg, #FF6B00, #FF8533, #FFa500, #FF6B00);
        background-size: 300% 300%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        margin: 0 auto 10px auto;
        letter-spacing: -1px;
        line-height: 1.0;
        animation: fadeInDown 0.8s ease-out, gradientShift 4s ease infinite;
        filter: drop-shadow(0 0 60px rgba(255, 107, 0, 0.5));
        width: 100%;
    }
    
    @keyframes gradientShift {
        0%, 100% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
    }
    
    @keyframes fadeInDown {
        from { opacity: 0; transform: translateY(-40px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(40px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes scaleIn {
        from { opacity: 0; transform: scale(0.9); }
        to { opacity: 1; transform: scale(1); }
    }
    
    .hero-subtitle {
        font-size: 1.3rem;
        color: #9ca3af;
        text-align: center;
        max-width: 700px;
        margin: 0 auto;
        line-height: 1.7;
        animation: fadeInUp 0.8s ease-out 0.2s both;
        display: block;
    }
    
    .centered-text {
        text-align: center;
        display: flex;
        justify-content: center;
        align-items: center;
        width: 100%;
    }
    
    .section-header {
        font-size: 2.5rem;
        font-weight: 800;
        color: #ffffff;
        text-align: center;
        margin: 50px 0 30px 0;
        animation: fadeInUp 0.6s ease-out;
    }
    
    /* Feature Cards */
    .feature-card {
        background: linear-gradient(145deg, #141414, #1a1a1a);
        border: 1px solid #262626;
        border-radius: 20px;
        padding: 35px 28px;
        text-align: center;
        height: 100%;
        transition: all 0.4s ease;
        animation: scaleIn 0.6s ease-out both;
    }
    
    .feature-card:hover {
        transform: translateY(-8px);
        border-color: #FF6B00;
        box-shadow: 0 20px 50px rgba(255, 107, 0, 0.15);
    }
    
    .feature-icon {
        width: 70px;
        height: 70px;
        margin: 0 auto 20px auto;
        background: linear-gradient(135deg, rgba(255, 107, 0, 0.2), rgba(255, 133, 51, 0.1));
        border-radius: 16px;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    
    .feature-icon svg {
        width: 36px;
        height: 36px;
        stroke: #FF6B00;
        fill: none;
        stroke-width: 1.5;
    }
    
    .feature-title {
        font-size: 1.25rem;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 12px;
    }
    
    .feature-desc {
        font-size: 0.95rem;
        color: #6b7280;
        line-height: 1.6;
    }
    
    /* Glass Cards */
    .glass-card {
        background: rgba(20, 20, 20, 0.8);
        backdrop-filter: blur(10px);
        border: 1px solid #262626;
        border-radius: 20px;
        padding: 30px;
        margin: 16px 0;
        animation: fadeInUp 0.5s ease-out;
    }
    
    /* Success/Danger Cards */
    .success-card {
        background: linear-gradient(135deg, rgba(34, 197, 94, 0.1), rgba(34, 197, 94, 0.05));
        border: 1px solid rgba(34, 197, 94, 0.3);
        border-radius: 20px;
        padding: 25px;
        animation: scaleIn 0.5s ease-out;
    }
    
    .danger-card {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.1), rgba(239, 68, 68, 0.05));
        border: 1px solid rgba(239, 68, 68, 0.3);
        border-radius: 20px;
        padding: 25px;
        animation: scaleIn 0.5s ease-out;
    }
    
    /* SHAP Container */
    .shap-container {
        background: #141414;
        border-radius: 16px;
        padding: 20px;
        border: 1px solid #262626;
        margin: 15px 0;
    }
    
    .section-header {
        font-size: 1.8rem;
        font-weight: 700;
        color: #ffffff;
        margin: 40px 0 20px 0;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    
    .section-header svg {
        stroke: #FF6B00;
    }
    
    /* Remedy Cards */
    .remedy-card {
        background: linear-gradient(135deg, #1a1a1a, #141414);
        border-left: 4px solid #FF6B00;
        padding: 18px 22px;
        margin: 12px 0;
        border-radius: 0 12px 12px 0;
        transition: all 0.3s ease;
    }
    
    .remedy-card:hover {
        transform: translateX(8px);
        box-shadow: 0 5px 20px rgba(255, 107, 0, 0.15);
    }
    
    /* Primary Button */
    .stButton > button {
        background: linear-gradient(135deg, #FF6B00, #FF8533) !important;
        color: white !important;
        border: none !important;
        padding: 14px 40px !important;
        border-radius: 12px !important;
        font-weight: 700 !important;
        font-size: 1.1rem !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 20px rgba(255, 107, 0, 0.3) !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px) !important;
        box-shadow: 0 8px 30px rgba(255, 107, 0, 0.4) !important;
    }
    
    /* Form Styling */
    .stSelectbox > div > div,
    .stNumberInput > div > div > input,
    .stSlider > div > div > div {
        background: #1a1a1a !important;
        border-color: #333 !important;
        color: white !important;
    }
    
    /* Small Glowing Analysis Button - Centered */
    .analysis-btn-container {
        display: flex;
        justify-content: center;
        align-items: center;
        width: 100%;
        margin-top: 20px;
    }
    
    .analysis-btn-container .stButton {
        display: flex;
        justify-content: center;
    }
    
    .analysis-btn-container .stButton > button {
        background: linear-gradient(135deg, #FF6B00, #FF8533) !important;
        color: white !important;
        border: none !important;
        padding: 12px 35px !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        width: auto !important;
        min-width: 200px !important;
        transition: all 0.3s ease !important;
    }
    
    .analysis-btn-container .stButton > button:hover {
        transform: translateY(-2px) scale(1.02) !important;
    }
    
    /* Form Submit Button - Orange Style */
    .analysis-btn-container button[kind="secondaryFormSubmit"],
    .analysis-btn-container [data-testid="stFormSubmitButton"] button,
    .analysis-btn-container button[data-testid="baseButton-secondaryFormSubmit"],
    .stForm button[kind="secondaryFormSubmit"],
    [data-testid="stFormSubmitButton"] button {
        background: linear-gradient(135deg, #FF6B00, #FF8533) !important;
        color: white !important;
        border: none !important;
        padding: 14px 40px !important;
        border-radius: 12px !important;
        font-weight: 700 !important;
        font-size: 1.1rem !important;
        min-width: 220px !important;
        transition: all 0.3s ease !important;
    }
    
    .analysis-btn-container button[kind="secondaryFormSubmit"]:hover,
    .analysis-btn-container [data-testid="stFormSubmitButton"] button:hover,
    [data-testid="stFormSubmitButton"] button:hover {
        transform: translateY(-3px) scale(1.03) !important;
    }
    
    /* Download Button - Orange Border Style */
    [data-testid="stDownloadButton"] button {
        background: transparent !important;
        background-color: transparent !important;
        border: 2px solid #FF6B00 !important;
        color: #FF6B00 !important;
        box-shadow: none !important;
        font-weight: 600 !important;
    }
    
    [data-testid="stDownloadButton"] button:hover {
        background: rgba(255, 107, 0, 0.15) !important;
        background-color: rgba(255, 107, 0, 0.15) !important;
        border-color: #FF8533 !important;
        color: #FF8533 !important;
        transform: translateY(-2px) !important;
    }
    
    [data-testid="stDownloadButton"] button:active,
    [data-testid="stDownloadButton"] button:focus {
        background: rgba(255, 107, 0, 0.2) !important;
        border-color: #FF6B00 !important;
        color: #FF6B00 !important;
    }
    
    /* Centered Animation Container */
    .lottie-container {
        display: flex;
        justify-content: center;
        align-items: center;
        width: 100%;
    }
    
    /* Remedy Icon Style */
    .remedy-icon {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 28px;
        height: 28px;
        background: rgba(255, 107, 0, 0.15);
        border-radius: 8px;
        margin-right: 12px;
        flex-shrink: 0;
    }
    
    .remedy-icon svg {
        width: 16px;
        height: 16px;
        stroke: #FF6B00;
        fill: none;
    }
    
    .remedy-content {
        display: flex;
        align-items: flex-start;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background: transparent;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: #1a1a1a;
        border-radius: 10px;
        color: #9ca3af;
        padding: 12px 24px;
        border: 1px solid #333;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #FF6B00, #FF8533) !important;
        color: white !important;
        border-color: #FF6B00 !important;
    }
    
    /* Page Transition Loader */
    .page-transition-loader {
        position: fixed;
        top: 0;
        left: 0;
        width: 100vw;
        height: 100vh;
        background: rgba(10, 10, 10, 0.98);
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        z-index: 9999;
        -webkit-backface-visibility: hidden;
        backface-visibility: hidden;
        -webkit-transform: translateZ(0);
        transform: translateZ(0);
    }
    
    .transition-brain {
        width: 120px;
        height: 120px;
        position: relative;
        -webkit-backface-visibility: hidden;
        backface-visibility: hidden;
        will-change: transform;
    }
    
    .transition-ring {
        position: absolute;
        width: 100%;
        height: 100%;
        border: 3px solid transparent;
        border-top-color: #FF6B00;
        border-radius: 50%;
        animation: spin-ring 1.2s linear infinite;
        -webkit-backface-visibility: hidden;
        backface-visibility: hidden;
        will-change: transform;
        -webkit-transform: translateZ(0);
        transform: translateZ(0);
    }
    
    .transition-ring:nth-child(2) {
        width: 80%;
        height: 80%;
        top: 10%;
        left: 10%;
        border-top-color: #FF8533;
        animation-duration: 1s;
        animation-direction: reverse;
    }
    
    .transition-ring:nth-child(3) {
        width: 60%;
        height: 60%;
        top: 20%;
        left: 20%;
        border-top-color: #FFa500;
        animation-duration: 0.8s;
    }
    
    .transition-core {
        position: absolute;
        width: 30px;
        height: 30px;
        background: linear-gradient(135deg, #FF6B00, #FF8533);
        border-radius: 50%;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        animation: pulse-core 1s ease-in-out infinite;
        -webkit-backface-visibility: hidden;
        backface-visibility: hidden;
        will-change: transform, opacity;
    }
    
    @keyframes spin-ring {
        0% { transform: rotate(0deg) translateZ(0); }
        100% { transform: rotate(360deg) translateZ(0); }
    }
    
    @keyframes pulse-core {
        0%, 100% { transform: translate(-50%, -50%) scale(1) translateZ(0); opacity: 1; }
        50% { transform: translate(-50%, -50%) scale(1.3) translateZ(0); opacity: 0.7; }
    }
    
    .transition-text {
        margin-top: 30px;
        font-size: 1.2rem;
        font-weight: 600;
        color: #FF6B00;
        animation: wave-text 1.5s ease-in-out infinite;
        -webkit-backface-visibility: hidden;
        backface-visibility: hidden;
    }
    
    @keyframes wave-text {
        0%, 100% { opacity: 0.5; }
        50% { opacity: 1; }
    }
    
    /* Tech Stack Badges */
    .tech-stack-badge {
        display: inline-block;
        padding: 8px 16px;
        margin: 5px;
        background: linear-gradient(135deg, #1a1a1a, #141414);
        border: 1px solid #333;
        border-radius: 25px;
        font-size: 0.85rem;
        color: #d1d5db;
        transition: all 0.3s ease;
    }
    
    .tech-stack-badge:hover {
        border-color: #FF6B00;
        color: #FF6B00;
        transform: translateY(-3px);
        box-shadow: 0 5px 15px rgba(255, 107, 0, 0.2);
    }
    
    /* Premium Footer */
    .footer {
        position: relative;
        width: 100%;
        background: linear-gradient(180deg, #0a0a0a 0%, #050505 100%);
        color: #ffffff;
        padding: 80px 0 30px 0;
        margin-top: 100px;
        overflow: hidden;
    }
    
    .footer::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, transparent, #FF6B00, #FF8533, #FF6B00, transparent);
        animation: shimmer 3s ease-in-out infinite;
    }
    
    @keyframes shimmer {
        0%, 100% { opacity: 0.5; }
        50% { opacity: 1; }
    }
    
    .footer-glow {
        position: absolute;
        top: -100px;
        left: 50%;
        transform: translateX(-50%);
        width: 600px;
        height: 200px;
        background: radial-gradient(ellipse, rgba(255, 107, 0, 0.12) 0%, transparent 70%);
        pointer-events: none;
    }
    
    .footer-container {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        flex-wrap: wrap;
        max-width: 1200px;
        margin: 0 auto;
        padding: 0 40px;
        position: relative;
        z-index: 1;
    }
    
    .footer-column {
        flex: 1;
        min-width: 280px;
        margin-bottom: 40px;
        padding: 0 25px;
    }
    
    .footer-brand {
        display: flex;
        align-items: center;
        margin-bottom: 20px;
    }
    
    .footer-brand-icon {
        width: 50px;
        height: 50px;
        background: linear-gradient(135deg, #FF6B00, #FF8533);
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        margin-right: 14px;
        box-shadow: 0 8px 25px rgba(255, 107, 0, 0.3);
    }
    
    .footer-brand-icon svg {
        width: 26px;
        height: 26px;
        stroke: #fff;
        fill: none;
    }
    
    .footer-brand-text {
        font-size: 1.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #FF6B00, #FF8533);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -1px;
    }
    
    .footer-column h3 {
        color: #ffffff;
        font-size: 1rem;
        margin-bottom: 20px;
        text-transform: uppercase;
        letter-spacing: 2px;
        font-weight: 700;
        position: relative;
        padding-bottom: 12px;
    }
    
    .footer-column h3::after {
        content: '';
        position: absolute;
        bottom: 0;
        left: 0;
        width: 35px;
        height: 3px;
        background: linear-gradient(90deg, #FF6B00, #FF8533);
        border-radius: 2px;
    }
    
    .footer-column p {
        font-size: 0.9rem;
        color: #6b7280;
        line-height: 1.7;
        margin-top: 0;
    }
    
    .footer-links {
        list-style: none;
        padding: 0;
        margin: 0;
    }
    
    .footer-links li {
        margin-bottom: 12px;
    }
    
    .footer-links a {
        color: #6b7280;
        text-decoration: none;
        font-size: 0.9rem;
        display: inline-flex;
        align-items: center;
        transition: all 0.3s ease;
    }
    
    .footer-links a:hover {
        color: #FF6B00;
        transform: translateX(5px);
    }
    
    .footer-links a svg {
        margin-right: 10px;
        width: 16px;
        height: 16px;
        stroke: currentColor;
    }
    
    .footer-social {
        display: flex;
        gap: 12px;
        margin-top: 20px;
    }
    
    .footer-social-icon {
        width: 42px;
        height: 42px;
        border-radius: 10px;
        background: rgba(255, 107, 0, 0.1);
        border: 1px solid rgba(255, 107, 0, 0.2);
        display: flex;
        align-items: center;
        justify-content: center;
        transition: all 0.3s ease;
        cursor: pointer;
        text-decoration: none;
    }
    
    .footer-social-icon:hover {
        background: #FF6B00;
        border-color: #FF6B00;
        transform: translateY(-4px);
        box-shadow: 0 8px 20px rgba(255, 107, 0, 0.3);
    }
    
    .footer-social-icon svg {
        width: 18px;
        height: 18px;
        stroke: #FF6B00;
        transition: all 0.3s ease;
    }
    
    .footer-social-icon:hover svg {
        stroke: #ffffff;
    }
    
    .tech-grid {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
    }
    
    .footer-bottom {
        text-align: center;
        padding: 25px 40px;
        border-top: 1px solid rgba(255, 107, 0, 0.15);
        margin-top: 40px;
    }
    
    .footer-bottom p {
        color: #4b5563;
        font-size: 0.85rem;
        margin: 0;
    }
    
    .footer-bottom-brand {
        color: #FF6B00;
        font-weight: 600;
    }
    
    /* ========== MOBILE RESPONSIVE STYLES ========== */
    @media screen and (max-width: 768px) {
        /* Hero Title - Mobile */
        .hero-title {
            font-size: 3.5rem !important;
            letter-spacing: -2px !important;
            line-height: 1.1 !important;
            word-break: keep-all !important;
            white-space: nowrap !important;
            padding: 0 10px !important;
        }
        
        .hero-subtitle {
            font-size: 1rem !important;
            padding: 0 20px !important;
            max-width: 100% !important;
        }
        
        .section-header {
            font-size: 1.8rem !important;
            padding: 0 15px !important;
        }
        
        /* Feature Cards - Mobile */
        .feature-card {
            margin-bottom: 24px !important;
            padding: 28px 22px !important;
            border-radius: 18px !important;
            border: 1px solid #333 !important;
            box-shadow: 0 8px 32px rgba(255, 107, 0, 0.1) !important;
        }
        
        .feature-icon {
            width: 60px !important;
            height: 60px !important;
        }
        
        .feature-title {
            font-size: 1.15rem !important;
            margin-bottom: 10px !important;
        }
        
        .feature-desc {
            font-size: 0.9rem !important;
            line-height: 1.5 !important;
        }
        
        /* Streamlit columns gap for mobile */
        [data-testid="column"] {
            padding: 8px 4px !important;
        }
        
        /* Glass Card - Mobile */
        .glass-card {
            padding: 20px !important;
            margin: 12px 0 !important;
        }
    }
    
    @media screen and (max-width: 480px) {
        .hero-title {
            font-size: 2.8rem !important;
            letter-spacing: -1px !important;
        }
        
        .hero-subtitle {
            font-size: 0.95rem !important;
        }
        
        .section-header {
            font-size: 1.5rem !important;
        }
        
        .feature-card {
            margin-bottom: 20px !important;
            padding: 24px 18px !important;
            background: linear-gradient(145deg, #181818, #1f1f1f) !important;
            border: 1px solid rgba(255, 107, 0, 0.2) !important;
        }
        
        [data-testid="column"] {
            padding: 6px 0 !important;
        }
    }
    
    /* ========== MOBILE SPECIFIC FIXES ========== */
    @media screen and (max-width: 768px) {
        /* Reduce top spacing on all pages - aggressive */
        .main .block-container {
            padding-top: 0.5rem !important;
        }
        
        [data-testid="stAppViewContainer"] > .main {
            padding-top: 0 !important;
        }
        
        .hero-section {
            padding-top: 0 !important;
            margin-top: 0 !important;
        }
        
        /* Tabs - fixed styling without messy indicator */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px !important;
            overflow-x: auto !important;
            scroll-behavior: smooth !important;
            -webkit-overflow-scrolling: touch !important;
            scrollbar-width: none !important;
            padding-bottom: 0 !important;
            border-bottom: none !important;
        }
        
        .stTabs [data-baseweb="tab-list"]::-webkit-scrollbar {
            display: none !important;
        }
        
        .stTabs [data-baseweb="tab"] {
            padding: 10px 16px !important;
            font-size: 0.85rem !important;
            white-space: nowrap !important;
            flex-shrink: 0 !important;
            border-radius: 10px !important;
        }
        
        /* Hide the tab highlight/indicator bar completely */
        .stTabs [data-baseweb="tab-highlight"] {
            display: none !important;
        }
        
        .stTabs [data-baseweb="tab-border"] {
            display: none !important;
        }
        
        /* Tab panel spacing */
        .stTabs [data-baseweb="tab-panel"] {
            padding-top: 10px !important;
        }
        
        /* Remove extra vertical spacing in columns */
        [data-testid="stVerticalBlock"] > [data-testid="stVerticalBlockBorderWrapper"] {
            margin-top: 0 !important;
            padding-top: 0 !important;
        }
    }
    
    @media screen and (max-width: 480px) {
        /* Title size for very small screens */
        .prediction-dashboard-title {
            font-size: 1.6rem !important;
        }
        
        /* Tabs - even more compact */
        .stTabs [data-baseweb="tab"] {
            padding: 8px 12px !important;
            font-size: 0.8rem !important;
        }
    }
</style>
""", unsafe_allow_html=True)

# --- SCROLL TO TOP ON PAGE LOAD ---
st.markdown("""
<style>
    /* Ensure content starts from top */
    html, body, [data-testid="stAppViewContainer"], .main, section.main {
        scroll-behavior: auto !important;
    }
</style>
<script>
    function scrollToTop() {
        // Multiple methods to ensure scroll works on mobile
        window.scrollTo(0, 0);
        document.documentElement.scrollTop = 0;
        document.body.scrollTop = 0;
        
        // Streamlit specific containers
        var containers = [
            'section.main',
            '.main',
            '[data-testid="stAppViewContainer"]',
            '[data-testid="stAppViewBlockContainer"]',
            '.stApp',
            '[data-testid="stVerticalBlock"]'
        ];
        
        containers.forEach(function(selector) {
            var el = document.querySelector(selector);
            if (el) el.scrollTop = 0;
        });
        
        // Also try parent document for iframes
        try {
            if (window.parent && window.parent.document) {
                window.parent.scrollTo(0, 0);
                var parentContainers = window.parent.document.querySelectorAll('section.main, .main, [data-testid="stAppViewContainer"]');
                parentContainers.forEach(function(el) {
                    el.scrollTop = 0;
                });
            }
        } catch(e) {}
        
        // Scroll to anchor if exists
        var anchor = document.getElementById('top-anchor');
        if (anchor) {
            anchor.scrollIntoView({behavior: 'instant', block: 'start'});
        }
    }
    
    // Execute immediately
    scrollToTop();
    
    // On DOM ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', scrollToTop);
    } else {
        scrollToTop();
    }
    
    // On window load with small delay
    window.addEventListener('load', function() {
        scrollToTop();
        setTimeout(scrollToTop, 50);
        setTimeout(scrollToTop, 150);
        setTimeout(scrollToTop, 300);
    });
</script>
""", unsafe_allow_html=True)

# --- LOTTIE HELPER ---
def load_lottieurl(url):
    try:
        r = requests.get(url, timeout=5)
        if r.status_code == 200:
            return r.json()
    except:
        pass
    return None

lottie_hero = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_fcfjwiyb.json")

# --- LOAD MODEL ASSETS ---
@st.cache_resource
def load_ml_assets():
    with open(os.path.join(SCRIPT_DIR, 'churn_model_assets.pkl'), 'rb') as f:
        assets = pickle.load(f)
    return assets['model'], assets['scaler'], assets['feature_names']

model, scaler, feature_names = load_ml_assets()

# --- CATEGORICAL MAPPINGS ---
categorical_mappings = {
    'gender': {'Female': 0, 'Male': 1},
    'Partner': {'No': 0, 'Yes': 1},
    'Dependents': {'No': 0, 'Yes': 1},
    'PhoneService': {'No': 0, 'Yes': 1},
    'MultipleLines': {'No': 0, 'No phone service': 1, 'Yes': 2},
    'InternetService': {'DSL': 0, 'Fiber optic': 1, 'No': 2},
    'OnlineSecurity': {'No': 0, 'No internet service': 1, 'Yes': 2},
    'OnlineBackup': {'No': 0, 'No internet service': 1, 'Yes': 2},
    'DeviceProtection': {'No': 0, 'No internet service': 1, 'Yes': 2},
    'TechSupport': {'No': 0, 'No internet service': 1, 'Yes': 2},
    'StreamingTV': {'No': 0, 'No internet service': 1, 'Yes': 2},
    'StreamingMovies': {'No': 0, 'No internet service': 1, 'Yes': 2},
    'Contract': {'Month-to-month': 0, 'One year': 1, 'Two year': 2},
    'PaperlessBilling': {'No': 0, 'Yes': 1},
    'PaymentMethod': {'Bank transfer (automatic)': 0, 'Credit card (automatic)': 1, 'Electronic check': 2, 'Mailed check': 3}
}

# --- DYNAMIC REMEDY KNOWLEDGE BASE ---
def get_remedies(user_data, shap_values, feature_names, churn_probability):
    """Generate truly personalized remedies based on SHAP analysis and user conditions"""
    remedies = []
    
    # Get feature importance from SHAP
    feature_importance = dict(zip(feature_names, shap_values))
    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    
    # Top factors contributing to churn (positive SHAP = increases churn risk)
    top_churn_factors = [(f, v) for f, v in sorted_features if v > 0][:6]
    
    for feature, impact in top_churn_factors:
        remedy = generate_dynamic_remedy(feature, user_data, impact, churn_probability)
        if remedy:
            remedies.append(remedy)
    
    # If no specific remedies found but high risk, add general strategies
    if not remedies and churn_probability > 0.5:
        remedies = get_fallback_remedies(user_data, churn_probability)
    
    return remedies[:4]

def generate_dynamic_remedy(feature, user_data, impact, churn_probability):
    """Generate highly personalized remedy based on specific user conditions"""
    
    # SVG icons for each remedy type
    icons = {
        'contract': '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/></svg>',
        'time': '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>',
        'money': '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="12" y1="1" x2="12" y2="23"/><path d="M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"/></svg>',
        'support': '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14.7 6.3a1 1 0 0 0 0 1.4l1.6 1.6a1 1 0 0 0 1.4 0l3.77-3.77a6 6 0 0 1-7.94 7.94l-6.91 6.91a2.12 2.12 0 0 1-3-3l6.91-6.91a6 6 0 0 1 7.94-7.94l-3.76 3.76z"/></svg>',
        'security': '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="11" width="18" height="11" rx="2" ry="2"/><path d="M7 11V7a5 5 0 0 1 10 0v4"/></svg>',
        'speed': '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/></svg>',
        'payment': '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="1" y="4" width="22" height="16" rx="2" ry="2"/><line x1="1" y1="10" x2="23" y2="10"/></svg>',
        'email': '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M4 4h16c1.1 0 2 .9 2 2v12c0 1.1-.9 2-2 2H4c-1.1 0-2-.9-2-2V6c0-1.1.9-2 2-2z"/><polyline points="22,6 12,13 2,6"/></svg>',
        'user': '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/></svg>',
        'phone': '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M22 16.92v3a2 2 0 0 1-2.18 2 19.79 19.79 0 0 1-8.63-3.07 19.5 19.5 0 0 1-6-6 19.79 19.79 0 0 1-3.07-8.67A2 2 0 0 1 4.11 2h3a2 2 0 0 1 2 1.72 12.84 12.84 0 0 0 .7 2.81 2 2 0 0 1-.45 2.11L8.09 9.91a16 16 0 0 0 6 6l1.27-1.27a2 2 0 0 1 2.11-.45 12.84 12.84 0 0 0 2.81.7A2 2 0 0 1 22 16.92z"/></svg>',
        'tv': '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="2" y="7" width="20" height="15" rx="2" ry="2"/><polyline points="17 2 12 7 7 2"/></svg>',
        'gift': '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="20 12 20 22 4 22 4 12"/><rect x="2" y="7" width="20" height="5"/><line x1="12" y1="22" x2="12" y2="7"/><path d="M12 7H7.5a2.5 2.5 0 0 1 0-5C11 2 12 7 12 7z"/><path d="M12 7h4.5a2.5 2.5 0 0 0 0-5C13 2 12 7 12 7z"/></svg>',
        'shield': '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>',
        'star': '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"/></svg>'
    }
    
    tenure = user_data.get('tenure', 0)
    monthly_charges = user_data.get('MonthlyCharges', 0)
    contract = user_data.get('Contract', '')
    internet = user_data.get('InternetService', '')
    tech_support = user_data.get('TechSupport', '')
    online_security = user_data.get('OnlineSecurity', '')
    payment = user_data.get('PaymentMethod', '')
    paperless = user_data.get('PaperlessBilling', '')
    streaming_tv = user_data.get('StreamingTV', '')
    senior = user_data.get('SeniorCitizen', 0)
    
    # Impact level determines urgency
    urgency = "URGENT" if impact > 0.3 else ("Important" if impact > 0.15 else "Recommended")
    
    # CONTRACT-based dynamic remedies
    if feature == 'Contract':
        if contract == 'Month-to-month':
            if tenure < 6:
                discount = "25%" if churn_probability > 0.7 else "20%"
                return f'<div class="remedy-content"><div class="remedy-icon">{icons["contract"]}</div><div><b>[{urgency}] New Customer Lock-in:</b> Offer {discount} discount for 1-year contract. Customer is in critical first 6 months with {tenure} months tenure. Add free premium feature for 3 months to sweeten the deal.</div></div>'
            elif tenure < 12:
                return f'<div class="remedy-content"><div class="remedy-icon">{icons["contract"]}</div><div><b>[{urgency}] Early Retention:</b> Customer has been with us {tenure} months. Offer 15% annual discount + waive first month on upgrade. Schedule personal account review call.</div></div>'
            else:
                return f'<div class="remedy-content"><div class="remedy-icon">{icons["contract"]}</div><div><b>[{urgency}] Loyalty Conversion:</b> Long-term month-to-month customer ({tenure} months). Offer exclusive "Loyal Customer" 2-year plan with 20% savings + priority support status.</div></div>'
        elif contract == 'One year':
            return f'<div class="remedy-content"><div class="remedy-icon">{icons["contract"]}</div><div><b>[{urgency}] Upgrade Path:</b> Current 1-year contract holder. Before renewal, offer 2-year plan with additional 10% savings + free service upgrade worth ${monthly_charges * 0.1:.0f}/month.</div></div>'
    
    # TENURE-based dynamic remedies
    if feature == 'tenure':
        if tenure <= 3:
            return f'<div class="remedy-content"><div class="remedy-icon">{icons["time"]}</div><div><b>[{urgency}] Critical Onboarding:</b> Only {tenure} month(s) tenure - highest churn risk period! Immediately assign dedicated success manager. Schedule weekly check-ins for first 90 days. Offer "Getting Started" bonus credits.</div></div>'
        elif tenure <= 6:
            return f'<div class="remedy-content"><div class="remedy-icon">{icons["time"]}</div><div><b>[{urgency}] Early Engagement:</b> Customer at {tenure} months - still in danger zone. Implement proactive "6-Month Milestone" rewards program. Send personalized usage tips and offer complimentary service review.</div></div>'
        elif tenure <= 12:
            return f'<div class="remedy-content"><div class="remedy-icon">{icons["time"]}</div><div><b>[{urgency}] First Year Success:</b> At {tenure} months, approaching 1-year mark. Prepare anniversary offer: exclusive discount + loyalty points + personal thank-you from account team.</div></div>'
        elif tenure <= 24:
            return f'<div class="remedy-content"><div class="remedy-icon">{icons["star"]}</div><div><b>[{urgency}] Loyalty Recognition:</b> {tenure}-month customer showing churn signals. Enroll in VIP loyalty program immediately. Offer exclusive access to beta features and priority support queue.</div></div>'
    
    # MONTHLY CHARGES-based dynamic remedies
    if feature == 'MonthlyCharges':
        if monthly_charges > 100:
            savings = monthly_charges * 0.15
            return f'<div class="remedy-content"><div class="remedy-icon">{icons["money"]}</div><div><b>[{urgency}] Premium Value Review:</b> High-value customer paying ${monthly_charges:.0f}/month. Conduct immediate service audit. Offer loyalty bundle saving ${savings:.0f}/month or add premium features at no cost to justify spend.</div></div>'
        elif monthly_charges > 70:
            savings = monthly_charges * 0.12
            return f'<div class="remedy-content"><div class="remedy-icon">{icons["money"]}</div><div><b>[{urgency}] Price Optimization:</b> Customer pays ${monthly_charges:.0f}/month. Review for unused services. Offer right-sized plan saving ${savings:.0f}/month or upgrade with same price + more features.</div></div>'
        elif monthly_charges > 50:
            return f'<div class="remedy-content"><div class="remedy-icon">{icons["money"]}</div><div><b>[{urgency}] Value Perception:</b> At ${monthly_charges:.0f}/month, ensure customer perceives full value. Send monthly value report showing benefits used. Offer free trial of premium features.</div></div>'
        else:
            return f'<div class="remedy-content"><div class="remedy-icon">{icons["gift"]}</div><div><b>[{urgency}] Upsell Opportunity:</b> Low-tier customer at ${monthly_charges:.0f}/month. Offer limited-time upgrade: premium features for only ${monthly_charges * 1.3:.0f}/month (normally ${monthly_charges * 1.8:.0f}).</div></div>'
    
    # TECH SUPPORT-based dynamic remedies
    if feature == 'TechSupport':
        if tech_support == 'No' and internet != 'No':
            if senior == 1:
                return f'<div class="remedy-content"><div class="remedy-icon">{icons["support"]}</div><div><b>[{urgency}] Senior Support Program:</b> Senior customer without tech support is high risk. Offer FREE lifetime tech support + dedicated senior helpline with extended hours. This demographic highly values support.</div></div>'
            elif tenure < 12:
                return f'<div class="remedy-content"><div class="remedy-icon">{icons["support"]}</div><div><b>[{urgency}] Support Onboarding:</b> Newer customer ({tenure} months) without tech support. Offer 6-month FREE premium support trial. Proactively reach out to help with any issues.</div></div>'
            else:
                return f'<div class="remedy-content"><div class="remedy-icon">{icons["support"]}</div><div><b>[{urgency}] Support Upgrade:</b> Long-term customer without tech support. Offer 50% off tech support package. Highlight 24/7 availability and priority queue benefits.</div></div>'
    
    # ONLINE SECURITY-based dynamic remedies
    if feature == 'OnlineSecurity':
        if online_security == 'No' and internet != 'No':
            if monthly_charges > 70:
                return f'<div class="remedy-content"><div class="remedy-icon">{icons["security"]}</div><div><b>[{urgency}] Premium Security Bundle:</b> High-value customer without security. Add FREE comprehensive security suite (worth $15/month) for 12 months. Include identity protection + antivirus + VPN.</div></div>'
            else:
                return f'<div class="remedy-content"><div class="remedy-icon">{icons["shield"]}</div><div><b>[{urgency}] Security Awareness:</b> Customer without online security. Offer FREE basic security for 6 months + educational webinar on online safety. Convert to paid after trial with 30% discount.</div></div>'
    
    # INTERNET SERVICE-based dynamic remedies  
    if feature == 'InternetService':
        if internet == 'Fiber optic':
            if monthly_charges > 80:
                return f'<div class="remedy-content"><div class="remedy-icon">{icons["speed"]}</div><div><b>[{urgency}] Fiber Premium Care:</b> High-paying fiber customer (${monthly_charges:.0f}/month). Schedule FREE professional network optimization visit. Offer speed upgrade or mesh WiFi system at cost price.</div></div>'
            else:
                return f'<div class="remedy-content"><div class="remedy-icon">{icons["speed"]}</div><div><b>[{urgency}] Fiber Value Check:</b> Fiber optic customer - ensure satisfaction. Send technician for FREE speed test & optimization. Proactively offer credits if speeds below promised levels.</div></div>'
        elif internet == 'DSL':
            return f'<div class="remedy-content"><div class="remedy-icon">{icons["speed"]}</div><div><b>[{urgency}] Upgrade Path:</b> DSL customer may want faster speeds. Offer fiber upgrade at DSL price for first 6 months. Highlight speed improvements for streaming & gaming.</div></div>'
    
    # PAYMENT METHOD-based dynamic remedies
    if feature == 'PaymentMethod':
        if payment == 'Electronic check':
            savings = 5 if monthly_charges < 50 else (8 if monthly_charges < 80 else 10)
            return f'<div class="remedy-content"><div class="remedy-icon">{icons["payment"]}</div><div><b>[{urgency}] Payment Optimization:</b> Electronic check users have higher churn. Offer ${savings}/month discount for switching to auto-pay. Emphasize convenience + no late fee worry + paper-free benefits.</div></div>'
        elif payment == 'Mailed check':
            return f'<div class="remedy-content"><div class="remedy-icon">{icons["payment"]}</div><div><b>[{urgency}] Modernize Payments:</b> Mailed check customer (often older/less engaged). Offer $10/month savings for 6 months for digital payment setup. Provide phone assistance for setup.</div></div>'
    
    # PAPERLESS BILLING-based dynamic remedies
    if feature == 'PaperlessBilling':
        if paperless == 'Yes':
            return f'<div class="remedy-content"><div class="remedy-icon">{icons["email"]}</div><div><b>[{urgency}] Digital Engagement:</b> Paperless customer may feel disconnected. Launch personalized monthly "Your Value Report" email showing savings, usage stats, and exclusive digital-only offers.</div></div>'
        else:
            return f'<div class="remedy-content"><div class="remedy-icon">{icons["email"]}</div><div><b>[{urgency}] Paper Bill Customer:</b> Traditional billing preference suggests lower digital engagement. Reach out via phone for personal service check. Offer small incentive for paperless switch.</div></div>'
    
    # STREAMING-based dynamic remedies
    if feature in ['StreamingTV', 'StreamingMovies']:
        if streaming_tv == 'No' and internet != 'No':
            return f'<div class="remedy-content"><div class="remedy-icon">{icons["tv"]}</div><div><b>[{urgency}] Entertainment Bundle:</b> Internet customer without streaming. Offer FREE 3-month streaming trial. Show content library highlights. Bundle savings of 20% if added permanently.</div></div>'
        elif streaming_tv == 'Yes':
            return f'<div class="remedy-content"><div class="remedy-icon">{icons["tv"]}</div><div><b>[{urgency}] Streaming Satisfaction:</b> Active streaming user showing churn signals. Ensure quality of experience. Offer FREE premium channel trial or exclusive content access. Check for buffering issues.</div></div>'
    
    # SENIOR CITIZEN-based dynamic remedies
    if feature == 'SeniorCitizen':
        if senior == 1:
            return f'<div class="remedy-content"><div class="remedy-icon">{icons["user"]}</div><div><b>[{urgency}] Senior Care Program:</b> Enroll in exclusive Senior Loyalty Program: 15% permanent discount + dedicated senior support line + simplified billing + annual in-home service check.</div></div>'
    
    # PHONE SERVICE-based dynamic remedies
    if feature in ['PhoneService', 'MultipleLines']:
        return f'<div class="remedy-content"><div class="remedy-icon">{icons["phone"]}</div><div><b>[{urgency}] Voice Service Review:</b> Phone service contributing to churn risk. Review call quality & coverage. Offer unlimited calling upgrade or bundled family plan with 25% multi-line discount.</div></div>'
    
    return None

def get_fallback_remedies(user_data, churn_probability):
    """Generate general remedies when no specific ones match"""
    icons = {
        'star': '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"/></svg>',
        'phone': '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M22 16.92v3a2 2 0 0 1-2.18 2 19.79 19.79 0 0 1-8.63-3.07 19.5 19.5 0 0 1-6-6 19.79 19.79 0 0 1-3.07-8.67A2 2 0 0 1 4.11 2h3a2 2 0 0 1 2 1.72 12.84 12.84 0 0 0 .7 2.81 2 2 0 0 1-.45 2.11L8.09 9.91a16 16 0 0 0 6 6l1.27-1.27a2 2 0 0 1 2.11-.45 12.84 12.84 0 0 0 2.81.7A2 2 0 0 1 22 16.92z"/></svg>',
        'gift': '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="20 12 20 22 4 22 4 12"/><rect x="2" y="7" width="20" height="5"/><line x1="12" y1="22" x2="12" y2="7"/><path d="M12 7H7.5a2.5 2.5 0 0 1 0-5C11 2 12 7 12 7z"/><path d="M12 7h4.5a2.5 2.5 0 0 0 0-5C13 2 12 7 12 7z"/></svg>'
    }
    
    remedies = []
    tenure = user_data.get('tenure', 0)
    monthly = user_data.get('MonthlyCharges', 0)
    
    if churn_probability > 0.7:
        remedies.append(f'<div class="remedy-content"><div class="remedy-icon">{icons["phone"]}</div><div><b>[URGENT] Immediate Outreach:</b> High churn risk ({churn_probability:.0%}). Schedule urgent retention call within 24 hours. Prepare personalized offer based on {tenure} months history and ${monthly:.0f}/month value.</div></div>')
        remedies.append(f'<div class="remedy-content"><div class="remedy-icon">{icons["gift"]}</div><div><b>[URGENT] Save Offer:</b> Prepare "Stay With Us" package: 25% discount for 6 months + free premium feature + waived fees. Total value: ${monthly * 0.25 * 6:.0f} savings.</div></div>')
    else:
        remedies.append(f'<div class="remedy-content"><div class="remedy-icon">{icons["star"]}</div><div><b>[Important] Proactive Care:</b> Customer showing moderate risk. Enroll in loyalty program + send appreciation message + offer exclusive early access to new features.</div></div>')
    
    return remedies

# --- SHOW FOOTER ---
def show_footer():
    """Display the premium footer section"""
    st.markdown("""
    <div class="footer">
        <div class="footer-glow"></div>
        <div class="footer-container">
            <div class="footer-column">
                <div class="footer-brand">
                    <div class="footer-brand-icon">
                        <svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                            <path d="M12 2L2 7l10 5 10-5-10-5z"/>
                            <path d="M2 17l10 5 10-5"/>
                            <path d="M2 12l10 5 10-5"/>
                        </svg>
                    </div>
                    <span class="footer-brand-text">ExplainAI</span>
                </div>
                <p>Empowering businesses with Explainable AI technology to predict customer churn, understand behavior patterns, and maximize retention.</p>
                <div class="footer-social">
                    <a href="https://github.com/ARUNULAGAPPAN" target="_blank" class="footer-social-icon" title="GitHub">
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <path d="M9 19c-5 1.5-5-2.5-7-3m14 6v-3.87a3.37 3.37 0 0 0-.94-2.61c3.14-.35 6.44-1.54 6.44-7A5.44 5.44 0 0 0 20 4.77 5.07 5.07 0 0 0 19.91 1S18.73.65 16 2.48a13.38 13.38 0 0 0-7 0C6.27.65 5.09 1 5.09 1A5.07 5.07 0 0 0 5 4.77a5.44 5.44 0 0 0-1.5 3.78c0 5.42 3.3 6.61 6.44 7A3.37 3.37 0 0 0 9 18.13V22"/>
                        </svg>
                    </a>
                    <a href="https://www.linkedin.com/in/arunulagappan2024/" target="_blank" class="footer-social-icon" title="LinkedIn">
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <path d="M16 8a6 6 0 0 1 6 6v7h-4v-7a2 2 0 0 0-2-2 2 2 0 0 0-2 2v7h-4v-7a6 6 0 0 1 6-6z"/>
                            <rect x="2" y="9" width="4" height="12"/>
                            <circle cx="4" cy="4" r="2"/>
                        </svg>
                    </a>
                    <a href="mailto:sarunulagappan@gmail.com" class="footer-social-icon" title="Email">
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <path d="M4 4h16c1.1 0 2 .9 2 2v12c0 1.1-.9 2-2 2H4c-1.1 0-2-.9-2-2V6c0-1.1.9-2 2-2z"/>
                            <polyline points="22,6 12,13 2,6"/>
                        </svg>
                    </a>
                    <a href="https://wa.me/918098368308" target="_blank" class="footer-social-icon" title="WhatsApp">
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <path d="M21 11.5a8.38 8.38 0 0 1-.9 3.8 8.5 8.5 0 0 1-7.6 4.7 8.38 8.38 0 0 1-3.8-.9L3 21l1.9-5.7a8.38 8.38 0 0 1-.9-3.8 8.5 8.5 0 0 1 4.7-7.6 8.38 8.38 0 0 1 3.8-.9h.5a8.48 8.48 0 0 1 8 8v.5z"/>
                        </svg>
                    </a>
                </div>
            </div>
            <div class="footer-column">
                <h3>Tech Stack</h3>
                <div class="tech-grid">
                    <span class="tech-stack-badge">XGBoost</span>
                    <span class="tech-stack-badge">SHAP</span>
                    <span class="tech-stack-badge">SMOTE</span>
                    <span class="tech-stack-badge">Streamlit</span>
                    <span class="tech-stack-badge">Scikit-Learn</span>
                    <span class="tech-stack-badge">Plotly</span>
                    <span class="tech-stack-badge">Pandas</span>
                    <span class="tech-stack-badge">NumPy</span>
                </div>
            </div>
            <div class="footer-column">
                <h3>Contact Us</h3>
                <ul class="footer-links">
                    <li><a href="https://github.com/ARUNULAGAPPAN" target="_blank">
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M9 19c-5 1.5-5-2.5-7-3m14 6v-3.87a3.37 3.37 0 0 0-.94-2.61c3.14-.35 6.44-1.54 6.44-7A5.44 5.44 0 0 0 20 4.77 5.07 5.07 0 0 0 19.91 1S18.73.65 16 2.48a13.38 13.38 0 0 0-7 0C6.27.65 5.09 1 5.09 1A5.07 5.07 0 0 0 5 4.77a5.44 5.44 0 0 0-1.5 3.78c0 5.42 3.3 6.61 6.44 7A3.37 3.37 0 0 0 9 18.13V22"/></svg>
                        GitHub
                    </a></li>
                    <li><a href="https://www.linkedin.com/in/arunulagappan2024/" target="_blank">
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M16 8a6 6 0 0 1 6 6v7h-4v-7a2 2 0 0 0-2-2 2 2 0 0 0-2 2v7h-4v-7a6 6 0 0 1 6-6z"/><rect x="2" y="9" width="4" height="12"/><circle cx="4" cy="4" r="2"/></svg>
                        LinkedIn
                    </a></li>
                    <li><a href="mailto:sarunulagappan@gmail.com">
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M4 4h16c1.1 0 2 .9 2 2v12c0 1.1-.9 2-2 2H4c-1.1 0-2-.9-2-2V6c0-1.1.9-2 2-2z"/><polyline points="22,6 12,13 2,6"/></svg>
                        sarunulagappan@gmail.com
                    </a></li>
                    <li><a href="https://wa.me/918098368308" target="_blank">
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 11.5a8.38 8.38 0 0 1-.9 3.8 8.5 8.5 0 0 1-7.6 4.7 8.38 8.38 0 0 1-3.8-.9L3 21l1.9-5.7a8.38 8.38 0 0 1-.9-3.8 8.5 8.5 0 0 1 4.7-7.6 8.38 8.38 0 0 1 3.8-.9h.5a8.48 8.48 0 0 1 8 8v.5z"/></svg>
                        WhatsApp: +91 8098368308
                    </a></li>
                </ul>
            </div>
        </div>
        <div class="footer-bottom">
            <p>© 2025 <span class="footer-bottom-brand">ExplainAI</span> — Intelligent Churn Prediction System</p>
            <p style="margin-top: 8px; font-size: 0.8rem;">Developed by Arun Ulagappan</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

# --- LANDING PAGE ---
def show_landing_page():
    # Anchor for scroll to top
    st.markdown('<div id="top-anchor"></div>', unsafe_allow_html=True)
    
    # Show Initializing Animation on first load
    if 'initialized' not in st.session_state:
        st.markdown('''
        <div class="page-transition-loader">
            <div class="transition-brain">
                <div class="transition-ring"></div>
                <div class="transition-ring"></div>
                <div class="transition-ring"></div>
                <div class="transition-core"></div>
            </div>
            <div class="transition-text">Initializing ExplainAI...</div>
        </div>
        ''', unsafe_allow_html=True)
        time.sleep(2)
        st.session_state.initialized = True
        st.rerun()
        return  # Exit function, don't render anything else
    
    # Hero Section - Centered
    st.markdown('''
    <div class="hero-section">
        <p class="hero-title">ExplainAI</p>
        <p class="hero-subtitle">Predict Customer Churn with Explainable AI. Understand WHY customers leave and get actionable strategies to retain them.</p>
    </div>
    ''', unsafe_allow_html=True)
    
    # Animation - Centered
    st.markdown('<div class="lottie-container">', unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if lottie_hero:
            st_lottie(lottie_hero, height=300, key="hero_anim")
        else:
            st.markdown('''<div style="height: 200px; display: flex; align-items: center; justify-content: center;">
                <svg width="100" height="100" viewBox="0 0 24 24" fill="none" stroke="#FF6B00" stroke-width="1.5">
                    <circle cx="12" cy="12" r="10"/>
                    <path d="M12 2a9 9 0 0 1 9 9c0 3.6-2.4 6.6-5.7 8.2l-.3.1V22h-6v-2.7l-.3-.1C5.4 17.6 3 14.6 3 11a9 9 0 0 1 9-9z"/>
                </svg>
            </div>''', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Start Button
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("Start Prediction →", use_container_width=True, key="start_btn"):
            st.session_state.loading = True
            st.rerun()
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # Features Section
    st.markdown('<p class="section-header">Why ExplainAI?</p>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">
                <svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                    <circle cx="12" cy="12" r="10"/>
                    <circle cx="12" cy="12" r="6"/>
                    <circle cx="12" cy="12" r="2" fill="#FF6B00"/>
                </svg>
            </div>
            <div class="feature-title">Precision Engine</div>
            <div class="feature-desc">Built with XGBoost & SMOTE for highly accurate churn predictions with 85%+ accuracy.</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">
                <svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                    <path d="M12 2a9 9 0 0 1 9 9c0 3.6-2.4 6.6-5.7 8.2l-.3.1V22h-6v-2.7l-.3-.1C5.4 17.6 3 14.6 3 11a9 9 0 0 1 9-9z"/>
                    <path d="M9 22h6M12 18v-4"/>
                </svg>
            </div>
            <div class="feature-title">Explainable AI</div>
            <div class="feature-desc">Powered by SHAP values to explain exactly WHY each prediction is made.</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">
                <svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                    <path d="M9 18l6-6-6-6"/>
                    <rect x="3" y="3" width="18" height="18" rx="2"/>
                </svg>
            </div>
            <div class="feature-title">Actionable Remedies</div>
            <div class="feature-desc">Get personalized retention strategies based on each customer's risk factors.</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">
                <svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                    <rect x="3" y="3" width="7" height="7" rx="1"/>
                    <rect x="14" y="3" width="7" height="7" rx="1"/>
                    <rect x="3" y="14" width="7" height="7" rx="1"/>
                    <rect x="14" y="14" width="7" height="7" rx="1"/>
                </svg>
            </div>
            <div class="feature-title">Bulk Analysis</div>
            <div class="feature-desc">Upload CSV files and get instant predictions for your entire customer base.</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">
                <svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                    <circle cx="12" cy="8" r="4"/>
                    <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/>
                </svg>
            </div>
            <div class="feature-title">Individual Analysis</div>
            <div class="feature-desc">Analyze single customers with detailed explainability and recommendations.</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">
                <svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                    <path d="M3 3v18h18"/>
                    <path d="M18 9l-5 5-4-4-3 3"/>
                </svg>
            </div>
            <div class="feature-title">Visual Insights</div>
            <div class="feature-desc">Beautiful visualizations including gauges, charts, and SHAP plots.</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Footer
    show_footer()

# --- PREDICTION PAGE ---
def show_prediction_page():
    # Centered title with minimal top spacing
    st.markdown('<h1 style="color: #FF6B00; font-weight: 800; margin: 0; padding-top: 10px; text-align: center;">Prediction Dashboard</h1>', unsafe_allow_html=True)
    
    # Tabs
    tab1, tab2 = st.tabs(["Single Customer Analysis", "Bulk CSV Upload"])
    
    # --- SINGLE CUSTOMER TAB ---
    with tab1:
        st.markdown("### Enter Customer Details")
        
        with st.form("customer_form"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**Account Info**")
                tenure = st.slider("Tenure (months)", 0, 72, 12)
                contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
                paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
                payment = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
            
            with col2:
                st.markdown("**Services**")
                internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
                security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
                tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
                streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
            
            with col3:
                st.markdown("**Billing & Demographics**")
                monthly = st.number_input("Monthly Charges ($)", 0.0, 150.0, 65.0)
                total = st.number_input("Total Charges ($)", 0.0, 10000.0, monthly * tenure)
                senior = st.selectbox("Senior Citizen", [0, 1])
                gender = st.selectbox("Gender", ["Male", "Female"])
            
            st.markdown('<div class="analysis-btn-container">', unsafe_allow_html=True)
            submitted = st.form_submit_button("Run AI Analysis", use_container_width=False)
            st.markdown('</div>', unsafe_allow_html=True)
        
        if submitted:
            with st.spinner("Analyzing with AI..."):
                # Prepare input
                user_data = {
                    'tenure': tenure,
                    'Contract': contract,
                    'PaperlessBilling': paperless,
                    'PaymentMethod': payment,
                    'InternetService': internet,
                    'OnlineSecurity': security,
                    'TechSupport': tech_support,
                    'StreamingTV': streaming_tv,
                    'MonthlyCharges': monthly,
                    'TotalCharges': total,
                    'SeniorCitizen': senior,
                    'gender': gender
                }
                
                # Create feature array
                input_dict = {f: 0 for f in feature_names}
                input_dict.update(user_data)
                
                encoded_vals = []
                for col in feature_names:
                    val = input_dict.get(col, 0)
                    if col in categorical_mappings:
                        encoded_vals.append(categorical_mappings[col].get(val, 0))
                    else:
                        encoded_vals.append(val if isinstance(val, (int, float)) else 0)
                
                final_input = np.array(encoded_vals).reshape(1, -1)
                final_scaled = scaler.transform(final_input)
                
                # Predict
                prob = model.predict_proba(final_scaled)[0][1]
                prediction = "Churn" if prob > 0.5 else "Stay"
                risk_level = "High" if prob > 0.7 else ("Medium" if prob > 0.4 else "Low")
            
            st.markdown("---")
            
            # Results
            col1, col2 = st.columns(2)
            
            with col1:
                # Gauge Chart
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=(1 - prob) * 100,
                    title={'text': "Loyalty Score", 'font': {'color': 'white', 'size': 20}},
                    number={'suffix': "%", 'font': {'color': 'white', 'size': 40}},
                    gauge={
                        'axis': {'range': [0, 100], 'tickcolor': 'white'},
                        'bar': {'color': '#FF6B00'},
                        'bgcolor': '#1a1a1a',
                        'bordercolor': '#333',
                        'steps': [
                            {'range': [0, 30], 'color': 'rgba(239, 68, 68, 0.3)'},
                            {'range': [30, 70], 'color': 'rgba(251, 191, 36, 0.3)'},
                            {'range': [70, 100], 'color': 'rgba(34, 197, 94, 0.3)'}
                        ]
                    }
                ))
                fig.update_layout(
                    paper_bgcolor='#0a0a0a',
                    plot_bgcolor='#0a0a0a',
                    font={'color': 'white'},
                    height=300,
                    margin=dict(t=80, b=30, l=30, r=30)
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Prediction Card
                if prediction == "Churn":
                    st.markdown(f'''
                    <div class="danger-card">
                        <h2 style="color: #ef4444; margin: 0; display: flex; align-items: center; gap: 10px;">
                            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#ef4444" stroke-width="2"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>
                            High Churn Risk
                        </h2>
                        <p style="font-size: 2rem; margin: 15px 0; color: white;">{prob:.1%} Probability</p>
                        <p style="color: #9ca3af;">This customer shows strong indicators of churning. Immediate retention action recommended.</p>
                    </div>
                    ''', unsafe_allow_html=True)
                else:
                    st.markdown(f'''
                    <div class="success-card">
                        <h2 style="color: #22c55e; margin: 0; display: flex; align-items: center; gap: 10px;">
                            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#22c55e" stroke-width="2"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>
                            Loyal Customer
                        </h2>
                        <p style="font-size: 2rem; margin: 15px 0; color: white;">{1-prob:.1%} Loyalty</p>
                        <p style="color: #9ca3af;">This customer shows strong retention indicators. Continue current engagement strategy.</p>
                    </div>
                    ''', unsafe_allow_html=True)
            
            # SHAP Explainability Section
            st.markdown('''
            <p class="section-header">
                <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#FF6B00" stroke-width="2">
                    <path d="M12 2a9 9 0 0 1 9 9c0 3.6-2.4 6.6-5.7 8.2V22h-6v-2.8C5.4 17.6 3 14.6 3 11a9 9 0 0 1 9-9z"/>
                    <path d="M9 22h6"/>
                </svg>
                Explainable AI - Why This Prediction?
            </p>
            ''', unsafe_allow_html=True)
            
            try:
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(final_scaled)
                
                col_shap1, col_shap2 = st.columns([1, 1])
                
                with col_shap1:
                    # SHAP Waterfall Plot
                    st.markdown("#### SHAP Waterfall Plot")
                    st.markdown('<div class="shap-container">', unsafe_allow_html=True)
                    
                    # Create SHAP Explanation object for waterfall plot
                    shap_explanation = shap.Explanation(
                        values=shap_values[0],
                        base_values=explainer.expected_value,
                        data=final_input[0],
                        feature_names=feature_names
                    )
                    
                    fig, ax = plt.subplots(figsize=(10, 8))
                    fig.patch.set_facecolor('#141414')
                    ax.set_facecolor('#141414')
                    
                    shap.plots.waterfall(shap_explanation, max_display=10, show=False)
                    
                    # Style the plot for dark theme
                    ax = plt.gca()
                    ax.set_facecolor('#141414')
                    for spine in ax.spines.values():
                        spine.set_color('#333333')
                    ax.tick_params(colors='#cccccc')
                    ax.xaxis.label.set_color('#cccccc')
                    ax.yaxis.label.set_color('#cccccc')
                    
                    plt.tight_layout()
                    st.pyplot(fig, use_container_width=True)
                    plt.close()
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col_shap2:
                    # Feature Importance Bar Chart
                    st.markdown("#### Top Contributing Factors")
                    st.markdown('<div class="shap-container">', unsafe_allow_html=True)
                    
                    # Create feature importance dataframe
                    importance_df = pd.DataFrame({
                        'Feature': feature_names,
                        'Impact': shap_values[0]
                    }).sort_values('Impact', key=abs, ascending=False).head(10)
                    
                    # Create horizontal bar chart
                    fig_bar = go.Figure(go.Bar(
                        x=importance_df['Impact'],
                        y=importance_df['Feature'],
                        orientation='h',
                        marker=dict(
                            color=importance_df['Impact'],
                            colorscale=[[0, '#22c55e'], [0.5, '#f59e0b'], [1, '#ef4444']],
                            line=dict(color='#333', width=1)
                        )
                    ))
                    
                    fig_bar.update_layout(
                        paper_bgcolor='#141414',
                        plot_bgcolor='#141414',
                        font=dict(color='#cccccc'),
                        height=400,
                        margin=dict(l=10, r=10, t=10, b=10),
                        xaxis=dict(gridcolor='#333', zerolinecolor='#FF6B00'),
                        yaxis=dict(gridcolor='#333')
                    )
                    
                    st.plotly_chart(fig_bar, use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Remedies
                st.markdown('''
                <h3 style="display: flex; align-items: center; gap: 10px; margin-top: 30px;">
                    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#FF6B00" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/></svg>
                    Personalized Retention Strategies
                </h3>
                ''', unsafe_allow_html=True)
                
                remedies = get_remedies(user_data, shap_values[0], feature_names, prob)
                
                if remedies:
                    for remedy in remedies:
                        st.markdown(f'<div class="remedy-card">{remedy}</div>', unsafe_allow_html=True)
                else:
                    st.info("No specific remedies needed - customer shows healthy engagement patterns.")
                    
            except Exception as e:
                st.warning(f"SHAP analysis unavailable: {str(e)}")
    
    # --- BULK UPLOAD TAB ---
    with tab2:
        st.markdown("### Upload Customer Data")
        
        # CSV Format Information
        st.markdown("""
        <div style="background: linear-gradient(135deg, rgba(59, 130, 246, 0.15), rgba(59, 130, 246, 0.05)); border: 1px solid rgba(59, 130, 246, 0.3); border-radius: 12px; padding: 20px; margin-bottom: 20px;">
            <h4 style="color: #3b82f6; margin: 0 0 15px 0; display: flex; align-items: center; gap: 8px;">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#3b82f6" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/></svg>
                CSV File Format Required
            </h4>
            <p style="color: #9ca3af; margin: 0 0 12px 0; font-size: 0.9rem;">Your CSV file must contain the following columns (in any order):</p>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 8px; margin-bottom: 15px;">
                <div style="background: rgba(0,0,0,0.2); padding: 8px 12px; border-radius: 6px; font-size: 0.85rem;">
                    <span style="color: #FF6B00; font-weight: 600;">gender</span>
                    <span style="color: #6b7280;"> — Male / Female</span>
                </div>
                <div style="background: rgba(0,0,0,0.2); padding: 8px 12px; border-radius: 6px; font-size: 0.85rem;">
                    <span style="color: #FF6B00; font-weight: 600;">SeniorCitizen</span>
                    <span style="color: #6b7280;"> — 0 / 1</span>
                </div>
                <div style="background: rgba(0,0,0,0.2); padding: 8px 12px; border-radius: 6px; font-size: 0.85rem;">
                    <span style="color: #FF6B00; font-weight: 600;">Partner</span>
                    <span style="color: #6b7280;"> — Yes / No</span>
                </div>
                <div style="background: rgba(0,0,0,0.2); padding: 8px 12px; border-radius: 6px; font-size: 0.85rem;">
                    <span style="color: #FF6B00; font-weight: 600;">Dependents</span>
                    <span style="color: #6b7280;"> — Yes / No</span>
                </div>
                <div style="background: rgba(0,0,0,0.2); padding: 8px 12px; border-radius: 6px; font-size: 0.85rem;">
                    <span style="color: #FF6B00; font-weight: 600;">tenure</span>
                    <span style="color: #6b7280;"> — Months (0-72)</span>
                </div>
                <div style="background: rgba(0,0,0,0.2); padding: 8px 12px; border-radius: 6px; font-size: 0.85rem;">
                    <span style="color: #FF6B00; font-weight: 600;">PhoneService</span>
                    <span style="color: #6b7280;"> — Yes / No</span>
                </div>
                <div style="background: rgba(0,0,0,0.2); padding: 8px 12px; border-radius: 6px; font-size: 0.85rem;">
                    <span style="color: #FF6B00; font-weight: 600;">MultipleLines</span>
                    <span style="color: #6b7280;"> — Yes / No / No phone service</span>
                </div>
                <div style="background: rgba(0,0,0,0.2); padding: 8px 12px; border-radius: 6px; font-size: 0.85rem;">
                    <span style="color: #FF6B00; font-weight: 600;">InternetService</span>
                    <span style="color: #6b7280;"> — DSL / Fiber optic / No</span>
                </div>
                <div style="background: rgba(0,0,0,0.2); padding: 8px 12px; border-radius: 6px; font-size: 0.85rem;">
                    <span style="color: #FF6B00; font-weight: 600;">OnlineSecurity</span>
                    <span style="color: #6b7280;"> — Yes / No / No internet service</span>
                </div>
                <div style="background: rgba(0,0,0,0.2); padding: 8px 12px; border-radius: 6px; font-size: 0.85rem;">
                    <span style="color: #FF6B00; font-weight: 600;">OnlineBackup</span>
                    <span style="color: #6b7280;"> — Yes / No / No internet service</span>
                </div>
                <div style="background: rgba(0,0,0,0.2); padding: 8px 12px; border-radius: 6px; font-size: 0.85rem;">
                    <span style="color: #FF6B00; font-weight: 600;">DeviceProtection</span>
                    <span style="color: #6b7280;"> — Yes / No / No internet service</span>
                </div>
                <div style="background: rgba(0,0,0,0.2); padding: 8px 12px; border-radius: 6px; font-size: 0.85rem;">
                    <span style="color: #FF6B00; font-weight: 600;">TechSupport</span>
                    <span style="color: #6b7280;"> — Yes / No / No internet service</span>
                </div>
                <div style="background: rgba(0,0,0,0.2); padding: 8px 12px; border-radius: 6px; font-size: 0.85rem;">
                    <span style="color: #FF6B00; font-weight: 600;">StreamingTV</span>
                    <span style="color: #6b7280;"> — Yes / No / No internet service</span>
                </div>
                <div style="background: rgba(0,0,0,0.2); padding: 8px 12px; border-radius: 6px; font-size: 0.85rem;">
                    <span style="color: #FF6B00; font-weight: 600;">StreamingMovies</span>
                    <span style="color: #6b7280;"> — Yes / No / No internet service</span>
                </div>
                <div style="background: rgba(0,0,0,0.2); padding: 8px 12px; border-radius: 6px; font-size: 0.85rem;">
                    <span style="color: #FF6B00; font-weight: 600;">Contract</span>
                    <span style="color: #6b7280;"> — Month-to-month / One year / Two year</span>
                </div>
                <div style="background: rgba(0,0,0,0.2); padding: 8px 12px; border-radius: 6px; font-size: 0.85rem;">
                    <span style="color: #FF6B00; font-weight: 600;">PaperlessBilling</span>
                    <span style="color: #6b7280;"> — Yes / No</span>
                </div>
                <div style="background: rgba(0,0,0,0.2); padding: 8px 12px; border-radius: 6px; font-size: 0.85rem;">
                    <span style="color: #FF6B00; font-weight: 600;">PaymentMethod</span>
                    <span style="color: #6b7280;"> — Electronic check / Mailed check / Bank transfer (automatic) / Credit card (automatic)</span>
                </div>
                <div style="background: rgba(0,0,0,0.2); padding: 8px 12px; border-radius: 6px; font-size: 0.85rem;">
                    <span style="color: #FF6B00; font-weight: 600;">MonthlyCharges</span>
                    <span style="color: #6b7280;"> — Numeric ($)</span>
                </div>
                <div style="background: rgba(0,0,0,0.2); padding: 8px 12px; border-radius: 6px; font-size: 0.85rem;">
                    <span style="color: #FF6B00; font-weight: 600;">TotalCharges</span>
                    <span style="color: #6b7280;"> — Numeric ($)</span>
                </div>
            </div>
            <p style="color: #6b7280; margin: 0; font-size: 0.8rem; font-style: italic; display: flex; align-items: center; gap: 6px;">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#f59e0b" stroke-width="2"><path d="M9 18h6"/><path d="M10 22h4"/><path d="M12 2a7 7 0 0 0-7 7c0 2.38 1.19 4.47 3 5.74V17a1 1 0 0 0 1 1h6a1 1 0 0 0 1-1v-2.26c1.81-1.27 3-3.36 3-5.74a7 7 0 0 0-7-7z"/></svg>
                <span>Tip: Column names are case-sensitive. The system will predict churn probability for each row.</span>
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        
        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file)
                st.markdown(f"**Loaded {len(df)} records**")
                st.dataframe(df.head(), use_container_width=True)
                
                if st.button("Run Batch Analysis", use_container_width=True):
                    with st.spinner("Processing all customers..."):
                        predictions = []
                        probabilities = []
                        
                        progress = st.progress(0)
                        for idx, row in df.iterrows():
                            # Prepare features
                            encoded_vals = []
                            for col in feature_names:
                                val = row.get(col, 0)
                                if col in categorical_mappings:
                                    encoded_vals.append(categorical_mappings[col].get(val, 0))
                                else:
                                    try:
                                        encoded_vals.append(float(val) if pd.notna(val) else 0)
                                    except:
                                        encoded_vals.append(0)
                            
                            final_input = np.array(encoded_vals).reshape(1, -1)
                            final_scaled = scaler.transform(final_input)
                            
                            prob = model.predict_proba(final_scaled)[0][1]
                            predictions.append("Churn" if prob > 0.5 else "Stay")
                            probabilities.append(prob)
                            
                            progress.progress((idx + 1) / len(df))
                        
                        df['Prediction'] = predictions
                        df['Churn_Probability'] = probabilities
                        df['Risk_Level'] = df['Churn_Probability'].apply(
                            lambda x: "High" if x > 0.7 else ("Medium" if x > 0.4 else "Low")
                        )
                    
                    st.success("✅ Analysis Complete!")
                    
                    # Summary Stats
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Total Customers", len(df))
                    col2.metric("High Risk", len(df[df['Risk_Level'] == 'High']), delta_color="inverse")
                    col3.metric("Medium Risk", len(df[df['Risk_Level'] == 'Medium']))
                    col4.metric("Low Risk", len(df[df['Risk_Level'] == 'Low']))
                    
                    # Risk Distribution Chart
                    fig_risk = px.pie(
                        df, names='Risk_Level',
                        title='Risk Distribution',
                        color='Risk_Level',
                        color_discrete_map={'High': '#ef4444', 'Medium': '#f59e0b', 'Low': '#22c55e'}
                    )
                    fig_risk.update_layout(
                        paper_bgcolor='#0a0a0a',
                        plot_bgcolor='#0a0a0a',
                        font={'color': 'white'}
                    )
                    st.plotly_chart(fig_risk, use_container_width=True)
                    
                    # Results Table
                    st.markdown("#### Detailed Results")
                    st.dataframe(
                        df[['Prediction', 'Churn_Probability', 'Risk_Level'] + 
                           [c for c in df.columns if c not in ['Prediction', 'Churn_Probability', 'Risk_Level']]],
                        use_container_width=True,
                        height=400
                    )
                    
                    # Download
                    csv = df.to_csv(index=False)
                    st.download_button(
                        "Download Results CSV",
                        data=csv,
                        file_name="churn_predictions.csv",
                        mime="text/csv",
                        use_container_width=True,
                        key="download_csv_btn"
                    )
                    
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    # Footer
    show_footer()

# --- MAIN APP ---
# Check loading state FIRST at the top level
if st.session_state.loading:
    st.markdown('''
    <div class="page-transition-loader">
        <div class="transition-brain">
            <div class="transition-ring"></div>
            <div class="transition-ring"></div>
            <div class="transition-ring"></div>
            <div class="transition-core"></div>
        </div>
        <div class="transition-text">Loading AI Engine...</div>
    </div>
    ''', unsafe_allow_html=True)
    time.sleep(1.5)
    st.session_state.loading = False
    st.session_state.page = 'prediction'
    st.rerun()
elif st.session_state.page == 'landing':
    show_landing_page()
else:
    show_prediction_page()
