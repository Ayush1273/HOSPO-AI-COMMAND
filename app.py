# ================================================================
# HOSP-AI COMMAND
# Hospital Operational Decision Support System
# ================================================================

import os
import base64
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np

from xgboost import XGBRegressor
from dotenv import load_dotenv

from google import genai
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# ================================================================
# 1. PAGE CONFIGURATION
# ================================================================

st.set_page_config(
    page_title="HOSP-AI COMMAND",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ================================================================
# 2. LOAD ENVIRONMENT VARIABLES
# ================================================================

BASE_DIR = Path(__file__).resolve().parent

load_dotenv(BASE_DIR / ".env")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")


# ================================================================
# 3. CUSTOM STYLING (HOSPITAL / CLINICAL GREEN & WHITE THEME)
# ================================================================

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }

    /* Main App & Kexsio Thread Background Stage */
    html, body {
        background-color: #f8fafc !important;
        overflow-x: hidden;
    }

    .stApp {
        background: transparent !important;
        color: #0f172a;
    }

    .main {
        background: transparent !important;
    }

    #kexsio-thread-canvas {
        position: fixed !important;
        top: 0 !important;
        left: 0 !important;
        width: 100vw !important;
        height: 100vh !important;
        z-index: 0 !important;
        pointer-events: none !important;
    }

    .kexsio-dot-grid {
        position: fixed !important;
        inset: 0;
        background-image: radial-gradient(rgba(5, 150, 105, 0.08) 1px, transparent 1px);
        background-size: 32px 32px;
        opacity: 0.65;
        z-index: 0 !important;
        pointer-events: none !important;
        mask-image: radial-gradient(ellipse at 50% 50%, black 50%, transparent 90%);
        -webkit-mask-image: radial-gradient(ellipse at 50% 50%, black 50%, transparent 90%);
    }

    div[data-testid="stCustomComponentV1"] {
        position: fixed !important;
        top: 0 !important;
        left: 0 !important;
        width: 100vw !important;
        height: 100vh !important;
        z-index: 0 !important;
        pointer-events: none !important;
        border: none !important;
        margin: 0 !important;
        padding: 0 !important;
        overflow: hidden !important;
    }

    div[data-testid="stCustomComponentV1"] iframe {
        width: 100% !important;
        height: 100% !important;
        border: none !important;
        pointer-events: none !important;
    }

    /* Streamlit Header Clearance */
    header[data-testid="stHeader"] {
        background: transparent !important;
    }

    .block-container {
        position: relative;
        z-index: 1;
        padding-top: 1.25rem !important;
        padding-bottom: 3.5rem !important;
        max-width: 1350px;
    }

    /* Floating Navbar Styling */
    .floating-navbar {
        position: sticky;
        top: 0.75rem;
        z-index: 999;
        background: rgba(255, 255, 255, 0.92);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(209, 250, 229, 0.85);
        border-radius: 9999px;
        padding: 0.75rem 1.6rem;
        box-shadow: 0 8px 30px -4px rgba(6, 78, 59, 0.08), 0 2px 8px -2px rgba(0, 0, 0, 0.03);
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 1.5rem;
        transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1);
    }

    .floating-navbar:hover {
        box-shadow: 0 12px 35px -4px rgba(6, 78, 59, 0.12), 0 4px 12px -2px rgba(0, 0, 0, 0.04);
        border-color: rgba(167, 243, 208, 0.95);
    }

    .nav-brand {
        display: flex;
        align-items: center;
        gap: 12px;
    }

    .nav-logo-wrap {
        width: 38px;
        height: 38px;
        display: flex;
        align-items: center;
        justify-content: center;
        flex-shrink: 0;
    }

    .nav-logo-img {
        width: 100%;
        height: 100%;
        object-fit: contain;
    }

    .nav-brand-text {
        display: flex;
        flex-direction: column;
    }

    .nav-title {
        font-size: 1.22rem;
        font-weight: 800;
        color: #064e3b;
        letter-spacing: -0.02em;
        line-height: 1.2;
    }

    .nav-title .brand-accent {
        color: #059669;
    }

    .nav-subtitle {
        font-size: 0.72rem;
        color: #64748b;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    .nav-pills {
        display: flex;
        align-items: center;
        gap: 8px;
    }

    .nav-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 5px 13px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        transition: all 0.2s ease;
    }

    .nav-pill.active-pill {
        background: #ecfdf5;
        color: #047857;
        border: 1px solid #a7f3d0;
        font-weight: 700;
        letter-spacing: 0.04em;
    }

    .nav-pill.info-pill {
        background: #f8fafc;
        color: #475569;
        border: 1px solid #e2e8f0;
    }

    .nav-pill.info-pill:hover {
        background: #f1f5f9;
        border-color: #cbd5e1;
        color: #1e293b;
    }

    .nav-pill .pulse-dot {
        width: 7px;
        height: 7px;
        background-color: #10b981;
        border-radius: 50%;
        box-shadow: 0 0 6px #10b981;
        animation: livePulse 1.8s infinite;
    }

    @media (max-width: 768px) {
        .floating-navbar {
            border-radius: 20px;
            flex-direction: column;
            gap: 10px;
            align-items: flex-start;
            padding: 1rem;
        }
        .nav-pills {
            flex-wrap: wrap;
            width: 100%;
        }
    }

    /* Hero Brand Text Graphic below Navbar */
    .hero-brand-wrap {
        display: flex;
        justify-content: center;
        align-items: center;
        margin-top: 0.25rem;
        margin-bottom: 1.25rem;
        text-align: center;
    }

    .hero-brand-img {
        max-width: 360px;
        width: 100%;
        height: auto;
        object-fit: contain;
        filter: drop-shadow(0 2px 8px rgba(6, 78, 59, 0.06));
    }

    /* Disclaimer Banner */
    .disclaimer-banner {
        background: rgba(236, 253, 245, 0.88);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(167, 243, 208, 0.9);
        border-left: 4px solid #059669;
        border-radius: 10px;
        padding: 0.85rem 1.25rem;
        color: #065f46;
        font-size: 0.88rem;
        font-weight: 500;
        line-height: 1.5;
        margin-bottom: 1.75rem;
        display: flex;
        align-items: center;
        gap: 12px;
    }

    /* Operational Conditions Card Box — Solid Opaque Hospital White */
    div[data-testid="stLayoutWrapper"]:has(.box-header-wrap) {
        position: relative !important;
        z-index: 10 !important;
        background-color: #ffffff !important;
        background: #ffffff !important;
        border: 1px solid rgba(209, 250, 229, 0.95) !important;
        border-radius: 18px !important;
        box-shadow: 0 10px 32px -4px rgba(6, 78, 59, 0.08), 0 2px 8px -2px rgba(0, 0, 0, 0.03) !important;
        padding: 1.75rem 1.75rem !important;
        transition: box-shadow 0.2s ease;
    }

    div[data-testid="stLayoutWrapper"]:has(.box-header-wrap) > div[data-testid="stVerticalBlock"] {
        border: none !important;
        background: #ffffff !important;
        box-shadow: none !important;
        padding: 0 !important;
    }

    div[data-testid="stLayoutWrapper"]:has(.box-header-wrap):hover {
        box-shadow: 0 14px 38px -4px rgba(6, 78, 59, 0.12), 0 4px 12px -2px rgba(0, 0, 0, 0.04) !important;
    }

    div[data-testid="stVerticalBlockBorderWrapper"]:has(.box-header-wrap),
    div[data-testid="stVerticalBlockBorderWrapper"] {
        position: relative !important;
        z-index: 10 !important;
        background-color: #ffffff !important;
        background: #ffffff !important;
        border: 1px solid rgba(209, 250, 229, 0.95) !important;
        border-radius: 18px !important;
        box-shadow: 0 10px 32px -4px rgba(6, 78, 59, 0.08), 0 2px 8px -2px rgba(0, 0, 0, 0.03) !important;
        padding: 1.75rem 1.75rem !important;
        margin-bottom: 1.5rem !important;
    }

    .box-header-wrap {
        margin-bottom: 1.25rem;
        padding-bottom: 0.85rem;
        border-bottom: 1px solid #f1f5f9;
    }

    .box-title {
        font-size: 1.15rem;
        font-weight: 700;
        color: #064e3b;
        letter-spacing: -0.01em;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    .box-subtitle {
        font-size: 0.84rem;
        color: #64748b;
        margin-top: 3px;
    }

    /* Clinical Box Typography */
    div[data-testid="stVerticalBlockBorderWrapper"] p,
    div[data-testid="stVerticalBlockBorderWrapper"] li {
        color: #1e293b !important;
        line-height: 1.75 !important;
        font-size: 0.95rem !important;
    }

    div[data-testid="stVerticalBlockBorderWrapper"] h1,
    div[data-testid="stVerticalBlockBorderWrapper"] h2,
    div[data-testid="stVerticalBlockBorderWrapper"] h3,
    div[data-testid="stVerticalBlockBorderWrapper"] h4 {
        color: #064e3b !important;
        font-weight: 700 !important;
        margin-top: 1.1rem !important;
        margin-bottom: 0.5rem !important;
    }

    /* Section Titles */
    .section-title {
        font-size: 1.2rem;
        font-weight: 700;
        color: #064e3b;
        margin-top: 1.75rem;
        margin-bottom: 0.9rem;
        letter-spacing: -0.01em;
    }

    /* Inputs Styling — Unified Hospital Container & Rounded Step Controls */
    div[data-testid="stNumberInput"] label,
    div[data-testid="stSelectbox"] label {
        font-size: 0.84rem !important;
        font-weight: 600 !important;
        color: #1e293b !important;
        margin-bottom: 4px !important;
    }

    /* Number Input Outer Wrapper */
    div[data-testid="stNumberInputContainer"] {
        background-color: #ffffff !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 10px !important;
        overflow: hidden !important;
        padding-right: 4px !important;
        transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
    }

    div[data-testid="stNumberInputContainer"]:focus-within {
        border-color: #059669 !important;
        box-shadow: 0 0 0 3px rgba(5, 150, 105, 0.15) !important;
    }

    /* Number Input Text Box Inside Container */
    div[data-testid="stNumberInput"] input,
    input[data-testid="stNumberInputField"] {
        background-color: transparent !important;
        border: none !important;
        border-radius: 0 !important;
        box-shadow: none !important;
        outline: none !important;
        color: #0f172a !important;
        font-weight: 600 !important;
        padding-left: 0.8rem !important;
    }

    div[data-testid="stNumberInput"] input:focus,
    input[data-testid="stNumberInputField"]:focus {
        border: none !important;
        box-shadow: none !important;
        outline: none !important;
    }

    /* Number Input Step Controls (+ / - Buttons) */
    div[data-testid="stNumberInput"] button {
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        border: none !important;
        border-radius: 7px !important;
        margin: 3px 2px !important;
        height: calc(100% - 6px) !important;
        width: 30px !important;
        background-color: #f1f5f9 !important;
        color: #334155 !important;
        cursor: pointer !important;
        transition: background-color 0.2s ease, color 0.2s ease, transform 0.1s ease !important;
    }

    div[data-testid="stNumberInput"] button * {
        color: inherit !important;
        fill: currentColor !important;
        stroke: currentColor !important;
    }

    div[data-testid="stNumberInput"] button:hover {
        background-color: #059669 !important;
        color: #ffffff !important;
    }

    div[data-testid="stNumberInput"] button:hover * {
        color: #ffffff !important;
        fill: #ffffff !important;
        stroke: #ffffff !important;
    }

    div[data-testid="stNumberInput"] button:active {
        background-color: #047857 !important;
        transform: scale(0.92) !important;
    }

    div[data-testid="stNumberInput"] button:disabled {
        background-color: transparent !important;
        color: #cbd5e1 !important;
        cursor: not-allowed !important;
    }

    div[data-testid="stNumberInput"] button:disabled * {
        color: #cbd5e1 !important;
        fill: #cbd5e1 !important;
    }

    /* Selectbox Styling to Match */
    div[data-testid="stSelectbox"] [data-baseweb="select"] {
        background-color: #ffffff !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 10px !important;
        color: #0f172a !important;
        font-weight: 600 !important;
        transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
    }

    div[data-testid="stSelectbox"] [data-baseweb="select"]:focus-within {
        border-color: #059669 !important;
        box-shadow: 0 0 0 3px rgba(5, 150, 105, 0.15) !important;
    }

    /* Analyze Button Styling: Centered in the middle, fits content width, green gradient */
    div[data-testid="stColumn"]:has(div.stButton),
    div[data-testid="stColumn"]:has([data-testid="stButton"]) {
        display: flex !important;
        justify-content: center !important;
        align-items: center !important;
    }

    div[data-testid="stElementContainer"]:has(div.stButton),
    div[data-testid="stElementContainer"]:has([data-testid="stButton"]) {
        display: flex !important;
        justify-content: center !important;
        align-items: center !important;
        width: 100% !important;
    }

    div.stButton,
    div[data-testid="stButton"] {
        display: flex !important;
        justify-content: center !important;
        align-items: center !important;
        text-align: center !important;
        width: 100% !important;
        margin-top: 1.25rem !important;
        margin-bottom: 0.5rem !important;
    }

    div.stButton button,
    div[data-testid="stButton"] button,
    button[data-testid="stBaseButton-primary"] {
        position: relative !important;
        overflow: hidden !important;
        margin-left: auto !important;
        margin-right: auto !important;
        width: 100% !important;
        min-width: 250px !important;
        max-width: 330px !important;
        padding: 0.8rem 2.25rem !important;
        border-radius: 9999px !important;
        font-size: 0.98rem !important;
        font-weight: 700 !important;
        letter-spacing: 0.03em !important;
        background: linear-gradient(135deg, #059669 0%, #047857 100%) !important;
        color: #ffffff !important;
        border: 1px solid #047857 !important;
        box-shadow: 0 4px 16px 0 rgba(5, 150, 105, 0.35) !important;
        transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1) !important;
        cursor: pointer !important;
        display: inline-flex !important;
        justify-content: center !important;
        align-items: center !important;
    }

    div.stButton button p,
    div[data-testid="stButton"] button p,
    button[data-testid="stBaseButton-primary"] p {
        margin: 0 !important;
        color: #ffffff !important;
        font-size: 0.98rem !important;
        font-weight: 700 !important;
        display: flex !important;
        align-items: center !important;
        gap: 8px !important;
        z-index: 2 !important;
        transition: all 0.2s ease !important;
    }

    /* Idle state: Hide loading graph until clicked */
    div.stButton button::after,
    div[data-testid="stButton"] button::after,
    button[data-testid="stBaseButton-primary"]::after {
        content: "";
        display: none !important;
        opacity: 0 !important;
    }

    div.stButton button:hover,
    div[data-testid="stButton"] button:hover,
    button[data-testid="stBaseButton-primary"]:hover {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%) !important;
        box-shadow: 0 6px 22px 0 rgba(5, 150, 105, 0.45) !important;
        transform: translateY(-1.5px) !important;
    }

    div.stButton button:active,
    div[data-testid="stButton"] button:active,
    button[data-testid="stBaseButton-primary"]:active {
        transform: translateY(1px) scale(0.98) !important;
    }

    /* Active Loading State: Shows when clicked and computing UNTIL FETCHED */
    div.stButton button.is-fetching,
    div[data-testid="stButton"] button.is-fetching,
    button[data-testid="stBaseButton-primary"].is-fetching {
        padding: 0.75rem 2rem 1.25rem !important;
        animation: medicalPulse 1.4s infinite ease-out !important;
        background: linear-gradient(135deg, #047857 0%, #065f46 100%) !important;
        opacity: 0.96 !important;
        cursor: wait !important;
        pointer-events: none !important;
    }

    div.stButton button.is-fetching p,
    div[data-testid="stButton"] button.is-fetching p,
    button[data-testid="stBaseButton-primary"].is-fetching p {
        font-size: 0 !important;
    }

    div.stButton button.is-fetching p::after,
    div[data-testid="stButton"] button.is-fetching p::after,
    button[data-testid="stBaseButton-primary"].is-fetching p::after {
        content: "🩺 ANALYZING CONDITION..." !important;
        font-size: 0.95rem !important;
        color: #ffffff !important;
        font-weight: 700 !important;
        letter-spacing: 0.03em !important;
        animation: heartBeat 1.2s infinite ease-in-out !important;
        display: inline-block !important;
    }

    /* ONLY show loading graph while clicked/fetching */
    div.stButton button.is-fetching::after,
    div[data-testid="stButton"] button.is-fetching::after,
    button[data-testid="stBaseButton-primary"].is-fetching::after {
        display: block !important;
        content: "" !important;
        position: absolute !important;
        bottom: 5px !important;
        left: 8% !important;
        right: 8% !important;
        height: 14px !important;
        background-image: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 500 70' preserveAspectRatio='none'><path d='M0,35 L120,35 L135,18 L150,52 L165,6 L180,64 L195,35 L320,35 L335,18 L350,52 L365,6 L380,64 L395,35 L500,35' fill='none' stroke='%23ffffff' stroke-width='3.2' stroke-linecap='round' stroke-linejoin='round'/></svg>") !important;
        background-size: 240px 100% !important;
        background-repeat: repeat-x !important;
        opacity: 1 !important;
        animation: ecgButtonFlow 1.1s linear infinite !important;
        filter: drop-shadow(0 0 4px rgba(255, 255, 255, 0.8)) !important;
        pointer-events: none !important;
        z-index: 1 !important;
    }

    @keyframes ecgButtonFlow {
        0% { background-position-x: 0; }
        100% { background-position-x: 240px; }
    }

    /* Metric Cards with Crisp Hospital Elevation */
    div[data-testid="stMetric"] {
        background: #f8fafc !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 14px !important;
        padding: 1.1rem 1.25rem !important;
        box-shadow: 0 2px 8px -2px rgba(0, 0, 0, 0.04) !important;
        transition: transform 0.15s ease, box-shadow 0.15s ease, border-color 0.15s ease !important;
    }

    div[data-testid="stMetric"]:hover {
        border-color: #a7f3d0 !important;
        background: #ffffff !important;
        box-shadow: 0 6px 20px -2px rgba(5, 150, 105, 0.1) !important;
        transform: translateY(-1px) !important;
    }

    div[data-testid="stMetricLabel"] {
        font-size: 0.82rem !important;
        font-weight: 600 !important;
        color: #475569 !important;
    }

    div[data-testid="stMetricValue"] {
        font-size: 1.65rem !important;
        font-weight: 800 !important;
        color: #064e3b !important;
    }

    /* AI Consultation Box with Solid White Card & Emerald Accent */
    .ai-box {
        padding: 1.75rem 2rem;
        border-radius: 18px;
        background-color: #ffffff !important;
        background: #ffffff !important;
        border: 1px solid rgba(209, 250, 229, 0.95) !important;
        border-left: 6px solid #059669 !important;
        box-shadow: 0 10px 32px -4px rgba(6, 78, 59, 0.08), 0 2px 8px -2px rgba(0, 0, 0, 0.03) !important;
        line-height: 1.75;
        color: #1e293b;
        font-size: 0.96rem;
        position: relative !important;
        z-index: 10 !important;
    }

    /* Loading Animation Card & Medical ECG line with Frosted Glassmorphism */
    .medical-loader-card {
        background: rgba(255, 255, 255, 0.92) !important;
        backdrop-filter: blur(16px) !important;
        -webkit-backdrop-filter: blur(16px) !important;
        border: 1px solid #a7f3d0 !important;
        border-radius: 14px !important;
        padding: 1.25rem 1.75rem !important;
        margin: 1.25rem 0 !important;
        box-shadow: 0 6px 24px -2px rgba(5, 150, 105, 0.14) !important;
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: 12px;
    }

    .ecg-loader-wrapper {
        display: flex;
        align-items: center;
        gap: 16px;
        width: 100%;
        max-width: 480px;
    }

    .heartbeat-icon {
        font-size: 1.8rem;
        animation: heartBeat 1.2s infinite ease-in-out;
    }

    .pulse-wave {
        flex: 1;
        height: 44px;
        background: #f0fdf4;
        border-radius: 8px;
        border: 1px solid #d1fae5;
        padding: 4px 8px;
        overflow: hidden;
    }

    .ecg-svg {
        width: 100%;
        height: 100%;
    }

    .ecg-line {
        fill: none;
        stroke: #059669;
        stroke-width: 2.5;
        stroke-linecap: round;
        stroke-linejoin: round;
        stroke-dasharray: 600;
        stroke-dashoffset: 600;
        animation: ecgFlow 2s linear infinite;
    }

    .loader-status-text {
        text-align: center;
    }

    .loader-title {
        display: block;
        font-size: 0.98rem;
        font-weight: 700;
        color: #064e3b;
        margin-bottom: 2px;
    }

    .loader-subtitle {
        display: block;
        font-size: 0.82rem;
        color: #475569;
    }

    /* Keyframes */
    @keyframes heartBeat {
        0% { transform: scale(1); }
        14% { transform: scale(1.18); }
        28% { transform: scale(1); }
        42% { transform: scale(1.14); }
        70% { transform: scale(1); }
    }

    @keyframes ecgFlow {
        0% { stroke-dashoffset: 600; }
        100% { stroke-dashoffset: 0; }
    }

    @keyframes livePulse {
        0% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }
        70% { box-shadow: 0 0 0 6px rgba(16, 185, 129, 0); }
        100% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
    }

    @keyframes medicalPulse {
        0% { box-shadow: 0 0 0 0 rgba(5, 150, 105, 0.6); }
        70% { box-shadow: 0 0 0 14px rgba(5, 150, 105, 0); }
        100% { box-shadow: 0 0 0 0 rgba(5, 150, 105, 0); }
    }

    /* Streamlit Native Spinner Customization */
    div[data-testid="stSpinner"] {
        background: #ecfdf5 !important;
        border: 1px solid #a7f3d0 !important;
        border-radius: 12px !important;
        padding: 0.85rem 1.25rem !important;
        color: #065f46 !important;
        font-weight: 600 !important;
    }

    div[data-testid="stSpinner"] i {
        border-top-color: #059669 !important;
    }

    /* Utility iframes (scripts & canvas) */
    iframe.stIFrame,
    iframe[data-testid="stIFrame"] {
        border: none !important;
    }

    iframe.stIFrame[style*="height: 1px"],
    iframe[data-testid="stIFrame"][style*="height: 1px"],
    iframe.stIFrame[height="1"],
    iframe[data-testid="stIFrame"][height="1"] {
        position: absolute !important;
        width: 0 !important;
        height: 0 !important;
        opacity: 0 !important;
        pointer-events: none !important;
        margin: 0 !important;
        padding: 0 !important;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ================================================================
# 4. PROJECT PATHS
# ================================================================

MODEL_PATH = BASE_DIR / "models" / "xgb_model.json"

DATA_PATH = BASE_DIR / "Operational data.xlsx"

RAG_DIR = BASE_DIR / "hosp_ai_embeddings"

LOGO_SOURCE_PATH = Path("/home/zoro/Downloads/hosp-ai logo.png")

LOGO_ASSET_PATH = BASE_DIR / "assets" / "logo.png"


@st.cache_data
def get_optimized_logo_base64():
    """Load or generate an optimized, bandwidth-friendly 96x96 logo data URI."""
    try:
        if not LOGO_ASSET_PATH.exists() and LOGO_SOURCE_PATH.exists():
            LOGO_ASSET_PATH.parent.mkdir(parents=True, exist_ok=True)
            from PIL import Image

            img = Image.open(LOGO_SOURCE_PATH)
            resized = img.resize((96, 96), Image.Resampling.LANCZOS)
            resized.save(LOGO_ASSET_PATH, "PNG", optimize=True)

        if LOGO_ASSET_PATH.exists():
            with open(LOGO_ASSET_PATH, "rb") as f:
                encoded = base64.b64encode(f.read()).decode("utf-8")
            return f"data:image/png;base64,{encoded}"
    except Exception:
        pass
    return None


TEXT_IMG_SOURCE_PATH = Path("/home/zoro/Downloads/hosp-ai text.png")

TEXT_IMG_ASSET_PATH = BASE_DIR / "assets" / "hosp_ai_text.png"


@st.cache_data
def get_optimized_text_img_base64():
    """Load or generate an optimized, bandwidth-friendly brand text image data URI."""
    try:
        if not TEXT_IMG_ASSET_PATH.exists() and TEXT_IMG_SOURCE_PATH.exists():
            TEXT_IMG_ASSET_PATH.parent.mkdir(parents=True, exist_ok=True)
            from PIL import Image

            img = Image.open(TEXT_IMG_SOURCE_PATH)
            bbox = img.getbbox()
            if bbox:
                img = img.crop(bbox)
            w, h = img.size
            target_w = 600
            target_h = int(h * (target_w / w))
            resized = img.resize((target_w, target_h), Image.Resampling.LANCZOS)
            resized.save(TEXT_IMG_ASSET_PATH, "PNG", optimize=True)

        if TEXT_IMG_ASSET_PATH.exists():
            with open(TEXT_IMG_ASSET_PATH, "rb") as f:
                encoded = base64.b64encode(f.read()).decode("utf-8")
            return f"data:image/png;base64,{encoded}"
    except Exception:
        pass
    return None


# ================================================================
# 5. LOAD XGBOOST MODEL
# ================================================================


@st.cache_resource
def load_xgboost_model():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"XGBoost model not found:\n{MODEL_PATH}")

    model = XGBRegressor()

    model.load_model(str(MODEL_PATH))

    return model


# ================================================================
# 6. LOAD HISTORICAL DATA
# ================================================================


@st.cache_data
def load_historical_data():

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Operational data file not found:\n{DATA_PATH}")

    df = pd.read_excel(DATA_PATH, sheet_name="Hospital_Operational_Data")

    df["Timestamp"] = pd.to_datetime(df["Timestamp"])

    df = df.sort_values("Timestamp").reset_index(drop=True)

    return df


# ================================================================
# 7. FEATURE ENGINEERING
# ================================================================


def create_model_features(history):

    history = history.copy()

    # ------------------------------------------------------------
    # Time features
    # ------------------------------------------------------------

    history["Hour"] = history["Timestamp"].dt.hour

    history["DayOfWeek"] = history["Timestamp"].dt.dayofweek

    history["Month"] = history["Timestamp"].dt.month

    history["IsWeekend"] = (history["DayOfWeek"] >= 5).astype(int)

    # ------------------------------------------------------------
    # ICU occupancy lag features
    # ------------------------------------------------------------

    for lag in [1, 3, 6, 12, 24, 48, 72]:
        history[f"ICU_Occupancy_Lag_{lag}h"] = history["ICU_Occupancy_Pct"].shift(lag)

    # ------------------------------------------------------------
    # ICU admission lag features
    # ------------------------------------------------------------

    for lag in [1, 6, 24]:
        history[f"ICU_Admissions_Lag_{lag}h"] = history["ICU_Admissions"].shift(lag)

    # ------------------------------------------------------------
    # Rolling features
    # ------------------------------------------------------------

    history["ICU_Occupancy_Rolling_6h"] = history["ICU_Occupancy_Pct"].rolling(6).mean()

    history["ICU_Occupancy_Rolling_12h"] = (
        history["ICU_Occupancy_Pct"].rolling(12).mean()
    )

    history["ICU_Occupancy_Rolling_24h"] = (
        history["ICU_Occupancy_Pct"].rolling(24).mean()
    )

    history["ER_Arrivals_Rolling_6h"] = history["ER_Arrivals"].rolling(6).mean()

    history["ICU_Admissions_Rolling_24h"] = history["ICU_Admissions"].rolling(24).mean()

    return history


# ================================================================
# 8. CREATE CURRENT INPUT ROW
# ================================================================


def create_current_row(
    history,
    er_arrivals,
    ambulance_arrivals,
    general_admissions,
    icu_admissions,
    icu_discharges,
    icu_transfers_in,
    icu_transfers_out,
    icu_occupied_beds,
    icu_available_beds,
    hdu_occupied_beds,
    general_occupied_beds,
    private_occupied_beds,
    ventilators_in_use,
    ventilators_available,
    icu_staff_available,
    staff_shortage_flag,
    elective_surgeries,
):

    # ------------------------------------------------------------
    # ICU occupancy
    # ------------------------------------------------------------

    total_icu_beds = icu_occupied_beds + icu_available_beds

    if total_icu_beds > 0:
        icu_occupancy_pct = (icu_occupied_beds / total_icu_beds) * 100

    else:
        icu_occupancy_pct = 0.0

    # ------------------------------------------------------------
    # Total occupied beds
    # ------------------------------------------------------------

    total_occupied_beds = (
        icu_occupied_beds
        + hdu_occupied_beds
        + general_occupied_beds
        + private_occupied_beds
    )

    # ------------------------------------------------------------
    # Hospital occupancy
    # ------------------------------------------------------------

    capacity_reference = history["Total_Occupied_Beds"].max()

    if capacity_reference > 0:
        hospital_occupancy_pct = (total_occupied_beds / capacity_reference) * 100

    else:
        hospital_occupancy_pct = 0.0

    # ------------------------------------------------------------
    # Timestamp
    # ------------------------------------------------------------

    latest_timestamp = pd.to_datetime(history["Timestamp"]).max()

    new_timestamp = latest_timestamp + pd.Timedelta(hours=1)

    # ------------------------------------------------------------
    # Current row
    # ------------------------------------------------------------

    row = pd.DataFrame(
        [
            {
                "Timestamp": new_timestamp,
                "ER_Arrivals": er_arrivals,
                "Ambulance_Arrivals": ambulance_arrivals,
                "General_Admissions": general_admissions,
                "ICU_Admissions": icu_admissions,
                "ICU_Discharges": icu_discharges,
                "ICU_Transfers_In": icu_transfers_in,
                "ICU_Transfers_Out": icu_transfers_out,
                "ICU_Occupied_Beds": icu_occupied_beds,
                "ICU_Available_Beds": icu_available_beds,
                "ICU_Occupancy_Pct": icu_occupancy_pct,
                "HDU_Occupied_Beds": hdu_occupied_beds,
                "General_Occupied_Beds": general_occupied_beds,
                "Private_Occupied_Beds": private_occupied_beds,
                "Total_Occupied_Beds": total_occupied_beds,
                "Hospital_Occupancy_Pct": hospital_occupancy_pct,
                "Ventilators_In_Use": ventilators_in_use,
                "Ventilators_Available": ventilators_available,
                "ICU_Staff_Available": icu_staff_available,
                "Staff_Shortage_Flag": staff_shortage_flag,
                "Elective_Surgeries": elective_surgeries,
            }
        ]
    )

    return row


# ================================================================
# 9. DETERMINISTIC OPERATIONAL RISK ENGINE
# ================================================================


def calculate_operational_risk(
    current_icu_occupancy,
    predicted_icu_occupancy,
    icu_available_beds,
    staff_shortage_flag,
    ventilators_available,
    icu_admissions,
    icu_discharges,
    er_arrivals,
    ambulance_arrivals,
):

    components = {}

    risk_factors = []

    positive_trends = []

    # ------------------------------------------------------------
    # Current ICU occupancy
    # ------------------------------------------------------------

    if current_icu_occupancy >= 90:
        occupancy_score = 3

        risk_factors.append("Critical ICU occupancy")

    elif current_icu_occupancy >= 80:
        occupancy_score = 2

        risk_factors.append("High ICU occupancy")

    elif current_icu_occupancy >= 70:
        occupancy_score = 1

    else:
        occupancy_score = 0

    components["Current ICU Occupancy"] = occupancy_score

    # ------------------------------------------------------------
    # Forecast trend
    # ------------------------------------------------------------

    forecast_change = predicted_icu_occupancy - current_icu_occupancy

    if forecast_change >= 10:
        forecast_score = 2

    elif forecast_change >= 3:
        forecast_score = 1

    elif forecast_change <= -20:
        forecast_score = -1

        positive_trends.append(
            "Forecast indicates a substantial decrease in ICU occupancy"
        )

    else:
        forecast_score = 0

    components["24h Forecast Trend"] = forecast_score

    # ------------------------------------------------------------
    # ICU beds
    # ------------------------------------------------------------

    if icu_available_beds <= 2:
        bed_score = 2

        risk_factors.append("Very limited ICU bed availability")

    elif icu_available_beds <= 5:
        bed_score = 1

        risk_factors.append("Limited ICU bed availability")

    else:
        bed_score = 0

    components["ICU Bed Availability"] = bed_score

    # ------------------------------------------------------------
    # Staffing
    # ------------------------------------------------------------

    if staff_shortage_flag == 1:
        staff_score = 1

        risk_factors.append("ICU staffing shortage is active")

    else:
        staff_score = 0

    components["Staff Shortage"] = staff_score

    # ------------------------------------------------------------
    # Ventilators
    # ------------------------------------------------------------

    if ventilators_available <= 1:
        ventilator_score = 2

        risk_factors.append("Critical ventilator availability")

    elif ventilators_available <= 4:
        ventilator_score = 1

        risk_factors.append("Limited ventilator availability")

    else:
        ventilator_score = 0

    components["Ventilator Availability"] = ventilator_score

    # ------------------------------------------------------------
    # Patient flow
    # ------------------------------------------------------------

    net_flow = icu_admissions - icu_discharges

    if net_flow >= 3:
        flow_score = 2

        risk_factors.append("Strong positive ICU patient-flow pressure")

    elif net_flow > 0:
        flow_score = 1

        risk_factors.append("Positive ICU patient-flow pressure")

    else:
        flow_score = 0

    components["ICU Patient Flow"] = flow_score

    # ------------------------------------------------------------
    # Emergency inflow
    # ------------------------------------------------------------

    if er_arrivals >= 50 or ambulance_arrivals >= 15:
        emergency_score = 2

        risk_factors.append("Very high emergency patient inflow")

    elif er_arrivals >= 40 or ambulance_arrivals >= 10:
        emergency_score = 1

        risk_factors.append("High emergency patient inflow")

    else:
        emergency_score = 0

    components["Emergency Inflow"] = emergency_score

    # ------------------------------------------------------------
    # Final score
    # ------------------------------------------------------------

    raw_score = sum(components.values())

    risk_score = round(min((raw_score / 14) * 10, 10), 1)

    # ------------------------------------------------------------
    # Risk level
    # ------------------------------------------------------------

    if risk_score >= 8:
        risk_level = "CRITICAL"

    elif risk_score >= 6:
        risk_level = "HIGH"

    elif risk_score >= 3:
        risk_level = "MODERATE"

    else:
        risk_level = "LOW"

    return {
        "risk_score": risk_score,
        "risk_level": risk_level,
        "raw_score": raw_score,
        "risk_components": components,
        "risk_factors": risk_factors,
        "positive_trends": positive_trends,
        "forecast_change": forecast_change,
    }


# ================================================================
# 10. LOAD RAG EMBEDDINGS + POLICY DATA
# ================================================================


@st.cache_resource
def load_rag_resources():

    if not RAG_DIR.exists():
        raise FileNotFoundError(f"RAG folder not found:\n{RAG_DIR}")

    # ------------------------------------------------------------
    # Find embedding file
    # ------------------------------------------------------------

    embedding_candidates = [
        RAG_DIR / "document_embeddings.npy",
        RAG_DIR / "policy_embeddings.npy",
        RAG_DIR / "embeddings.npy",
    ]

    embedding_path = None

    for path in embedding_candidates:
        if path.exists():
            embedding_path = path

            break

    if embedding_path is None:
        npy_files = list(RAG_DIR.glob("*.npy"))

        if len(npy_files) == 1:
            embedding_path = npy_files[0]

        else:
            raise FileNotFoundError("Could not identify the RAG embedding .npy file.")

    embeddings = np.load(embedding_path)

    # ------------------------------------------------------------
    # Find policy/document CSV
    # ------------------------------------------------------------

    csv_candidates = [
        RAG_DIR / "policy_documents.csv",
        RAG_DIR / "policies.csv",
        RAG_DIR / "knowledge_base.csv",
        RAG_DIR / "embedded_knowledge_base.csv",
    ]

    csv_path = None

    for path in csv_candidates:
        if path.exists():
            csv_path = path

            break

    if csv_path is None:
        csv_files = list(RAG_DIR.glob("*.csv"))

        if len(csv_files) == 1:
            csv_path = csv_files[0]

        else:
            raise FileNotFoundError("Could not identify the RAG policy CSV file.")

    policy_df = pd.read_csv(csv_path)

    # ------------------------------------------------------------
    # Find text column
    # ------------------------------------------------------------

    possible_text_columns = [
        "Policy_Content",
        "policy_content",
        "Content",
        "content",
        "text",
        "Text",
        "document",
        "Document",
    ]

    text_column = None

    for column in possible_text_columns:
        if column in policy_df.columns:
            text_column = column

            break

    if text_column is None:
        raise ValueError(
            "Could not identify the policy text column. "
            f"Available columns: {list(policy_df.columns)}"
        )

    # ------------------------------------------------------------
    # Validate embedding/document count
    # ------------------------------------------------------------

    if len(embeddings) != len(policy_df):
        raise ValueError(
            "RAG embedding count does not match "
            "the number of policy documents.\n"
            f"Embeddings: {len(embeddings)}\n"
            f"Documents: {len(policy_df)}"
        )

    # ------------------------------------------------------------
    # Load sentence transformer
    # ------------------------------------------------------------

    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    return (embeddings, policy_df, text_column, embedding_model)


# ================================================================
# 11. RETRIEVE RELEVANT POLICIES
# ================================================================


def retrieve_policies(
    query, embeddings, policy_df, text_column, embedding_model, top_k=5
):

    query_embedding = embedding_model.encode([query], convert_to_numpy=True)

    similarity_scores = cosine_similarity(query_embedding, embeddings)[0]

    top_indices = np.argsort(similarity_scores)[::-1][:top_k]

    results = []

    for index in top_indices:
        row = policy_df.iloc[index]

        result = {
            "document": row.get(
                "Document",
                row.get(
                    "document", row.get("Policy", row.get("policy", "Hospital Policy"))
                ),
            ),
            "category": row.get("Category", row.get("category", "OPERATIONAL")),
            "content": row[text_column],
            "similarity": float(similarity_scores[index]),
        }

        results.append(result)

    return results


# ================================================================
# 12. BUILD RISK CONTEXT
# ================================================================


def build_risk_context(
    risk_result,
    current_icu_occupancy,
    predicted_icu_occupancy,
    icu_available_beds,
    ventilators_available,
    icu_staff_available,
    staff_shortage_input,
    icu_admissions,
    icu_discharges,
    er_arrivals,
    ambulance_arrivals,
):

    risk_factors_text = "\n".join(
        [f"- {factor}" for factor in risk_result["risk_factors"]]
    )

    if not risk_factors_text:
        risk_factors_text = "- No major operational risk factors detected."

    return f"""
OPERATIONAL RISK ASSESSMENT

Risk Score:
{risk_result["risk_score"]} / 10

Risk Level:
{risk_result["risk_level"]}

Current ICU Occupancy:
{current_icu_occupancy:.2f}%

Predicted 24-Hour ICU Occupancy:
{predicted_icu_occupancy:.2f}%

ICU Beds Available:
{icu_available_beds}

Ventilators Available:
{ventilators_available}

ICU Staff Available:
{icu_staff_available}

ICU Staff Shortage:
{staff_shortage_input}

ICU Admissions:
{icu_admissions}

ICU Discharges:
{icu_discharges}

Emergency Department Arrivals:
{er_arrivals}

Ambulance Arrivals:
{ambulance_arrivals}

Key Risk Factors:
{risk_factors_text}
"""


# ================================================================
# 13. BUILD GROUNDED GEMINI PROMPT
# ================================================================


def build_grounded_prompt(
    risk_context, current_icu_occupancy, predicted_icu_occupancy, retrieved_policies
):

    policy_context_parts = []

    for i, policy in enumerate(retrieved_policies, start=1):
        policy_context_parts.append(
            f"""
--- POLICY {i} ---

Document:
{policy["document"]}

Category:
{policy["category"]}

Similarity Score:
{policy["similarity"]:.4f}

Policy Content:
{policy["content"]}
"""
        )

    policy_context = "\n".join(policy_context_parts)

    prompt = f"""
You are the explanation and recommendation layer of
HOSP-AI COMMAND, an administrative hospital operational
decision-support prototype.

IMPORTANT ARCHITECTURE RULES:

1. The XGBoost model produces the numerical 24-hour
   ICU occupancy forecast.

2. The deterministic Operational Risk Engine produces
   the numerical risk score and risk level.

3. Retrieved hospital policies provide the operational
   knowledge context for recommendations.

4. You MUST NOT invent, change, or recalculate the
   numerical risk score.

5. You MUST NOT present this system as a clinical
   decision-making system.

6. Recommendations must be operational and grounded
   in the retrieved policies.

7. Do not invent hospital policies that are not present
   in the retrieved policy context.

8. Clearly distinguish current operational pressure from
   the 24-hour forecast.

9. Human review is required before any real operational
   action.

------------------------------------------------------------
CURRENT HOSPITAL OBSERVATION
------------------------------------------------------------

Current ICU Occupancy:
{current_icu_occupancy:.2f}%

Predicted 24-Hour ICU Occupancy:
{predicted_icu_occupancy:.2f}%

{risk_context}

------------------------------------------------------------
RETRIEVED HOSPITAL POLICY CONTEXT
------------------------------------------------------------

{policy_context}

------------------------------------------------------------
TASK
------------------------------------------------------------

Generate a concise administrative operational
decision-support summary using ONLY this structure:

1. Operational Situation

Briefly explain the current operational situation using
the observed values.

2. Predicted Risk

Report the exact ML forecast, exact deterministic
risk score, and exact deterministic risk level.

3. Key Risk Factors

List the important operational risk factors.

4. Recommended Actions

Provide practical administrative recommendations
grounded in the retrieved hospital policies.

5. Forecast Trend

Briefly explain whether the 24-hour forecast indicates
increasing, stable, or decreasing ICU occupancy.

Do NOT include:

- Supporting Policies section
- Technical details
- Model details
- Risk component calculations
- Similarity scores
- Clinical treatment recommendations
- Long limitation sections

Keep the response concise and suitable for a hospital
operations dashboard.
"""

    return prompt


# ================================================================
# 14. GEMINI CLIENT
# ================================================================


@st.cache_resource
def load_gemini_client():

    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY was not found in the .env file.")

    return genai.Client(api_key=GEMINI_API_KEY)


# ================================================================
# 15. GENERATE AI DECISION SUPPORT
# ================================================================


def generate_ai_decision_support(prompt):

    client = load_gemini_client()

    response = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)

    if not response.text:
        raise ValueError("Gemini returned an empty response.")

    return response.text


# ================================================================
# 16. KEXSIO THREAD BACKGROUND & FLOATING NAVBAR
# ================================================================

st.markdown(
    """
    <div class="kexsio-dot-grid" aria-hidden="true"></div>
    """,
    unsafe_allow_html=True,
)

st.iframe(
    """
    <!DOCTYPE html>
    <html>
    <head>
    <meta charset="UTF-8">
    <style>
      * { margin: 0; padding: 0; box-sizing: border-box; }
      html, body {
        width: 100%;
        height: 100%;
        overflow: hidden;
        background: transparent;
      }
      #kexsio-thread-canvas {
        position: fixed;
        top: 0;
        left: 0;
        width: 100vw;
        height: 100vh;
        z-index: 0;
        pointer-events: none;
      }
    </style>
    </head>
    <body>
    <canvas id="kexsio-thread-canvas"></canvas>
    <script>
    (function() {
      var targetDoc = document;
      var targetWin = window;
      var isParent = false;

      try {
        if (window.parent && window.parent.document && window.parent.document.body) {
          targetDoc = window.parent.document;
          targetWin = window.parent;
          isParent = true;
        }
      } catch (e) {
        // Sandboxed iframe mode fallback
      }

      if (targetDoc.getElementById('kexsio-thread-canvas')) {
        return;
      }

      var canvas;
      if (isParent) {
        canvas = targetDoc.createElement('canvas');
        canvas.id = 'kexsio-thread-canvas';
        canvas.style.position = 'fixed';
        canvas.style.top = '0';
        canvas.style.left = '0';
        canvas.style.width = '100vw';
        canvas.style.height = '100vh';
        canvas.style.zIndex = '0';
        canvas.style.pointerEvents = 'none';
        targetDoc.body.insertBefore(canvas, targetDoc.body.firstChild);
      } else {
        canvas = document.getElementById('kexsio-thread-canvas');
      }

      var ctx = canvas.getContext('2d');
      var width = 0;
      var height = 0;
      var dpr = targetWin.devicePixelRatio || 1;

      function resize() {
        width = targetWin.innerWidth || document.documentElement.clientWidth;
        height = targetWin.innerHeight || document.documentElement.clientHeight;
        canvas.width = Math.floor(width * dpr);
        canvas.height = Math.floor(height * dpr);
        canvas.style.width = width + 'px';
        canvas.style.height = height + 'px';
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      }
      targetWin.addEventListener('resize', resize);
      resize();

      var mouse = {
        x: width * 0.5,
        y: height * 0.5,
        targetX: width * 0.5,
        targetY: height * 0.5,
        active: false
      };

      targetWin.addEventListener('mousemove', function(e) {
        mouse.targetX = e.clientX;
        mouse.targetY = e.clientY;
        mouse.active = true;
      });

      targetWin.addEventListener('mouseleave', function() {
        mouse.active = false;
      });

      var THREAD_COUNT = 48;
      var POINTS_PER_LINE = 55;

      var colorPalette = [
        { r: 5, g: 150, b: 105 },
        { r: 16, g: 185, b: 129 },
        { r: 13, g: 148, b: 136 },
        { r: 4, g: 120, b: 87 },
        { r: 52, g: 211, b: 153 }
      ];

      var threads = [];
      for (var i = 0; i < THREAD_COUNT; i++) {
        var p = i / (THREAD_COUNT - 1);
        var colorIdx = Math.floor(p * (colorPalette.length - 1));
        var nextIdx = Math.min(colorIdx + 1, colorPalette.length - 1);
        var blend = (p * (colorPalette.length - 1)) - colorIdx;

        var c1 = colorPalette[colorIdx];
        var c2 = colorPalette[nextIdx];
        var r = Math.round(c1.r + (c2.r - c1.r) * blend);
        var g = Math.round(c1.g + (c2.g - c1.g) * blend);
        var b = Math.round(c1.b + (c2.b - c1.b) * blend);

        var distFromCenter = Math.abs(p - 0.5) * 2;
        var baseAlpha = 0.55 * (1 - Math.pow(distFromCenter, 1.4)) + 0.12;
        var lineWidth = 0.9 + (1 - distFromCenter) * 0.8;

        threads.push({
          p: p,
          r: r, g: g, b: b,
          baseAlpha: baseAlpha,
          lineWidth: lineWidth,
          phaseOffset: p * Math.PI * 1.85,
          freqMultiplier: 0.85 + (p * 0.3),
          speedMultiplier: 0.9 + (p * 0.2)
        });
      }

      var startTime = performance.now();
      var reducedMotion = false;
      try {
        if (targetWin.matchMedia && targetWin.matchMedia('(prefers-reduced-motion: reduce)').matches) {
          reducedMotion = true;
        }
      } catch (e) {}

      function animate(now) {
        targetWin.requestAnimationFrame(animate);

        var elapsed = reducedMotion ? 1.5 : (now - startTime) * 0.001;

        if (mouse.active) {
          mouse.x += (mouse.targetX - mouse.x) * 0.04;
          mouse.y += (mouse.targetY - mouse.y) * 0.04;
        } else {
          mouse.x += (width * 0.5 - mouse.x) * 0.02;
          mouse.y += (height * 0.5 - mouse.y) * 0.02;
        }

        ctx.clearRect(0, 0, width, height);

        var centerY = height * 0.52;
        var baseWaveSpeed = reducedMotion ? 0 : 0.45;

        for (var i = 0; i < THREAD_COUNT; i++) {
          var th = threads[i];
          var p = th.p;
          var centeredP = (p - 0.5) * 2;

          ctx.beginPath();
          var points = [];
          for (var s = 0; s <= POINTS_PER_LINE; s++) {
            var u = s / POINTS_PER_LINE;
            var x = u * (width + 120) - 60;

            var w1 = Math.sin(u * Math.PI * 2.2 + elapsed * baseWaveSpeed * th.speedMultiplier + th.phaseOffset) * 65;
            var w2 = Math.cos(u * Math.PI * 3.6 - elapsed * (baseWaveSpeed * 0.7) + th.phaseOffset * 0.6) * 35;
            var w3 = Math.sin(u * Math.PI * 1.1 + elapsed * 0.3) * 25;

            var pinchDist = Math.abs(u - 0.28);
            var spreadEnvelope = 40 + (Math.sin(pinchDist * Math.PI * 1.4) + 1.0) * 85;
            var threadSpreadY = centeredP * spreadEnvelope;

            var mouseDisplacement = 0;
            if (mouse.active) {
              var dx = x - mouse.x;
              var dy = centerY - mouse.y;
              var distSq = dx * dx + dy * dy;
              var maxDist = 320;
              if (distSq < maxDist * maxDist) {
                var dist = Math.sqrt(distSq);
                var force = (1 - dist / maxDist);
                mouseDisplacement = (mouse.y - centerY) * force * 0.38 * (1 - Math.abs(centeredP) * 0.5);
              }
            }

            var y = centerY + w1 + w2 + w3 + threadSpreadY + mouseDisplacement;
            points.push({ x: x, y: y });
          }

          ctx.moveTo(points[0].x, points[0].y);
          for (var j = 1; j < points.length - 1; j++) {
            var xc = (points[j].x + points[j + 1].x) * 0.5;
            var yc = (points[j].y + points[j + 1].y) * 0.5;
            ctx.quadraticCurveTo(points[j].x, points[j].y, xc, yc);
          }
          ctx.lineTo(points[points.length - 1].x, points[points.length - 1].y);

          var grad = ctx.createLinearGradient(0, 0, width, 0);
          grad.addColorStop(0, 'rgba(' + th.r + ',' + th.g + ',' + th.b + ', 0)');
          grad.addColorStop(0.12, 'rgba(' + th.r + ',' + th.g + ',' + th.b + ',' + (th.baseAlpha * 0.65) + ')');
          grad.addColorStop(0.5, 'rgba(' + th.r + ',' + th.g + ',' + th.b + ',' + th.baseAlpha + ')');
          grad.addColorStop(0.88, 'rgba(' + th.r + ',' + th.g + ',' + th.b + ',' + (th.baseAlpha * 0.65) + ')');
          grad.addColorStop(1, 'rgba(' + th.r + ',' + th.g + ',' + th.b + ', 0)');

          ctx.strokeStyle = grad;
          ctx.lineWidth = th.lineWidth;
          ctx.stroke();
        }
      }

      // Real-time click listener to display the ECG loading graph on the button until fetched
      function setupButtonListener(doc) {
        if (!doc) return;
        try {
          // Clear any leftover is-fetching class from previous runs
          doc.querySelectorAll('.is-fetching').forEach(function(b) {
            b.classList.remove('is-fetching');
          });

          doc.addEventListener('click', function(e) {
            var btn = e.target.closest('button[data-testid="stBaseButton-primary"], div.stButton button');
            if (btn) {
              btn.classList.add('is-fetching');
              setTimeout(function() {
                btn.classList.remove('is-fetching');
              }, 25000);
            }
          }, true);
        } catch (err) {}
      }

      setupButtonListener(targetDoc);
      if (isParent) setupButtonListener(document);

      targetWin.requestAnimationFrame(animate);
    })();
    </script>
    </body>
    </html>
    """,
    height=1,
)

logo_uri = get_optimized_logo_base64()
logo_element = (
    f'<img src="{logo_uri}" class="nav-logo-img" alt="HOSP-AI Logo" />'
    if logo_uri
    else '<span style="font-size: 1.35rem;">🏥</span>'
)

st.markdown(
    f"""
    <div class="floating-navbar">
        <div class="nav-brand">
            <div class="nav-logo-wrap">
                {logo_element}
            </div>
            <div class="nav-brand-text">
                <span class="nav-title">HOSP<span class="brand-accent">-AI</span> COMMAND</span>
            </div>
        </div>
        <div class="nav-pills">
            <div class="nav-pill active-pill">
                <span class="pulse-dot"></span>
                <span>SYSTEM ACTIVE</span>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

text_img_uri = get_optimized_text_img_base64()
if text_img_uri:
    st.markdown(
        f"""
        <div class="hero-brand-wrap">
            <img src="{text_img_uri}" class="hero-brand-img" alt="HOSP-AI COMMAND" />
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown(
    """
    <div class="disclaimer-banner">
        <span style="font-size: 1.25rem;">⚕️</span>
        <div>
            <strong>Operational Prototype:</strong> Administrative operational decision-support system based on synthetic research data. Not intended for individual clinical decision-making.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ================================================================
# 17. LOAD MODEL + DATA + RAG
# ================================================================

try:
    model = load_xgboost_model()

    historical_data = load_historical_data()

    (rag_embeddings, policy_df, rag_text_column, embedding_model) = load_rag_resources()

except Exception as e:
    st.error("Unable to load HOSP-AI COMMAND resources.")

    st.exception(e)

    st.stop()


# ================================================================
# 18. HOSPITAL INPUT SECTION & CONTROL PANEL (ENCLOSED BOX)
# ================================================================

with st.container(border=True):
    st.markdown(
        """
        <div class="box-header-wrap">
            <div class="box-title">
                🏥 CURRENT HOSPITAL OPERATIONAL CONDITIONS
            </div>
            <div class="box-subtitle">
                Configure current patient census, intake inflow, and critical care resource availability
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        er_arrivals = st.number_input("ER Arrivals", min_value=0, value=42, step=1)

        ambulance_arrivals = st.number_input(
            "Ambulance Arrivals", min_value=0, value=11, step=1
        )

        general_admissions = st.number_input(
            "General Admissions", min_value=0, value=30, step=1
        )

        icu_admissions = st.number_input("ICU Admissions", min_value=0, value=8, step=1)

    with col2:
        icu_discharges = st.number_input("ICU Discharges", min_value=0, value=5, step=1)

        icu_transfers_in = st.number_input(
            "ICU Transfers In", min_value=0, value=5, step=1
        )

        icu_transfers_out = st.number_input(
            "ICU Transfers Out", min_value=0, value=2, step=1
        )

        icu_occupied_beds = st.number_input(
            "ICU Occupied Beds", min_value=0, value=43, step=1
        )

    with col3:
        icu_available_beds = st.number_input(
            "ICU Available Beds", min_value=0, value=5, step=1
        )

        hdu_occupied_beds = st.number_input(
            "HDU Occupied Beds", min_value=0, value=4, step=1
        )

        general_occupied_beds = st.number_input(
            "General Occupied Beds", min_value=0, value=3, step=1
        )

        private_occupied_beds = st.number_input(
            "Private Occupied Beds", min_value=0, value=6, step=1
        )

    with col4:
        ventilators_in_use = st.number_input(
            "Ventilators In Use", min_value=0, value=2, step=1
        )

        ventilators_available = st.number_input(
            "Ventilators Available", min_value=0, value=1, step=1
        )

        icu_staff_available = st.number_input(
            "ICU Staff Available", min_value=0, value=12, step=1
        )

        staff_shortage_input = st.selectbox("Staff Shortage", ["YES", "NO"])

        elective_surgeries = st.number_input(
            "Elective Surgeries", min_value=0, value=3, step=1
        )

    staff_shortage_flag = 1 if staff_shortage_input == "YES" else 0

    # ================================================================
    # 19. ANALYZE BUTTON (CENTERED IN MIDDLE & CONTENT-FIT WIDTH)
    # ================================================================

    _, col_center, _ = st.columns([1.2, 1, 1.2])

    with col_center:
        analyze = st.button(
            "🩺 ANALYZE CONDITION", type="primary", use_container_width=True
        )


# ================================================================
# 20. RUN COMPLETE HOSP-AI SYSTEM
# ================================================================

if analyze:
    try:
        # ====================================================
        # STEP 1 — CURRENT OBSERVATION
        # ====================================================

        current_row = create_current_row(
            historical_data,
            er_arrivals,
            ambulance_arrivals,
            general_admissions,
            icu_admissions,
            icu_discharges,
            icu_transfers_in,
            icu_transfers_out,
            icu_occupied_beds,
            icu_available_beds,
            hdu_occupied_beds,
            general_occupied_beds,
            private_occupied_beds,
            ventilators_in_use,
            ventilators_available,
            icu_staff_available,
            staff_shortage_flag,
            elective_surgeries,
        )

        # ====================================================
        # STEP 2 — APPEND CURRENT OBSERVATION
        # ====================================================

        inference_history = pd.concat([historical_data, current_row], ignore_index=True)

        # ====================================================
        # STEP 3 — FEATURE ENGINEERING
        # ====================================================

        inference_history = create_model_features(inference_history)

        # ====================================================
        # STEP 4 — MODEL FEATURES
        # ====================================================

        feature_names = list(model.feature_names_in_)

        X_current = inference_history.tail(1)[feature_names].copy()

        if X_current.shape[1] != 39:
            raise ValueError(
                f"Expected 39 model features, received {X_current.shape[1]}."
            )

        if X_current.isna().any().any():
            missing_features = X_current.columns[X_current.isna().any()].tolist()

            raise ValueError("Missing model features: " + str(missing_features))

        # ====================================================
        # STEP 5 — XGBOOST FORECAST
        # ====================================================

        predicted_icu_occupancy = float(model.predict(X_current)[0])

        predicted_icu_occupancy = float(np.clip(predicted_icu_occupancy, 0, 100))

        # ====================================================
        # STEP 6 — CURRENT ICU OCCUPANCY
        # ====================================================

        total_icu_beds = icu_occupied_beds + icu_available_beds

        if total_icu_beds > 0:
            current_icu_occupancy = (icu_occupied_beds / total_icu_beds) * 100

        else:
            current_icu_occupancy = 0.0

        # ====================================================
        # STEP 7 — DETERMINISTIC RISK ENGINE
        # ====================================================

        risk_result = calculate_operational_risk(
            current_icu_occupancy,
            predicted_icu_occupancy,
            icu_available_beds,
            staff_shortage_flag,
            ventilators_available,
            icu_admissions,
            icu_discharges,
            er_arrivals,
            ambulance_arrivals,
        )

        # ====================================================
        # STEP 8 — BUILD RAG QUERY
        # ====================================================

        rag_query = f"""
Hospital operational decision support.

Current ICU occupancy:
{current_icu_occupancy:.2f}%

Predicted 24-hour ICU occupancy:
{predicted_icu_occupancy:.2f}%

Risk level:
{risk_result["risk_level"]}

Risk score:
{risk_result["risk_score"]}/10

ICU beds available:
{icu_available_beds}

Ventilators available:
{ventilators_available}

Staff shortage:
{staff_shortage_input}

ICU admissions:
{icu_admissions}

ICU discharges:
{icu_discharges}

Emergency arrivals:
{er_arrivals}

Ambulance arrivals:
{ambulance_arrivals}

Find hospital policies relevant to ICU capacity,
staffing, bed management, emergency inflow,
patient flow and operational escalation.
"""

        # ====================================================
        # STEP 9 — RAG RETRIEVAL
        # ====================================================

        retrieved_policies = retrieve_policies(
            rag_query,
            rag_embeddings,
            policy_df,
            rag_text_column,
            embedding_model,
            top_k=5,
        )

        # ====================================================
        # STEP 10 — RISK CONTEXT
        # ====================================================

        risk_context = build_risk_context(
            risk_result,
            current_icu_occupancy,
            predicted_icu_occupancy,
            icu_available_beds,
            ventilators_available,
            icu_staff_available,
            staff_shortage_input,
            icu_admissions,
            icu_discharges,
            er_arrivals,
            ambulance_arrivals,
        )

        # ====================================================
        # STEP 11 — GROUNDED GEMINI PROMPT
        # ====================================================

        final_prompt = build_grounded_prompt(
            risk_context,
            current_icu_occupancy,
            predicted_icu_occupancy,
            retrieved_policies,
        )

        # ====================================================
        # STEP 12 — GEMINI
        # ====================================================

        ai_response = generate_ai_decision_support(final_prompt)

        # ====================================================
        # RESULTS
        # ====================================================

        st.divider()

        # ====================================================
        # HOSP-AI COMMAND — SYSTEM RESULTS (ENCLOSED BOX)
        # ====================================================

        with st.container(border=True):
            st.markdown(
                """
                <div class="box-header-wrap">
                    <div class="box-title">
                        📊 HOSP-AI COMMAND — SYSTEM RESULTS
                    </div>
                    <div class="box-subtitle">
                        Predicted ICU census, projected bed occupancy shift, and composite operational risk assessment
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            m1, m2, m3, m4 = st.columns(4)

            m1.metric("Current ICU Occupancy", f"{current_icu_occupancy:.2f}%")

            m2.metric(
                "24h Predicted Occupancy",
                f"{predicted_icu_occupancy:.2f}%",
                delta=(f"{predicted_icu_occupancy - current_icu_occupancy:.2f} pp"),
            )

            m3.metric("Risk Score", f"{risk_result['risk_score']:.1f} / 10")

            m4.metric("Risk Level", risk_result["risk_level"])

        # ====================================================
        # CURRENT HOSPITAL STATE (ENCLOSED BOX)
        # ====================================================

        with st.container(border=True):
            st.markdown(
                """
                <div class="box-header-wrap">
                    <div class="box-title">
                        🏥 CURRENT HOSPITAL STATE
                    </div>
                    <div class="box-subtitle">
                        Real-time bed availability, ventilator deployment, and critical care staffing status
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            state1, state2, state3, state4 = st.columns(4)

            state1.metric("ICU Beds Available", icu_available_beds)

            state2.metric("Ventilators Available", ventilators_available)

            state3.metric("ICU Staff Available", icu_staff_available)

            state4.metric("Staff Shortage", staff_shortage_input)

        # ====================================================
        # KEY RISK FACTORS & FORECAST TREND (ENCLOSED BOXES)
        # ====================================================

        with st.container(border=True):
            st.markdown(
                """
                <div class="box-header-wrap">
                    <div class="box-title">
                        ⚠️ KEY RISK FACTORS
                    </div>
                    <div class="box-subtitle">
                        Identified capacity bottlenecks and operational alerts
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if risk_result["risk_factors"]:
                for factor in risk_result["risk_factors"]:
                    st.warning(factor)

            else:
                st.success("No major operational risk factors detected.")

        with st.container(border=True):
            st.markdown(
                """
                <div class="box-header-wrap">
                    <div class="box-title">
                        📈 FORECAST TREND
                    </div>
                    <div class="box-subtitle">
                        24-hour predictive trajectory and census velocity
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if risk_result["positive_trends"]:
                for trend in risk_result["positive_trends"]:
                    st.success(trend)

            else:
                if risk_result["forecast_change"] > 0:
                    st.warning(
                        "Forecast indicates an increase "
                        "in ICU occupancy over the next 24 hours."
                    )

                elif risk_result["forecast_change"] < 0:
                    st.success(
                        "Forecast indicates a decrease "
                        "in ICU occupancy over the next 24 hours."
                    )

                else:
                    st.info(
                        "Forecast indicates stable ICU occupancy over the next 24 hours."
                    )

        # ====================================================
        # AI DECISION SUPPORT (ENCLOSED BOX)
        # ====================================================

        with st.container(border=True):
            st.markdown(
                """
                <div class="box-header-wrap">
                    <div class="box-title">
                        🤖 AI DECISION SUPPORT
                    </div>
                    <div class="box-subtitle">
                        Automated clinical guidance and operational recommendations synthesized by Gemini Clinical AI
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown(ai_response)

        # Reset analyze button state immediately now that results are fetched
        st.iframe(
            """
            <script>
            (function() {
                function clearButtonLoading() {
                    try {
                        var pDoc = window.parent ? window.parent.document : document;
                        var fetchingButtons = pDoc.querySelectorAll('.is-fetching');
                        fetchingButtons.forEach(function(b) {
                            b.classList.remove('is-fetching');
                        });
                    } catch(e) {}
                }
                clearButtonLoading();
                setTimeout(clearButtonLoading, 60);
                setTimeout(clearButtonLoading, 200);
                setTimeout(clearButtonLoading, 600);
                setTimeout(clearButtonLoading, 1500);
            })();
            </script>
            """,
            height=1,
        )

    except Exception as e:
        st.iframe(
            """
            <script>
            try {
                var pDoc = window.parent ? window.parent.document : document;
                pDoc.querySelectorAll('.is-fetching').forEach(function(b) {
                    b.classList.remove('is-fetching');
                });
            } catch(e) {}
            </script>
            """,
            height=1,
        )
        st.error("The HOSP-AI COMMAND analysis could not be completed.")

        st.exception(e)
