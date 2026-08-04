import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import tempfile
import os
import time
import base64
from io import BytesIO
import streamlit.components.v1 as components

# ══════════════════════════════════════════════════════════════════
#  PAGE CONFIG  (must be first Streamlit call)
# ══════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="TrafficVision AI",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════
#  GLOBAL CSS  – premium dark dashboard theme
# ══════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

:root {
    --bg-primary:   #060B18;
    --bg-card:      #0D1525;
    --bg-card2:     #111E32;
    --accent:       #3B82F6;
    --accent2:      #6366F1;
    --accent3:      #10B981;
    --danger:       #EF4444;
    --warning:      #F59E0B;
    --text-primary: #F1F5F9;
    --text-muted:   #94A3B8;
    --border:       rgba(59,130,246,0.15);
    --glow:         rgba(59,130,246,0.25);
}

/* ── App shell ── */
.stApp {
    background: var(--bg-primary) !important;
    font-family: 'Inter', sans-serif !important;
}
.block-container { padding: 1.5rem 2rem 2rem 2rem !important; }

/* ── Hide chrome ── */
#MainMenu, footer { visibility: hidden !important; }
header { background: transparent !important; }
.stDeployButton { display: none !important; }

[data-testid="collapsedControl"] { display:flex!important; visibility:visible!important; opacity:1!important; }
[data-testid="collapsedControl"] button, [data-testid="collapsedControl"] button * { font-size:0!important; color:transparent!important; }
[data-testid="collapsedControl"] button::before { content:"Menu"; color:#60A5FA!important; font-size:0.78rem!important; font-weight:700!important; }
[data-testid="stSidebarCollapseButton"] button, [data-testid="stSidebarCollapseButton"] button *, [data-testid="stSidebarCollapseButton"] span { font-size:0!important; color:transparent!important; }
[data-testid="stSidebarCollapseButton"] button::before { content:"Close"; color:#60A5FA!important; font-size:0.72rem!important; font-weight:700!important; }
[data-baseweb="tooltip"], div[role="tooltip"] { display:none!important; }

/* ── Animated gradient background ── */
@keyframes gradientShift {
    0%   { background-position: 0% 50%; }
    50%  { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}
@keyframes float {
    0%,100% { transform: translateY(0px); }
    50%      { transform: translateY(-6px); }
}
@keyframes pulse-ring {
    0%   { transform: scale(0.8); opacity: 1; }
    100% { transform: scale(2.2); opacity: 0; }
}
@keyframes shimmer {
    0%   { background-position: -200% 0; }
    100% { background-position: 200% 0; }
}
@keyframes fadeInUp {
    from { opacity:0; transform:translateY(20px); }
    to   { opacity:1; transform:translateY(0); }
}
@keyframes pulse-dot {
    0%,100% { opacity:1; box-shadow: 0 0 8px currentColor; }
    50%      { opacity:0.4; box-shadow: 0 0 2px currentColor; }
}
@keyframes scanline {
    0%   { top: -10%; }
    100% { top: 110%; }
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #080F20 0%, #0A1228 100%) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { font-family: 'Inter', sans-serif !important; }
[data-testid="stSidebarContent"] { padding: 1.5rem 1rem !important; }

/* ── Sidebar logo ── */
.sidebar-logo {
    background: linear-gradient(135deg, #0f2044 0%, #0a1628 60%, #111E32 100%);
    border: 1px solid rgba(59,130,246,0.25);
    border-radius: 18px;
    padding: 22px 16px;
    text-align: center;
    margin-bottom: 20px;
    position: relative;
    overflow: hidden;
}
.sidebar-logo::before {
    content: '';
    position: absolute;
    top: 0; left: -100%;
    width: 200%; height: 100%;
    background: linear-gradient(90deg, transparent, rgba(59,130,246,0.06), transparent);
    animation: shimmer 3s infinite;
}
.sidebar-logo .logo-icon {
    font-size: 2.6rem;
    display: block;
    margin-bottom: 8px;
    animation: float 3s ease-in-out infinite;
}
.sidebar-logo .logo-title {
    color: #F1F5F9;
    font-size: 0.92rem;
    font-weight: 800;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    background: linear-gradient(135deg, #60A5FA, #818CF8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.sidebar-logo .logo-sub { color:#475569; font-size:0.68rem; margin-top:3px; }

/* ── Sidebar widgets ── */
.nav-section {
    color: #334155;
    font-size: 0.67rem;
    font-weight: 700;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    margin: 20px 4px 8px 4px;
    display: flex;
    align-items: center;
    gap: 8px;
}
.nav-section::after {
    content: '';
    flex: 1;
    height: 1px;
    background: rgba(59,130,246,0.10);
}

.sidebar-stat {
    background: linear-gradient(135deg, var(--bg-card2), #0D1525);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 14px 16px;
    margin-bottom: 10px;
    transition: border-color 0.2s;
}
.sidebar-stat:hover { border-color: rgba(59,130,246,0.30); }
.sidebar-stat .stat-label { color:#4B5563; font-size:0.70rem; font-weight:600; text-transform:uppercase; letter-spacing:0.09em; }
.sidebar-stat .stat-value { color:#F1F5F9; font-size:1.4rem; font-weight:800; margin-top:3px; }
.stat-badge { display:inline-block; padding:2px 10px; border-radius:20px; font-size:0.67rem; font-weight:700; margin-top:5px; }
.badge-green  { background:rgba(16,185,129,0.15); color:#10B981; border:1px solid rgba(16,185,129,0.25); }
.badge-blue   { background:rgba(59,130,246,0.15); color:#60A5FA; border:1px solid rgba(59,130,246,0.25); }
.badge-yellow { background:rgba(245,158,11,0.15); color:#F59E0B; border:1px solid rgba(245,158,11,0.25); }
.badge-red    { background:rgba(239,68,68,0.15);  color:#EF4444; border:1px solid rgba(239,68,68,0.25); }

/* ── Hero header ── */
.page-header {
    background: linear-gradient(135deg, #0a1628 0%, #0d1f3c 40%, #0f2044 70%, #080F20 100%);
    background-size: 300% 300%;
    animation: gradientShift 8s ease infinite;
    border: 1px solid rgba(59,130,246,0.20);
    border-radius: 24px;
    padding: 36px 40px;
    margin-bottom: 28px;
    position: relative;
    overflow: hidden;
    animation: gradientShift 8s ease infinite, fadeInUp 0.6s ease;
}
.page-header::before {
    content: '';
    position: absolute;
    top: -60%; left: -20%;
    width: 55%; height: 220%;
    background: radial-gradient(ellipse, rgba(59,130,246,0.10) 0%, transparent 70%);
    pointer-events: none;
}
.page-header::after {
    content: '';
    position: absolute;
    top: -10%; right: 5%;
    width: 2px; height: 120%;
    background: linear-gradient(180deg, transparent, rgba(59,130,246,0.3), transparent);
    animation: scanline 4s linear infinite;
}
.header-badge {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    background: rgba(59,130,246,0.10);
    border: 1px solid rgba(59,130,246,0.28);
    border-radius: 20px;
    padding: 5px 16px;
    color: #60A5FA;
    font-size: 0.73rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 14px;
}
.header-badge .live-dot {
    width: 6px; height: 6px;
    background: #10B981;
    border-radius: 50%;
    animation: pulse-dot 1.5s infinite;
}
.page-header h1 {
    color: #F1F5F9 !important;
    font-size: 2.2rem !important;
    font-weight: 900 !important;
    margin: 0 0 8px 0 !important;
    letter-spacing: -0.03em;
    line-height: 1.1;
}
.page-header h1 span {
    background: linear-gradient(135deg, #60A5FA, #818CF8, #34D399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.page-header p { color:#64748B; font-size:0.92rem; margin:0; line-height:1.6; }
.header-stats {
    display: flex;
    gap: 24px;
    margin-top: 20px;
    padding-top: 18px;
    border-top: 1px solid rgba(59,130,246,0.10);
}
.hstat { display:flex; align-items:center; gap:8px; }
.hstat-dot { width:8px; height:8px; border-radius:50%; }
.hstat-text { color:#64748B; font-size:0.78rem; font-weight:500; }
.hstat-text strong { color:#94A3B8; font-weight:700; }

/* ── Metric cards ── */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 16px;
    margin-bottom: 28px;
}
.metric-card {
    background: linear-gradient(135deg, var(--bg-card), var(--bg-card2));
    border: 1px solid var(--border);
    border-radius: 18px;
    padding: 22px;
    position: relative;
    overflow: hidden;
    transition: transform 0.25s, border-color 0.25s, box-shadow 0.25s;
    animation: fadeInUp 0.5s ease both;
}
.metric-card:hover {
    transform: translateY(-4px);
    border-color: rgba(59,130,246,0.40);
    box-shadow: 0 12px 40px rgba(0,0,0,0.4), 0 0 20px var(--glow);
}
.metric-card::after {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: var(--card-accent, linear-gradient(90deg, #3B82F6, #6366F1));
    border-radius: 18px 18px 0 0;
}
.metric-card::before {
    content: '';
    position: absolute;
    bottom: -30px; right: -20px;
    width: 90px; height: 90px;
    border-radius: 50%;
    background: var(--icon-glow, rgba(59,130,246,0.05));
    pointer-events: none;
}
.metric-card .mc-icon {
    width: 42px; height: 42px;
    border-radius: 12px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.25rem;
    margin-bottom: 16px;
    background: var(--icon-bg, rgba(59,130,246,0.12));
    border: 1px solid var(--icon-border, rgba(59,130,246,0.20));
}
.metric-card .mc-value { color:#F1F5F9; font-size:1.9rem; font-weight:900; line-height:1; margin-bottom:5px; }
.metric-card .mc-label { color:#4B5563; font-size:0.75rem; font-weight:600; text-transform:uppercase; letter-spacing:0.08em; }
.metric-card .mc-sub   { color:#60A5FA; font-size:0.70rem; font-weight:500; margin-top:4px; }

/* ── Upload zone ── */
.upload-zone {
    background: linear-gradient(135deg, var(--bg-card), var(--bg-card2));
    border: 2px dashed rgba(59,130,246,0.28);
    border-radius: 22px;
    padding: 52px 28px;
    text-align: center;
    transition: all 0.3s;
    position: relative;
    overflow: hidden;
}
.upload-zone:hover {
    border-color: rgba(59,130,246,0.55);
    background: linear-gradient(135deg, rgba(59,130,246,0.04), rgba(99,102,241,0.04));
    box-shadow: 0 0 30px rgba(59,130,246,0.08);
}
.upload-zone .uz-icon { font-size:3.2rem; margin-bottom:16px; display:block; animation:float 3s ease-in-out infinite; }
.upload-zone .uz-title { color:#CBD5E1; font-size:1.05rem; font-weight:700; }
.upload-zone .uz-sub   { color:#475569; font-size:0.80rem; margin-top:6px; }
.upload-zone .uz-formats {
    display: flex;
    gap: 8px;
    justify-content: center;
    margin-top: 14px;
}
.format-tag {
    background: rgba(59,130,246,0.10);
    border: 1px solid rgba(59,130,246,0.20);
    border-radius: 6px;
    padding: 2px 10px;
    color: #60A5FA;
    font-size: 0.70rem;
    font-weight: 700;
    letter-spacing: 0.06em;
}

/* ── Section heading ── */
.section-heading {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 18px;
}
.section-heading .sh-dot {
    width: 4px; height: 24px;
    background: linear-gradient(180deg, #3B82F6, #6366F1);
    border-radius: 2px;
    box-shadow: 0 0 10px rgba(59,130,246,0.4);
}
.section-heading .sh-title { color:#F1F5F9; font-size:1.05rem; font-weight:800; }
.section-heading .sh-badge {
    margin-left: auto;
    background: rgba(59,130,246,0.10);
    border: 1px solid rgba(59,130,246,0.20);
    border-radius: 20px;
    padding: 2px 12px;
    color: #60A5FA;
    font-size: 0.70rem;
    font-weight: 700;
}

/* ── Panel card ── */
.panel-card {
    background: linear-gradient(135deg, var(--bg-card), var(--bg-card2));
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 22px;
}
.panel-card-title {
    color: #4B5563;
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.10em;
    margin-bottom: 14px;
    display: flex;
    align-items: center;
    gap: 8px;
}
.panel-card-title::before {
    content: '';
    width: 3px; height: 14px;
    background: linear-gradient(180deg, #3B82F6, #6366F1);
    border-radius: 2px;
}

/* ── Violation tag ── */
.vtag {
    display: flex;
    align-items: center;
    gap: 10px;
    background: rgba(239,68,68,0.08);
    border: 1px solid rgba(239,68,68,0.22);
    border-radius: 12px;
    padding: 12px 16px;
    margin: 6px 0;
    transition: transform 0.2s, box-shadow 0.2s;
    animation: fadeInUp 0.4s ease both;
}
.vtag:hover { transform: translateX(4px); box-shadow: 0 4px 20px rgba(0,0,0,0.3); }
.vtag .vtag-dot {
    width: 9px; height: 9px;
    border-radius: 50%;
    background: #EF4444;
    flex-shrink: 0;
    animation: pulse-dot 2s infinite;
    position: relative;
}
.vtag .vtag-dot::after {
    content: '';
    position: absolute;
    top: 50%; left: 50%;
    transform: translate(-50%, -50%);
    width: 100%; height: 100%;
    border-radius: 50%;
    background: inherit;
    animation: pulse-ring 2s infinite;
}
.vtag .vtag-name { color:#FCA5A5; font-size:0.84rem; font-weight:600; }
.vtag-safe { background:rgba(16,185,129,0.08)!important; border-color:rgba(16,185,129,0.22)!important; }
.vtag-safe .vtag-dot { background:#10B981!important; animation:none!important; }
.vtag-safe .vtag-dot::after { display:none; }
.vtag-safe .vtag-name { color:#6EE7B7!important; }

/* ── Confidence bars ── */
.conf-row { display:flex; align-items:center; gap:12px; margin-bottom:12px; }
.conf-label { color:#94A3B8; font-size:0.78rem; width:160px; flex-shrink:0; }
.conf-bar-bg { flex:1; height:7px; background:rgba(255,255,255,0.05); border-radius:4px; overflow:hidden; border:1px solid rgba(255,255,255,0.04); }
.conf-bar-fill { height:100%; border-radius:4px; transition:width 1s ease; }
.conf-pct { color:#F1F5F9; font-size:0.78rem; font-weight:700; width:40px; text-align:right; }

/* ── Confidence badge coloring ── */
.conf-high { color: #10B981 !important; }
.conf-mid  { color: #F59E0B !important; }
.conf-low  { color: #EF4444 !important; }

/* ── Image comparison slider ── */
.compare-wrapper {
    position: relative;
    border-radius: 18px;
    overflow: hidden;
    border: 1px solid var(--border);
    background: #000;
    user-select: none;
    box-shadow: 0 8px 32px rgba(0,0,0,0.4);
}
.compare-wrapper img { display: block; width: 100%; height: auto; }
.compare-overlay {
    position: absolute;
    top: 0; left: 0;
    width: 50%;
    height: 100%;
    overflow: hidden;
}
.compare-overlay img { position: absolute; top:0; left:0; width: 200%; max-width:none; }
.compare-handle {
    position: absolute;
    top: 0;
    left: 50%;
    transform: translateX(-50%);
    width: 2px;
    height: 100%;
    background: linear-gradient(180deg, transparent, #60A5FA, #818CF8, #60A5FA, transparent);
    cursor: ew-resize;
    z-index: 10;
}
.compare-handle-circle {
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    width: 40px; height: 40px;
    border-radius: 50%;
    background: linear-gradient(135deg, #3B82F6, #6366F1);
    border: 2px solid rgba(255,255,255,0.3);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.75rem;
    color: white;
    box-shadow: 0 4px 16px rgba(59,130,246,0.5);
}
.compare-label {
    position: absolute;
    bottom: 12px;
    background: rgba(0,0,0,0.6);
    backdrop-filter: blur(8px);
    border: 1px solid rgba(255,255,255,0.10);
    border-radius: 6px;
    padding: 3px 10px;
    color: white;
    font-size: 0.70rem;
    font-weight: 700;
    letter-spacing: 0.06em;
}
.label-left  { left: 10px; }
.label-right { right: 10px; }

/* ── Video timeline ── */
.timeline-wrap {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 18px 20px;
    margin-top: 16px;
}
.timeline-title {
    color:#4B5563; font-size:0.72rem; font-weight:700;
    text-transform:uppercase; letter-spacing:0.10em; margin-bottom:14px;
}
.timeline-bar {
    position: relative;
    height: 36px;
    background: rgba(255,255,255,0.04);
    border-radius: 8px;
    margin-bottom: 6px;
    overflow: hidden;
    border: 1px solid rgba(255,255,255,0.05);
}
.timeline-fill {
    height: 100%;
    border-radius: 8px;
    opacity: 0.85;
    display: flex;
    align-items: center;
    padding-left: 10px;
}
.timeline-fill span {
    font-size: 0.70rem;
    font-weight: 700;
    color: white;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.timeline-time {
    color:#4B5563; font-size:0.65rem;
    margin-top: 2px; text-align:right;
}

/* ── Donut chart card ── */
.chart-card {
    background: linear-gradient(135deg, var(--bg-card), var(--bg-card2));
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 22px;
    margin-top: 0;
}

/* ── Model not found card ── */
.model-error {
    background: linear-gradient(135deg, rgba(239,68,68,0.06), rgba(239,68,68,0.02));
    border: 1px solid rgba(239,68,68,0.25);
    border-radius: 20px;
    padding: 40px 32px;
    text-align: center;
}
.model-error .me-icon { font-size:3rem; margin-bottom:16px; }
.model-error .me-title { color:#FCA5A5; font-size:1.1rem; font-weight:700; margin-bottom:8px; }
.model-error .me-sub { color:#64748B; font-size:0.84rem; line-height:1.6; }
.model-error code {
    background: rgba(239,68,68,0.12);
    border: 1px solid rgba(239,68,68,0.20);
    border-radius: 6px;
    padding: 2px 8px;
    color: #FCA5A5;
    font-size: 0.82rem;
}

/* ── Streamlit widget overrides ── */
.stSlider > div { color:#94A3B8 !important; }
[data-testid="stSlider"] .rc-slider-track { background: linear-gradient(90deg, #3B82F6, #6366F1) !important; }
[data-testid="stSlider"] .rc-slider-handle { border-color:#3B82F6!important; background:#3B82F6!important; box-shadow:0 0 8px rgba(59,130,246,0.5)!important; }
.stButton > button {
    background: linear-gradient(135deg, #1D4ED8, #4338CA) !important;
    color: white !important;
    border: none !important;
    border-radius: 14px !important;
    padding: 0.65rem 1.8rem !important;
    font-weight: 700 !important;
    font-size: 0.88rem !important;
    letter-spacing: 0.03em !important;
    transition: all 0.2s !important;
    box-shadow: 0 4px 20px rgba(59,130,246,0.35) !important;
}
.stButton > button:hover { opacity:0.88!important; transform:translateY(-2px)!important; box-shadow:0 8px 28px rgba(59,130,246,0.45)!important; }
[data-testid="stFileUploader"] {
    background: var(--bg-card) !important;
    border: 2px dashed rgba(59,130,246,0.25) !important;
    border-radius: 18px !important;
    padding: 12px !important;
    transition: border-color 0.2s !important;
}
[data-testid="stFileUploader"]:hover { border-color: rgba(59,130,246,0.5) !important; }
[data-testid="stFileUploader"] label { color:#94A3B8!important; font-size:0.85rem!important; }
.stTabs [data-baseweb="tab-list"] {
    background: #0D1525 !important;
    border-radius: 14px !important;
    padding: 5px !important;
    gap: 3px !important;
    border: 1px solid var(--border) !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: #4B5563 !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 0.87rem !important;
    padding: 9px 22px !important;
    transition: all 0.2s !important;
}
.stTabs [data-baseweb="tab"]:hover { color:#94A3B8 !important; background: rgba(59,130,246,0.06) !important; }
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #1D4ED8, #4338CA) !important;
    color: white !important;
    box-shadow: 0 4px 16px rgba(59,130,246,0.35) !important;
}
.stTabs [data-baseweb="tab-panel"] { background:transparent!important; padding-top:24px!important; }
.stProgress > div > div > div { background: linear-gradient(90deg, #3B82F6, #6366F1, #10B981) !important; border-radius:4px!important; }
[data-testid="stDownloadButton"] > button {
    background: linear-gradient(135deg, #059669, #047857) !important;
    box-shadow: 0 4px 16px rgba(5,150,105,0.35) !important;
    border-radius: 14px !important;
}
.stSelectbox label, .stSlider label { color:#64748B!important; font-size:0.82rem!important; font-weight:600!important; }
hr { border-color: rgba(59,130,246,0.10) !important; }
[data-testid="stImage"] { border-radius:16px!important; overflow:hidden!important; box-shadow:0 4px 24px rgba(0,0,0,0.3)!important; }
[data-testid="stMetric"] { background: var(--bg-card)!important; border-radius:12px!important; padding:10px!important; border:1px solid var(--border)!important; }
[data-testid="stMetricLabel"] { color:#64748B!important; font-size:0.75rem!important; }
[data-testid="stMetricValue"] { color:#F1F5F9!important; font-size:1.4rem!important; font-weight:800!important; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
#  MODEL & CONSTANTS
# ══════════════════════════════════════════════════════════════════
CLASS_META = {
    'Number_plate':                         {'icon': '🔢', 'color': '#3B82F6', 'short': 'Number Plate'},
    'mobile_usage':                         {'icon': '📱', 'color': '#F59E0B', 'short': 'Mobile Usage'},
    'pillion_rider_not_wearing_helmet':     {'icon': '🪖', 'color': '#8B5CF6', 'short': 'Pillion No Helmet'},
    'rider_and_pillion_not_wearing_helmet': {'icon': '⛑️', 'color': '#EC4899', 'short': 'Both No Helmet'},
    'rider_not_wearing_helmet':             {'icon': '🚫', 'color': '#EF4444', 'short': 'Rider No Helmet'},
    'triple_riding':                        {'icon': '👥', 'color': '#10B981', 'short': 'Triple Riding'},
    'vehicle_with_offence':                 {'icon': '🚗', 'color': '#F97316', 'short': 'Vehicle Offence'},
}
CLASS_NAMES = list(CLASS_META.keys())

@st.cache_resource(show_spinner=False)
def load_model():
    try:
        return YOLO("best.pt")
    except Exception:
        return None

# Loading indicator
with st.spinner("🚦 Initializing TrafficVision AI..."):
    model = load_model()

# ══════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════
def fmt(name: str) -> str:
    return CLASS_META.get(name, {}).get('short', name.replace('_', ' ').title())

def hex_to_bgr(hex_c: str):
    h = hex_c.lstrip('#')
    r, g, b = (int(h[i:i+2], 16) for i in (0, 2, 4))
    return (b, g, r)

def hex_to_rgb(hex_c: str):
    h = hex_c.lstrip('#')
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

def conf_class(v):
    if v >= 0.70: return "conf-high"
    if v >= 0.45: return "conf-mid"
    return "conf-low"

def draw_boxes(image: np.ndarray, results, conf_thresh: float = 0.25) -> np.ndarray:
    img = image.copy()
    h_img, w_img = img.shape[:2]
    if results[0].boxes is None:
        return img

    boxes  = results[0].boxes.xyxy.cpu().numpy().astype(int)
    clss   = results[0].boxes.cls.cpu().numpy().astype(int)
    confs  = results[0].boxes.conf.cpu().numpy()
    used_areas = []

    for box, cls_id, conf in zip(boxes, clss, confs):
        if conf < conf_thresh:
            continue
        x1, y1, x2, y2 = box
        name  = CLASS_NAMES[cls_id]
        label = fmt(name)
        bgr   = hex_to_bgr(CLASS_META.get(name, {}).get('color', '#3B82F6'))

        # semi-transparent fill
        overlay = img.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), bgr, -1)
        cv2.addWeighted(overlay, 0.06, img, 0.94, 0, img)

        # box border
        cv2.rectangle(img, (x1, y1), (x2, y2), bgr, 2)

        # corner accents
        L = min(22, (x2-x1)//4, (y2-y1)//4)
        for cx, cy, dx, dy in [(x1,y1,1,1),(x2,y1,-1,1),(x1,y2,1,-1),(x2,y2,-1,-1)]:
            cv2.line(img, (cx, cy), (cx+dx*L, cy), bgr, 3)
            cv2.line(img, (cx, cy), (cx, cy+dy*L), bgr, 3)

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 1)
        positions = [(x1, y1-6), (x1, y2+th+8), (x1+10, y1-26), (x1+10, y2+26)]
        fx, fy = x1, y1
        for px, py in positions:
            px = max(0, min(px, w_img-tw))
            py = max(th, min(py, h_img))
            overlap = any(
                not (px+tw < ux1 or px > ux2 or py < uy1 or py-th > uy2)
                for ux1,uy1,ux2,uy2 in used_areas
            )
            if not overlap:
                fx, fy = px, py
                break

        used_areas.append((fx, fy-th, fx+tw, fy))
        cv2.rectangle(img, (fx-4, fy-th-5), (fx+tw+5, fy+3), bgr, -1)
        cv2.putText(img, label, (fx, fy-3), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255,255,255), 1, cv2.LINE_AA)

    return img


# ── Per-ID color palette for tracking ──────────────────────────────
_TRACK_PALETTE = [
    '#3B82F6','#10B981','#F59E0B','#EF4444','#8B5CF6',
    '#EC4899','#F97316','#06B6D4','#84CC16','#6366F1',
    '#14B8A6','#F43F5E','#A855F7','#22C55E','#FB923C',
]

def track_color(track_id: int):
    hex_c = _TRACK_PALETTE[track_id % len(_TRACK_PALETTE)]
    return hex_to_bgr(hex_c), hex_c

def draw_tracked_boxes(image, tracks, conf_thresh=0.25, trails=None):
    img = image.copy()
    h_img, w_img = img.shape[:2]

    if tracks is None or tracks.id is None:
        return img

    boxes = tracks.xyxy.cpu().numpy().astype(int)
    clss  = tracks.cls.cpu().numpy().astype(int)
    confs = tracks.conf.cpu().numpy()
    ids   = tracks.id.cpu().numpy().astype(int)

    for box, cls_id, conf, tid in zip(boxes, clss, confs, ids):
        if conf < conf_thresh:
            continue
        x1, y1, x2, y2 = box
        bgr, hex_c = track_color(tid)
        name  = CLASS_NAMES[cls_id]
        label = f"#{tid} {fmt(name)}"

        # Trail
        if trails is not None:
            cx_t, cy_t = (x1 + x2) // 2, (y1 + y2) // 2
            if tid not in trails:
                from collections import deque as _deque
                trails[tid] = _deque(maxlen=30)
            trails[tid].append((cx_t, cy_t))
            pts = list(trails[tid])
            for i in range(1, len(pts)):
                alpha = i / max(len(pts), 1)
                cv2.line(img, pts[i-1], pts[i], bgr, max(1, int(3*alpha)), cv2.LINE_AA)

        # Semi-transparent fill
        overlay = img.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), bgr, -1)
        cv2.addWeighted(overlay, 0.07, img, 0.93, 0, img)

        # Box + corner accents
        cv2.rectangle(img, (x1, y1), (x2, y2), bgr, 2)
        L = min(18, (x2-x1)//4, (y2-y1)//4)
        for cx2, cy2, dx, dy in [(x1,y1,1,1),(x2,y1,-1,1),(x1,y2,1,-1),(x2,y2,-1,-1)]:
            cv2.line(img, (cx2, cy2), (cx2+dx*L, cy2), bgr, 3)
            cv2.line(img, (cx2, cy2), (cx2, cy2+dy*L), bgr, 3)

        # Label
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
        lx = max(0, min(x1, w_img - tw - 10))
        ly = max(th + 6, y1)
        cv2.rectangle(img, (lx-4, ly-th-5), (lx+tw+5, ly+3), bgr, -1)
        cv2.putText(img, label, (lx, ly-3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255,255,255), 1, cv2.LINE_AA)

    return img

def violation_summary(results):
    if results[0].boxes is None:
        return []
    out = {}
    for cls_id, conf in zip(results[0].boxes.cls, results[0].boxes.conf):
        name = CLASS_NAMES[int(cls_id)]
        if name not in out or conf > out[name]:
            out[name] = float(conf)
    return sorted(out.items(), key=lambda x: -x[1])

def img_to_b64(arr: np.ndarray) -> str:
    pil = Image.fromarray(arr)
    buf = BytesIO()
    pil.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

def render_compare_slider(orig_arr: np.ndarray, annotated_arr: np.ndarray):
    """Render an interactive before/after comparison slider."""
    orig_b64 = img_to_b64(orig_arr)
    ann_b64  = img_to_b64(annotated_arr)
    h_img, w_img = orig_arr.shape[:2]
    aspect = round(h_img / max(w_img, 1) * 100, 2)

    html = f"""
    <style>
      body {{ margin:0; padding:0; background:transparent; overflow:hidden; }}
      #cmp {{
        position: relative;
        width: 100%;
        padding-top: {aspect}%;
        cursor: ew-resize;
        border-radius: 14px;
        overflow: hidden;
        background: #000;
        box-shadow: 0 8px 32px rgba(0,0,0,0.5);
      }}
      #cmp img.base {{
        position: absolute; top:0; left:0; width:100%; height:100%; object-fit:cover;
      }}
      #cmpOverlay {{
        position: absolute; top:0; left:0; width:50%; height:100%; overflow:hidden;
      }}
      #cmpOverlay img {{
        position: absolute; top:0; left:0; height:100%; width:auto; min-width:200%;
      }}
      #cmpHandle {{
        position: absolute; top:0; left:50%; transform:translateX(-50%);
        width:3px; height:100%;
        background: linear-gradient(180deg, transparent 5%, #60A5FA, #818CF8, #60A5FA, transparent 95%);
        z-index: 10; pointer-events:none;
      }}
      #cmpCircle {{
        position:absolute; top:50%; left:50%;
        transform:translate(-50%,-50%);
        width:38px; height:38px; border-radius:50%;
        background: linear-gradient(135deg,#3B82F6,#6366F1);
        border:2px solid rgba(255,255,255,0.3);
        display:flex; align-items:center; justify-content:center;
        font-size:14px; color:white;
        box-shadow: 0 4px 16px rgba(59,130,246,0.6);
      }}
      .clabel {{
        position:absolute; bottom:12px;
        background:rgba(0,0,0,0.65);
        backdrop-filter:blur(6px);
        border:1px solid rgba(255,255,255,0.12);
        border-radius:5px; padding:2px 10px;
        color:white; font-size:10px; font-weight:700;
        letter-spacing:0.06em; font-family:Inter,sans-serif;
        pointer-events:none;
      }}
    </style>
    <div id="cmp">
      <img class="base" src="data:image/png;base64,{ann_b64}" alt="detected">
      <div id="cmpOverlay">
        <img src="data:image/png;base64,{orig_b64}" alt="original">
      </div>
      <div id="cmpHandle"><div id="cmpCircle">⇔</div></div>
      <div class="clabel" style="left:10px;">ORIGINAL</div>
      <div class="clabel" style="right:10px;">DETECTED</div>
    </div>
    <script>
    (function(){{
      const wrap    = document.getElementById('cmp');
      const overlay = document.getElementById('cmpOverlay');
      const handle  = document.getElementById('cmpHandle');
      let dragging  = false;
      function setPos(clientX){{
        const rect = wrap.getBoundingClientRect();
        let pct = Math.max(0.01, Math.min(0.99, (clientX - rect.left) / rect.width));
        overlay.style.width = (pct * 100) + '%';
        handle.style.left   = (pct * 100) + '%';
        // keep overlay image at full width by inverse-scaling
        const img = overlay.querySelector('img');
        img.style.width = 'auto';
        img.style.minWidth = (1 / pct * 100) + '%';
      }}
      wrap.addEventListener('mousedown',  e => {{ dragging=true; setPos(e.clientX); e.preventDefault(); }});
      document.addEventListener('mousemove', e => {{ if(dragging) setPos(e.clientX); }});
      document.addEventListener('mouseup',   () => dragging=false);
      wrap.addEventListener('touchstart', e => {{ dragging=true; setPos(e.touches[0].clientX); }}, {{passive:true}});
      document.addEventListener('touchmove', e => {{ if(dragging) setPos(e.touches[0].clientX); }}, {{passive:true}});
      document.addEventListener('touchend',  () => dragging=false);
      // init at 50%
      setTimeout(() => setPos(wrap.getBoundingClientRect().left + wrap.getBoundingClientRect().width * 0.5), 50);
    }})();
    </script>
    """
    # height = width * aspect_ratio; iframe width ≈ column width. Use a safe fixed height.
    slider_height = min(500, max(300, int(500 * aspect / 100)))
    components.html(html, height=slider_height + 20, scrolling=False)

def render_donut_chart(violations):
    """Render donut chart via st.components so SVG is never stripped by st.markdown."""
    if not violations:
        return
    total = sum(v for _, v in violations)
    colors = [CLASS_META.get(n, {}).get('color', '#3B82F6') for n, _ in violations]
    labels = [fmt(n) for n, _ in violations]
    pcts   = [v / total for _, v in violations]

    cx, cy, r_out, r_in = 110, 110, 88, 52
    circumference = 2 * 3.14159 * r_out
    slices = []
    offset = 0
    for i, pct in enumerate(pcts):
        dash = pct * circumference
        slices.append((colors[i], dash, offset, labels[i], violations[i][1]))
        offset += dash

    circles_svg = ""
    for color, dash, off, label, val in slices:
        circles_svg += f'<circle cx="{cx}" cy="{cy}" r="{r_out}" fill="none" stroke="{color}" stroke-width="30" stroke-dasharray="{dash:.2f} {circumference:.2f}" stroke-dashoffset="{-off:.2f}" transform="rotate(-90 {cx} {cy})" />'

    legend_rows = ""
    for i, (color, _, _, label, val) in enumerate(slices):
        y = 28 + i * 26
        legend_rows += f'''
        <rect x="240" y="{y-9}" width="11" height="11" rx="3" fill="{color}"/>
        <text x="258" y="{y}" fill="#94A3B8" font-size="12" font-family="Inter,sans-serif">{label}</text>'''

    n_types = len(violations)
    svg_height = max(220, 30 + n_types * 26 + 20)
    html = f"""
    <div style="background:transparent;">
    <svg viewBox="0 0 430 {svg_height}" xmlns="http://www.w3.org/2000/svg" style="width:100%;display:block;">
        <rect width="430" height="{svg_height}" fill="transparent"/>
        <g>{circles_svg}</g>
        <circle cx="{cx}" cy="{cy}" r="{r_in}" fill="#0D1525"/>
        <text x="{cx}" y="{cy-6}" text-anchor="middle" fill="#F1F5F9" font-size="20" font-weight="800" font-family="Inter,sans-serif">{n_types}</text>
        <text x="{cx}" y="{cy+13}" text-anchor="middle" fill="#4B5563" font-size="9" font-family="Inter,sans-serif" font-weight="700" letter-spacing="1">TYPES</text>
        <g>{legend_rows}</g>
    </svg>
    </div>
    """
    components.html(html, height=svg_height + 10, scrolling=False)

# ══════════════════════════════════════════════════════════════════
#  SESSION STATE
# ══════════════════════════════════════════════════════════════════
if 'total_processed'   not in st.session_state:
    st.session_state.total_processed   = 0
if 'total_violations'  not in st.session_state:
    st.session_state.total_violations  = 0
if 'violation_counts'  not in st.session_state:
    st.session_state.violation_counts  = {k: 0 for k in CLASS_NAMES}
if 'processed_ids'     not in st.session_state:
    st.session_state.processed_ids     = set()
# ── Layout-stability keys (prevent jiggle on file select) ─────────
if 'img_has_file'      not in st.session_state:
    st.session_state.img_has_file      = False
if 'vid_has_file'      not in st.session_state:
    st.session_state.vid_has_file      = False
# ── Detection result cache (prevents box jitter on Streamlit reruns) ──
if 'img_cache_id'      not in st.session_state:
    st.session_state.img_cache_id      = None
if 'img_cache_result'  not in st.session_state:
    st.session_state.img_cache_result  = None
# ── File bytes cache (survives reruns after file uploader resets) ──
if 'img_file_bytes'    not in st.session_state:
    st.session_state.img_file_bytes    = None
if 'img_file_name'     not in st.session_state:
    st.session_state.img_file_name     = None
if 'img_file_size'     not in st.session_state:
    st.session_state.img_file_size     = 0
if 'vid_file_bytes'    not in st.session_state:
    st.session_state.vid_file_bytes    = None
if 'vid_file_name'     not in st.session_state:
    st.session_state.vid_file_name     = None
if 'vid_file_size'     not in st.session_state:
    st.session_state.vid_file_size     = 0

def reset_session_stats():
    st.session_state.total_processed  = 0
    st.session_state.total_violations = 0
    st.session_state.violation_counts = {k: 0 for k in CLASS_NAMES}
    st.session_state.processed_ids    = set()

def render_session_stats(container):
    alert = st.session_state.total_violations > 0
    container.markdown(f"""
    <div class="sidebar-stat">
        <div class="stat-label">Files Processed</div>
        <div class="stat-value">{st.session_state.total_processed}</div>
        <span class="stat-badge badge-blue">Session</span>
    </div>
    <div class="sidebar-stat">
        <div class="stat-label">Violations Found</div>
        <div class="stat-value">{st.session_state.total_violations}</div>
        <span class="stat-badge {'badge-red' if alert else 'badge-green'}">
            {'⚠ Alert' if alert else '✓ Clear'}
        </span>
    </div>
    """, unsafe_allow_html=True)

def render_metric_cards(container, confidence_value, iou_value):
    top_violation = "—"
    if st.session_state.total_violations > 0:
        top = max(st.session_state.violation_counts, key=st.session_state.violation_counts.get)
        if st.session_state.violation_counts[top] > 0:
            top_violation = CLASS_META[top]['icon'] + " " + fmt(top)

    container.markdown(f"""
<div class="metric-grid">
    <div class="metric-card" style="--card-accent:linear-gradient(90deg,#3B82F6,#6366F1);--icon-bg:rgba(59,130,246,0.12);--icon-border:rgba(59,130,246,0.20);--icon-glow:rgba(59,130,246,0.06);">
        <div class="mc-icon">🎚️</div>
        <div class="mc-value">{int(confidence_value * 100)}%</div>
        <div class="mc-label">Confidence Threshold</div>
        <div class="mc-sub">Min detection score</div>
    </div>
    <div class="metric-card" style="--card-accent:linear-gradient(90deg,#10B981,#059669);--icon-bg:rgba(16,185,129,0.12);--icon-border:rgba(16,185,129,0.20);--icon-glow:rgba(16,185,129,0.06);">
        <div class="mc-icon">📐</div>
        <div class="mc-value">{int(iou_value * 100)}%</div>
        <div class="mc-label">IoU Threshold</div>
        <div class="mc-sub">NMS overlap filter</div>
    </div>
    <div class="metric-card" style="--card-accent:linear-gradient(90deg,#6366F1,#8B5CF6);--icon-bg:rgba(99,102,241,0.12);--icon-border:rgba(99,102,241,0.20);--icon-glow:rgba(99,102,241,0.06);">
        <div class="mc-icon">🔍</div>
        <div class="mc-value">7</div>
        <div class="mc-label">Violation Classes</div>
        <div class="mc-sub">Custom YOLOv8 model</div>
    </div>
    <div class="metric-card" style="--card-accent:linear-gradient(90deg,#F59E0B,#F97316);--icon-bg:rgba(245,158,11,0.12);--icon-border:rgba(245,158,11,0.20);--icon-glow:rgba(245,158,11,0.06);">
        <div class="mc-icon">📊</div>
        <div class="mc-value">{st.session_state.total_violations}</div>
        <div class="mc-label">Session Violations</div>
        <div class="mc-sub">{top_violation}</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">
        <span class="logo-icon">🚦</span>
        <div class="logo-title">TrafficVision AI</div>
        <div class="logo-sub">YOLOv8 · Real-Time Detection</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="nav-section">Detection Settings</div>', unsafe_allow_html=True)
    confidence = st.slider("Confidence Threshold", 0.10, 1.0, 0.40, 0.05,
                           help="Minimum confidence score for a detection to be shown")
    iou_thresh = st.slider("IoU Threshold (NMS)", 0.10, 1.0, 0.45, 0.05,
                           help="Overlap threshold for Non-Maximum Suppression")

    st.markdown('<div class="nav-section">Video Settings</div>', unsafe_allow_html=True)
    frame_skip = st.slider("Frame Skip", 1, 5, 1, 1,
                           help="Process every Nth frame — higher = faster but less precise")

    st.markdown('<div class="nav-section">Model Status</div>', unsafe_allow_html=True)
    model_status = "🟢 Loaded" if model else "🔴 Not Found"
    st.markdown(f"""
    <div class="sidebar-stat">
        <div class="stat-label">Model</div>
        <div class="stat-value" style="font-size:0.95rem;">{model_status}</div>
        <span class="stat-badge badge-blue">YOLOv8 Nano</span>
    </div>
    <div class="sidebar-stat">
        <div class="stat-label">Classes</div>
        <div class="stat-value">7</div>
        <span class="stat-badge badge-green">Active</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="nav-section">Session Stats</div>', unsafe_allow_html=True)
    sidebar_stats = st.empty()
    render_session_stats(sidebar_stats)

    if st.button("↺  Reset Session", use_container_width=True, key="clear_stats"):
        reset_session_stats()
        st.rerun()

    st.markdown('<div class="nav-section">Violation Classes</div>', unsafe_allow_html=True)
    sidebar_classes_html = ""
    for name, meta in CLASS_META.items():
        count      = st.session_state.violation_counts.get(name, 0)
        bg         = "rgba(59,130,246,0.06)" if count > 0 else "transparent"
        txt_color  = "#CBD5E1" if count > 0 else "#94A3B8"
        txt_weight = "600" if count > 0 else "500"
        badge      = f'<span style="margin-left:auto;background:rgba(239,68,68,0.15);color:#FCA5A5;border-radius:10px;padding:1px 7px;font-size:0.65rem;font-weight:700;">{count}</span>' if count > 0 else ""
        sidebar_classes_html += (
            f'<div style="display:flex;align-items:center;gap:9px;padding:7px 8px;border-radius:10px;margin-bottom:2px;background:{bg};">'
            f'<div style="width:9px;height:9px;border-radius:50%;background:{meta["color"]};flex-shrink:0;box-shadow:0 0 5px {meta["color"]}60;"></div>'
            f'<span style="color:{txt_color};font-size:0.77rem;font-weight:{txt_weight};">{meta["icon"]} {meta["short"]}</span>'
            f'{badge}'
            f'</div>'
        )
    st.markdown(sidebar_classes_html, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="color:#1E293B;font-size:0.68rem;text-align:center;padding:8px;
                border-top:1px solid rgba(59,130,246,0.07);margin-top:4px;">
        <span style="color:#334155;">BBD University · CSE-AI · 2026</span>
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
#  MAIN CONTENT
# ══════════════════════════════════════════════════════════════════
st.markdown("""
<div class="page-header">
    <div class="header-badge">
        <div class="live-dot"></div>
        AI-Powered &nbsp;·&nbsp; Real-Time &nbsp;·&nbsp; Multi-Class
    </div>
    <h1>Traffic <span>Violation</span> Detection</h1>
    <p>Automated surveillance analysis using YOLOv8 deep learning — detect helmet violations,
       triple riding, mobile usage, and more from images or video feeds.</p>
    <div class="header-stats">
        <div class="hstat">
            <div class="hstat-dot" style="background:#10B981;box-shadow:0 0 6px #10B98160;"></div>
            <div class="hstat-text"><strong>YOLOv8 Nano</strong> · Real-time inference</div>
        </div>
        <div class="hstat">
            <div class="hstat-dot" style="background:#6366F1;box-shadow:0 0 6px #6366F160;"></div>
            <div class="hstat-text"><strong>7 Classes</strong> · Custom Roboflow dataset</div>
        </div>
        <div class="hstat">
            <div class="hstat-dot" style="background:#F59E0B;box-shadow:0 0 6px #F59E0B60;"></div>
            <div class="hstat-text"><strong>Image + Video</strong> · Drag & drop upload</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

metric_cards = st.empty()
render_metric_cards(metric_cards, confidence, iou_thresh)

tab1, tab2, tab3 = st.tabs(["📸  Image Detection", "🎥  Video Detection", "📋  About & Classes"])

# ════════════════════════════════════════════════════════════
#  TAB 1 — IMAGE
# ════════════════════════════════════════════════════════════
with tab1:
    col_up, col_res = st.columns([1, 1.5], gap="large")

    with col_up:
        st.markdown("""
        <div class="section-heading">
            <div class="sh-dot"></div>
            <div class="sh-title">Upload Image</div>
            <div class="sh-badge">JPG · PNG · JPEG</div>
        </div>
        """, unsafe_allow_html=True)

        # ── File uploader OUTSIDE form (st.file_uploader breaks inside st.form)
        #    Button inside form so detection only fires on explicit click,
        #    not on every Streamlit rerun. ──
        uploaded_file = st.file_uploader(
            "Drag & drop or browse",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed",
            key="img_uploader"
        )

        if uploaded_file:
            # Store bytes in session_state so the image survives reruns
            st.session_state.img_file_bytes = uploaded_file.getvalue()
            st.session_state.img_file_name  = uploaded_file.name
            st.session_state.img_file_size  = uploaded_file.size
            st.image(uploaded_file, caption="Original Image", use_container_width=True)
        elif st.session_state.get("img_file_bytes"):
            st.image(st.session_state.img_file_bytes, caption="Original Image", use_container_width=True)
        else:
            st.markdown("""
            <div class="upload-zone">
                <span class="uz-icon">📂</span>
                <div class="uz-title">Upload a surveillance image</div>
                <div class="uz-sub">Drag & drop or click to browse</div>
                <div class="uz-formats">
                    <span class="format-tag">JPG</span>
                    <span class="format-tag">PNG</span>
                    <span class="format-tag">JPEG</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Button in a form — only reruns on explicit click, stops constant reruns
        with st.form("img_form", clear_on_submit=False):
            run_btn = st.form_submit_button(
                "🔍  Run Detection",
                use_container_width=True,
            )

    with col_res:
        st.markdown("""
        <div class="section-heading">
            <div class="sh-dot"></div>
            <div class="sh-title">Detection Results</div>
        </div>
        """, unsafe_allow_html=True)

        # Resolve file from uploader or from session_state bytes (survives reruns)
        _img_bytes = (uploaded_file.getvalue() if uploaded_file
                      else st.session_state.get("img_file_bytes"))
        _img_name  = (uploaded_file.name if uploaded_file
                      else st.session_state.get("img_file_name", "image"))
        _img_size  = (uploaded_file.size if uploaded_file
                      else st.session_state.get("img_file_size", 0))

        if run_btn and _img_bytes:
            if model is None:
                st.markdown("""
                <div class="model-error">
                    <div class="me-icon">⚠️</div>
                    <div class="me-title">Model Not Found</div>
                    <div class="me-sub">
                        Place your trained weights file <code>best.pt</code> in the same
                        directory as this app and restart.<br><br>
                        Make sure you've trained a YOLOv8 model on the traffic violation dataset.
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                file_id = f"img_{_img_name}_{_img_size}"

                # ── Cache detection result — model.predict is non-deterministic;
                #    re-running it on reruns gives slightly different box coords
                #    causing visible jitter. Cache locks boxes permanently. ──
                if st.session_state.img_cache_id != file_id:
                    with st.spinner("🔍 Analyzing image..."):
                        img_np     = np.array(Image.open(BytesIO(_img_bytes)).convert("RGB"))
                        results    = model.predict(img_np, conf=confidence, iou=iou_thresh, verbose=False)
                        annotated  = draw_boxes(img_np, results, confidence)
                        violations = violation_summary(results)
                    st.session_state.img_cache_id     = file_id
                    st.session_state.img_cache_result = (annotated, violations)
                else:
                    annotated, violations = st.session_state.img_cache_result

                # Update session stats (guarded so re-clicks don't double-count)
                if file_id not in st.session_state.processed_ids:
                    st.session_state.processed_ids.add(file_id)
                    st.session_state.total_processed += 1
                    st.session_state.total_violations += len(violations)
                    for name, _ in violations:
                        st.session_state.violation_counts[name] += 1
                render_session_stats(sidebar_stats)
                render_metric_cards(metric_cards, confidence, iou_thresh)

                st.markdown("""<div class="panel-card-title">🖼️ Detection Output</div>""",
                            unsafe_allow_html=True)
                st.image(annotated, caption="Detection Output", use_container_width=True)
                st.markdown("<br>", unsafe_allow_html=True)

                if violations:
                    viol_col, chart_col = st.columns([1, 1], gap="medium")
                    with viol_col:
                        st.markdown(f"""
                        <div class="panel-card">
                            <div class="panel-card-title">⚠ {len(violations)} Violation{'s' if len(violations)>1 else ''} Detected</div>
                        """, unsafe_allow_html=True)
                        for name, conf_val in violations:
                            meta  = CLASS_META.get(name, {})
                            col_a = meta.get('color', '#EF4444')
                            st.markdown(f"""
                            <div class="vtag" style="border-color:{col_a}35;background:rgba({",".join(str(x) for x in hex_to_rgb(col_a))},0.07);">
                                <div class="vtag-dot" style="background:{col_a};"></div>
                                <div class="vtag-name" style="color:{col_a};">{meta.get('icon','')} {fmt(name)}</div>
                            </div>
                            """, unsafe_allow_html=True)
                        st.markdown("""<br><div class="panel-card-title">📊 Confidence Scores</div>""",
                                    unsafe_allow_html=True)
                        for name, conf_val in violations:
                            pct   = int(conf_val * 100)
                            col_a = CLASS_META.get(name, {}).get('color', '#3B82F6')
                            st.markdown(f"""
                            <div class="conf-row">
                                <div class="conf-label">{fmt(name)}</div>
                                <div class="conf-bar-bg">
                                    <div class="conf-bar-fill" style="width:{pct}%;background:linear-gradient(90deg,{col_a},{col_a}BB);"></div>
                                </div>
                                <div class="conf-pct">{pct}%</div>
                            </div>
                            """, unsafe_allow_html=True)
                        st.markdown("</div>", unsafe_allow_html=True)

                    with chart_col:
                        st.markdown("""
                        <div class="chart-card">
                            <div class="panel-card-title">🍩 Violation Distribution</div>
                        """, unsafe_allow_html=True)
                        render_donut_chart(violations)
                        st.markdown("</div>", unsafe_allow_html=True)

                else:
                    st.markdown("""
                    <div class="vtag vtag-safe" style="padding:16px 20px;">
                        <div class="vtag-dot" style="background:#10B981;"></div>
                        <div class="vtag-name" style="font-size:0.92rem;">✅ No violations detected in this image</div>
                    </div>
                    """, unsafe_allow_html=True)

        elif st.session_state.img_cache_result and not run_btn:
            # Show previously cached result so it doesn't disappear on rerun
            annotated, violations = st.session_state.img_cache_result
            st.markdown("""<div class="panel-card-title">🖼️ Detection Output</div>""",
                        unsafe_allow_html=True)
            st.image(annotated, caption="Detection Output", use_container_width=True)

        else:
            st.markdown("""
            <div style="background:linear-gradient(135deg,#0D1525,#111E32);
                        border:1px solid rgba(59,130,246,0.12);border-radius:20px;
                        padding:80px 28px;text-align:center;height:100%;">
                <div style="font-size:3.5rem;margin-bottom:16px;opacity:0.4;">🖼️</div>
                <div style="color:#334155;font-size:0.90rem;font-weight:500;">
                    Upload an image and click Run Detection
                </div>
            </div>
            """, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
#  TAB 2 — VIDEO  (ByteTrack)
# ════════════════════════════════════════════════════════════
with tab2:
    col_v1, col_v2 = st.columns([1, 1.5], gap="large")

    with col_v1:
        st.markdown("""
        <div class="section-heading">
            <div class="sh-dot"></div>
            <div class="sh-title">Upload Video</div>
            <div class="sh-badge">MP4 · AVI · MOV</div>
        </div>
        """, unsafe_allow_html=True)

        uploaded_video = st.file_uploader(
            "Upload video file",
            type=["mp4", "avi", "mov"],
            label_visibility="collapsed",
            key="vid_uploader"
        )

        if uploaded_video:
            st.session_state.vid_file_bytes = uploaded_video.getvalue()
            st.session_state.vid_file_name  = uploaded_video.name
            st.session_state.vid_file_size  = uploaded_video.size
            st.video(uploaded_video)
            st.markdown(f"""
            <div style="background:rgba(59,130,246,0.07);border:1px solid rgba(59,130,246,0.15);
                        border-radius:12px;padding:10px 14px;margin-top:8px;font-size:0.78rem;color:#64748B;">
                ⚡ Frame skip: every <strong style="color:#60A5FA;">{frame_skip}</strong> frame(s)
                &nbsp;·&nbsp; 🎯 ByteTrack enabled
            </div>
            """, unsafe_allow_html=True)
        elif st.session_state.get("vid_file_bytes"):
            st.video(st.session_state.vid_file_bytes)
        else:
            st.markdown("""
            <div class="upload-zone">
                <span class="uz-icon">🎬</span>
                <div class="uz-title">Upload a surveillance video</div>
                <div class="uz-sub">Drag & drop or click to browse</div>
                <div class="uz-formats">
                    <span class="format-tag">MP4</span>
                    <span class="format-tag">AVI</span>
                    <span class="format-tag">MOV</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with st.form("vid_form", clear_on_submit=False):
            run_video = st.form_submit_button(
                "▶  Start Tracking",
                use_container_width=True,
            )

    with col_v2:
        st.markdown("""
        <div class="section-heading">
            <div class="sh-dot"></div>
            <div class="sh-title">Live Tracking Feed</div>
            <div class="sh-badge">ByteTrack</div>
        </div>
        """, unsafe_allow_html=True)

        _vid_bytes = (uploaded_video.getvalue() if uploaded_video
                      else st.session_state.get("vid_file_bytes"))
        _vid_name  = (uploaded_video.name if uploaded_video
                      else st.session_state.get("vid_file_name", "video"))
        _vid_size  = (uploaded_video.size if uploaded_video
                      else st.session_state.get("vid_file_size", 0))

        if _vid_bytes and run_video:
            if model is None:
                st.markdown("""
                <div class="model-error">
                    <div class="me-icon">⚠️</div>
                    <div class="me-title">Model Not Found</div>
                    <div class="me-sub">Place <code>best.pt</code> in the app directory and restart.</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                tfile.write(_vid_bytes)
                tfile.flush()

                cap     = cv2.VideoCapture(tfile.name)
                total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps     = int(cap.get(cv2.CAP_PROP_FPS)) or 25
                w_v     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h_v     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                output_path = "output_tracked.mp4"
                out = cv2.VideoWriter(output_path,
                                      cv2.VideoWriter_fourcc(*"mp4v"),
                                      fps, (w_v, h_v))

                feed_ph  = st.empty()
                prog_bar = st.progress(0, text="Initializing ByteTrack…")

                stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
                frames_ph  = stat_col1.empty()
                tracks_ph  = stat_col2.empty()
                viol_ph    = stat_col3.empty()
                fps_ph     = stat_col4.empty()

                all_violations: dict = {}          # class_name -> max_conf
                video_session_added  = set()
                trails: dict         = {}           # track_id -> deque of (cx,cy)
                active_ids: set      = set()        # all track IDs seen so far
                frame_idx   = 0
                _METRIC_EVERY = 15
                vid_file_id = f"vid_{_vid_name}_{_vid_size}"
                t0 = time.time()

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    if frame_idx % frame_skip == 0:
                        # ── model.track() runs ByteTrack internally ──
                        # persist=True keeps the tracker state across frames
                        results = model.track(
                            frame,
                            conf=confidence,
                            iou=iou_thresh,
                            tracker="bytetrack.yaml",
                            persist=True,
                            verbose=False,
                        )
                        annotated = draw_tracked_boxes(frame, results[0].boxes,
                                                       conf_thresh=confidence,
                                                       trails=trails)

                        # Collect violations and track IDs from this frame
                        if results[0].boxes is not None and results[0].boxes.id is not None:
                            for cls_id, conf_val, tid in zip(
                                results[0].boxes.cls,
                                results[0].boxes.conf,
                                results[0].boxes.id
                            ):
                                if float(conf_val) < confidence:
                                    continue
                                n = CLASS_NAMES[int(cls_id)]
                                conf_float = float(conf_val)
                                track_id   = int(tid)
                                active_ids.add(track_id)
                                video_session_added.add(n)
                                if n not in all_violations or conf_float > all_violations[n]:
                                    all_violations[n] = conf_float
                    else:
                        annotated = frame

                    out.write(annotated)

                    # Base64 JPEG — zero layout shift
                    rgb_frame = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                    pil_frame = Image.fromarray(rgb_frame)
                    buf = BytesIO()
                    pil_frame.save(buf, format="JPEG", quality=75)
                    b64 = base64.b64encode(buf.getvalue()).decode()
                    feed_ph.markdown(
                        f'<img src="data:image/jpeg;base64,{b64}" '
                        f'style="width:100%;border-radius:12px;display:block;" />',
                        unsafe_allow_html=True,
                    )

                    frame_idx += 1
                    elapsed  = time.time() - t0
                    live_fps = frame_idx / elapsed if elapsed > 0 else 0
                    pct      = min(frame_idx / max(total_f, 1), 1.0)
                    prog_bar.progress(pct, text=f"Tracking… {int(pct*100)}%  |  {live_fps:.1f} FPS")

                    if frame_idx % _METRIC_EVERY == 0:
                        frames_ph.metric("Frames",    frame_idx)
                        tracks_ph.metric("Tracks",    len(active_ids))
                        viol_ph.metric("Violations",  len(all_violations))
                        fps_ph.metric("Live FPS",     f"{live_fps:.1f}")

                cap.release()
                out.release()
                prog_bar.progress(1.0, text="✅ Tracking complete!")
                frames_ph.metric("Frames",   frame_idx)
                tracks_ph.metric("Tracks",   len(active_ids))
                viol_ph.metric("Violations", len(all_violations))
                fps_ph.metric("Live FPS",    f"{live_fps:.1f}")

                # Update session stats
                if vid_file_id not in st.session_state.processed_ids:
                    st.session_state.processed_ids.add(vid_file_id)
                    st.session_state.total_processed += 1
                    for n in video_session_added:
                        st.session_state.total_violations += 1
                        st.session_state.violation_counts[n] += 1
                render_session_stats(sidebar_stats)
                render_metric_cards(metric_cards, confidence, iou_thresh)

                st.markdown("<br>", unsafe_allow_html=True)

                if all_violations:
                    vv_col, chart_col = st.columns([1, 1], gap="medium")

                    with vv_col:
                        st.markdown(f"""
                        <div class="panel-card">
                            <div class="panel-card-title">⚠ {len(all_violations)} Violation Type{'s' if len(all_violations)>1 else ''}</div>
                        """, unsafe_allow_html=True)
                        for name, conf_val in sorted(all_violations.items(), key=lambda x: -x[1]):
                            meta  = CLASS_META.get(name, {})
                            col_a = meta.get('color', '#EF4444')
                            st.markdown(f"""
                            <div class="vtag" style="border-color:{col_a}35;background:rgba({",".join(str(x) for x in hex_to_rgb(col_a))},0.07);">
                                <div class="vtag-dot" style="background:{col_a};"></div>
                                <div class="vtag-name" style="color:{col_a};">{meta.get('icon','')} {fmt(name)}</div>
                            </div>
                            """, unsafe_allow_html=True)

                        st.markdown("""<br><div class="panel-card-title">📊 Confidence</div>""",
                                    unsafe_allow_html=True)
                        for name, conf_val in sorted(all_violations.items(), key=lambda x: -x[1]):
                            pct_v = int(conf_val * 100)
                            col_a = CLASS_META.get(name, {}).get('color', '#3B82F6')
                            st.markdown(f"""
                            <div class="conf-row">
                                <div class="conf-label">{fmt(name)}</div>
                                <div class="conf-bar-bg">
                                    <div class="conf-bar-fill" style="width:{pct_v}%;background:linear-gradient(90deg,{col_a},{col_a}BB);"></div>
                                </div>
                                <div class="conf-pct">{pct_v}%</div>
                            </div>
                            """, unsafe_allow_html=True)

                        # Track summary
                        st.markdown(f"""<br><div class="panel-card-title">🎯 {len(active_ids)} Unique Track ID{'s' if len(active_ids)!=1 else ''}</div>""",
                                    unsafe_allow_html=True)
                        ids_html = " ".join(
                            f'<span style="background:rgba(59,130,246,0.12);border:1px solid rgba(59,130,246,0.25);'
                            f'border-radius:6px;padding:2px 8px;color:#60A5FA;font-size:0.72rem;font-weight:700;">#{i}</span>'
                            for i in sorted(active_ids)
                        )
                        st.markdown(f'<div style="display:flex;flex-wrap:wrap;gap:6px;margin-top:8px;">{ids_html}</div>',
                                    unsafe_allow_html=True)
                        st.markdown("</div>", unsafe_allow_html=True)

                    with chart_col:
                        st.markdown("""
                        <div class="chart-card">
                            <div class="panel-card-title">🍩 Type Distribution</div>
                        """, unsafe_allow_html=True)
                        render_donut_chart(list(all_violations.items()))
                        st.markdown("</div>", unsafe_allow_html=True)

                else:
                    st.markdown("""
                    <div class="vtag vtag-safe" style="padding:16px 20px;">
                        <div class="vtag-dot" style="background:#10B981;"></div>
                        <div class="vtag-name" style="font-size:0.92rem;">✅ No violations detected in video</div>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)
                if os.path.exists(output_path):
                    with open(output_path, "rb") as f:
                        st.download_button(
                            "⬇️  Download Tracked Video",
                            f,
                            file_name="traffic_tracked_output.mp4",
                            mime="video/mp4",
                            use_container_width=True
                        )
        else:
            st.markdown("""
            <div style="background:linear-gradient(135deg,#0D1525,#111E32);
                        border:1px solid rgba(59,130,246,0.12);border-radius:20px;
                        padding:80px 28px;text-align:center;">
                <div style="font-size:3.5rem;margin-bottom:16px;opacity:0.4;">📹</div>
                <div style="color:#334155;font-size:0.90rem;font-weight:500;">
                    Upload a video and click Start Tracking
                </div>
            </div>
            """, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
#  TAB 3 — ABOUT
# ════════════════════════════════════════════════════════════
with tab3:
    c1, c2 = st.columns([1.2, 1], gap="large")

    with c1:
        st.markdown("""
        <div class="section-heading">
            <div class="sh-dot"></div>
            <div class="sh-title">About This Project</div>
        </div>
        <div class="panel-card">
            <div style="color:#F1F5F9;font-size:1.0rem;font-weight:800;margin-bottom:10px;">
                🚦 ML-Powered Traffic Violation Detection
            </div>
            <div style="color:#64748B;font-size:0.86rem;line-height:1.75;">
                This system uses <strong style="color:#60A5FA;">YOLOv8</strong> — a
                real-time object detection model — to automatically identify traffic violations from
                surveillance imagery and video feeds.<br><br>
                Trained on a <strong style="color:#60A5FA;">customised Roboflow dataset</strong>
                for traffic violation detection, the model detects 7 distinct violation categories,
                enabling scalable, automated traffic enforcement without manual oversight.
            </div>
            <hr style="border-color:rgba(59,130,246,0.10);margin:18px 0;">
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;">
        """, unsafe_allow_html=True)

        specs = [
            ("Model", "YOLOv8 Nano", "#3B82F6"),
            ("Classes", "7 Violation Types", "#10B981"),
            ("Dataset", "Custom Roboflow", "#6366F1"),
            ("Controls", "Confidence · IoU", "#F59E0B"),
            ("Input", "Image + Video", "#EC4899"),
            ("Output", "Annotated + Download", "#F97316"),
        ]
        for label, val, color in specs:
            st.markdown(f"""
            <div style="background:linear-gradient(135deg,#111E32,#0D1525);border:1px solid {color}20;
                        border-left:3px solid {color};border-radius:12px;padding:13px 14px;">
                <div style="color:#334155;font-size:0.68rem;font-weight:700;text-transform:uppercase;letter-spacing:0.09em;">{label}</div>
                <div style="color:#F1F5F9;font-size:0.88rem;font-weight:700;margin-top:3px;">{val}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("</div></div>", unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="section-heading">
            <div class="sh-dot"></div>
            <div class="sh-title">Violation Classes</div>
        </div>
        """, unsafe_allow_html=True)

        about_classes_html = ""
        for name, meta in CLASS_META.items():
            col_a = meta['color']
            count = st.session_state.violation_counts.get(name, 0)
            r, g, b = hex_to_rgb(col_a)
            count_badge = f'<div style="color:#FCA5A5;font-size:0.70rem;font-weight:700;">{count}x</div>' if count > 0 else ""
            about_classes_html += (
                f'<div style="background:linear-gradient(135deg,rgba({r},{g},{b},0.06),rgba({r},{g},{b},0.02));'
                f'border:1px solid {col_a}25;border-left:3px solid {col_a};'
                f'border-radius:14px;padding:14px 18px;margin-bottom:10px;'
                f'display:flex;align-items:center;gap:14px;">'
                f'<div style="font-size:1.6rem;flex-shrink:0;">{meta["icon"]}</div>'
                f'<div style="flex:1;">'
                f'<div style="color:#F1F5F9;font-size:0.88rem;font-weight:700;">{meta["short"]}</div>'
                f'<div style="color:#334155;font-size:0.72rem;margin-top:2px;font-family:monospace;">{name}</div>'
                f'</div>'
                f'<div style="text-align:right;">'
                f'<div style="width:10px;height:10px;border-radius:50%;background:{col_a};box-shadow:0 0 8px {col_a}70;margin:0 auto 4px;"></div>'
                f'{count_badge}'
                f'</div>'
                f'</div>'
            )
        st.markdown(about_classes_html, unsafe_allow_html=True)

# ── Footer ──
st.markdown("""
<div style="text-align:center;padding:28px 0 8px 0;
            border-top:1px solid rgba(59,130,246,0.08);margin-top:32px;">
    <div style="color:#1E3A5F;font-size:0.78rem;letter-spacing:0.05em;">
        <span style="color:#334155;">🚦 TrafficVision AI</span>
        <span style="color:#1E293B;"> &nbsp;·&nbsp; </span>
        <span style="color:#334155;">YOLOv8</span>
        <span style="color:#1E293B;"> &nbsp;·&nbsp; </span>
        <span style="color:#334155;">BBD University</span>
        <span style="color:#1E293B;"> &nbsp;·&nbsp; </span>
        <span style="color:#334155;">CSE-AI · 2026</span>
    </div>
</div>
""", unsafe_allow_html=True)
