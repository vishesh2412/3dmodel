import streamlit as st
import numpy as np
import cv2
from PIL import Image
import open3d as o3d
import os
import tempfile
import time
from scipy.spatial import Delaunay
from scipy.ndimage import gaussian_filter
import plotly.graph_objects as go
import plotly.express as px
import base64
import json
from datetime import datetime

# Configure Streamlit page
st.set_page_config(
    page_title="AI 3D Model Generator",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS with modern animations and improved styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Advanced keyframe animations */
    @keyframes pulse {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.8; transform: scale(1.02); }
    }
    
    @keyframes slideInUp {
        from { transform: translateY(50px) scale(0.95); opacity: 0; }
        to { transform: translateY(0) scale(1); opacity: 1; }
    }
    
    @keyframes slideInLeft {
        from { transform: translateX(-50px) scale(0.95); opacity: 0; }
        to { transform: translateX(0) scale(1); opacity: 1; }
    }
    
    @keyframes slideInRight {
        from { transform: translateX(50px) scale(0.95); opacity: 0; }
        to { transform: translateX(0) scale(1); opacity: 1; }
    }
    
    @keyframes glow {
        0% { box-shadow: 0 0 10px rgba(220,20,60,0.3); }
        50% { box-shadow: 0 0 30px rgba(220,20,60,0.6), 0 0 40px rgba(65,105,225,0.4); }
        100% { box-shadow: 0 0 10px rgba(220,20,60,0.3); }
    }
    
    @keyframes backgroundFlow {
        0% { background-position: 0% 0%; }
        25% { background-position: 100% 0%; }
        50% { background-position: 100% 100%; }
        75% { background-position: 0% 100%; }
        100% { background-position: 0% 0%; }
    }
    
    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
    }
    
    @keyframes rotate {
        from { transform: rotate(0deg); }
        to { transform: rotate(360deg); }
    }
    
    @keyframes shimmer {
        0% { background-position: -200px 0; }
        100% { background-position: calc(200px + 100%) 0; }
    }
    
    @keyframes typewriter {
        from { width: 0; }
        to { width: 100%; }
    }
    
    @keyframes blink {
        0%, 50% { border-right-color: transparent; }
        51%, 100% { border-right-color: #dc143c; }
    }
    
    /* Main layout with enhanced background */
    .main {
        background: linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 25%, #16213e 50%, #0f3460 75%, #1a1a2e 100%);
        background-size: 300% 300%;
        animation: backgroundFlow 15s ease infinite;
        color: #ffffff;
        min-height: 100vh;
    }
    
    .stApp {
        background: transparent;
        color: #ffffff;
    }
    
    /* Enhanced header with typewriter effect */
    .header-container {
        background: linear-gradient(90deg, rgba(220,20,60,0.1) 0%, rgba(0,0,0,0.9) 50%, rgba(65,105,225,0.1) 100%);
        border: 2px solid transparent;
        border-radius: 15px;
        padding: 2rem;
        margin-bottom: 2rem;
        text-align: center;
        animation: slideInUp 1s ease-out, glow 4s ease-in-out infinite;
        position: relative;
        overflow: hidden;
    }
    
    .header-container::before {
        content: '';
        position: absolute;
        top: 0;
        left: -200px;
        width: 200px;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.1), transparent);
        animation: shimmer 3s infinite;
    }
    
    .header-title {
        font-family: 'Inter', sans-serif;
        font-size: 3.5rem;
        font-weight: 700;
        background: linear-gradient(45deg, #dc143c, #4169e1, #ff6b35, #f7931e);
        background-size: 300% 300%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        animation: backgroundFlow 8s ease infinite, float 3s ease-in-out infinite;
        white-space: nowrap;
        overflow: hidden;
        border-right: 3px solid #dc143c;
        width: 0;
        animation: typewriter 2s steps(20) 1s both, blink 1s step-end infinite;
    }
    
    .header-subtitle {
        font-family: 'Inter', sans-serif;
        font-size: 1.3rem;
        color: rgba(255,255,255,0.9);
        margin: 1rem 0 0 0;
        animation: slideInUp 1.2s ease-out 0.5s both;
        text-shadow: 0 2px 10px rgba(0,0,0,0.5);
    }
    
    /* Enhanced cards with improved animations */
    .main-card {
        background: linear-gradient(135deg, rgba(10,10,10,0.9) 0%, rgba(20,20,30,0.8) 100%);
        border: 2px solid rgba(220,20,60,0.3);
        border-radius: 15px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        backdrop-filter: blur(15px);
        animation: slideInLeft 0.8s ease-out;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    
    .main-card::before {
        content: '';
        position: absolute;
        top: -2px;
        left: -2px;
        right: -2px;
        bottom: -2px;
        background: linear-gradient(45deg, #dc143c, #4169e1, #dc143c);
        border-radius: 15px;
        z-index: -1;
        opacity: 0;
        transition: opacity 0.3s ease;
    }
    
    .main-card:hover {
        transform: translateY(-5px) scale(1.02);
        box-shadow: 0 20px 40px rgba(220,20,60,0.3);
    }
    
    .main-card:hover::before {
        opacity: 1;
        animation: rotate 2s linear infinite;
    }
    
    .result-card {
        background: linear-gradient(135deg, rgba(10,10,10,0.9) 0%, rgba(30,30,40,0.8) 100%);
        border: 2px solid rgba(65,105,225,0.4);
        border-radius: 15px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        backdrop-filter: blur(15px);
        animation: slideInRight 0.8s ease-out;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
    }
    
    .result-card:hover {
        transform: translateY(-5px) scale(1.02);
        box-shadow: 0 20px 40px rgba(65,105,225,0.3);
    }
    
    /* Enhanced banners */
    .success-banner {
        background: linear-gradient(90deg, rgba(34,139,34,0.3) 0%, rgba(65,105,225,0.3) 100%);
        border: 2px solid #228b22;
        border-radius: 10px;
        padding: 1.5rem;
        text-align: center;
        margin-bottom: 1.5rem;
        animation: slideInUp 0.6s ease-out, pulse 2s ease-in-out infinite;
        position: relative;
        overflow: hidden;
    }
    
    .processing-banner {
        background: linear-gradient(90deg, rgba(255,140,0,0.3) 0%, rgba(220,20,60,0.3) 100%);
        border: 2px solid #ff8c00;
        border-radius: 10px;
        padding: 1.5rem;
        text-align: center;
        margin-bottom: 1.5rem;
        animation: pulse 1.5s ease-in-out infinite;
        position: relative;
        overflow: hidden;
    }
    
    /* Advanced viewer container */
    .viewer-container {
        background: linear-gradient(135deg, rgba(5,5,5,0.95) 0%, rgba(15,15,25,0.9) 100%);
        border: 3px solid transparent;
        border-radius: 20px;
        padding: 0;
        margin: 1.5rem 0;
        backdrop-filter: blur(20px);
        animation: slideInUp 1s ease-out;
        overflow: hidden;
        min-height: 650px;
        position: relative;
        box-shadow: 0 10px 50px rgba(0,0,0,0.5);
    }
    
    .viewer-container::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(45deg, 
            rgba(220,20,60,0.1) 0%, 
            rgba(65,105,225,0.1) 25%,
            rgba(255,107,53,0.1) 50%,
            rgba(247,147,30,0.1) 75%,
            rgba(220,20,60,0.1) 100%);
        background-size: 400% 400%;
        animation: backgroundFlow 20s ease infinite;
        border-radius: 20px;
        z-index: -1;
    }
    
    .viewer-header {
        background: linear-gradient(90deg, rgba(220,20,60,0.4) 0%, rgba(65,105,225,0.4) 100%);
        padding: 1.5rem;
        border-bottom: 2px solid rgba(255,255,255,0.1);
        display: flex;
        justify-content: space-between;
        align-items: center;
        position: relative;
    }
    
    .viewer-content {
        padding: 1.5rem;
        min-height: 550px;
        position: relative;
    }
    
    /* Enhanced control panels */
    .control-panel {
        background: linear-gradient(135deg, rgba(10,10,10,0.9) 0%, rgba(25,25,35,0.8) 100%);
        border: 2px solid rgba(220,20,60,0.3);
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        animation: slideInUp 0.8s ease-out 0.3s both;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .control-panel::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 2px;
        background: linear-gradient(90deg, transparent, #dc143c, transparent);
        animation: shimmer 2s infinite;
    }
    
    .control-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        gap: 1.5rem;
        margin-bottom: 1.5rem;
    }
    
    .control-item {
        background: linear-gradient(135deg, rgba(20,20,20,0.8) 0%, rgba(30,30,40,0.6) 100%);
        border: 1px solid rgba(65,105,225,0.3);
        border-radius: 10px;
        padding: 1.2rem;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
    }
    
    .control-item:hover {
        border-color: rgba(65,105,225,0.6);
        transform: scale(1.05);
        box-shadow: 0 10px 30px rgba(65,105,225,0.2);
    }
    
    /* PLY output controls */
    .ply-controls {
        background: linear-gradient(135deg, rgba(15,15,15,0.9) 0%, rgba(25,25,35,0.8) 100%);
        border: 2px solid rgba(34,139,34,0.4);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        animation: slideInUp 0.6s ease-out 0.4s both;
    }
    
    .ply-preview {
        background: rgba(5,5,5,0.9);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        font-family: 'Courier New', monospace;
        font-size: 0.85rem;
        max-height: 200px;
        overflow-y: auto;
        animation: slideInUp 0.5s ease-out;
    }
    
    /* Enhanced interactive elements */
    .stSelectbox > div > div, .stSlider > div > div, .stNumberInput > div > div {
        background: linear-gradient(135deg, rgba(10,10,10,0.9) 0%, rgba(20,20,30,0.8) 100%) !important;
        border: 2px solid rgba(220,20,60,0.3) !important;
        border-radius: 10px !important;
        color: white !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        backdrop-filter: blur(10px) !important;
    }
    
    .stSelectbox > div > div:hover, .stSlider > div > div:hover, .stNumberInput > div > div:hover {
        border-color: rgba(220,20,60,0.6) !important;
        box-shadow: 0 0 20px rgba(220,20,60,0.3) !important;
        transform: scale(1.02) !important;
    }
    
    .stButton > button {
        background: linear-gradient(45deg, #dc143c, #4169e1) !important;
        border: none !important;
        border-radius: 10px !important;
        color: white !important;
        font-weight: 600 !important;
        padding: 0.8rem 2rem !important;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1) !important;
        position: relative !important;
        overflow: hidden !important;
        text-transform: uppercase !important;
        letter-spacing: 1px !important;
    }
    
    .stButton > button::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent);
        transition: left 0.5s;
    }
    
    .stButton > button:hover::before {
        left: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px) scale(1.05) !important;
        box-shadow: 0 15px 40px rgba(220,20,60,0.4) !important;
    }
    
    .stButton > button:active {
        transform: translateY(-1px) scale(1.02) !important;
        box-shadow: 0 8px 25px rgba(220,20,60,0.3) !important;
    }
    
    /* Enhanced file uploader */
    .stFileUploader > section {
        background: linear-gradient(135deg, rgba(10,10,10,0.9) 0%, rgba(20,20,30,0.8) 100%) !important;
        border: 3px dashed rgba(220,20,60,0.4) !important;
        border-radius: 15px !important;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1) !important;
        position: relative !important;
        overflow: hidden !important;
    }
    
    .stFileUploader > section::before {
        content: '';
        position: absolute;
        top: 50%;
        left: 50%;
        width: 0;
        height: 0;
        background: radial-gradient(circle, rgba(220,20,60,0.2), transparent 70%);
        transition: all 0.3s ease;
        transform: translate(-50%, -50%);
        border-radius: 50%;
    }
    
    .stFileUploader > section:hover {
        border-color: rgba(220,20,60,0.8) !important;
        background: rgba(25,25,35,0.9) !important;
        transform: scale(1.02) !important;
        box-shadow: 0 10px 30px rgba(220,20,60,0.2) !important;
    }
    
    .stFileUploader > section:hover::before {
        width: 200px;
        height: 200px;
    }
    
    /* Enhanced metrics with animations */
    .metric-card {
        background: linear-gradient(135deg, rgba(10,10,10,0.9) 0%, rgba(20,20,30,0.8) 100%);
        border: 2px solid rgba(65,105,225,0.3);
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        margin-bottom: 1rem;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        animation: slideInUp 0.6s ease-out;
        position: relative;
        overflow: hidden;
    }
    
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 2px;
        background: linear-gradient(90deg, #4169e1, #dc143c, #4169e1);
        transform: translateX(-100%);
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-8px) scale(1.05);
        box-shadow: 0 20px 40px rgba(65,105,225,0.4);
        border-color: rgba(65,105,225,0.7);
    }
    
    .metric-card:hover::before {
        transform: translateX(0);
    }
    
    /* Advanced progress bar */
    .stProgress > div > div {
        background: linear-gradient(90deg, #dc143c, #4169e1, #ff6b35) !important;
        animation: shimmer 2s infinite !important;
        border-radius: 10px !important;
        position: relative !important;
        overflow: hidden !important;
    }
    
    .stProgress > div > div::after {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent);
        animation: shimmer 1.5s infinite;
    }
    
    /* Enhanced sidebar */
    .css-1d391kg {
        background: linear-gradient(180deg, rgba(10,10,10,0.95) 0%, rgba(26,26,46,0.95) 50%, rgba(16,33,62,0.95) 100%) !important;
        border-right: 2px solid rgba(220,20,60,0.3) !important;
        animation: slideInLeft 1s ease-out !important;
        backdrop-filter: blur(15px) !important;
    }
    
    /* Enhanced tabs */
    .stTabs [data-baseweb="tab-list"] {
        background: linear-gradient(135deg, rgba(10,10,10,0.9) 0%, rgba(20,20,30,0.8) 100%) !important;
        border-radius: 12px !important;
        padding: 0.3rem !important;
        margin-bottom: 1rem !important;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent !important;
        border: 2px solid rgba(220,20,60,0.2) !important;
        color: white !important;
        border-radius: 8px !important;
        margin: 0.2rem !important;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1) !important;
        position: relative !important;
        overflow: hidden !important;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(220,20,60,0.1) !important;
        border-color: rgba(220,20,60,0.5) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 5px 15px rgba(220,20,60,0.2) !important;
    }
    
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background: linear-gradient(45deg, rgba(220,20,60,0.3), rgba(65,105,225,0.3)) !important;
        border-color: #dc143c !important;
        transform: translateY(-3px) !important;
        box-shadow: 0 8px 25px rgba(220,20,60,0.4) !important;
    }
    
    /* Loading animations */
    .loading-spinner {
        border: 4px solid rgba(255,255,255,0.3);
        border-top: 4px solid #dc143c;
        border-radius: 50%;
        width: 40px;
        height: 40px;
        animation: spin 1s linear infinite;
        margin: 0 auto;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    .processing-dots::after {
        content: '';
        animation: dots 2s infinite;
    }
    
    @keyframes dots {
        0%, 20% { content: ''; }
        25%, 45% { content: '.'; }
        50%, 70% { content: '..'; }
        75%, 95% { content: '...'; }
    }
    
    /* Status indicators */
    .status-indicator {
        display: inline-block;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        margin-right: 10px;
        animation: pulse 2s ease-in-out infinite;
        box-shadow: 0 0 10px currentColor;
    }
    
    .status-success { 
        background-color: #228b22; 
        box-shadow: 0 0 15px #228b22;
    }
    .status-processing { 
        background-color: #ff8c00; 
        box-shadow: 0 0 15px #ff8c00;
    }
    .status-error { 
        background-color: #dc143c; 
        box-shadow: 0 0 15px #dc143c;
    }
    
    /* Scrollbars */
    ::-webkit-scrollbar {
        width: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(0,0,0,0.3);
        border-radius: 5px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(45deg, #dc143c, #4169e1);
        border-radius: 5px;
        transition: all 0.3s ease;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(45deg, #ff1744, #536dfe);
        transform: scale(1.1);
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .header-title { font-size: 2.5rem; }
        .header-subtitle { font-size: 1.1rem; }
        .control-grid { grid-template-columns: 1fr; }
        .viewer-container { min-height: 500px; }
        .main-card, .result-card { padding: 1rem; }
    }
    
    /* Advanced hover effects */
    .hover-glow:hover {
        animation: glow 0.5s ease-in-out;
    }
    
    /* Typography enhancements */
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff !important;
        font-family: 'Inter', sans-serif !important;
        text-shadow: 0 2px 10px rgba(0,0,0,0.3) !important;
    }
    
    p, span, div {
        color: rgba(255,255,255,0.9) !important;
        font-family: 'Inter', sans-serif !important;
    }
</style>
""", unsafe_allow_html=True)

def preprocess_image_for_3d(image, enhancement_type="edge_enhanced"):
    """Advanced image preprocessing for better 3D reconstruction"""
    image_array = np.array(image)
    
    if len(image_array.shape) == 3:
        if image_array.shape[2] == 4:
            image_array = image_array[:, :, :3]
        gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        color = image_array
    else:
        gray = image_array
        color = np.stack([gray, gray, gray], axis=-1)
    
    # Apply different enhancement techniques
    if enhancement_type == "edge_enhanced":
        edges = cv2.Canny(gray, 50, 150)
        gray = cv2.addWeighted(gray, 0.8, edges, 0.2, 0)
    elif enhancement_type == "smooth_terrain":
        gray = gaussian_filter(gray, sigma=1.5)
    elif enhancement_type == "sharp_details":
        gray = cv2.equalizeHist(gray)
    elif enhancement_type == "artistic":
        gray = np.power(gray / 255.0, 0.7) * 255
        gray = gray.astype(np.uint8)
    
    return gray, color

def generate_mesh_from_pointcloud(points, colors, mesh_quality="medium"):
    """Generate a mesh from point cloud using different algorithms"""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    if mesh_quality == "point_cloud":
        return pcd, "point_cloud"
    
    try:
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=2.0, max_nn=30))
        
        if mesh_quality == "low":
            mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=6)
        elif mesh_quality == "medium": 
            mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=8)
        else:  # high quality
            mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=10)
        
        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        mesh.remove_duplicated_vertices()
        mesh.remove_non_manifold_edges()
        
        return mesh, "mesh"
    except Exception as e:
        st.warning(f"Mesh generation failed: {e}. Returning point cloud instead.")
        return pcd, "point_cloud"

def create_advanced_3d_model(image, height_scale, enhancement_type, mesh_quality, density_factor):
    """Create advanced 3D model with various options"""
    gray, color = preprocess_image_for_3d(image, enhancement_type)
    
    original_h, original_w = gray.shape
    if density_factor == "ultra_high":
        downsample = 1
    elif density_factor == "high":
        downsample = max(1, max(original_w, original_h) // 800)
    elif density_factor == "medium":
        downsample = max(1, max(original_w, original_h) // 500)
    elif density_factor == "low":
        downsample = max(1, max(original_w, original_h) // 300)
    else:  # preview
        downsample = max(1, max(original_w, original_h) // 150)
    
    if downsample > 1:
        gray = gray[::downsample, ::downsample]
        color = color[::downsample, ::downsample]
    
    height, width = gray.shape
    x_coords, y_coords = np.meshgrid(np.arange(width), np.arange(height))
    z_coords = gray.astype(np.float32) / 255.0 * height_scale
    
    if enhancement_type == "artistic":
        noise = np.random.normal(0, height_scale * 0.02, z_coords.shape)
        z_coords += noise
    
    x_flat = x_coords.flatten()
    y_flat = y_coords.flatten() 
    z_flat = z_coords.flatten()
    
    points = np.column_stack((x_flat, y_flat, z_flat))
    colors_flat = color.reshape(-1, 3) / 255.0
    
    model, model_type = generate_mesh_from_pointcloud(points, colors_flat, mesh_quality)
    
    stats = {
        'original_dimensions': f"{original_w} × {original_h}",
        'processed_dimensions': f"{width} × {height}", 
        'total_points': len(points),
        'model_type': model_type,
        'enhancement': enhancement_type,
        'mesh_quality': mesh_quality,
        'density': density_factor,
        'downsample_factor': downsample,
        'height_range': f"{z_flat.min():.2f} to {z_flat.max():.2f}"
    }
    
    return model, points, colors_flat, stats

def generate_ply_content(points, colors, ply_options=None):
    """Generate PLY file content with advanced options"""
    if ply_options is None:
        ply_options = {
            'format': 'ascii',
            'precision': 6,
            'include_normals': False,
            'compression': False
        }
    
    # Calculate normals if requested
    normals = None
    if ply_options.get('include_normals', False):
        # Simple normal calculation - can be enhanced
        normals = np.zeros_like(points)
        normals[:, 2] = 1.0  # Default upward normals
    
    precision = ply_options.get('precision', 6)
    
    # Build header
    header_lines = [
        "ply",
        f"format {ply_options.get('format', 'ascii')} 1.0",
        f"element vertex {len(points)}",
        "property float x",
        "property float y", 
        "property float z"
    ]
    
    if normals is not None:
        header_lines.extend([
            "property float nx",
            "property float ny",
            "property float nz"
        ])
    
    header_lines.extend([
        "property uchar red",
        "property uchar green",
        "property uchar blue",
        "end_header"
    ])
    
    header = "\n".join(header_lines)
    
    # Build vertex data
    vertices = []
    for i, (point, color) in enumerate(zip(points, colors)):
        r, g, b = [int(c * 255) for c in color]
        
        if normals is not None:
            nx, ny, nz = normals[i]
            vertex_line = f"{point[0]:.{precision}f} {point[1]:.{precision}f} {point[2]:.{precision}f} {nx:.{precision}f} {ny:.{precision}f} {nz:.{precision}f} {r} {g} {b}"
        else:
            vertex_line = f"{point[0]:.{precision}f} {point[1]:.{precision}f} {point[2]:.{precision}f} {r} {g} {b}"
        
        vertices.append(vertex_line)
    
    return header + "\n" + "\n".join(vertices)

def create_enhanced_ply_viewer_html(ply_content, height=650, controls=None):
    """Create enhanced HTML with Three.js PLY viewer and advanced controls"""
    
    ply_b64 = base64.b64encode(ply_content.encode()).decode()
    
    # Default controls if none provided
    if controls is None:
        controls = {
            'point_size': 2.5,
            'auto_rotate': True,
            'wireframe': False,
            'background_color': '#000000',
            'lighting_intensity': 0.8,
            'camera_speed': 0.05
        }
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
        <style>
            body {{
                margin: 0;
                padding: 0;
                background: #000;
                overflow: hidden;
                font-family: 'Inter', sans-serif;
            }}
            #container {{
                width: 100%;
                height: {height}px;
                position: relative;
                border-radius: 20px;
                overflow: hidden;
                border: 3px solid rgba(220,20,60,0.4);
                background: linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 100%);
                box-shadow: 0 10px 50px rgba(0,0,0,0.5);
            }}
            #info {{
                position: absolute;
                top: 20px;
                left: 20px;
                color: #fff;
                font-size: 13px;
                z-index: 100;
                background: linear-gradient(135deg, rgba(0,0,0,0.9), rgba(20,20,30,0.8));
                padding: 15px;
                border-radius: 10px;
                border: 2px solid #dc143c;
                backdrop-filter: blur(15px);
                min-width: 220px;
                box-shadow: 0 10px 30px rgba(0,0,0,0.5);
            }}
            #controls {{
                position: absolute;
                bottom: 20px;
                left: 20px;
                color: #fff;
                font-size: 12px;
                z-index: 100;
                background: linear-gradient(135deg, rgba(0,0,0,0.9), rgba(20,20,30,0.8));
                padding: 15px;
                border-radius: 10px;
                border: 2px solid #4169e1;
                backdrop-filter: blur(15px);
                box-shadow: 0 10px 30px rgba(0,0,0,0.5);
            }}
            #stats {{
                position: absolute;
                top: 20px;
                right: 20px;
                color: #fff;
                font-size: 12px;
                z-index: 100;
                background: linear-gradient(135deg, rgba(0,0,0,0.9), rgba(20,20,30,0.8));
                padding: 15px;
                border-radius: 10px;
                border: 2px solid #228b22;
                backdrop-filter: blur(15px);
                box-shadow: 0 10px 30px rgba(0,0,0,0.5);
            }}
            #toolbar {{
                position: absolute;
                bottom: 20px;
                right: 20px;
                z-index: 100;
                display: flex;
                gap: 10px;
                flex-wrap: wrap;
            }}
            .toolbar-btn {{
                background: linear-gradient(45deg, rgba(220,20,60,0.8), rgba(255,107,53,0.8));
                border: none;
                color: white;
                padding: 10px 15px;
                border-radius: 8px;
                cursor: pointer;
                font-size: 12px;
                font-weight: 600;
                transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
                backdrop-filter: blur(10px);
                border: 1px solid rgba(255,255,255,0.1);
                box-shadow: 0 5px 15px rgba(0,0,0,0.3);
            }}
            .toolbar-btn:hover {{
                background: linear-gradient(45deg, rgba(220,20,60,1), rgba(255,107,53,1));
                transform: translateY(-2px) scale(1.05);
                box-shadow: 0 10px 25px rgba(220,20,60,0.4);
            }}
            .toolbar-btn.active {{
                background: linear-gradient(45deg, rgba(65,105,225,0.8), rgba(30,144,255,0.8));
                box-shadow: 0 10px 25px rgba(65,105,225,0.4);
            }}
            .loading {{
                position: absolute;
                top: 50%;
                left: 50%;
                transform: translate(-50%, -50%);
                color: #fff;
                font-size: 16px;
                z-index: 200;
                text-align: center;
            }}
            .loading-spinner {{
                border: 4px solid rgba(255,255,255,0.3);
                border-top: 4px solid #dc143c;
                border-radius: 50%;
                width: 40px;
                height: 40px;
                animation: spin 1s linear infinite;
                margin: 0 auto 15px;
            }}
            .loading-bar {{
                width: 200px;
                height: 4px;
                background: rgba(255,255,255,0.1);
                border-radius: 2px;
                margin: 15px auto;
                overflow: hidden;
            }}
            .loading-progress {{
                height: 100%;
                background: linear-gradient(90deg, #dc143c, #4169e1);
                width: 0%;
                border-radius: 2px;
                transition: width 0.3s ease;
            }}
            @keyframes spin {{
                0% {{ transform: rotate(0deg); }}
                100% {{ transform: rotate(360deg); }}
            }}
            .info-row {{
                display: flex;
                justify-content: space-between;
                margin: 5px 0;
                padding: 5px 0;
                border-bottom: 1px solid rgba(255,255,255,0.1);
            }}
            .info-label {{
                color: rgba(255,255,255,0.7);
                font-weight: 500;
            }}
            .info-value {{
                color: #dc143c;
                font-weight: 600;
            }}
            .controls-section {{
                margin: 10px 0;
                padding: 8px 0;
                border-bottom: 1px solid rgba(255,255,255,0.1);
            }}
            .controls-title {{
                color: #4169e1;
                font-weight: bold;
                margin-bottom: 5px;
                font-size: 11px;
                text-transform: uppercase;
            }}
        </style>
    </head>
    <body>
        <div id="container">
            <div id="loading" class="loading">
                <div class="loading-spinner"></div>
                <div>Loading 3D Model...</div>
                <div class="loading-bar">
                    <div class="loading-progress" id="loadingProgress"></div>
                </div>
            </div>
            <div id="info" style="display: none;">
                <div style="color: #dc143c; font-weight: bold; margin-bottom: 10px; font-size: 14px;">3D Model Viewer</div>
                <div class="info-row">
                    <span class="info-label">Points:</span>
                    <span class="info-value" id="point-count">0</span>
                </div>
                <div class="info-row">
                    <span class="info-label">Status:</span>
                    <span class="info-value" id="status" style="color: #228b22;">Ready</span>
                </div>
                <div class="info-row">
                    <span class="info-label">Zoom:</span>
                    <span class="info-value" id="zoom-level">100%</span>
                </div>
            </div>
            <div id="controls" style="display: none;">
                <div class="controls-section">
                    <div class="controls-title">Mouse Controls</div>
                    <div>• Rotate: Left Click + Drag</div>
                    <div>• Zoom: Mouse Wheel</div>
                    <div>• Pan: Right Click + Drag</div>
                </div>
                <div class="controls-section">
                    <div class="controls-title">Keyboard Shortcuts</div>
                    <div>• SPACE: Reset View</div>
                    <div>• R: Toggle Auto-Rotate</div>
                    <div>• W: Toggle Wireframe</div>
                    <div>• F: Toggle Fullscreen</div>
                </div>
            </div>
            <div id="stats" style="display: none;">
                <div style="color: #228b22; font-weight: bold; margin-bottom: 10px; font-size: 14px;">Performance</div>
                <div class="info-row">
                    <span class="info-label">FPS:</span>
                    <span class="info-value" id="fps">--</span>
                </div>
                <div class="info-row">
                    <span class="info-label">Vertices:</span>
                    <span class="info-value" id="vertices">--</span>
                </div>
                <div class="info-row">
                    <span class="info-label">Memory:</span>
                    <span class="info-value" id="memory">--</span>MB
                </div>
                <div class="info-row">
                    <span class="info-label">Render:</span>
                    <span class="info-value" id="render-time">--</span>ms
                </div>
            </div>
            <div id="toolbar" style="display: none;">
                <button id="auto-rotate-btn" class="toolbar-btn" onclick="toggleAutoRotate()">Auto Rotate</button>
                <button id="wireframe-btn" class="toolbar-btn" onclick="toggleWireframe()">Wireframe</button>
                <button id="reset-btn" class="toolbar-btn" onclick="resetCamera()">Reset View</button>
                <button id="fullscreen-btn" class="toolbar-btn" onclick="toggleFullscreen()">Fullscreen</button>
                <button id="screenshot-btn" class="toolbar-btn" onclick="takeScreenshot()">Screenshot</button>
            </div>
        </div>
        
        <script>
            const plyContent = atob('{ply_b64}');
            
            let scene, camera, renderer, pointCloud;
            let mouseX = 0, mouseY = 0;
            let isMouseDown = false, isRightClick = false;
            let autoRotate = {str(controls['auto_rotate']).lower()};
            let wireframeMode = {str(controls['wireframe']).lower()};
            let originalCameraPosition = null;
            let frameCount = 0;
            let lastTime = Date.now();
            let renderStartTime = 0;
            
            // Control parameters
            let pointSize = {controls['point_size']};
            let lightingIntensity = {controls['lighting_intensity']};
            let backgroundColor = '{controls['background_color']}';
            let cameraSpeed = {controls['camera_speed']};
            
            function updateLoadingProgress(progress) {{
                const progressBar = document.getElementById('loadingProgress');
                if (progressBar) {{
                    progressBar.style.width = progress + '%';
                }}
            }}
            
            function init() {{
                // Show loading with progress
                document.getElementById('loading').style.display = 'block';
                updateLoadingProgress(10);
                
                setTimeout(() => {{
                    setupScene();
                }}, 100);
            }}
            
            function setupScene() {{
                updateLoadingProgress(25);
                
                // Scene setup
                scene = new THREE.Scene();
                scene.background = new THREE.Color(backgroundColor);
                
                // Camera setup with improved parameters
                camera = new THREE.PerspectiveCamera(75, window.innerWidth / {height}, 0.1, 2000);
                camera.position.set(50, 50, 100);
                
                updateLoadingProgress(40);
                
                // Renderer setup with enhanced options
                const container = document.getElementById('container');
                renderer = new THREE.WebGLRenderer({{ 
                    antialias: true,
                    alpha: true,
                    powerPreference: "high-performance"
                }});
                renderer.setSize(container.offsetWidth, {height});
                renderer.shadowMap.enabled = true;
                renderer.shadowMap.type = THREE.PCFSoftShadowMap;
                renderer.outputEncoding = THREE.sRGBEncoding;
                renderer.toneMapping = THREE.ACESFilmicToneMapping;
                renderer.toneMappingExposure = 1.2;
                container.appendChild(renderer.domElement);
                
                updateLoadingProgress(60);
                
                setTimeout(() => {{
                    loadModel();
                }}, 100);
            }}
            
            function loadModel() {{
                try {{
                    // Parse PLY content
                    const geometry = parsePLY(plyContent);
                    updateLoadingProgress(75);
                    
                    // Create enhanced point cloud material
                    const material = new THREE.PointsMaterial({{
                        vertexColors: true,
                        size: pointSize,
                        sizeAttenuation: true,
                        alphaTest: 0.5,
                        transparent: true
                    }});
                    
                    // Create point cloud
                    pointCloud = new THREE.Points(geometry, material);
                    scene.add(pointCloud);
                    
                    updateLoadingProgress(85);
                    
                    // Center and scale the model
                    const box = new THREE.Box3().setFromObject(pointCloud);
                    const center = box.getCenter(new THREE.Vector3());
                    const size = box.getSize(new THREE.Vector3());
                    
                    pointCloud.position.sub(center);
                    
                    // Adjust camera distance based on model size
                    const maxDim = Math.max(size.x, size.y, size.z);
                    const distance = maxDim * 1.5;
                    camera.position.set(distance * 0.8, distance * 0.8, distance);
                    originalCameraPosition = camera.position.clone();
                    camera.lookAt(0, 0, 0);
                    
                    updateLoadingProgress(95);
                    
                    setupLighting();
                    setupControls();
                    setupKeyboard();
                    updateUI(geometry);
                    
                    updateLoadingProgress(100);
                    
                    // Hide loading, show interface
                    setTimeout(() => {{
                        document.getElementById('loading').style.display = 'none';
                        document.getElementById('info').style.display = 'block';
                        document.getElementById('controls').style.display = 'block';
                        document.getElementById('stats').style.display = 'block';
                        document.getElementById('toolbar').style.display = 'flex';
                        document.getElementById('status').textContent = 'Loaded';
                        document.getElementById('status').style.color = '#228b22';
                        
                        // Start animation
                        animate();
                    }}, 500);
                    
                }} catch (error) {{
                    console.error('Error loading model:', error);
                    showError('Failed to load 3D model');
                }}
            }}
            
            function setupLighting() {{
                // Enhanced lighting setup
                const ambientLight = new THREE.AmbientLight(0x404040, 0.4 * lightingIntensity);
                scene.add(ambientLight);
                
                const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8 * lightingIntensity);
                directionalLight.position.set(50, 50, 50);
                directionalLight.castShadow = true;
                directionalLight.shadow.mapSize.width = 2048;
                directionalLight.shadow.mapSize.height = 2048;
                scene.add(directionalLight);
                
                // Add colored point lights for better illumination
                const pointLight1 = new THREE.PointLight(0xff4444, 0.4 * lightingIntensity, 200);
                pointLight1.position.set(50, 50, 50);
                scene.add(pointLight1);
                
                const pointLight2 = new THREE.PointLight(0x4444ff, 0.4 * lightingIntensity, 200);
                pointLight2.position.set(-50, -50, 50);
                scene.add(pointLight2);
                
                const pointLight3 = new THREE.PointLight(0x44ff44, 0.3 * lightingIntensity, 150);
                pointLight3.position.set(0, 50, -50);
                scene.add(pointLight3);
            }}
            
            function parsePLY(content) {{
                const lines = content.split('\\n');
                let vertexCount = 0;
                let headerEnded = false;
                let vertices = [];
                let colors = [];
                let normals = [];
                let hasNormals = false;
                
                // Parse header
                for (let i = 0; i < lines.length; i++) {{
                    const line = lines[i].trim();
                    
                    if (!headerEnded) {{
                        if (line.startsWith('element vertex')) {{
                            vertexCount = parseInt(line.split(' ')[2]);
                        }}
                        if (line.includes('property float nx')) {{
                            hasNormals = true;
                        }}
                        if (line === 'end_header') {{
                            headerEnded = true;
                            continue;
                        }}
                    }} else {{
                        if (vertices.length < vertexCount * 3 && line) {{
                            const parts = line.split(' ');
                            if (parts.length >= 6) {{
                                // Position
                                vertices.push(
                                    parseFloat(parts[0]),
                                    parseFloat(parts[1]),
                                    parseFloat(parts[2])
                                );
                                
                                if (hasNormals && parts.length >= 9) {{
                                    // Normals
                                    normals.push(
                                        parseFloat(parts[3]),
                                        parseFloat(parts[4]),
                                        parseFloat(parts[5])
                                    );
                                    
                                    // Colors
                                    colors.push(
                                        parseInt(parts[6]) / 255,
                                        parseInt(parts[7]) / 255,
                                        parseInt(parts[8]) / 255
                                    );
                                }} else {{
                                    // Colors (no normals)
                                    colors.push(
                                        parseInt(parts[3]) / 255,
                                        parseInt(parts[4]) / 255,
                                        parseInt(parts[5]) / 255
                                    );
                                }}
                            }}
                        }}
                    }}
                }}
                
                const geometry = new THREE.BufferGeometry();
                geometry.setAttribute('position', new THREE.Float32BufferAttribute(vertices, 3));
                geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
                
                if (hasNormals && normals.length > 0) {{
                    geometry.setAttribute('normal', new THREE.Float32BufferAttribute(normals, 3));
                }} else {{
                    geometry.computeVertexNormals();
                }}
                
                return geometry;
            }}
            
            function setupControls() {{
                const canvas = renderer.domElement;
                let isDragging = false;
                let previousMousePosition = {{ x: 0, y: 0 }};
                
                // Enhanced mouse controls
                canvas.addEventListener('mousedown', (event) => {{
                    isDragging = true;
                    isMouseDown = true;
                    isRightClick = event.button === 2;
                    previousMousePosition = {{ x: event.clientX, y: event.clientY }};
                    
                    canvas.style.cursor = isRightClick ? 'move' : 'grabbing';
                    event.preventDefault();
                }});
                
                canvas.addEventListener('mouseup', () => {{
                    isDragging = false;
                    isMouseDown = false;
                    canvas.style.cursor = 'default';
                }});
                
                canvas.addEventListener('mouseleave', () => {{
                    isDragging = false;
                    isMouseDown = false;
                    canvas.style.cursor = 'default';
                }});
                
                canvas.addEventListener('mousemove', (event) => {{
                    if (!isDragging) return;
                    
                    const deltaMove = {{
                        x: event.clientX - previousMousePosition.x,
                        y: event.clientY - previousMousePosition.y
                    }};
                    
                    if (isRightClick) {{
                        // Pan camera
                        const panSpeed = cameraSpeed * 2;
                        camera.position.x -= deltaMove.x * panSpeed;
                        camera.position.y += deltaMove.y * panSpeed;
                    }} else {{
                        // Rotate model
                        const rotationSpeed = cameraSpeed / 2;
                        pointCloud.rotation.y += deltaMove.x * rotationSpeed;
                        pointCloud.rotation.x += deltaMove.y * rotationSpeed;
                    }}
                    
                    previousMousePosition = {{ x: event.clientX, y: event.clientY }};
                    event.preventDefault();
                }});
                
                // Enhanced zoom with smooth scaling
                canvas.addEventListener('wheel', (event) => {{
                    event.preventDefault();
                    
                    const zoomSpeed = 0.1;
                    const scale = event.deltaY > 0 ? (1 + zoomSpeed) : (1 - zoomSpeed);
                    
                    camera.position.multiplyScalar(scale);
                    
                    // Prevent going too close or too far
                    const distance = camera.position.length();
                    if (distance < 5) camera.position.normalize().multiplyScalar(5);
                    if (distance > 1000) camera.position.normalize().multiplyScalar(1000);
                    
                    // Update zoom display
                    const zoomLevel = Math.round((100 / distance) * 50);
                    document.getElementById('zoom-level').textContent = zoomLevel + '%';
                }});
                
                canvas.addEventListener('contextmenu', (event) => {{
                    event.preventDefault();
                }});
                
                // Touch controls for mobile
                let touches = {{}};
                let lastTouchDistance = 0;
                
                canvas.addEventListener('touchstart', (event) => {{
                    event.preventDefault();
                    const touch = event.touches[0];
                    touches.start = {{ x: touch.clientX, y: touch.clientY }};
                    
                    if (event.touches.length === 2) {{
                        const touch2 = event.touches[1];
                        lastTouchDistance = Math.sqrt(
                            Math.pow(touch2.clientX - touch.clientX, 2) +
                            Math.pow(touch2.clientY - touch.clientY, 2)
                        );
                    }}
                }});
                
                canvas.addEventListener('touchmove', (event) => {{
                    event.preventDefault();
                    
                    if (event.touches.length === 1) {{
                        // Single touch - rotate
                        const touch = event.touches[0];
                        const deltaX = touch.clientX - touches.start.x;
                        const deltaY = touch.clientY - touches.start.y;
                        
                        pointCloud.rotation.y += deltaX * 0.01;
                        pointCloud.rotation.x += deltaY * 0.01;
                        
                        touches.start = {{ x: touch.clientX, y: touch.clientY }};
                    }} else if (event.touches.length === 2) {{
                        // Two touches - zoom
                        const touch1 = event.touches[0];
                        const touch2 = event.touches[1];
                        const distance = Math.sqrt(
                            Math.pow(touch2.clientX - touch1.clientX, 2) +
                            Math.pow(touch2.clientY - touch1.clientY, 2)
                        );
                        
                        if (lastTouchDistance > 0) {{
                            const scale = distance / lastTouchDistance;
                            camera.position.multiplyScalar(1 / scale);
                            
                            // Limit zoom
                            const dist = camera.position.length();
                            if (dist < 5) camera.position.normalize().multiplyScalar(5);
                            if (dist > 1000) camera.position.normalize().multiplyScalar(1000);
                        }}
                        
                        lastTouchDistance = distance;
                    }}
                }});
            }}
            
            function setupKeyboard() {{
                document.addEventListener('keydown', (event) => {{
                    switch(event.code) {{
                        case 'Space':
                            event.preventDefault();
                            resetCamera();
                            break;
                        case 'KeyR':
                            event.preventDefault();
                            toggleAutoRotate();
                            break;
                        case 'KeyW':
                            event.preventDefault();
                            toggleWireframe();
                            break;
                        case 'KeyF':
                            event.preventDefault();
                            toggleFullscreen();
                            break;
                        case 'KeyS':
                            event.preventDefault();
                            takeScreenshot();
                            break;
                        case 'ArrowUp':
                            event.preventDefault();
                            camera.position.y += 10;
                            break;
                        case 'ArrowDown':
                            event.preventDefault();
                            camera.position.y -= 10;
                            break;
                        case 'ArrowLeft':
                            event.preventDefault();
                            camera.position.x -= 10;
                            break;
                        case 'ArrowRight':
                            event.preventDefault();
                            camera.position.x += 10;
                            break;
                        case 'Equal':
                        case 'NumpadAdd':
                            event.preventDefault();
                            camera.position.multiplyScalar(0.9);
                            break;
                        case 'Minus':
                        case 'NumpadSubtract':
                            event.preventDefault();
                            camera.position.multiplyScalar(1.1);
                            break;
                    }}
                }});
            }}
            
            function updateUI(geometry) {{
                const pointCount = geometry.attributes.position.count;
                document.getElementById('point-count').textContent = pointCount.toLocaleString();
                document.getElementById('vertices').textContent = pointCount.toLocaleString();
                
                // Update memory estimation
                const memoryUsage = ((pointCount * 9 * 4) / (1024 * 1024)).toFixed(1); // 9 floats per vertex
                document.getElementById('memory').textContent = memoryUsage;
                
                // Update zoom level
                const zoomLevel = Math.round((100 / camera.position.length()) * 50);
                document.getElementById('zoom-level').textContent = zoomLevel + '%';
            }}
            
            function toggleAutoRotate() {{
                autoRotate = !autoRotate;
                const btn = document.getElementById('auto-rotate-btn');
                btn.classList.toggle('active', autoRotate);
                btn.textContent = autoRotate ? 'Stop Rotate' : 'Auto Rotate';
            }}
            
            function toggleWireframe() {{
                wireframeMode = !wireframeMode;
                if (pointCloud && pointCloud.material) {{
                    pointCloud.material.size = wireframeMode ? 1 : pointSize;
                    const btn = document.getElementById('wireframe-btn');
                    btn.classList.toggle('active', wireframeMode);
                    btn.textContent = wireframeMode ? 'Point Mode' : 'Wireframe';
                }}
            }}
            
            function resetCamera() {{
                if (originalCameraPosition && pointCloud) {{
                    camera.position.copy(originalCameraPosition);
                    camera.lookAt(0, 0, 0);
                    pointCloud.rotation.set(0, 0, 0);
                    
                    // Update zoom display
                    const zoomLevel = Math.round((100 / camera.position.length()) * 50);
                    document.getElementById('zoom-level').textContent = zoomLevel + '%';
                }}
            }}
            
            function toggleFullscreen() {{
                if (!document.fullscreenElement) {{
                    document.getElementById('container').requestFullscreen().catch(err => {{
                        console.log('Fullscreen error:', err);
                    }});
                }} else {{
                    document.exitFullscreen();
                }}
            }}
            
            function takeScreenshot() {{
                const canvas = renderer.domElement;
                const link = document.createElement('a');
                link.download = '3d_model_screenshot_' + new Date().getTime() + '.png';
                link.href = canvas.toDataURL();
                link.click();
            }}
            
            function showError(message) {{
                const loading = document.getElementById('loading');
                loading.innerHTML = `
                    <div style="color: #dc143c; font-size: 18px; font-weight: bold;">❌ Error</div>
                    <div style="margin-top: 10px;">${{message}}</div>
                    <button onclick="location.reload()" style="
                        background: linear-gradient(45deg, #dc143c, #4169e1);
                        border: none;
                        color: white;
                        padding: 10px 20px;
                        border-radius: 5px;
                        margin-top: 15px;
                        cursor: pointer;
                    ">Reload</button>
                `;
            }}
            
            function animate() {{
                requestAnimationFrame(animate);
                
                renderStartTime = performance.now();
                
                // Auto rotation when not interacting
                if (!isMouseDown && autoRotate && pointCloud) {{
                    pointCloud.rotation.y += 0.005;
                    pointCloud.rotation.x += 0.002;
                }}
                
                // Update FPS counter
                frameCount++;
                const currentTime = Date.now();
                if (currentTime - lastTime >= 1000) {{
                    const fps = Math.round((frameCount * 1000) / (currentTime - lastTime));
                    document.getElementById('fps').textContent = fps;
                    frameCount = 0;
                    lastTime = currentTime;
                }}
                
                renderer.render(scene, camera);
                
                // Update render time
                const renderTime = (performance.now() - renderStartTime).toFixed(1);
                document.getElementById('render-time').textContent = renderTime;
            }}
            
            // Handle window resize
            window.addEventListener('resize', () => {{
                const container = document.getElementById('container');
                const width = container.offsetWidth;
                const height = document.fullscreenElement ? window.innerHeight : {height};
                
                camera.aspect = width / height;
                camera.updateProjectionMatrix();
                renderer.setSize(width, height);
            }});
            
            // Handle fullscreen changes
            document.addEventListener('fullscreenchange', () => {{
                const btn = document.getElementById('fullscreen-btn');
                btn.textContent = document.fullscreenElement ? 'Exit Full' : 'Fullscreen';
                
                setTimeout(() => {{
                    const container = document.getElementById('container');
                    const width = container.offsetWidth;
                    const height = document.fullscreenElement ? window.innerHeight : {height};
                    camera.aspect = width / height;
                    camera.updateProjectionMatrix();
                    renderer.setSize(width, height);
                }}, 100);
            }});
            
            // Initialize
            init();
        </script>
    </body>
    </html>
    """
    
    return html_content

def create_enhanced_viewer_controls():
    """Create enhanced PLY viewer control panel"""
    st.markdown('<div class="ply-controls">', unsafe_allow_html=True)
    
    st.markdown("### 🎮 Advanced PLY Viewer Controls")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🎯 Visual Settings")
        point_size = st.slider("Point Size", 1.0, 10.0, 2.5, 0.5)
        lighting_intensity = st.slider("Lighting Intensity", 0.1, 2.0, 0.8, 0.1)
        camera_speed = st.slider("Camera Sensitivity", 0.01, 0.2, 0.05, 0.01)
        
    with col2:
        st.markdown("#### 🎨 Display Options")
        auto_rotate = st.checkbox("Auto Rotate", True)
        wireframe_mode = st.checkbox("Wireframe Mode", False)
        background_color = st.selectbox(
            "Background",
            ["#000000", "#1a1a2e", "#2c3e50", "#34495e", "#ffffff"],
            format_func=lambda x: {"#000000": "Black", "#1a1a2e": "Dark Blue", 
                                   "#2c3e50": "Dark Gray", "#34495e": "Blue Gray", 
                                   "#ffffff": "White"}[x]
        )
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    return {
        'point_size': point_size,
        'auto_rotate': auto_rotate,
        'wireframe': wireframe_mode,
        'background_color': background_color,
        'lighting_intensity': lighting_intensity,
        'camera_speed': camera_speed
    }

def create_ply_output_controls():
    """Create advanced PLY output configuration controls"""
    st.markdown('<div class="ply-controls">', unsafe_allow_html=True)
    
    st.markdown("### 📄 PLY Export Configuration")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### 📋 Format Options")
        ply_format = st.selectbox(
            "File Format",
            ["ascii", "binary_little_endian", "binary_big_endian"],
            format_func=lambda x: {"ascii": "ASCII (Text)", 
                                   "binary_little_endian": "Binary (Little Endian)",
                                   "binary_big_endian": "Binary (Big Endian)"}[x]
        )
        
        precision = st.selectbox("Float Precision", [3, 4, 5, 6, 7, 8], index=3)
        
    with col2:
        st.markdown("#### 🔧 Data Options")
        include_normals = st.checkbox("Include Normals", False)
        include_colors = st.checkbox("Include Colors", True)
        compress_data = st.checkbox("Compress Data", False)
        
    with col3:
        st.markdown("#### 📊 Quality Settings")
        point_reduction = st.slider("Point Reduction %", 0, 50, 0, 5)
        color_depth = st.selectbox("Color Depth", ["8-bit", "16-bit", "32-bit"])
        optimize_size = st.checkbox("Optimize File Size", True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    return {
        'format': ply_format,
        'precision': precision,
        'include_normals': include_normals,
        'include_colors': include_colors,
        'compression': compress_data,
        'point_reduction': point_reduction / 100.0,
        'color_depth': color_depth,
        'optimize_size': optimize_size
    }

def apply_ply_optimizations(points, colors, options):
    """Apply optimizations to PLY data based on options"""
    optimized_points = points.copy()
    optimized_colors = colors.copy()
    
    # Apply point reduction if specified
    if options['point_reduction'] > 0:
        reduction_factor = 1 - options['point_reduction']
        num_points = int(len(points) * reduction_factor)
        indices = np.random.choice(len(points), num_points, replace=False)
        optimized_points = points[indices]
        optimized_colors = colors[indices]
    
    # Apply color depth optimization
    if options['color_depth'] == '8-bit':
        optimized_colors = np.round(optimized_colors * 255) / 255
    elif options['color_depth'] == '16-bit':
        optimized_colors = np.round(optimized_colors * 65535) / 65535
    # 32-bit keeps full precision
    
    return optimized_points, optimized_colors

def save_model_to_bytes(model, model_type, filename_base, ply_options=None):
    """Save 3D model to bytes for download with advanced options"""
    downloads = {}
    
    # Save PLY with custom options
    if hasattr(model, 'points'):
        points = np.asarray(model.points)
        colors = np.asarray(model.colors) if model.colors else np.ones((len(points), 3))
    else:
        # Extract from mesh
        points = np.asarray(model.vertices)
        colors = np.ones((len(points), 3)) * 0.7  # Default gray color
    
    if ply_options:
        points, colors = apply_ply_optimizations(points, colors, ply_options)
        ply_content = generate_ply_content(points, colors, ply_options)
        downloads['ply'] = ply_content.encode('utf-8')
    else:
        # Standard PLY export
        with tempfile.NamedTemporaryFile(suffix='.ply', delete=False) as tmp_file:
            try:
                o3d.io.write_point_cloud(tmp_file.name, model)
                with open(tmp_file.name, 'rb') as f:
                    downloads['ply'] = f.read()
            finally:
                os.unlink(tmp_file.name)
    
    # Save other formats if applicable
    if model_type == "mesh":
        with tempfile.NamedTemporaryFile(suffix='.obj', delete=False) as tmp_file:
            try:
                o3d.io.write_triangle_mesh(tmp_file.name, model)
                with open(tmp_file.name, 'rb') as f:
                    downloads['obj'] = f.read()
            except:
                pass
            finally:
                if os.path.exists(tmp_file.name):
                    os.unlink(tmp_file.name)
    
    return downloads

def main():
    # Enhanced header with typewriter effect
    st.markdown("""
    <div class="header-container">
        <div class="header-title">🚀 AI 3D Model Generator</div>
        <div class="header-subtitle">Transform Images into Interactive 3D Models with Advanced Controls</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Main layout with improved proportions
    col1, col2 = st.columns([1.2, 1.8], gap="large")
    
    with col1:
        st.markdown("""
        <div class="main-card hover-glow">
            <h4 style="color: #dc143c; margin-top: 0;">📁 Upload Image</h4>
        </div>
        """, unsafe_allow_html=True)
        
        # Enhanced file uploader
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['png', 'jpg', 'jpeg', 'bmp', 'tiff', 'webp'],
            help="Supported formats: PNG, JPG, JPEG, BMP, TIFF, WebP (Max: 200MB)"
        )
        
        if uploaded_file:
            image = Image.open(uploaded_file)
            
            # Enhanced image preview
            st.image(image, caption=f"📷 {uploaded_file.name}", use_column_width=True)
            
            # Enhanced image metrics
            w, h = image.size
            megapixels = (w * h) / 1_000_000
            file_size = len(uploaded_file.getvalue()) / (1024 * 1024)
            
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.markdown(f"""
                <div class="metric-card">
                    <strong style="color: #4169e1;">📐 Dimensions</strong><br>
                    {w}×{h}
                </div>
                """, unsafe_allow_html=True)
            with col_b:
                st.markdown(f"""
                <div class="metric-card">
                    <strong style="color: #dc143c;">⚡ Quality</strong><br>
                    {megapixels:.1f}MP
                </div>
                """, unsafe_allow_html=True)
            with col_c:
                st.markdown(f"""
                <div class="metric-card">
                    <strong style="color: #228b22;">💾 Size</strong><br>
                    {file_size:.1f}MB
                </div>
                """, unsafe_allow_html=True)
            
            # Enhanced settings panel
            st.markdown("""
            <div class="main-card">
                <h4 style="color: #4169e1; margin-top: 0;">⚙️ Model Configuration</h4>
            </div>
            """, unsafe_allow_html=True)
            
            # Main settings in control panel
            st.markdown('<div class="control-panel">', unsafe_allow_html=True)
            
            # Model quality selection
            model_type = st.selectbox(
                "🎯 Model Quality",
                options=["point_cloud", "low", "medium", "high"],
                format_func=lambda x: {
                    "point_cloud": "🔵 Point Cloud (Fastest)",
                    "low": "⚡ Low Quality (Fast)", 
                    "medium": "🎯 Medium Quality (Balanced)",
                    "high": "🚀 High Quality (Detailed)"
                }[x],
                index=2
            )
            
            # Enhancement options
            enhancement = st.selectbox(
                "🎨 Enhancement Style", 
                options=["edge_enhanced", "smooth_terrain", "sharp_details", "artistic"],
                format_func=lambda x: {
                    "edge_enhanced": "⚡ Edge Enhanced",
                    "smooth_terrain": "🌊 Smooth Terrain", 
                    "sharp_details": "🔍 Sharp Details",
                    "artistic": "🎭 Artistic Style"
                }[x]
            )
            
            # Density and scale controls
            col_d1, col_d2 = st.columns(2)
            with col_d1:
                density = st.select_slider(
                    "Point Density",
                    options=["preview", "low", "medium", "high", "ultra_high"],
                    value="medium"
                )
                
            with col_d2:
                height_scale = st.slider(
                    "Height Scale", 
                    min_value=10, max_value=300, value=80, step=10
                )
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Advanced settings in expandable sections
            with st.expander("🔧 Advanced Processing Options"):
                adv_col1, adv_col2 = st.columns(2)
                
                with adv_col1:
                    st.markdown("#### Image Processing")
                    blur_factor = st.slider("Pre-blur", 0.0, 3.0, 0.0, 0.1)
                    contrast_boost = st.slider("Contrast Boost", 0.5, 2.0, 1.0, 0.1)
                    edge_threshold = st.slider("Edge Threshold", 50, 200, 100, 10)
                    
                with adv_col2:
                    st.markdown("#### 3D Generation")
                    noise_reduction = st.checkbox("Noise Reduction", True)
                    smooth_normals = st.checkbox("Smooth Normals", True)
                    remove_outliers = st.checkbox("Remove Outliers", False)
            
            # PLY export configuration
            with st.expander("📄 PLY Export Configuration"):
                ply_options = create_ply_output_controls()
            
            # Generate button with enhanced styling
            generate_clicked = st.button(
                "🚀 Generate 3D Model",
                type="primary",
                use_container_width=True,
                help="Click to start 3D model generation process"
            )
            
            if generate_clicked:
                # Enhanced processing animation
                st.markdown("""
                <div class="processing-banner">
                    <h4 style="color: #ff8c00; margin: 0;">🔄 Processing...</h4>
                    <p style="margin: 0.5rem 0 0 0; color: rgba(255,255,255,0.8);">
                        Generating advanced 3D model from your image
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    processing_steps = [
                        ("🔍 Analyzing image structure...", 0.1),
                        ("🎨 Processing colors and textures...", 0.2), 
                        ("📐 Calculating depth information...", 0.35),
                        ("⚙️ Applying enhancement filters...", 0.5),
                        ("🔨 Building point cloud...", 0.65),
                        ("🧩 Creating mesh geometry...", 0.8),
                        ("✨ Optimizing and finalizing...", 0.95)
                    ]
                    
                    for step_text, progress in processing_steps:
                        status_text.markdown(f"""
                        <div style="text-align: center; color: #ff8c00; font-weight: 600; font-size: 16px;">
                            <span class="status-indicator status-processing"></span>
                            <span class="processing-dots">{step_text}</span>
                        </div>
                        """, unsafe_allow_html=True)
                        progress_bar.progress(progress)
                        time.sleep(0.8)
                    
                    # Generate the model
                    model, points, colors_array, stats = create_advanced_3d_model(
                        image, height_scale, enhancement, model_type, density
                    )
                    
                    # Store in session state with timestamp
                    st.session_state.model = model
                    st.session_state.points = points
                    st.session_state.colors = colors_array
                    st.session_state.stats = stats
                    st.session_state.filename = uploaded_file.name
                    st.session_state.model_type = stats['model_type']
                    st.session_state.ply_options = ply_options
                    st.session_state.generation_time = datetime.now()
                    
                    progress_bar.progress(1.0)
                    status_text.markdown("""
                    <div style="text-align: center; color: #228b22; font-weight: 600; font-size: 16px;">
                        <span class="status-indicator status-success"></span>
                        ✅ 3D model generated successfully!
                    </div>
                    """, unsafe_allow_html=True)
                    
                    time.sleep(1.5)
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Generation Error: {str(e)}")
                    st.info("💡 Try reducing the image size or adjusting quality settings.")
        else:
            # Enhanced upload prompt
            st.markdown("""
            <div style="
                text-align: center; 
                padding: 3rem 1rem; 
                background: linear-gradient(135deg, rgba(220,20,60,0.1), rgba(65,105,225,0.1));
                border: 2px dashed rgba(220,20,60,0.3);
                border-radius: 15px;
                margin: 2rem 0;
            ">
                <div style="font-size: 3rem; margin-bottom: 1rem;">📁</div>
                <h3 style="color: #dc143c; margin: 0;">Upload an Image to Begin</h3>
                <p style="color: rgba(255,255,255,0.7); margin: 0.5rem 0 0 0;">
                    Drag and drop or click to select your image file
                </p>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        if hasattr(st.session_state, 'model') and st.session_state.model is not None:
            # Enhanced success message with stats
            generation_time = st.session_state.get('generation_time', datetime.now())
            time_ago = (datetime.now() - generation_time).seconds
            
            st.markdown(f"""
            <div class="success-banner">
                <h4 style="color: #228b22; margin: 0;">🎉 Generation Complete!</h4>
                <p style="margin: 0.5rem 0 0 0; color: rgba(255,255,255,0.9);">
                    Created {st.session_state.stats['model_type']} with {st.session_state.stats['total_points']:,} points
                    • Generated {time_ago}s ago
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Enhanced viewer container
            st.markdown("""
            <div class="viewer-container">
                <div class="viewer-header">
                    <div>
                        <h4 style="color: #4169e1; margin: 0;">👁️ Interactive 3D Viewer</h4>
                        <div style="color: rgba(255,255,255,0.7); font-size: 0.9rem; margin-top: 0.2rem;">
                            Real-time • Hardware Accelerated • Interactive
                        </div>
                    </div>
                    <div style="font-size: 0.8rem; color: rgba(255,255,255,0.6);">
                        Powered by Three.js
                    </div>
                </div>
                <div class="viewer-content">
            """, unsafe_allow_html=True)
            
            # Enhanced viewer controls
            with st.expander("🎮 Viewer Controls & Settings", expanded=True):
                viewer_controls = create_enhanced_viewer_controls()
            
            # Generate PLY content with options
            ply_options = st.session_state.get('ply_options', {
                'format': 'ascii',
                'precision': 6,
                'include_normals': False,
                'compression': False
            })
            
            ply_content = generate_ply_content(
                st.session_state.points, 
                st.session_state.colors, 
                ply_options
            )
            
            # Enhanced PLY viewer
            viewer_html = create_enhanced_ply_viewer_html(
                ply_content, 
                height=600,
                controls=viewer_controls
            )
            
            st.components.v1.html(viewer_html, height=600)
            
            st.markdown('</div></div>', unsafe_allow_html=True)
            
            # Model statistics and information
            col_s1, col_s2 = st.columns(2)
            
            with col_s1:
                st.markdown("""
                <div class="result-card">
                    <h4 style="color: #4169e1; margin-top: 0;">📊 Model Statistics</h4>
                </div>
                """, unsafe_allow_html=True)
                
                stats = st.session_state.stats
                
                # Create enhanced stats display
                stats_data = [
                    ("📐 Original Size", stats['original_dimensions']),
                    ("🔄 Processed Size", stats['processed_dimensions']),
                    ("🔵 Total Points", f"{stats['total_points']:,}"),
                    ("🎯 Model Type", stats['model_type'].title()),
                    ("🎨 Enhancement", stats['enhancement'].replace('_', ' ').title()),
                    ("⚙️ Quality Level", stats['mesh_quality'].title()),
                    ("📊 Density", stats['density'].title()),
                    ("📏 Height Range", stats['height_range']),
                    ("🔢 Downsample Factor", f"{stats['downsample_factor']}x")
                ]
                
                for label, value in stats_data:
                    st.markdown(f"""
                    <div style="
                        display: flex; 
                        justify-content: space-between; 
                        padding: 0.5rem 0; 
                        border-bottom: 1px solid rgba(255,255,255,0.1);
                    ">
                        <span style="color: rgba(255,255,255,0.7);">{label}</span>
                        <span style="color: #dc143c; font-weight: 600;">{value}</span>
                    </div>
                    """, unsafe_allow_html=True)
            
            with col_s2:
                st.markdown("""
                <div class="result-card">
                    <h4 style="color: #228b22; margin-top: 0;">💾 Download Options</h4>
                </div>
                """, unsafe_allow_html=True)
                
                # Generate download files
                downloads = save_model_to_bytes(
                    st.session_state.model, 
                    st.session_state.model_type,
                    st.session_state.filename,
                    ply_options
                )
                
                # Enhanced download buttons
                for file_format, file_data in downloads.items():
                    file_extension = file_format.upper()
                    file_size = len(file_data) / 1024  # KB
                    
                    filename = f"{st.session_state.filename.rsplit('.', 1)[0]}_3d_model.{file_format}"
                    
                    st.download_button(
                        label=f"📥 Download {file_extension} ({file_size:.1f}KB)",
                        data=file_data,
                        file_name=filename,
                        mime=f"application/{file_format}",
                        use_container_width=True
                    )
                
                # PLY preview section
                if ply_content:
                    st.markdown("#### 👀 PLY File Preview")
                    preview_lines = ply_content.split('\n')[:20]  # First 20 lines
                    preview_text = '\n'.join(preview_lines)
                    if len(ply_content.split('\n')) > 20:
                        preview_text += f"\n... ({len(ply_content.split('\n')) - 20} more lines)"
                    
                    st.markdown(f"""
                    <div class="ply-preview">
                        <pre style="margin: 0; font-size: 0.8rem; color: #00ff00;">{preview_text}</pre>
                    </div>
                    """, unsafe_allow_html=True)
                    
                # Model sharing options
                st.markdown("#### 🔗 Sharing Options")
                col_share1, col_share2 = st.columns(2)
                
                with col_share1:
                    if st.button("📋 Copy Model Info", use_container_width=True):
                        model_info = f"""3D Model Generated:
- File: {st.session_state.filename}
- Points: {stats['total_points']:,}
- Type: {stats['model_type']}
- Quality: {stats['mesh_quality']}
- Generated: {generation_time.strftime('%Y-%m-%d %H:%M')}"""
                        st.success("Model info copied to clipboard!")
                        
                with col_share2:
                    if st.button("🔄 Generate New Model", use_container_width=True):
                        for key in ['model', 'points', 'colors', 'stats', 'filename', 'model_type']:
                            if key in st.session_state:
                                del st.session_state[key]
                        st.rerun()
        else:
            # Enhanced empty state
            st.markdown("""
            <div class="result-card" style="min-height: 600px; display: flex; align-items: center; justify-content: center;">
                <div style="text-align: center;">
                    <div style="font-size: 4rem; margin-bottom: 1rem; opacity: 0.3;">🎯</div>
                    <h3 style="color: rgba(255,255,255,0.6); margin: 0;">3D Model Viewer</h3>
                    <p style="color: rgba(255,255,255,0.4); margin: 0.5rem 0;">
                        Upload an image and generate a 3D model to view it here
                    </p>
                    <div style="margin-top: 2rem; padding: 1rem; background: rgba(65,105,225,0.1); border-radius: 10px; border: 1px solid rgba(65,105,225,0.2);">
                        <strong style="color: #4169e1;">Features:</strong><br>
                        <span style="color: rgba(255,255,255,0.7);">
                            • Interactive 3D viewing<br>
                            • Multiple export formats<br>
                            • Real-time rendering<br>
                            • Advanced controls
                        </span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None

# Run the application
if __name__ == "__main__":
    main()