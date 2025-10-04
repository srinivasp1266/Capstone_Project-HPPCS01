#!/usr/bin/env python3
"""
HAAI++ Capstone Project - CV Creation using Multi-LLM Architecture
Project ID: HPPCS[01]
Author: Srinivas P
Date: October 2025

This is the main entry point for the CV Builder application that uses
multiple Large Language Models for enhanced CV creation and tailoring.

MULTI-LLM ARCHITECTURE:
- Gemma 2B: Content Generation Specialist (fast, creative)
- Llama2 7B: Data Analysis Specialist (powerful, analytical)

Usage:
    python main.py [--host HOST] [--timeout TIMEOUT]
    
Arguments:
    --host: Ollama server host (default: http://localhost:11434)
    --timeout: Request timeout in seconds (default: 120)
"""

import streamlit as st
import os
import sys
import json
import logging
import argparse
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import asdict
import tempfile
import io

# All files are now in the same directory per HAAI++ requirements
try:
    from document_extractor import DocumentExtractor
    from llm_client import OllamaClient, LLMService
    from multi_llm_service import MultiLLMService  # New multi-LLM service
    from resume_parser import ResumeParser, ResumeData
    from job_parser import JobDescriptionParser
    from cv_generator import CVTailoringEngine
    from document_generator import CVDocumentGenerator
    from config import settings
except ImportError as e:
    st.error(f"Import error: {e}")
    st.error("Please make sure all dependencies are installed and all files are in the same directory.")
    st.stop()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="HAAI++ Multi-LLM CV Builder",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - Unique HAAI++ Design
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
    
    /* Global Styling - Unified Green Theme */
    .main {
        background: linear-gradient(135deg, #f0fdf4 0%, #ecfdf5 50%, #d1fae5 100%);
        min-height: 100vh;
    }
    
    .main-header {
        font-family: 'Inter', sans-serif;
        font-size: 3.5rem;
        font-weight: 800;
        text-align: center;
        margin-bottom: 2.5rem;
        /* Primary color fallback */
        color: #065f46 !important;
        text-shadow: 2px 2px 4px rgba(6, 95, 70, 0.3), 0 0 20px rgba(16, 185, 129, 0.4);
        padding: 3rem 0;
        position: relative;
        z-index: 10;
        /* Add a subtle background to ensure visibility */
        background: linear-gradient(135deg, rgba(240, 253, 244, 0.8) 0%, rgba(220, 252, 231, 0.8) 100%);
        border-radius: 20px;
        margin: 2rem auto;
        max-width: 90%;
        backdrop-filter: blur(10px);
        border: 2px solid rgba(16, 185, 129, 0.3);
        box-shadow: 0 8px 32px rgba(16, 185, 129, 0.2);
    }
    
    .main-header::after {
        content: '';
        position: absolute;
        bottom: 0;
        left: 50%;
        transform: translateX(-50%);
        width: 120px;
        height: 4px;
        background: linear-gradient(90deg, #10b981, #34d399, #6ee7b7);
        border-radius: 2px;
    }
    
    .section-header {
        font-family: 'Inter', sans-serif;
        font-size: 1.6rem;
        font-weight: 600;
        color: #065f46;
        margin: 2.5rem 0 1.5rem 0;
        padding: 1rem 1.5rem;
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.1) 0%, rgba(52, 211, 153, 0.1) 100%);
        border-radius: 12px;
        border-left: 6px solid #10b981;
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.15);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(16, 185, 129, 0.2);
    }
    
    .info-box {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.08) 0%, rgba(52, 211, 153, 0.08) 100%);
        padding: 2rem;
        border-radius: 16px;
        border: 1px solid rgba(16, 185, 129, 0.2);
        margin: 1.5rem 0;
        box-shadow: 0 8px 24px rgba(16, 185, 129, 0.1);
        backdrop-filter: blur(20px);
        color: #064e3b;
    }
    
    .success-box {
        background: linear-gradient(135deg, rgba(34, 197, 94, 0.1) 0%, rgba(16, 185, 129, 0.1) 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid rgba(34, 197, 94, 0.3);
        margin: 1.5rem 0;
        color: #065f46;
        box-shadow: 0 4px 12px rgba(34, 197, 94, 0.15);
    }
    
    .warning-box {
        background: linear-gradient(135deg, rgba(246, 173, 85, 0.1) 0%, rgba(245, 158, 11, 0.1) 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid rgba(246, 173, 85, 0.3);
        margin: 1.5rem 0;
        color: #744210;
        box-shadow: 0 4px 12px rgba(246, 173, 85, 0.15);
    }
    
    .metric-card {
        background: linear-gradient(135deg, rgba(240, 253, 244, 0.95) 0%, rgba(220, 252, 231, 0.95) 100%);
        padding: 1.5rem;
        border-radius: 16px;
        text-align: center;
        border: 1px solid rgba(16, 185, 129, 0.2);
        box-shadow: 0 8px 24px rgba(16, 185, 129, 0.12);
        backdrop-filter: blur(20px);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        color: #064e3b;
    }
    
    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 32px rgba(16, 185, 129, 0.25);
        border-color: #10b981;
    }
    
    /* Button Styling - Green Theme */
    .stButton > button {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-family: 'Inter', sans-serif;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.3);
        text-transform: none;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #059669 0%, #047857 100%);
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(16, 185, 129, 0.4);
    }
    
    /* Primary Button Variant - Green Theme */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #34d399 0%, #10b981 100%);
        box-shadow: 0 4px 12px rgba(52, 211, 153, 0.3);
    }
    
    .stButton > button[kind="primary"]:hover {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        box-shadow: 0 8px 20px rgba(52, 211, 153, 0.4);
    }
    
    /* Input Styling - Green Theme */
    .stSelectbox > div > div {
        background: linear-gradient(135deg, rgba(240, 253, 244, 0.9) 0%, rgba(220, 252, 231, 0.9) 100%);
        border-radius: 12px;
        border: 2px solid #86efac;
        backdrop-filter: blur(10px);
        color: #064e3b;
        transition: all 0.3s ease;
    }
    
    .stSelectbox > div > div:hover {
        border-color: #10b981;
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.15);
    }
    
    .stTextArea > div > div > textarea {
        background: linear-gradient(135deg, rgba(240, 253, 244, 0.9) 0%, rgba(220, 252, 231, 0.9) 100%);
        border-radius: 12px;
        border: 2px solid #86efac;
        backdrop-filter: blur(10px);
        color: #064e3b;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .stTextArea > div > div > textarea:focus {
        border-color: #10b981;
        box-shadow: 0 0 0 4px rgba(16, 185, 129, 0.2), 0 6px 20px rgba(16, 185, 129, 0.15);
        background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
    }
    
    .stFileUploader > div {
        background: linear-gradient(135deg, rgba(240, 253, 244, 0.9) 0%, rgba(220, 252, 231, 0.9) 100%);
        border-radius: 16px;
        border: 3px dashed #86efac;
        backdrop-filter: blur(10px);
        padding: 2rem;
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .stFileUploader > div:hover {
        border-color: #10b981;
        background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
        box-shadow: 0 8px 24px rgba(16, 185, 129, 0.2);
    }
    
    /* Progress Bar - Green Theme */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #10b981, #34d399, #6ee7b7);
        border-radius: 6px;
    }
    
    /* Text Input Fields - Green Theme */
    .stTextInput > div > div > input {
        background: linear-gradient(135deg, rgba(240, 253, 244, 0.9) 0%, rgba(220, 252, 231, 0.9) 100%);
        border: 2px solid #86efac;
        border-radius: 12px;
        color: #064e3b;
        font-size: 1rem;
        font-weight: 500;
        padding: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.1);
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #10b981;
        box-shadow: 0 0 0 4px rgba(16, 185, 129, 0.2), 0 6px 20px rgba(16, 185, 129, 0.15);
        background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
    }
    
    /* Number Input Fields - Green Theme */
    .stNumberInput > div > div > input {
        background: linear-gradient(135deg, rgba(240, 253, 244, 0.9) 0%, rgba(220, 252, 231, 0.9) 100%);
        border: 2px solid #86efac;
        border-radius: 12px;
        color: #064e3b;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .stNumberInput > div > div > input:focus {
        border-color: #10b981;
        box-shadow: 0 0 0 4px rgba(16, 185, 129, 0.2);
        background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
    }
    
    /* Multiselect and Tags - Green Theme */
    .stMultiSelect > div > div {
        background: linear-gradient(135deg, rgba(240, 253, 244, 0.9) 0%, rgba(220, 252, 231, 0.9) 100%);
        border: 2px solid #86efac;
        border-radius: 12px;
        color: #064e3b;
    }
    
    .stMultiSelect > div > div:focus-within {
        border-color: #10b981;
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.15);
    }
    
    /* Checkbox and Radio - Green Theme */
    .stCheckbox > label > div:first-child {
        background-color: rgba(240, 253, 244, 0.9);
        border: 2px solid #86efac;
        border-radius: 6px;
    }
    
    .stCheckbox > label > div:first-child:hover {
        border-color: #10b981;
    }
    
    .stRadio > div > label > div:first-child {
        background-color: rgba(240, 253, 244, 0.9);
        border: 2px solid #86efac;
    }
    
    .stRadio > div > label > div:first-child:hover {
        border-color: #10b981;
    }
    
    /* Date Input - Green Theme */
    .stDateInput > div > div > input {
        background: linear-gradient(135deg, rgba(240, 253, 244, 0.9) 0%, rgba(220, 252, 231, 0.9) 100%);
        border: 2px solid #86efac;
        border-radius: 12px;
        color: #064e3b;
        font-weight: 500;
    }
    
    .stDateInput > div > div > input:focus {
        border-color: #10b981;
        box-shadow: 0 0 0 4px rgba(16, 185, 129, 0.2);
    }
    
    /* Hide Streamlit default deploy and rerun buttons */
    .stToolbar {
        display: none !important;
    }
    
    /* Hide deploy button specifically */
    button[title="Deploy this app"],
    button[data-testid="stToolbarActionButton"],
    .stActionButton,
    [data-testid="stDeployButton"],
    [data-testid="stToolbar"] {
        display: none !important;
    }
    
    /* Hide the entire header toolbar area */
    .stAppHeader,
    header[data-testid="stHeader"],
    .css-18ni7ap,
    .css-vk3wp9 {
        display: none !important;
    }
    
    /* Hide rerun button and deploy options */
    .stApp > header,
    .stApp > div:first-child > div:first-child {
        display: none !important;
    }
    
    /* Alternative selectors for deploy button */
    button[kind="secondary"][title*="Deploy"],
    button[kind="secondary"][title*="deploy"],
    .element-container button[title*="Deploy"] {
        display: none !important;
    }
    
    /* Sidebar Styling - Force Right Side with Green Theme */
    .css-1d391kg, [data-testid="stSidebar"], .css-1cypcdb, .css-17eq0hr, 
    section[data-testid="stSidebar"], .css-1lcbmhc, .css-6qob1r {
        background: linear-gradient(180deg, rgba(34, 197, 94, 0.15) 0%, rgba(16, 185, 129, 0.15) 100%);
        backdrop-filter: blur(20px);
        border-left: 3px solid #10b981;
        border-right: none !important;
        box-shadow: -4px 0 12px rgba(16, 185, 129, 0.1);
    }
    
    /* Force sidebar to right side using comprehensive selectors */
    section[data-testid="stSidebar"], 
    [data-testid="stSidebar"],
    .css-1d391kg,
    .css-1lcbmhc,
    .css-6qob1r,
    .css-17eq0hr {
        position: fixed !important;
        right: 0 !important;
        left: unset !important;
        top: 0 !important;
        height: 100vh !important;
        z-index: 999999 !important;
        transform: translateX(0) !important;
        width: 21rem !important;
    }
    
    /* Additional sidebar positioning */
    .stSidebar, .css-1cypcdb {
        position: fixed !important;
        right: 0 !important;
        left: unset !important;
        top: 0 !important;
        height: 100vh !important;
        z-index: 999999 !important;
    }
    
    /* Adjust main app content - Right sidebar layout */
    .main, .stApp > div {
        margin-right: 22rem !important;
        margin-left: 1rem !important;
        max-width: none !important;
        transition: margin 0.3s ease;
    }
    
    /* Adjust main content container - Optimized for right sidebar */
    .main .block-container, [data-testid="stMainBlockContainer"] {
        padding-right: 2rem !important;
        padding-left: 2rem !important;
        max-width: 1200px !important;
        margin: 0 auto !important;
    }
    
    /* Sidebar content styling */
    [data-testid="stSidebar"] .css-17eq0hr,
    [data-testid="stSidebar"] .element-container {
        background: rgba(34, 197, 94, 0.05);
        border-radius: 8px;
        margin: 0.5rem 0;
        padding: 0.5rem;
    }
    
    /* Sidebar headers */
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3 {
        color: #065f46 !important;
        border-bottom: 2px solid #10b981;
        padding-bottom: 0.5rem;
    }
    
    /* Sidebar buttons */
    [data-testid="stSidebar"] .stButton > button {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        border: none;
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(16, 185, 129, 0.3);
    }
    
    [data-testid="stSidebar"] .stButton > button:hover {
        background: linear-gradient(135deg, #059669 0%, #047857 100%);
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.4);
    }
    
    /* Sidebar content styling */
    .css-1d391kg .css-17eq0hr {
        background: rgba(34, 197, 94, 0.05);
        border-radius: 8px;
        margin: 0.5rem 0;
        padding: 0.5rem;
    }
    
    /* Sidebar headers */
    .css-1d391kg h1, .css-1d391kg h2, .css-1d391kg h3 {
        color: #065f46;
        border-bottom: 2px solid #10b981;
        padding-bottom: 0.5rem;
    }
    
    /* Metrics Styling */
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.95) 0%, rgba(248, 250, 252, 0.95) 100%);
        border-radius: 12px;
        padding: 1rem;
        border: 1px solid rgba(102, 126, 234, 0.1);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.1);
        backdrop-filter: blur(20px);
    }
    
    /* Expander Styling */
    .streamlit-expanderHeader {
        background: rgba(255, 255, 255, 0.9);
        border-radius: 8px;
        backdrop-filter: blur(10px);
    }
    
    /* Code Block Styling */
    .stCode {
        background: rgba(45, 55, 72, 0.95);
        border-radius: 8px;
        font-family: 'JetBrains Mono', monospace;
    }
    
    /* Custom Animations */
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .section-header {
        animation: fadeInUp 0.6s ease-out;
    }
    
    /* Custom Badge */
    .haai-badge {
        display: inline-block;
        background: linear-gradient(45deg, #667eea, #764ba2);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-size: 0.9rem;
        font-weight: 600;
        margin: 0.5rem;
        box-shadow: 0 2px 8px rgba(102, 126, 234, 0.3);
    }
    
    /* Custom Status Indicators */
    .status-active {
        color: #48bb78;
        font-weight: 600;
    }
    
    .status-pending {
        color: #ed8936;
        font-weight: 600;
    }
    
    .status-complete {
        color: #667eea;
        font-weight: 600;
    }
</style>

<script>
    // Enhanced sidebar positioning with multiple selectors and robust approach
    function moveSidebarToRight() {
        // Multiple selectors to catch all possible sidebar variations
        const selectors = [
            '[data-testid="stSidebar"]',
            '.css-1d391kg',
            '.css-1lcbmhc', 
            '.css-6qob1r',
            '.css-17eq0hr',
            '.css-1cypcdb',
            'section[data-testid="stSidebar"]',
            '.stSidebar'
        ];
        
        let sidebarFound = false;
        
        selectors.forEach(selector => {
            const elements = document.querySelectorAll(selector);
            elements.forEach(sidebar => {
                if (sidebar && !sidebarFound) {
                    // Apply comprehensive right-side positioning
                    sidebar.style.setProperty('position', 'fixed', 'important');
                    sidebar.style.setProperty('right', '0', 'important');
                    sidebar.style.setProperty('left', 'unset', 'important');
                    sidebar.style.setProperty('top', '0', 'important');
                    sidebar.style.setProperty('height', '100vh', 'important');
                    sidebar.style.setProperty('z-index', '999999', 'important');
                    sidebar.style.setProperty('width', '21rem', 'important');
                    sidebar.style.setProperty('transform', 'translateX(0)', 'important');
                    
                    sidebarFound = true;
                    console.log('Sidebar moved to right using selector:', selector);
                }
            });
        });
        
        // Center main content with proper spacing for right sidebar
        const mainSelectors = ['.main', '.stApp > div', '[data-testid="stAppViewContainer"]'];
        mainSelectors.forEach(selector => {
            const elements = document.querySelectorAll(selector);
            elements.forEach(main => {
                if (main) {
                    main.style.setProperty('margin-right', '22rem', 'important');
                    main.style.setProperty('margin-left', '1rem', 'important');
                    main.style.setProperty('max-width', 'none', 'important');
                }
            });
        });
        
        // Center main block container
        const containerSelectors = ['[data-testid="stMainBlockContainer"]', '.main .block-container'];
        containerSelectors.forEach(selector => {
            const elements = document.querySelectorAll(selector);
            elements.forEach(container => {
                if (container) {
                    container.style.setProperty('max-width', '1200px', 'important');
                    container.style.setProperty('margin', '0 auto', 'important');
                    container.style.setProperty('padding-left', '2rem', 'important');
                    container.style.setProperty('padding-right', '2rem', 'important');
                }
            });
        });
        
        return sidebarFound;
    }
    
    // Enhanced monitoring with mutation observer
    function setupSidebarMonitoring() {
        // Initial attempts
        moveSidebarToRight();
        
        // Mutation observer to catch dynamic changes
        const observer = new MutationObserver((mutations) => {
            let shouldReposition = false;
            
            mutations.forEach((mutation) => {
                if (mutation.type === 'childList' || mutation.type === 'attributes') {
                    const addedNodes = Array.from(mutation.addedNodes);
                    const hasRelevantChanges = addedNodes.some(node => 
                        node.nodeType === Node.ELEMENT_NODE && 
                        (node.querySelector && (
                            node.querySelector('[data-testid="stSidebar"]') ||
                            node.matches && node.matches('[data-testid="stSidebar"]')
                        ))
                    );
                    
                    if (hasRelevantChanges) {
                        shouldReposition = true;
                    }
                }
            });
            
            if (shouldReposition) {
                setTimeout(moveSidebarToRight, 50);
            }
        });
        
        // Observe the entire document for changes
        observer.observe(document.body, {
            childList: true,
            subtree: true,
            attributes: true,
            attributeFilter: ['class', 'style']
        });
        
        // Periodic checks as fallback
        setInterval(moveSidebarToRight, 2000);
    }
    
    // Run setup when page loads and on updates
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', setupSidebarMonitoring);
    } else {
        setupSidebarMonitoring();
    }
    
    // Function to hide deploy and rerun buttons
    function hideDeployButtons() {
        const deploySelectors = [
            'button[title="Deploy this app"]',
            'button[data-testid="stToolbarActionButton"]',
            '[data-testid="stDeployButton"]',
            '[data-testid="stToolbar"]',
            'button[title*="Deploy"]',
            'button[title*="deploy"]',
            '.stToolbar',
            '.stActionButton',
            '.stAppHeader',
            'header[data-testid="stHeader"]',
            '.css-18ni7ap',
            '.css-vk3wp9',
            '.stApp > header'
        ];
        
        deploySelectors.forEach(selector => {
            const elements = document.querySelectorAll(selector);
            elements.forEach(element => {
                if (element) {
                    element.style.setProperty('display', 'none', 'important');
                    element.style.setProperty('visibility', 'hidden', 'important');
                    element.remove(); // Completely remove the element
                }
            });
        });
    }
    
    // Hide deploy buttons on load and periodically
    hideDeployButtons();
    setInterval(hideDeployButtons, 1000);
    
    // Also hide on mutations
    const deployObserver = new MutationObserver(() => {
        hideDeployButtons();
    });
    
    deployObserver.observe(document.body, {
        childList: true,
        subtree: true
    });
    
    // Additional timeout-based attempts for robustness
    setTimeout(moveSidebarToRight, 100);
    setTimeout(moveSidebarToRight, 500);
    setTimeout(moveSidebarToRight, 1000);
    setTimeout(moveSidebarToRight, 2000);
</script>
""", unsafe_allow_html=True)


def init_session_state():
    """Initialize session state variables."""
    if 'llm_client' not in st.session_state:
        st.session_state.llm_client = None
    if 'resume_data' not in st.session_state:
        st.session_state.resume_data = None
    if 'job_requirements' not in st.session_state:
        st.session_state.job_requirements = None
    if 'tailored_cv' not in st.session_state:
        st.session_state.tailored_cv = None
    if 'generated_document' not in st.session_state:
        st.session_state.generated_document = None
    if 'ollama_host' not in st.session_state:
        st.session_state.ollama_host = settings.ollama_host

# Initialize session state immediately when module is loaded
init_session_state()


def initialize_services():
    """Initialize Multi-LLM and other services."""
    if st.session_state.llm_client is None:
        try:
            with st.spinner("Initializing AI services with dual-model architecture..."):
                # Initialize Multi-LLM service for HAAI++ capstone requirements
                st.session_state.llm_client = MultiLLMService(
                    host=settings.ollama_host,
                    timeout=settings.ollama_timeout
                )
            st.success("✅ Multi-LLM AI services initialized successfully!")
            st.markdown('''
                <div class="info-box">
                    <strong>🎓 HAAI++ Neural Architecture Active</strong><br/>
                    <span class="status-active">●</span> Gemma 2B: Content Generation Specialist<br/>
                    <span class="status-active">●</span> Llama2 7B: Data Analysis Specialist<br/>
                    <span class="status-complete">●</span> Multi-model coordination enabled
                </div>
            ''', unsafe_allow_html=True)
        except Exception as e:
            st.error(f"❌ Failed to initialize AI services: {str(e)}")
            st.error("Please make sure Ollama is running and both models (gemma:2b, llama2:7b) are available.")
            return False
    return True


def generate_output_filename(base_name: str, template_style: str, file_extension: str) -> str:
    """Generate output filename including original input filename for easy identification."""
    # Get original filename if available
    original_filename = getattr(st.session_state, 'original_filename', None)
    
    # Clean the original filename (remove extension and special chars)
    if original_filename:
        # Extract name without extension and clean it
        clean_name = os.path.splitext(original_filename)[0]
        clean_name = re.sub(r'[^\w\-_]', '_', clean_name)  # Replace special chars with underscore
        clean_name = re.sub(r'_+', '_', clean_name)  # Remove multiple underscores
        clean_name = clean_name.strip('_')  # Remove leading/trailing underscores
    else:
        # Fallback to generic name if no original filename
        clean_name = "resume"
    
    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create filename: original_name_templatestyle_timestamp.extension
    filename = f"{clean_name}_{base_name}_{template_style}_{timestamp}.{file_extension}"
    
    return filename


def handle_resume_upload():
    """Handle resume file upload and processing."""
    st.markdown('''
        <div class="section-header">
            � Neural Resume Analysis
            <div style="font-size: 0.9rem; font-weight: 400; opacity: 0.7; margin-top: 0.3rem;">
                Upload your resume for AI-powered content extraction
            </div>
        </div>
    ''', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Choose your resume file",
        type=['pdf', 'docx', 'txt'],
        help="Upload your existing resume in PDF, DOCX, or TXT format"
    )
    
    if uploaded_file is not None:
        with st.spinner("Processing your resume..."):
            try:
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_file_path = tmp_file.name
                
                # Extract text from document
                extractor = DocumentExtractor()
                resume_text = extractor.extract_text(tmp_file_path)
                
                # Clean up temporary file
                os.unlink(tmp_file_path)
                
                if not resume_text.strip():
                    st.error("❌ Could not extract text from the uploaded file. Please try a different format.")
                    return
                
                # Parse resume using Multi-LLM service
                parser = ResumeParser(st.session_state.llm_client)
                
                # Debug: Show raw text length
                st.info(f"📄 Extracted {len(resume_text)} characters from resume")
                
                # Parse the resume with enhanced error handling
                try:
                    with st.spinner("🤖 Processing resume with AI models..."):
                        st.session_state.resume_data = parser.parse_resume(resume_text)
                        # Store original filename for output file naming
                        st.session_state.original_filename = uploaded_file.name
                    
                    # Check what was parsed
                    if st.session_state.resume_data:
                        logger.info(f"Resume parsing completed successfully")
                        st.success("✅ Resume processed successfully!")
                        
                        # Show parsing method used
                        if hasattr(st.session_state.resume_data, '_parsing_method'):
                            method = getattr(st.session_state.resume_data, '_parsing_method', 'unknown')
                            if method == 'basic':
                                st.info("ℹ️ Used basic text extraction (LLM unavailable)")
                            elif method == 'fallback':
                                st.warning("⚠️ Used fallback parsing due to LLM timeout")
                            else:
                                st.success("🎯 Used advanced AI parsing")
                    else:
                        st.warning("⚠️ Resume processed but limited data extracted")
                        # Create a minimal resume data structure
                        from resume_parser import ResumeData, PersonalInfo
                        st.session_state.resume_data = ResumeData()
                        st.session_state.resume_data.personal_info = PersonalInfo()
                        
                except ConnectionError as conn_error:
                    st.error("🔌 **Connection Error**: Cannot reach AI services")
                    st.info("💡 **Tip**: Make sure Ollama is running with: `ollama serve`")
                    st.info("📋 **Fallback**: Using basic text extraction instead")
                    
                    # Use basic extraction as fallback
                    from llm_client import LLMService
                    basic_service = LLMService(None)  # No client = basic extraction only
                    st.session_state.resume_data = basic_service._extract_resume_data_basic(resume_text)
                    
                except TimeoutError as timeout_error:
                    st.error("⏰ **Timeout Error**: AI processing took too long")
                    st.info("💡 **Tip**: Try with a shorter resume or restart Ollama")
                    st.info("📋 **Fallback**: Using basic text extraction instead")
                    
                    # Use basic extraction as fallback
                    from llm_client import LLMService
                    basic_service = LLMService(None)
                    st.session_state.resume_data = basic_service._extract_resume_data_basic(resume_text)
                    
                except Exception as parse_error:
                    st.error(f"❌ **Parsing Error**: {str(parse_error)}")
                    logger.error(f"Resume parsing error: {str(parse_error)}")
                    
                    # Try basic extraction as last resort
                    try:
                        st.info("📋 **Attempting basic extraction...**")
                        from llm_client import LLMService
                        basic_service = LLMService(None)
                        basic_data = basic_service._extract_resume_data_basic(resume_text)
                        
                        # Convert to proper format
                        from resume_parser import ResumeData, PersonalInfo
                        st.session_state.resume_data = ResumeData()
                        st.session_state.resume_data.personal_info = PersonalInfo(
                            name=basic_data.get('name', ''),
                            email=basic_data.get('email', ''),
                            phone=basic_data.get('phone', ''),
                            address=basic_data.get('address', ''),
                            linkedin=basic_data.get('linkedin', ''),
                            website=basic_data.get('website', ''),
                        )
                        st.success("✅ Basic extraction completed!")
                        
                    except Exception as fallback_error:
                        st.error(f"❌ All extraction methods failed: {str(fallback_error)}")
                        from resume_parser import ResumeData, PersonalInfo
                        st.session_state.resume_data = ResumeData()
                        st.session_state.resume_data.personal_info = PersonalInfo()
                
                # Display extracted information
                with st.expander("📋 Extracted Resume Information", expanded=True):
                    if st.session_state.resume_data and st.session_state.resume_data.personal_info:
                        st.subheader("👤 Personal Information")
                        info = st.session_state.resume_data.personal_info
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write(f"**Name:** {info.name or 'Not found'}")
                            st.write(f"**Email:** {info.email or 'Not found'}")
                            st.write(f"**Phone:** {info.phone or 'Not found'}")
                        with col2:
                            st.write(f"**Location:** {info.address or 'Not found'}")
                            st.write(f"**LinkedIn:** {info.linkedin or 'Not found'}")
                            st.write(f"**Website:** {info.website or 'Not found'}")
                    
                    if st.session_state.resume_data and st.session_state.resume_data.experience:
                        st.subheader("💼 Work Experience")
                        for i, exp in enumerate(st.session_state.resume_data.experience):
                            st.write(f"**{exp.position or 'Position'}** at {exp.company or 'Company'} ({exp.start_date or 'Start'} - {exp.end_date or 'End'})")
                    else:
                        st.subheader("💼 Work Experience")
                        st.write("No work experience found in the resume")
                    
                    if st.session_state.resume_data and st.session_state.resume_data.skills:
                        st.subheader("🛠️ Skills")
                        skills = st.session_state.resume_data.skills
                        if isinstance(skills, dict):
                            for category, skill_list in skills.items():
                                if skill_list:
                                    st.write(f"**{category.title()}:** {', '.join(skill_list)}")
                        elif isinstance(skills, list):
                            st.write(f"**Skills:** {', '.join(skills)}")
                        else:
                            st.write(f"**Skills:** {skills}")
                    else:
                        st.subheader("🛠️ Skills")
                        st.write("No skills found in the resume")
                
                # Add manual continue button
                st.markdown("---")
                if st.button("➡️ Continue to Job Description", type="primary", key="continue_to_job"):
                    st.rerun()
                            
            except Exception as e:
                st.error(f"❌ Error processing resume: {str(e)}")
                logger.error(f"Resume processing error: {str(e)}")


def handle_job_description():
    """Handle job description input and processing."""
    st.markdown('''
        <div class="section-header">
            🎯 Smart Job Requirements Analysis
            <div style="font-size: 0.9rem; font-weight: 400; opacity: 0.7; margin-top: 0.3rem;">
                Input job description for intelligent keyword extraction
            </div>
        </div>
    ''', unsafe_allow_html=True)
    
    job_text = st.text_area(
        "Paste the job description here:",
        height=200,
        placeholder="Copy and paste the job description, requirements, and qualifications here..."
    )
    
    if st.button("🔍 Analyze Job Requirements", disabled=not job_text.strip()):
        with st.spinner("Analyzing job requirements..."):
            try:
                # Parse job description using Multi-LLM service
                parser = JobDescriptionParser(st.session_state.llm_client)
                st.session_state.job_requirements = parser.parse_job_description(job_text)
                
                st.success("✅ Job requirements analyzed successfully!")
                
                # Display extracted job information
                with st.expander("📊 Extracted Job Requirements", expanded=True):
                    reqs = st.session_state.job_requirements
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("📌 Basic Information")
                        st.write(f"**Title:** {reqs.title}")
                        st.write(f"**Company:** {reqs.company}")
                        st.write(f"**Experience Level:** {reqs.experience_level}")
                        
                        if reqs.required_skills:
                            st.subheader("🎯 Required Skills")
                            for skill in reqs.required_skills[:10]:  # Show top 10
                                st.write(f"• {skill}")
                    
                    with col2:
                        if reqs.responsibilities:
                            st.subheader("📝 Key Responsibilities")
                            for resp in reqs.responsibilities[:5]:  # Show top 5
                                st.write(f"• {resp}")
                        
                        if reqs.keywords:
                            st.subheader("🔑 Important Keywords")
                            keywords_text = ", ".join(reqs.keywords[:15])  # Show top 15
                            st.write(keywords_text)
                
                # Add manual continue button after successful analysis
                st.markdown("---")
                if st.button("➡️ Continue to AI Tailoring", type="primary", key="continue_to_ai"):
                    st.rerun()
                
            except Exception as e:
                st.error(f"❌ Error analyzing job description: {str(e)}")
                logger.error(f"Job analysis error: {str(e)}")
                
                # Even on error, allow manual continuation with basic job data
                st.warning("⚠️ Job analysis failed, but you can still proceed with basic information")
                if st.button("➡️ Continue Anyway", type="secondary", key="continue_anyway"):
                    # Create minimal job requirements
                    from job_parser import JobRequirements
                    st.session_state.job_requirements = JobRequirements(
                        title="Software Developer",
                        company="Unknown Company",
                        description=job_text,
                        required_skills=[],
                        responsibilities=[],
                        keywords=[]
                    )
                    st.rerun()


def handle_cv_tailoring():
    """Handle CV tailoring process."""
    st.markdown('''
        <div class="section-header">
            � Neural CV Optimization Engine
            <div style="font-size: 0.9rem; font-weight: 400; opacity: 0.7; margin-top: 0.3rem;">
                Advanced AI tailoring using dual-model architecture
            </div>
        </div>
    ''', unsafe_allow_html=True)
    
    if not st.session_state.resume_data:
        st.warning("⚠️ Please upload and process your resume first.")
        return
    
    if not st.session_state.job_requirements:
        st.warning("⚠️ Please analyze the job description first.")
        return
    
    # Tailoring options
    st.markdown('''
        <div style="margin: 2rem 0;">
            <h3 style="color: #2d3748; font-family: 'Inter', sans-serif; margin-bottom: 1rem;">
                🎛️ Neural Optimization Settings
            </h3>
        </div>
    ''', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        optimize_summary = st.checkbox("🧠 Neural Summary Enhancement", value=True)
        highlight_skills = st.checkbox("⚡ Smart Skill Prioritization", value=True)
    with col2:
        optimize_projects = st.checkbox("� Project Intelligence Boost", value=True)
        add_achievements = st.checkbox("� Achievement Amplification", value=True)
    
    if st.button("🚀 Activate Neural Tailoring", type="primary", disabled=st.session_state.tailored_cv is not None):
        with st.spinner("🧠 Neural networks analyzing and optimizing your CV... This may take a few minutes."):
            try:
                # Create tailoring engine with Multi-LLM service
                engine = CVTailoringEngine(st.session_state.llm_client)
                
                # Progress tracking
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                status_text.markdown('<p class="status-active">🔍 Neural analysis in progress...</p>', unsafe_allow_html=True)
                progress_bar.progress(20)
                
                # Tailor the CV
                tailored_cv = engine.tailor_cv(
                    st.session_state.resume_data,
                    st.session_state.job_requirements
                )
                
                progress_bar.progress(60)
                status_text.markdown('<p class="status-active">✨ Content optimization active...</p>', unsafe_allow_html=True)
                
                # Store result
                st.session_state.tailored_cv = tailored_cv
                
                progress_bar.progress(100)
                status_text.markdown('<p class="status-complete">✅ Neural optimization complete!</p>', unsafe_allow_html=True)
                
                st.markdown('''
                    <div class="success-box">
                        🎉 <strong>Neural CV Optimization Complete!</strong><br/>
                        Your resume has been intelligently tailored using advanced AI models.
                    </div>
                ''', unsafe_allow_html=True)
                
                # Show tailoring score
                score = tailored_cv.tailoring_score
                st.metric(
                    "🎯 Neural Compatibility Score", 
                    f"{score:.1f}%",
                    delta=f"{score-75:.1f}%" if score > 75 else None
                )
                
                # Add continue button after successful tailoring
                st.markdown("---")
                if st.button("➡️ Continue to Document Generation", type="primary", key="continue_to_doc"):
                    st.rerun()
                
            except Exception as e:
                error_msg = str(e)
                st.error(f"❌ Error during CV tailoring: {error_msg}")
                
                # Provide specific troubleshooting tips
                if "timeout" in error_msg.lower() or "connection" in error_msg.lower():
                    st.info("💡 **Tip**: Connection issue with the AI service. Please check that Ollama is running (`ollama serve`) and try again.")
                elif "JSON parse failed" in error_msg or "Extra data" in error_msg:
                    st.info("💡 **Tip**: The AI model's response couldn't be parsed properly. This often happens with complex resumes. Try simplifying your resume or job description.")
                elif "Connection" in error_msg or "timeout" in error_msg.lower():
                    st.info("💡 **Tip**: Connection issue with the AI service. Please check that Ollama is running (`ollama serve`) and try again.")
                else:
                    st.info("💡 **Tip**: An unexpected error occurred. Please check the logs for more details and try again.")
                
                # Provide debugging information if available
                with st.expander("🔧 Debug Information (for developers)"):
                    st.code(f"""
Error Type: {type(e).__name__}
Error Message: {error_msg}
                    
Please check the application logs for more detailed information.
                    """, language="text")
    
    # Display current results if available
    if st.session_state.tailored_cv:
        st.markdown("### 📈 Tailoring Results")
        display_tailoring_results(st.session_state.tailored_cv)


def display_tailoring_results(tailored_cv):
    """Display tailoring results."""
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("🎯 Match Score", f"{tailored_cv.tailoring_score:.1f}%")
    
    with col2:
        skills_count = sum(len(skills) for skills in tailored_cv.skills.values()) if tailored_cv.skills else 0
        st.metric("🛠️ Skills", skills_count)
    
    with col3:
        exp_count = len(tailored_cv.experience) if tailored_cv.experience else 0
        st.metric("💼 Experiences", exp_count)
    
    with col4:
        proj_count = len(tailored_cv.projects) if tailored_cv.projects else 0
        st.metric("🚀 Projects", proj_count)
    
    # Show optimized content
    with st.expander("📝 Optimized Professional Summary", expanded=True):
        st.write(tailored_cv.summary)
    
    if tailored_cv.optimization_notes:
        with st.expander("💡 Optimization Notes"):
            for note in tailored_cv.optimization_notes:
                st.info(note)
    
    # Add continue button for already completed tailoring
    st.markdown("---")
    if st.button("➡️ Continue to Document Generation", type="primary", key="continue_from_results"):
        st.rerun()


def handle_document_generation():
    """Handle final document generation."""
    st.markdown('''
        <div class="section-header">
            � Professional Document Synthesis
            <div style="font-size: 0.9rem; font-weight: 400; opacity: 0.7; margin-top: 0.3rem;">
                Generate polished CV documents in multiple formats
            </div>
        </div>
    ''', unsafe_allow_html=True)
    
    if not st.session_state.tailored_cv:
        st.warning("⚠️ Please complete the CV tailoring step first.")
        return
    
    # Check PDF availability
    try:
        from reportlab.pdfgen import canvas
        from reportlab.lib.pagesizes import letter, A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
        pdf_available = True
    except ImportError:
        pdf_available = False
    
    # Document format selection
    st.markdown('''
        <div style="margin: 2rem 0;">
            <h3 style="color: #2d3748; font-family: 'Inter', sans-serif; margin-bottom: 1rem;">
                🎨 Output Configuration
            </h3>
        </div>
    ''', unsafe_allow_html=True)
    
    # Format selection: HTML always available, PDF conditionally
    format_options = ["HTML"]
    if pdf_available:
        format_options.append("PDF")
    else:
        st.markdown('''
            <div class="warning-box">
                ⚠️ PDF generation is not available. Please install reportlab: <code>pip install reportlab</code>
            </div>
        ''', unsafe_allow_html=True)
    
    format_type = st.selectbox(
        "📄 Choose Output Format:",
        format_options,
        index=0,
        help="Select the output format for your CV"
    )
    
    if format_type == "HTML":
        st.markdown('''
            <div class="info-box">
                🌐 <strong>HTML Format</strong><br/>
                Interactive web-based resume with modern responsive styling
            </div>
        ''', unsafe_allow_html=True)
    else:
        st.markdown('''
            <div class="info-box">
                📄 <strong>PDF Format</strong><br/>
                Print-ready document with professional formatting and layout
            </div>
        ''', unsafe_allow_html=True)
    
    # Template selection
    template_style = st.selectbox(
        "🎨 Choose Template Style:",
        ["modern", "professional", "clean"],
        index=0,
        help="Select the visual style for your CV"
    )
    
    if st.button("📄 Generate CV Document", type="primary"):
        with st.spinner(f"Generating your professional CV in {format_type} format..."):
            try:
                # Generate document
                generator = CVDocumentGenerator()
                
                # Convert TailoredCV object to dictionary for Jinja2 template
                cv_dict = asdict(st.session_state.tailored_cv)
                
                if format_type == "HTML":
                    document_content = generator.generate_html(
                        cv_dict,
                        template_name="cv_template.html"
                    )
                    
                    # Generate filename with original input name
                    html_filename = generate_output_filename("tailored_cv", template_style, "html")
                    
                    # Save to current directory
                    output_path = Path(".") / html_filename
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    try:
                        with open(output_path, 'w', encoding='utf-8') as f:
                            f.write(document_content)
                        
                        # Verify the file was created
                        if not output_path.exists():
                            st.error(f"❌ Failed to save HTML file to {output_path}")
                    except Exception as e:
                        st.error(f"❌ Error saving HTML file: {str(e)}")
                        logger.error(f"Error saving HTML file: {str(e)}")
                    
                    # Store the generated document
                    st.session_state.generated_document = {
                        'html_content': document_content,
                        'html_filename': html_filename,
                        'output_path': str(output_path),
                        'format': 'HTML'
                    }
                    
                    # Preview
                    with st.expander("👁️ Document Preview", expanded=True):
                        st.components.v1.html(document_content, height=600, scrolling=True)
                    
                    # Download button for HTML
                    st.download_button(
                        label="⬇️ Download HTML CV",
                        data=document_content,
                        file_name=st.session_state.generated_document['html_filename'],
                        mime='text/html',
                        help="Click to download your tailored CV as HTML"
                    )
                
                elif format_type == "PDF":
                    # Generate both HTML (for preview) and PDF
                    html_content = generator.generate_html(
                        cv_dict,
                        template_name="cv_template.html"
                    )
                    
                    pdf_bytes = generator.generate_pdf_from_html(
                        cv_dict,
                        template_name="cv_template.html"
                    )
                    
                    # Generate filenames with original input name
                    html_filename = generate_output_filename("tailored_cv", template_style, "html")
                    pdf_filename = generate_output_filename("tailored_cv", template_style, "pdf")
                    
                    # Save both files to current directory
                    output_dir = Path(".")
                    output_dir.mkdir(parents=True, exist_ok=True)
                    
                    html_output_path = output_dir / html_filename
                    pdf_output_path = output_dir / pdf_filename
                    
                    # Save HTML file
                    try:
                        with open(html_output_path, 'w', encoding='utf-8') as f:
                            f.write(html_content)
                        
                        # Verify the file was created
                        if not html_output_path.exists():
                            st.error(f"❌ Failed to save HTML file to {html_output_path}")
                    except Exception as e:
                        st.error(f"❌ Error saving HTML file: {str(e)}")
                        logger.error(f"Error saving HTML file: {str(e)}")
                    
                    # Save PDF file
                    try:
                        with open(pdf_output_path, 'wb') as f:
                            f.write(pdf_bytes)
                        
                        # Verify the file was created
                        if not pdf_output_path.exists():
                            st.error(f"❌ Failed to save PDF file to {pdf_output_path}")
                    except Exception as e:
                        st.error(f"❌ Error saving PDF file: {str(e)}")
                        logger.error(f"Error saving PDF file: {str(e)}")
                    
                    # Store the generated documents
                    st.session_state.generated_document = {
                        'html_content': html_content,
                        'pdf_content': pdf_bytes,
                        'html_filename': html_filename,
                        'pdf_filename': pdf_filename,
                        'html_output_path': str(html_output_path),
                        'pdf_output_path': str(pdf_output_path),
                        'format': 'PDF'
                    }
                    
                    st.success("✅ PDF CV document with HTML preview generated successfully!")
                    st.info(f"📁 Both files saved to current directory")
                    
                    # Show files in output directory
                    try:
                        output_files = list(Path(".").glob("*.html")) + list(Path(".").glob("*.pdf"))
                        if output_files:
                            st.markdown("**Recent files in output directory:**")
                            for file_path in sorted(output_files, key=lambda x: x.stat().st_mtime, reverse=True)[:5]:
                                file_size = file_path.stat().st_size
                                st.markdown(f"- {file_path.name} ({file_size:,} bytes)")
                    except Exception as e:
                        logger.error(f"Error listing output files: {str(e)}")
                    
                    # Preview (show HTML version)
                    with st.expander("👁️ Document Preview (HTML version)", expanded=True):
                        st.components.v1.html(html_content, height=600, scrolling=True)
                    
                    # Download buttons for both formats
                    col1, col2 = st.columns(2)
                    with col1:
                        st.download_button(
                            label="⬇️ Download PDF CV",
                            data=pdf_bytes,
                            file_name=st.session_state.generated_document['pdf_filename'],
                            mime='application/pdf',
                            help="Click to download your tailored CV as PDF"
                        )
                    with col2:
                        st.download_button(
                            label="⬇️ Download HTML CV",
                            data=html_content,
                            file_name=st.session_state.generated_document['html_filename'],
                            mime='text/html',
                            help="Click to download your tailored CV as HTML"
                        )
                
            except Exception as e:
                st.error(f"❌ Error generating document: {str(e)}")
                logger.error(f"Document generation error: {str(e)}")


def main():
    """Main application function."""
    # Handle command line arguments if run directly
    if len(sys.argv) > 1:
        parser = argparse.ArgumentParser(
            description="HAAI++ Capstone Project - Multi-LLM CV Builder"
        )
        parser.add_argument(
            "--host", 
            default="http://localhost:11434",
            help="Ollama server host (default: http://localhost:11434)"
        )
        parser.add_argument(
            "--timeout", 
            type=int, 
            default=120,
            help="Request timeout in seconds (default: 120)"
        )
        
        args = parser.parse_args()
        
        # Set environment variables for the application
        os.environ["OLLAMA_HOST"] = args.host
        os.environ["OLLAMA_TIMEOUT"] = str(args.timeout)
        
        print("=" * 60)
        print("🎓 HAAI++ Capstone Project - CV Builder")
        print("📋 Project ID: HPPCS[01]")
        print("🤖 Multi-LLM Architecture: Gemma 2B + Llama2 7B")
        print("=" * 60)
        print(f"🔧 Ollama Host: {args.host}")
        print(f"⏱️  Timeout: {args.timeout}s")
        print("🚀 Starting CV Builder application...")
        print("=" * 60)
    
    # Header
    st.markdown('''
        <div class="main-header">
            ⚡ CV Builder
            <div style="font-size: 1.2rem; font-weight: 1000; margin-top: 0.5rem; opacity: 0.8;">
                <span class="haai-badge">Multi-LLM</span>
                <span class="haai-badge">AI-Powered</span>
                <span class="haai-badge">Gemma + Llama2</span>
            </div>
        </div>
    ''', unsafe_allow_html=True)
    
    # Ensure session state is initialized
    init_session_state()
    
    st.markdown("""
    <div class="info-box">
    <h4>Welcome to the Multi-LLM AI-Powered CV Builder! 🎓</h4>
    <p>This application uses a specialized dual-model architecture for optimal CV tailoring:</p>
    <ul>
    <li><strong>Gemma 2B</strong>: Fast, efficient content generation and professional writing</li>
    <li><strong>Llama2 7B</strong>: Advanced data analysis and requirement matching</li>
    </ul>
    <p>Simply upload your existing resume and provide a job description to get started.</p>
    <p><em>Built for HAAI++ Capstone Project with justified multi-LLM architecture.</em></p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar for navigation and settings
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Ollama configuration
        st.subheader("🤖 Multi-LLM Configuration")
        ollama_host = st.text_input(
            "Ollama Host", 
            value=st.session_state.get("ollama_host", "http://localhost:11434"),
            key="ollama_host_input"
        )
        
        # Show current multi-model setup
        st.markdown("""
        **Active Models:**
        - 🎨 **Gemma 2B**: Content Generation
        - 🔍 **Llama2 7B**: Data Analysis
        """)
        
        if st.button("🔄 Initialize/Reconnect AI"):
            try:
                st.session_state.llm_client = None
                st.session_state.ollama_host = ollama_host
                settings.ollama_host = ollama_host
                initialize_services()
                st.success("✅ AI services reconnected successfully!")
            except Exception as e:
                st.error(f"❌ Failed to connect to AI services: {str(e)}")
                st.info("💡 Make sure Ollama is running with: `ollama serve`")
        
        st.divider()
        
        # Process steps indicator
        st.subheader("📋 Process Steps")
        steps = [
            ("1. Upload Resume", st.session_state.resume_data),
            ("2. Add Job Description", st.session_state.job_requirements), 
            ("3. AI Tailoring", st.session_state.tailored_cv),
            ("4. Generate CV", st.session_state.generated_document)
        ]
        
        for step_name, is_completed in steps:
            if is_completed:
                st.success(f"✅ {step_name}")
            else:
                st.write(f"⏳ {step_name}")
        
        # Current step indicator
        if not st.session_state.resume_data:
            st.info("👆 **Current Step**: Upload Resume")
        elif not st.session_state.job_requirements:
            st.info("👆 **Current Step**: Add Job Description")
        elif not st.session_state.tailored_cv:
            st.info("👆 **Current Step**: AI Tailoring")
        elif not st.session_state.generated_document:
            st.info("👆 **Current Step**: Generate CV")
        else:
            st.success("🎉 **All Steps Complete!**")
    
    # Initialize services
    if not initialize_services():
        st.stop()
    
    # Main content - conditional rendering based on progress
    if not st.session_state.resume_data:
        # Step 1: Resume Upload
        handle_resume_upload()
    elif not st.session_state.job_requirements:
        # Step 2: Job Description
        st.markdown("---")
        col1, col2 = st.columns([3, 1])
        with col1:
            st.success("✅ Resume uploaded successfully!")
        with col2:
            if st.button("� Upload Different Resume"):
                st.session_state.resume_data = None
                st.rerun()
        
        handle_job_description()
    elif not st.session_state.tailored_cv:
        # Step 3: AI Tailoring
        st.markdown("---")
        col1, col2 = st.columns([3, 1])
        with col1:
            st.success("✅ Resume and job requirements processed!")
        with col2:
            if st.button("🔄 Restart Process"):
                st.session_state.resume_data = None
                st.session_state.job_requirements = None
                st.rerun()
        
        handle_cv_tailoring()
    elif not st.session_state.generated_document:
        # Step 4: Document Generation
        st.markdown("---")
        col1, col2 = st.columns([3, 1])
        with col1:
            st.success("✅ CV tailoring completed!")
        with col2:
            if st.button("🔄 Restart Process"):
                st.session_state.resume_data = None
                st.session_state.job_requirements = None
                st.session_state.tailored_cv = None
                st.rerun()
        
        handle_document_generation()
    else:
        # Step 5: Final Results
        st.markdown("---")
        col1, col2 = st.columns([3, 1])
        with col1:
            st.success("✅ All steps completed! Your CV is ready!")
        with col2:
            if st.button("🔄 Start New CV"):
                st.session_state.resume_data = None
                st.session_state.job_requirements = None
                st.session_state.tailored_cv = None
                st.session_state.generated_document = None
                st.rerun()
        
        # Show final document
        handle_document_generation()


if __name__ == "__main__":
    main()