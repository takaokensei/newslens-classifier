"""
NewsLens AI Classifier - UI Components
Reusable UI components built with the Design System.
"""
import streamlit as st
from apps.design_system import DESIGN_TOKENS

def status_badge(status: str, text: str):
    """
    Renders a status badge with semantic colors.
    
    Args:
        status: 'success', 'warning', 'error', 'info', or 'neutral'
        text: The text to display inside the badge
    """
    colors = DESIGN_TOKENS['colors']
    
    # Map status to color hex
    color_map = {
        'success': colors['success'],
        'warning': colors['warning'],
        'error': colors['error'],
        'info': colors['info'],
        'neutral': colors['neutral']['medium']
    }
    
    bg_color = color_map.get(status, colors['info'])
    
    # Use a slightly transparent background for the badge look
    # We'll use inline styles for simplicity here
    st.markdown(f"""
        <span style="
            background-color: {bg_color}20;
            color: {bg_color};
            padding: 0.25rem 0.75rem;
            border-radius: 9999px;
            font-size: 0.75rem;
            font-weight: 600;
            border: 1px solid {bg_color}40;
            display: inline-flex;
            align-items: center;
            gap: 0.25rem;
        ">
            {text}
        </span>
    """, unsafe_allow_html=True)

def info_card(title: str, value: str, delta: str = None, icon: str = None):
    """
    Renders a metric card with optional delta and icon.
    """
    delta_html = ""
    if delta:
        is_positive = not delta.startswith('-')
        color = DESIGN_TOKENS['colors']['success'] if is_positive else DESIGN_TOKENS['colors']['error']
        delta_html = f'<span style="color: {color}; font-size: 0.875rem; font-weight: 500;">{delta}</span>'
    
    icon_html = ""
    if icon:
        icon_html = f'<span class="material-icons" style="font-size: 24px; color: {DESIGN_TOKENS["colors"]["primary"]};">{icon}</span>'

    st.markdown(f"""
        <div style="
            background-color: var(--color-card-bg);
            border: 1px solid var(--color-border);
            border-radius: 0.5rem;
            padding: 1.25rem;
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
            box-shadow: var(--shadow-md);
            height: 100%;
        ">
            <div style="display: flex; justify-content: space-between; align-items: start;">
                <span style="color: var(--color-text-secondary); font-size: 0.875rem; font-weight: 500; text-transform: uppercase; letter-spacing: 0.05em;">{title}</span>
                {icon_html}
            </div>
            <div style="display: flex; align-items: baseline; gap: 0.75rem;">
                <span style="color: var(--color-text); font-size: 1.875rem; font-weight: 700;">{value}</span>
                {delta_html}
            </div>
        </div>
    """, unsafe_allow_html=True)

def result_card(predicted_class: int, label: str, confidence: float, probabilities: dict):
    """
    Displays the classification result in a prominent card.
    """
    # Determine confidence color
    if confidence > 0.8:
        conf_color = DESIGN_TOKENS['colors']['success']
    elif confidence > 0.5:
        conf_color = DESIGN_TOKENS['colors']['warning']
    else:
        conf_color = DESIGN_TOKENS['colors']['error']
        
    st.markdown(f"""
        <div style="
            background-color: var(--color-card-bg);
            border: 1px solid {conf_color};
            border-radius: 0.75rem;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            box-shadow: 0 4px 20px -5px {conf_color}30;
        ">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
                <span style="color: var(--color-text-secondary); font-size: 0.875rem; font-weight: 600; text-transform: uppercase;">Resultado da Classificação</span>
                <span style="background-color: {conf_color}20; color: {conf_color}; padding: 0.25rem 0.75rem; border-radius: 9999px; font-size: 0.75rem; font-weight: 700;">
                    {confidence:.1%} Confiança
                </span>
            </div>
            
            <div style="margin-bottom: 1.5rem;">
                <h2 style="
                    color: {conf_color}; 
                    font-size: 2rem; 
                    font-weight: 800; 
                    margin: 0; 
                    line-height: 1.2;
                ">
                    {label}
                </h2>
                <div style="color: var(--color-text-secondary); font-size: 0.9rem; margin-top: 0.25rem;">
                    Classe {predicted_class}
                </div>
            </div>
            
            <!-- Progress Bar -->
            <div style="background-color: var(--color-bg); height: 8px; border-radius: 4px; overflow: hidden; margin-bottom: 0.5rem;">
                <div style="
                    width: {confidence * 100}%; 
                    height: 100%; 
                    background-color: {conf_color}; 
                    transition: width 0.5s ease-out;
                "></div>
            </div>
        </div>
    """, unsafe_allow_html=True)

def empty_state(icon: str, title: str, description: str):
    """
    Renders an empty state placeholder.
    """
    st.markdown(f"""
        <div style="
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            padding: 3rem 1rem;
            text-align: center;
            color: var(--color-text-secondary);
            border: 2px dashed var(--color-border);
            border-radius: 0.75rem;
            background-color: var(--color-card-bg);
        ">
            <span class="material-icons" style="font-size: 48px; margin-bottom: 1rem; opacity: 0.5;">{icon}</span>
            <h3 style="color: var(--color-text); font-size: 1.125rem; font-weight: 600; margin-bottom: 0.5rem;">{title}</h3>
            <p style="font-size: 0.9rem; max-width: 400px; margin: 0;">{description}</p>
        </div>
    """, unsafe_allow_html=True)
