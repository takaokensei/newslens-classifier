"""
NewsLens AI Classifier - Design System
Defines the visual language (tokens) for the application.
"""

DESIGN_TOKENS = {
    'spacing': {
        'xs': '0.25rem',
        'sm': '0.5rem',
        'md': '1rem',
        'lg': '1.5rem',
        'xl': '2rem',
        'xxl': '3rem'
    },
    'colors': {
        'primary': '#3b82f6',  # Blue 500
        'secondary': '#8b5cf6', # Violet 500
        'accent': '#ffffff',    # White for dark mode text/icons
        'success': '#10b981',   # Emerald 500
        'warning': '#f59e0b',   # Amber 500
        'error': '#ef4444',     # Red 500
        'info': '#3b82f6',      # Blue 500
        'neutral': {
            'light': '#f9fafb',
            'medium': '#9ca3af', # Gray 400
            'dark': '#111827'    # Gray 900
        },
        'dark_mode': {
            'bg': '#0e1117',
            'card_bg': '#1e2130',
            'border': '#464b5f',
            'text': '#fafafa',
            'secondary_text': '#b0b0b0',
            'sidebar_bg': '#262730',
            'sidebar_text': '#ffffff'
        }
    },
    'typography': {
        'font_family': {
            'sans': '"Inter", sans-serif',
            'icons': '"Material Icons"'
        },
        'scale': {
            'xs': '0.75rem',
            'sm': '0.875rem',
            'base': '1rem',
            'lg': '1.125rem',
            'xl': '1.25rem',
            '2xl': '1.5rem',
            '3xl': '1.875rem',
            '4xl': '2.25rem'
        },
        'weight': {
            'normal': 400,
            'medium': 500,
            'semibold': 600,
            'bold': 700
        }
    },
    'shadows': {
        'sm': '0 1px 2px 0 rgba(0, 0, 0, 0.05)',
        'md': '0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)',
        'lg': '0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05)'
    },
    'border_radius': {
        'sm': '0.125rem',
        'md': '0.375rem',
        'lg': '0.5rem',
        'full': '9999px'
    }
}

def get_css_variables():
    """Generates CSS variables from design tokens."""
    c = DESIGN_TOKENS['colors']['dark_mode']
    t = DESIGN_TOKENS['typography']
    
    return f"""
        :root {{
            --color-bg: {c['bg']};
            --color-card-bg: {c['card_bg']};
            --color-border: {c['border']};
            --color-text: {c['text']};
            --color-text-secondary: {c['secondary_text']};
            --color-primary: {DESIGN_TOKENS['colors']['primary']};
            --color-success: {DESIGN_TOKENS['colors']['success']};
            --color-warning: {DESIGN_TOKENS['colors']['warning']};
            --color-error: {DESIGN_TOKENS['colors']['error']};
            --font-sans: {t['font_family']['sans']};
            --shadow-md: {DESIGN_TOKENS['shadows']['md']};
        }}
    """
