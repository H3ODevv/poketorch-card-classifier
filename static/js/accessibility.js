/**
 * Accessibility enhancements for PokéTorch Card Classifier
 * Improves keyboard navigation, screen reader support, and more
 */

// DOM Elements
let focusableElements;
let firstFocusableElement;
let lastFocusableElement;

// Constants
const FOCUSABLE_SELECTORS = 'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])';
const SKIP_LINK_ID = 'skip-to-content';

/**
 * Initialize accessibility enhancements
 */
function initAccessibility() {
    // Add skip to content link
    addSkipToContentLink();
    
    // Enhance keyboard navigation
    enhanceKeyboardNavigation();
    
    // Add ARIA attributes
    addAriaAttributes();
    
    // Add keyboard shortcuts
    addKeyboardShortcuts();
}

/**
 * Add skip to content link for keyboard users
 */
function addSkipToContentLink() {
    // Check if skip link already exists
    if (document.getElementById(SKIP_LINK_ID)) {
        return;
    }
    
    // Create skip link
    const skipLink = document.createElement('a');
    skipLink.id = SKIP_LINK_ID;
    skipLink.href = '#drop-area';
    skipLink.textContent = 'Skip to content';
    skipLink.className = 'skip-link';
    
    // Add skip link to the DOM
    document.body.insertBefore(skipLink, document.body.firstChild);
    
    // Add CSS for the skip link if it doesn't exist
    if (!document.getElementById('accessibility-styles')) {
        const styleSheet = document.createElement('style');
        styleSheet.id = 'accessibility-styles';
        styleSheet.textContent = `
            .skip-link {
                position: absolute;
                top: -40px;
                left: 0;
                background: var(--primary-color);
                color: white;
                padding: 8px;
                z-index: 100;
                transition: top 0.3s;
            }
            
            .skip-link:focus {
                top: 0;
            }
            
            /* Focus styles */
            :focus {
                outline: 2px solid var(--accent-color);
                outline-offset: 2px;
            }
            
            /* High contrast focus for keyboard users */
            :focus-visible {
                outline: 3px solid var(--accent-color);
                outline-offset: 3px;
                box-shadow: 0 0 0 2px var(--background-color), 0 0 0 5px var(--accent-color);
            }
        `;
        document.head.appendChild(styleSheet);
    }
}

/**
 * Enhance keyboard navigation
 */
function enhanceKeyboardNavigation() {
    // Get all focusable elements
    focusableElements = document.querySelectorAll(FOCUSABLE_SELECTORS);
    
    // Convert NodeList to Array for easier manipulation
    focusableElements = Array.prototype.slice.call(focusableElements);
    
    // Filter out hidden elements
    focusableElements = focusableElements.filter(element => {
        return element.offsetParent !== null; // Element is visible
    });
    
    // Get first and last focusable elements
    firstFocusableElement = focusableElements[0];
    lastFocusableElement = focusableElements[focusableElements.length - 1];
    
    // Add event listener for tab key to trap focus in modals
    document.addEventListener('keydown', handleTabKey);
}

/**
 * Handle tab key for keyboard navigation
 * @param {KeyboardEvent} e - The keyboard event
 */
function handleTabKey(e) {
    // Check if tab key was pressed
    if (e.key !== 'Tab') {
        return;
    }
    
    // Check if loading overlay is visible
    const loadingOverlay = document.getElementById('loading-overlay');
    if (loadingOverlay && !loadingOverlay.classList.contains('hidden')) {
        // Prevent tabbing while loading
        e.preventDefault();
        return;
    }
}

/**
 * Add ARIA attributes for better screen reader support
 */
function addAriaAttributes() {
    // Add ARIA attributes to the drop area
    const dropArea = document.getElementById('drop-area');
    if (dropArea) {
        dropArea.setAttribute('role', 'region');
        dropArea.setAttribute('aria-label', 'Image upload area');
    }
    
    // Add ARIA attributes to the result section
    const resultSection = document.getElementById('result-section');
    if (resultSection) {
        resultSection.setAttribute('aria-live', 'polite');
        resultSection.setAttribute('aria-atomic', 'true');
    }
    
    // Add ARIA attributes to the loading overlay
    const loadingOverlay = document.getElementById('loading-overlay');
    if (loadingOverlay) {
        loadingOverlay.setAttribute('role', 'alert');
        loadingOverlay.setAttribute('aria-busy', 'true');
        loadingOverlay.setAttribute('aria-label', 'Loading, please wait');
    }
    
    // Add ARIA attributes to the example cards
    const exampleCards = document.querySelectorAll('.example-card');
    exampleCards.forEach(card => {
        card.setAttribute('role', 'button');
        card.setAttribute('aria-label', `Load ${card.querySelector('span').textContent} example`);
        card.setAttribute('tabindex', '0');
    });
}

/**
 * Add keyboard shortcuts for common actions
 */
function addKeyboardShortcuts() {
    document.addEventListener('keydown', e => {
        // Ctrl+U to upload an image
        if (e.ctrlKey && e.key === 'u') {
            e.preventDefault();
            document.getElementById('file-input').click();
        }
        
        // Ctrl+C to classify (if button is enabled)
        if (e.ctrlKey && e.key === 'c') {
            e.preventDefault();
            const classifyBtn = document.getElementById('classify-btn');
            if (classifyBtn && !classifyBtn.disabled) {
                classifyBtn.click();
            }
        }
        
        // Escape to clear the preview
        if (e.key === 'Escape') {
            const clearBtn = document.getElementById('clear-btn');
            const preview = document.getElementById('preview');
            if (clearBtn && preview && !preview.classList.contains('hidden')) {
                clearBtn.click();
            }
        }
    });
}

// Initialize accessibility enhancements when the DOM is loaded
document.addEventListener('DOMContentLoaded', initAccessibility);
