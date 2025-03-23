/**
 * Animations and micro-interactions for PokéTorch Card Classifier
 * Enhances user experience with subtle animations and feedback
 */

// Animation variables
let uploadIcon;
let resultCard;
let predictionBars;
// Note: We're using DOM elements already defined in main.js
// (dropArea, previewImage, classifyBtn, exampleCards)

// Animation Constants
const ANIMATION_DURATION = 300; // in milliseconds
const STAGGER_DELAY = 50; // in milliseconds

/**
 * Initialize animations
 */
function initAnimations() {
    // Get DOM elements
    uploadIcon = document.querySelector('.upload-icon i');
    resultCard = document.querySelector('.result-card');
    // We're using DOM elements already defined in main.js
    
    // Add event listeners
    addHoverAnimations();
    addClickAnimations();
    addResultAnimations();
    
    // Add pulse animation to upload icon
    addPulseAnimation();
}

/**
 * Add hover animations to interactive elements
 */
function addHoverAnimations() {
    // Add hover animation to example cards
    exampleCards.forEach(card => {
        card.addEventListener('mouseenter', () => {
            card.style.transform = 'translateY(-8px) scale(1.03)';
            card.style.boxShadow = 'var(--shadow-strong)';
        });
        
        card.addEventListener('mouseleave', () => {
            card.style.transform = '';
            card.style.boxShadow = '';
        });
    });
    
    // Add hover animation to classify button
    if (classifyBtn) {
        classifyBtn.addEventListener('mouseenter', () => {
            if (!classifyBtn.disabled) {
                classifyBtn.style.transform = 'translateY(-3px)';
                classifyBtn.style.boxShadow = 'var(--shadow-strong)';
            }
        });
        
        classifyBtn.addEventListener('mouseleave', () => {
            classifyBtn.style.transform = '';
            classifyBtn.style.boxShadow = '';
        });
    }
}

/**
 * Add click animations to interactive elements
 */
function addClickAnimations() {
    // Add click animation to classify button
    if (classifyBtn) {
        classifyBtn.addEventListener('mousedown', () => {
            if (!classifyBtn.disabled) {
                classifyBtn.style.transform = 'translateY(2px)';
                classifyBtn.style.boxShadow = 'var(--shadow-light)';
            }
        });
        
        classifyBtn.addEventListener('mouseup', () => {
            if (!classifyBtn.disabled) {
                classifyBtn.style.transform = 'translateY(-3px)';
                classifyBtn.style.boxShadow = 'var(--shadow-strong)';
            }
        });
    }
    
    // Add click animation to example cards
    exampleCards.forEach(card => {
        card.addEventListener('mousedown', () => {
            card.style.transform = 'translateY(-2px) scale(0.98)';
            card.style.boxShadow = 'var(--shadow-light)';
        });
        
        card.addEventListener('mouseup', () => {
            card.style.transform = 'translateY(-8px) scale(1.03)';
            card.style.boxShadow = 'var(--shadow-strong)';
        });
    });
}

/**
 * Add animations to result display
 */
function addResultAnimations() {
    // Add observer for result section
    const resultSection = document.getElementById('result-section');
    
    if (resultSection) {
        // Create a mutation observer to watch for changes to the result section
        const observer = new MutationObserver((mutations) => {
            mutations.forEach((mutation) => {
                if (mutation.attributeName === 'class' && 
                    !resultSection.classList.contains('hidden')) {
                    // Result section is now visible, animate the results
                    animateResults();
                }
            });
        });
        
        // Start observing the result section
        observer.observe(resultSection, { attributes: true });
    }
}

/**
 * Animate the results when they are displayed
 */
function animateResults() {
    if (resultCard) {
        // Animate the result card
        resultCard.style.opacity = '0';
        resultCard.style.transform = 'translateY(20px)';
        
        setTimeout(() => {
            resultCard.style.transition = `opacity ${ANIMATION_DURATION}ms ease, transform ${ANIMATION_DURATION}ms ease`;
            resultCard.style.opacity = '1';
            resultCard.style.transform = 'translateY(0)';
        }, 50);
        
        // Animate the prediction bars with a staggered delay
        setTimeout(() => {
            const predictionBars = document.querySelectorAll('.prediction-bar');
            
            predictionBars.forEach((bar, index) => {
                const originalWidth = bar.style.width;
                bar.style.width = '0';
                
                setTimeout(() => {
                    bar.style.transition = `width ${ANIMATION_DURATION}ms ease`;
                    bar.style.width = originalWidth;
                }, index * STAGGER_DELAY);
            });
        }, ANIMATION_DURATION);
    }
}

/**
 * Add pulse animation to upload icon
 */
function addPulseAnimation() {
    if (uploadIcon) {
        // Add CSS animation class
        uploadIcon.classList.add('pulse-animation');
        
        // Add CSS for the animation if it doesn't exist
        if (!document.getElementById('animation-styles')) {
            const styleSheet = document.createElement('style');
            styleSheet.id = 'animation-styles';
            styleSheet.textContent = `
                @keyframes pulse {
                    0% { transform: scale(1); }
                    50% { transform: scale(1.1); }
                    100% { transform: scale(1); }
                }
                
                .pulse-animation {
                    animation: pulse 2s infinite ease-in-out;
                }
            `;
            document.head.appendChild(styleSheet);
        }
    }
}

// Initialize animations when the DOM is loaded
document.addEventListener('DOMContentLoaded', initAnimations);
