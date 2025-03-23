/**
 * Scroll behavior for PokéTorch Card Classifier
 * Handles header transformations on scroll
 */

// DOM Elements
let header;

// Scroll Constants
const SCROLL_THRESHOLD = 50; // in pixels

/**
 * Initialize scroll behavior
 */
function initScroll() {
    // Get DOM elements
    header = document.querySelector('header');
    
    // Add scroll event listener
    window.addEventListener('scroll', handleScroll);
}

/**
 * Handle scroll events
 */
function handleScroll() {
    if (window.scrollY > SCROLL_THRESHOLD) {
        header.classList.add('header-scrolled');
    } else {
        header.classList.remove('header-scrolled');
    }
}

// Initialize scroll behavior when the DOM is loaded
document.addEventListener('DOMContentLoaded', initScroll);
