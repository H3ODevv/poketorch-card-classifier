/**
 * Scroll behavior for PokéTorch Card Classifier
 * Handles header transformations on scroll with debouncing for smoother transitions
 */

// DOM Elements
let header;

// Scroll Constants
const SCROLL_THRESHOLD = 50; // in pixels
const BUFFER_ZONE = 10; // buffer zone to prevent jittering
const DEBOUNCE_DELAY = 10; // debounce delay in milliseconds

// Variables for debouncing
let scrollTimeout;
let lastScrollY = 0;
let isHeaderScrolled = false;

/**
 * Initialize scroll behavior
 */
function initScroll() {
    // Get DOM elements
    header = document.querySelector('header');
    
    // Add scroll event listener with debouncing
    window.addEventListener('scroll', debounceScroll);
    
    // Initial check
    handleScroll();
}

/**
 * Debounce the scroll event to prevent excessive function calls
 */
function debounceScroll() {
    // Store the current scroll position
    lastScrollY = window.scrollY;
    
    // Clear the timeout if it exists
    if (scrollTimeout) {
        clearTimeout(scrollTimeout);
    }
    
    // Set a timeout to call handleScroll after a delay
    scrollTimeout = setTimeout(handleScroll, DEBOUNCE_DELAY);
}

/**
 * Handle scroll events with buffer zone to prevent jittering
 */
function handleScroll() {
    // Use the stored scroll position
    const scrollY = lastScrollY;
    
    // Add buffer zone to prevent jittering
    if (!isHeaderScrolled && scrollY > SCROLL_THRESHOLD + BUFFER_ZONE) {
        header.classList.add('header-scrolled');
        isHeaderScrolled = true;
    } else if (isHeaderScrolled && scrollY < SCROLL_THRESHOLD - BUFFER_ZONE) {
        header.classList.remove('header-scrolled');
        isHeaderScrolled = false;
    }
}

// Initialize scroll behavior when the DOM is loaded
document.addEventListener('DOMContentLoaded', initScroll);
