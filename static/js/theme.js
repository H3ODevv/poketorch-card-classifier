/**
 * Theme toggle functionality for PokéTorch Card Classifier
 * Handles switching between light and dark themes
 */

// Theme constants
const THEME_STORAGE_KEY = 'poketorch-theme';
const DARK_THEME = 'dark';
const LIGHT_THEME = 'light';

// DOM Elements
let themeToggleBtn;
let themeIcon;

/**
 * Initialize the theme functionality
 */
function initTheme() {
    // Create theme toggle button if it doesn't exist
    if (!document.getElementById('theme-toggle')) {
        createThemeToggle();
    } else {
        themeToggleBtn = document.getElementById('theme-toggle');
        themeIcon = document.getElementById('theme-icon');
    }

    // Set the initial theme based on user preference or system preference
    const savedTheme = localStorage.getItem(THEME_STORAGE_KEY);
    if (savedTheme) {
        setTheme(savedTheme);
    } else {
        // Check if user prefers dark mode
        const prefersDarkMode = window.matchMedia('(prefers-color-scheme: dark)').matches;
        setTheme(prefersDarkMode ? DARK_THEME : LIGHT_THEME);
    }

    // Add event listener to theme toggle button
    themeToggleBtn.addEventListener('click', toggleTheme);
}

/**
 * Create the theme toggle button and add it to the DOM
 */
function createThemeToggle() {
    // Create the button
    themeToggleBtn = document.createElement('button');
    themeToggleBtn.id = 'theme-toggle';
    themeToggleBtn.className = 'theme-toggle';
    themeToggleBtn.setAttribute('aria-label', 'Toggle theme');
    themeToggleBtn.setAttribute('title', 'Toggle theme');

    // Create the icon
    themeIcon = document.createElement('i');
    themeIcon.id = 'theme-icon';
    themeIcon.className = 'fas fa-moon';
    
    // Append the icon to the button
    themeToggleBtn.appendChild(themeIcon);
    
    // Add the button to the header
    const header = document.querySelector('header');
    header.appendChild(themeToggleBtn);
}

/**
 * Toggle between light and dark themes
 */
function toggleTheme() {
    const currentTheme = document.documentElement.getAttribute('data-theme') || LIGHT_THEME;
    const newTheme = currentTheme === LIGHT_THEME ? DARK_THEME : LIGHT_THEME;
    
    setTheme(newTheme);
}

/**
 * Set the theme to light or dark
 * @param {string} theme - The theme to set (light or dark)
 */
function setTheme(theme) {
    // Set the data-theme attribute on the root element
    document.documentElement.setAttribute('data-theme', theme);
    
    // Update the icon
    if (themeIcon) {
        themeIcon.className = theme === DARK_THEME ? 'fas fa-sun' : 'fas fa-moon';
    }
    
    // Save the theme preference to localStorage
    localStorage.setItem(THEME_STORAGE_KEY, theme);
}

// Initialize the theme when the DOM is loaded
document.addEventListener('DOMContentLoaded', initTheme);
