// DOM Elements
const dropArea = document.getElementById('drop-area');
const fileInput = document.getElementById('file-input');
const preview = document.getElementById('preview');
const previewImage = document.getElementById('preview-image');
const clearBtn = document.getElementById('clear-btn');
const classifyBtn = document.getElementById('classify-btn');
const resultSection = document.getElementById('result-section');
const pokemonName = document.getElementById('pokemon-name');
const confidence = document.getElementById('confidence');
const top5List = document.getElementById('top5-list');
const loadingOverlay = document.getElementById('loading-overlay');
const modelType = document.getElementById('model-type');
const numClasses = document.getElementById('num-classes');
const exampleCards = document.querySelectorAll('.example-card');

// API Endpoints
const API_URL = window.location.origin;
const HEALTH_ENDPOINT = `${API_URL}/health`;
const INFO_ENDPOINT = `${API_URL}/info`;
const PREDICT_ENDPOINT = `${API_URL}/predict`;

// Variables
let selectedFile = null;

// Initialize the application
async function init() {
    try {
        // Check if the API is available
        const healthResponse = await fetch(HEALTH_ENDPOINT);
        if (!healthResponse.ok) {
            throw new Error('API is not available');
        }
        
        // Get model information
        const infoResponse = await fetch(INFO_ENDPOINT);
        if (!infoResponse.ok) {
            throw new Error('Failed to get model information');
        }
        
        const modelInfo = await infoResponse.json();
        modelType.textContent = `Model: ${modelInfo.model_type}`;
        numClasses.textContent = `Classes: ${modelInfo.num_classes}`;
        
        console.log('Application initialized successfully');
    } catch (error) {
        console.error('Failed to initialize application:', error);
        alert('Failed to connect to the API. Please make sure the server is running.');
    }
}

// Event Listeners
// Prevent default drag behaviors
['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
    dropArea.addEventListener(eventName, preventDefaults, false);
    document.body.addEventListener(eventName, preventDefaults, false);
});

// Highlight drop area when dragging over it
['dragenter', 'dragover'].forEach(eventName => {
    dropArea.addEventListener(eventName, highlight, false);
});

['dragleave', 'drop'].forEach(eventName => {
    dropArea.addEventListener(eventName, unhighlight, false);
});

// Handle dropped files
dropArea.addEventListener('drop', handleDrop, false);

// Handle file input change
fileInput.addEventListener('change', handleFileInput, false);

// Handle clear button click
clearBtn.addEventListener('click', clearImage, false);

// Handle classify button click
classifyBtn.addEventListener('click', classifyImage, false);

// Handle example card clicks
exampleCards.forEach(card => {
    card.addEventListener('click', () => loadExampleImage(card.dataset.example));
});

// Functions
function preventDefaults(e) {
    e.preventDefault();
    e.stopPropagation();
}

function highlight() {
    dropArea.classList.add('highlight');
}

function unhighlight() {
    dropArea.classList.remove('highlight');
}

function handleDrop(e) {
    const dt = e.dataTransfer;
    const files = dt.files;
    
    if (files.length > 0) {
        handleFiles(files);
    }
}

function handleFileInput() {
    const files = fileInput.files;
    
    if (files.length > 0) {
        handleFiles(files);
    }
}

function handleFiles(files) {
    const file = files[0];
    
    if (file.type.startsWith('image/')) {
        selectedFile = file;
        displayPreview(file);
        classifyBtn.disabled = false;
    } else {
        alert('Please select an image file');
        clearImage();
    }
}

function displayPreview(file) {
    const reader = new FileReader();
    
    reader.onload = function(e) {
        previewImage.src = e.target.result;
        preview.classList.remove('hidden');
        document.querySelector('.upload-form').classList.add('hidden');
    };
    
    reader.readAsDataURL(file);
}

function clearImage() {
    selectedFile = null;
    previewImage.src = '';
    preview.classList.add('hidden');
    document.querySelector('.upload-form').classList.remove('hidden');
    classifyBtn.disabled = true;
    resultSection.classList.add('hidden');
}

async function classifyImage() {
    if (!selectedFile) {
        alert('Please select an image first');
        return;
    }
    
    try {
        // Show loading overlay
        loadingOverlay.classList.remove('hidden');
        
        // Create form data
        const formData = new FormData();
        formData.append('file', selectedFile);
        
        // Send the request
        const response = await fetch(PREDICT_ENDPOINT, {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            throw new Error('Failed to classify image');
        }
        
        // Parse the response
        const result = await response.json();
        
        // Display the result
        displayResult(result);
    } catch (error) {
        console.error('Error classifying image:', error);
        alert('Failed to classify image. Please try again.');
    } finally {
        // Hide loading overlay
        loadingOverlay.classList.add('hidden');
    }
}

function displayResult(result) {
    // Display the predicted class and confidence
    pokemonName.textContent = result.predicted_class;
    confidence.textContent = `Confidence: ${(result.confidence * 100).toFixed(2)}%`;
    
    // Clear the top 5 list
    top5List.innerHTML = '';
    
    // Add the top 5 predictions to the list
    for (let i = 0; i < result.top5_classes.length; i++) {
        const li = document.createElement('li');
        
        // Create the prediction text
        const predictionText = document.createElement('span');
        predictionText.textContent = `${result.top5_classes[i]}`;
        
        // Create the probability text
        const probabilityText = document.createElement('span');
        const probability = (result.top5_probabilities[i] * 100).toFixed(2);
        probabilityText.textContent = `${probability}%`;
        
        // Create the prediction bar
        const predictionBar = document.createElement('div');
        predictionBar.className = 'prediction-bar';
        predictionBar.style.width = `${probability}%`;
        
        // Add the elements to the list item
        li.appendChild(predictionText);
        li.appendChild(probabilityText);
        li.appendChild(predictionBar);
        
        // Add the list item to the top 5 list
        top5List.appendChild(li);
    }
    
    // Show the result section
    resultSection.classList.remove('hidden');
}

async function loadExampleImage(example) {
    try {
        // Fetch the example image
        const response = await fetch(`examples/${example}.jpg`);
        
        if (!response.ok) {
            throw new Error(`Failed to load example image: ${example}`);
        }
        
        // Convert the response to a blob
        const blob = await response.blob();
        
        // Create a file from the blob
        const file = new File([blob], `${example}.jpg`, { type: 'image/jpeg' });
        
        // Handle the file
        selectedFile = file;
        displayPreview(file);
        classifyBtn.disabled = false;
    } catch (error) {
        console.error('Error loading example image:', error);
        alert('Failed to load example image. Please try again.');
    }
}

// Initialize the application
init();
