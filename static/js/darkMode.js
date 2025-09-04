// Function to initialize dark mode from localStorage
function initDarkMode() {
  const darkMode = localStorage.getItem('darkMode') === 'true';
  document.documentElement.setAttribute('data-theme', darkMode ? 'dark' : 'light');
}

// Function to toggle dark mode
function toggleDarkMode() {
  document.body.classList.toggle('dark-mode');
  const isDarkMode = document.body.classList.contains('dark-mode');
  localStorage.setItem('darkMode', isDarkMode);
}

// Menu toggle functionality
function toggleMenu() {
  const menu = document.querySelector('.menu');
  const menuIcon = menu.querySelector('.menu-icon');
  menuIcon.classList.toggle('active');
  
  // Add your menu open/close logic here
  // For example, show/hide a navigation panel
}

// Initialize dark mode when the DOM is loaded
document.addEventListener('DOMContentLoaded', initDarkMode);

// If there's a dark mode toggle in the page, set its initial state
document.addEventListener('DOMContentLoaded', () => {
  const darkModeToggle = document.getElementById('darkModeToggle');
  if (darkModeToggle) {
    darkModeToggle.checked = localStorage.getItem('darkMode') === 'true';
  }
});

// Check for saved dark mode preference
document.addEventListener('DOMContentLoaded', () => {
  const isDarkMode = localStorage.getItem('darkMode') === 'true';
  if (isDarkMode) {
    document.body.classList.add('dark-mode');
  }
}); 