/* Loaded first, as a classic script, so that the chosen theme is in place before the first
   paint. (The page's Content-Security-Policy allows no inline script.) */
(function () {
  try {
    var theme = localStorage.getItem('po-theme');
    if (theme === 'light' || theme === 'dark') document.documentElement.setAttribute('data-theme', theme);
  } catch (e) { /* storage is unavailable: the system's preference applies */ }
})();
