
# F1 Race Stint & Animated Lap Performance Visualizer

This repository contains a Streamlit web application for visualizing Formula 1 race data, focusing on tire stints and lap-by-lap performance. It includes an interactive dashboard to analyze past F1 races, view driver stint overviews, and animate lap time progression for specific stints.

## Features
- **Stint Overview**: Horizontal bar chart showing tire compounds used by each driver across laps (using Matplotlib).
- **Animated Lap Performance**: Interactive Plotly chart with a slider to simulate lap-by-lap progression, including trend lines for degradation.
- **Data Source**: Powered by FastF1 library for real-time F1 data fetching.
- **Deployment**: Hosted on Streamlit Cloud at [f1-visualizer-leev75.streamlit.app](https://f1-visualizer-leev75.streamlit.app).

## Files
- **`app.py`**: The main Streamlit application script. Run locally with `streamlit run app.py`.
- **`report.ipynb`**: Jupyter notebook report detailing the code structure, functionality, and how-to-run instructions.
- **`requirements.txt`**: Dependencies for installation and deployment.

## Installation and Local Setup
enter this link for trying the website :[ https://f1-visualizer-leev75.streamlit.app ]/

## Usage
- Select a year and F1 event from the sidebar.
- View stint overview by clicking the button.
- Choose a driver and stint for detailed lap analysis with the animation slider.

## Libraries Used
- Streamlit
- FastF1
- Matplotlib
- Pandas
- NumPy
- Plotly

## Notes
- Data caching is enabled in `fastf1_cache_stint` for faster loads.
- Ensure writable permissions for the cache directory.
- For questions or contributions, open an issue or PR.

