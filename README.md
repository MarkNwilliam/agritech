# AI-Powered Farm Management Platform Backend

## Project Overview

This backend powers an innovative farm management platform that leverages AI and machine learning to provide farmers with data-driven insights. Our system integrates time series forecasting, natural language processing, and soil data analysis to offer a comprehensive suite of tools for modern agriculture.

![Create-Next-App](https://github.com/user-attachments/assets/c3d2db91-313f-484e-97a5-0fabccbb41c1)
![happy-desert-0cf68191e-5-azurestaticapps-net-agronomist](https://github.com/user-attachments/assets/f7d9361e-4152-4419-9547-18a883bbf219)

Certainly! Here's an updated README for the backend, focusing on Prophet, Watson AI, and the iSDAsoil API:

markdown

Copy
# AI-Powered Farm Management Platform Backend

## Project Overview

This backend powers an innovative farm management platform that leverages AI and machine learning to provide farmers with data-driven insights. Our system integrates time series forecasting, natural language processing, and soil data analysis to offer a comprehensive suite of tools for modern agriculture.

![Dashboard Screenshot](link_to_dashboard_screenshot.png)
![Forecast Screenshot](link_to_forecast_screenshot.png)
![Agronomist Screenshot](link_to_agronomist_screenshot.png)

## Key Technologies

### Prophet for Time Series Forecasting

We use Facebook's Prophet library for robust time series forecasting, particularly for sales and yield predictions. Prophet is especially powerful for our use case because:

- It handles seasonal effects with weekly, monthly, and yearly seasonality.
- It's robust to missing data and shifts in trends.
- It can model holiday effects, which is crucial for agricultural cycles.

Prophet works by decomposing time series into trend, seasonality, and holiday components:

1. Trend: Captures non-periodic changes in the time series.
2. Seasonality: Represents periodic changes (e.g., weekly or yearly cycles).
3. Holidays: Accounts for irregularly scheduled events.

In our `/forecast` endpoint, we use Prophet to generate future predictions based on historical farm data, providing farmers with valuable insights for planning and decision-making.

### IBM Watson AI for Natural Language Processing

We leverage IBM Watson's foundation models, specifically the GRANITE_13B_CHAT_V2 model, for various natural language processing tasks:

- Generating insightful reports from numerical data.
- Creating engaging social media content for farms.
- Analyzing customer feedback and providing actionable insights.
- Offering crop planning recommendations based on farm-specific data.

Watson AI's advanced language understanding and generation capabilities allow us to translate complex agricultural data into clear, actionable advice for farmers.

### iSDAsoil API for Soil Analysis

The iSDAsoil API is a crucial component of our virtual agronomist feature. It provides detailed soil property and agronomy information for locations across Africa. Key features include:

- High-resolution data: Most layers are at 30m resolution.
- Comprehensive coverage: Includes all of Africa (excluding water bodies and deserts).
- Rich soil information: Provides data on various soil properties and agronomic factors.

#### How we use the iSDAsoil API:

1. We query the `/layers` endpoint to retrieve metadata on available soil properties.
2. In our `/agronomist_ai` endpoint, we use the `/soilproperty` endpoint to fetch specific soil data based on latitude and longitude.
3. This soil data is then analyzed by our AI to provide tailored agronomic advice.
