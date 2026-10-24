# DataWorkbench

A local Python application for data engineering, analysis, and data science. Upload a CSV or TSV and use the same dataset in all three workspaces.

## Run

Install dependencies using python -m pip install -r requirements.txt, then start the app using streamlit run app.py. It opens in your browser. Sample marketing data loads when no file is uploaded.

## Deploy to Streamlit Community Cloud

Push this folder to a GitHub repository, then create an app at share.streamlit.io using the repository, branch, and app.py entrypoint. You need admin access to that repository. The app visibility follows the repository visibility.

## Features

- **Data Engineering:** standardize column names, handle missing values, keep columns, filter numeric ranges, remove duplicate rows, preview the result, and download cleaned CSV and SQL.
- **Data Analysis:** chart measures by category or numeric variables, inspect correlations, and review descriptive statistics.
- **Data Science:** select a target and features, train random forest or linear/logistic regression, inspect held-out metrics and predictions, and view feature importance.

The SQL output is a query template for the selected steps. Pandas performs the actual transformations in the app.

