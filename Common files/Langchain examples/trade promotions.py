import pandas as pd

df = pd.read_csv("C:/Users/cs25/OneDrive - Capgemini/Trade Promotions POC/ML Team Data/input files csv 1/Sample input files/weekly_sales.csv")

X = ['retailer_national_account_name', 'retailer_banner_geography', 'promoted_group', 'category_name', 'base_price', 'promoted_price', 'week_number', 'plan_year', 'is_feature', 'is_display', 'is_tpr', 'is_promo_sale']

out = df.columns([X])

out.to_csv("xvalues.csv")