import mysql.connector
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer

# 🔹 Connect to MySQL
conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="Bala@2006",
    database="datas"
)
cursor = conn.cursor()

# Execute Query
cursor.execute("SELECT * FROM sales_data;")
data = cursor.fetchall()

# Get column names
columns = [desc[0] for desc in cursor.description]

# Close connection
cursor.close()
conn.close()

# Convert to DataFrame
df = pd.DataFrame(data, columns=columns)
null_data=df.isnull().sum()
print("Befor Cleaning: \n",null_data)

# data cleaning
# cleaning null datas Quantity_Ordered
n_Quantity_Ordered=SimpleImputer(strategy='mean')
n_Quantity_Ordered.fit(df[['Quantity_Ordered']])
df[['Quantity_Ordered']]=n_Quantity_Ordered.transform(df[['Quantity_Ordered']])

# Price_Each
n_Price_Each=SimpleImputer(strategy='mean')
n_Price_Each.fit(df[['Price_Each']])
df[['Price_Each']]=n_Price_Each.transform(df[['Price_Each']])

# Month 
n_Month=SimpleImputer(strategy='median')
n_Month.fit(df[['Month']])
df[['Month']]=n_Month.transform(df[['Month']])

# Sales
n_Sales=SimpleImputer(strategy='mean')
n_Sales.fit(df[['Sales']])
df[['Sales']]=n_Sales.transform(df[['Sales']])

n_Hour=SimpleImputer(strategy='mean')
n_Hour.fit(df[['Hour']])
df[['Hour']]=n_Hour.transform(df[['Hour']])

n_Order_ID =SimpleImputer(strategy='most_frequent')
n_Order_ID.fit(df[['Order_ID']])
df[['Order_ID']]=n_Order_ID .transform(df[['Order_ID']])

print("After Cleaning: \n",df.isnull().sum())

# data_types
print("Data Types:")
print(df.dtypes)

# change datatypes
df['Order_Date'] = pd.to_datetime(df['Order_Date'], format='%d-%m-%Y %H:%M')
df['Month']=df['Month'].astype(int)
df['Quantity_Ordered']=df['Quantity_Ordered'].astype(int)

# Unique Products
unique_products=df['Product'].unique()
unique_City =df['City'].unique()
print("Unique Products \n",unique_City )

# removing duplicates
df.drop_duplicates(inplace=True)
print("Shape of dataset: ",df.shape)

# numeric columns
numeric_df = df.select_dtypes(include=['number'])

# Correlation Matrix
plt.figure(figsize=(10, 6))
sns.heatmap(numeric_df.corr(), annot=True)
plt.title("Correlation Matrix of Numerical Columns")
plt.show()

# EDA
# 1-->Top 10 Most Sold Products
group_products = df.groupby('Product')['Sales'].mean()
group_products = group_products.sort_values(ascending=False).head(10)  #top 10
print('Grouped products \n',group_products)
plt.figure(figsize=(10, 6))
group_products.plot(kind='bar')
plt.title("Top 10 Most Sold Products")
plt.ylabel("Sales")
plt.xlabel("Products")
plt.xticks(rotation=45, ha='right')
plt.show()

# 2-->Monthly Sales
monthly_sales = df.groupby('Month')['Sales'].sum()
print(monthly_sales)
plt.figure(figsize=(10, 6))
monthly_sales.plot(kind='line', marker='o')
plt.title("Monthly Sales")  # ✅ Correct function call
plt.xlabel("Month")  # ✅ Correct function call
plt.ylabel("Sales")
plt.show()

# 3-->Top 5 City Sales
city_sales = df.groupby('City')['Sales'].sum().sort_values(ascending=False).head(5)
print("Top 5 City Sales \n",city_sales)
plt.figure(figsize=(10, 6))
city_sales.plot(kind='bar')
plt.title("Top 5 City Sales")
plt.ylabel('Sales Frequency')
plt.xlabel('City')
plt.show()

# 4-->Top-Selling Product in Each City
# Group by 'City' and 'Product', summing the 'Quantity_Ordered'
city_product_sales = df.groupby(['City', 'Product'])['Quantity_Ordered'].sum()
top_selling_products = city_product_sales.groupby('City').idxmax()
# Extract City and Product separately for plotting
city_names = [city for city, product in top_selling_products]
product_names = [product[1] for product in top_selling_products]

plt.figure(figsize=(10,6))
sns.barplot(x=city_names, y=city_product_sales.groupby('City').max().values, hue=product_names)
plt.title("Top-Selling Product in Each City")
plt.xlabel("City")
plt.ylabel("Quantity Ordered")
plt.xticks(rotation=45, ha='right')
plt.show()

# 5--># Top 3 Month Sales
df['Year-Month'] = df['Order_Date'].dt.to_period('M')
monthly_sales = df.groupby('Year-Month')['Sales'].sum()
top_3_months = monthly_sales.sort_values(ascending=False).head(3)
print("Top 3 Month Sales in a Year \n",top_3_months)
plt.figure(figsize=(10, 6))
top_3_months.plot(kind='bar')
plt.title("Top 3 Month Sales")
plt.xlabel("Year-Month")
plt.ylabel("Sales")
plt.xticks(rotation=360, ha='right')
plt.show()

# save in csv file
df.to_csv("cleaned_data.csv", index=False)

print("✅ Data saved successfully as cleaned_data.csv")