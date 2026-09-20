import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score,mean_absolute_error ,mean_squared_error
import joblib
df =pd.read_csv("train3.csv")

stores =pd.read_csv("stores.csv")

df["date"] =pd.to_datetime(df["date"])
df["year"] =df["date"].dt.year
df["month"] =df["date"].dt.month
df["day"] = df["date"].dt.day
df["day_of_week"] =df["date"].dt.dayofweek


df =df.merge(stores,on="store_nbr",how="left")


df["sales_clean"] = df["sales"].copy()
median_value =df[(df["store_nbr"]==39)& (df["family"]=="MEATS") & (df["sales"]<10000)]["sales"].median()


df.loc[(df["store_nbr"]== 39)&(df["family"] == "MEATS") & (df["date"]== "2016-10-07"),"sales_clean"] = median_value

df =df.sort_values(["store_nbr","family","date"])
df["sales_lag1"] =df.groupby(["store_nbr","family"])["sales_clean"].shift(1)


df["sales_lag7"] = df.groupby(["store_nbr","family"])["sales_clean"].shift(7)
df["sales_lag30"] =df.groupby(["store_nbr","family"])["sales_clean"].shift(30)
df["sales_rolling7"] =(df.groupby(["store_nbr","family"])["sales_clean"].shift(1).rolling(7).mean())
df["sales_rolling30"] =(df.groupby(["store_nbr","family"])["sales_clean"].transform(lambda x:x.shift(1).rolling(30).mean()))
df= df.dropna(subset=["sales_lag1","sales_lag7","sales_lag30"])

y =df["sales"]


x =df.drop(["sales","id","date"],axis=1)

categorical_features =["family","city","state","type"]
numeric_features =["store_nbr","onpromotion","year","month","day","day_of_week","cluster","sales_lag1",
                   "sales_lag7","sales_lag30","sales_rolling7","sales_rolling30"]

preprocessor =ColumnTransformer(transformers=[("cat",OneHotEncoder(handle_unknown="ignore"),categorical_features),("num","passthrough",numeric_features)])


split_data =df["date"].quantile(0.8)
x_train =x[df["date"] <= split_data]
x_test =x[df["date"] > split_data]
y_train =y[df['date']<= split_data]
y_test =y[df["date"] > split_data]



x_train_processed =preprocessor.fit_transform(x_train)

joblib.dump(preprocessor,"preprocessor.pkl")
x_test_processed =preprocessor.transform(x_test)


model =RandomForestRegressor(n_estimators=100,max_depth=15,random_state=42,n_jobs=-1)
model.fit(x_train_processed[:300000],y_train[:300000])
joblib.dump(model,"model.pkl")

y_pred =model.predict(x_test_processed)


mae =mean_absolute_error(y_test,y_pred)
mse =mean_squared_error(y_test,y_pred)
r2 =r2_score(y_test,y_pred)

print("mae : ",mae)
print("mse : ",mse)
print("r2 : ",r2)
error =abs(y_test - y_pred)
print("Mean Error : ", error.mean())
print("Median : ",error.median())
print("90% Error : ",error.quantile(0.90))
print("Max Error :",error.max())


results =df.loc[x_test.index, ["date","store_nbr","family","sales","sales_lag1",
             "sales_lag7","sales_lag30"]].copy()

results["predicted"] =y_pred
results["error"] = abs(results["sales"] - results["predicted"])

print(results.sort_values("error",ascending = False).head(10).to_string(index=False))
check =df[(df["store_nbr"] == 39)&(df["family"] == "MEATS")& (df["date"].between("2016-10-01","2016-10-15"))]
print(check[["date","sales","sales_lag1","sales_lag7","sales_lag30","sales_rolling7","sales_rolling30"]].to_string(index=False))

print(df[df["date"]== "2016-10-07"].sort_values("sales",ascending=False)[["store_nbr","family","sales"]].head(20).to_string(index=False))


plt.figure(figsize=(10,6)) 
plt.scatter(y_test[:5000],y_pred[:5000],alpha=0.3)
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted Sales")
plt.show()

