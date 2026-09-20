import pandas as pd
import joblib
test =pd.read_csv("test3.csv")
stores =pd.read_csv("stores.csv")

test =test.merge(stores,on="store_nbr",how="left")

test["date"] = pd.to_datetime(test["date"])
test["year"] =test["date"].dt.year
test["month"] =test["date"] .dt.month
test["day"] = test["date"].dt.day
test["day_of_week"] = test["date"].dt.day_of_week

train =pd.read_csv("train3.csv")
train["date"] =pd.to_datetime(train["date"])

history =train[["date","store_nbr","family","sales"]].copy()
history =history.sort_values(["store_nbr","family","date"])


history["sales_lag1"] =(history.groupby(["store_nbr","family"])["sales"].shift(1))



history["sales_lag7"] = history.groupby(["store_nbr","family"])["sales"].shift(7)
history["sales_lag30"] =history.groupby(["store_nbr","family"])["sales"].shift(30)
history["sales_rolling7"] =(history.groupby(["store_nbr","family"])["sales"].shift(1).rolling(7).mean())
history["sales_rolling30"] =(history.groupby(["store_nbr","family"])["sales"].transform(lambda x:x.shift(1).rolling(30).mean()))

first_day =test[test["date"] == "2017-08-16"].copy() 
previous_day = history[history["date"] == "2017-08-15"][["store_nbr","family","sales"]].copy()
first_day = first_day.merge(
    previous_day,on=["store_nbr","family"],how="left"
)

first_day =first_day.rename(
    columns={"sales" :"sales_lag1"}
)


seven_days_ago = history[history["date"] == "2017-08-09"][["store_nbr","family","sales"]].copy()

first_day =first_day.merge(seven_days_ago,on=["store_nbr","family"],how="left")

first_day =first_day.rename(columns={"sales" : "sales_lag7"})

thirty_days_ago =history[history["date"] == "2017-07-17"][["store_nbr","family",'sales']].copy()
first_day =first_day.merge(thirty_days_ago,on=["store_nbr","family"],how="left")
first_day =first_day.rename(columns={"sales":"sales_lag30"})


rolling_days_ago7 =history[(history["date"] >="2017-08-09")& (history["date"] <= "2017-08-15")][["store_nbr","family","sales"]].copy()
rolling7 =rolling_days_ago7.groupby(["store_nbr","family"])["sales"].mean().reset_index()


first_day =first_day.merge(rolling7,on=["store_nbr","family"],how="left")
first_day =first_day.rename(columns={"sales":"sales_rolling7"})


rolling_days_ago30 =history[(history["date"] >= "2017-07-17")& (history["date"]<= "2017-08-15")][["store_nbr","family","sales"]].copy()
rolling30 =rolling_days_ago30.groupby(["store_nbr","family"])["sales"].mean().reset_index()


first_day=first_day.merge(rolling30,on=["store_nbr","family"],how="left")
first_day =first_day.rename(columns={"sales":"sales_rolling30"})


first_day =first_day.drop(['id','date'],axis=1)

preprocessor =joblib.load("preprocessor.pkl")
first_day_processed =preprocessor.transform(first_day)
model =joblib.load("model.pkl")
predictions =model.predict(first_day_processed)

first_day_history =first_day[["store_nbr","family"]].copy()

first_day_history['date'] =pd.to_datetime("2017-08-16")
first_day_history = pd.DataFrame({"date":pd.to_datetime("2017-08-16"),
                                  "store_nbr": first_day["store_nbr"].values ,
                                  "family" :first_day["family"].values,
                                  "sales" :predictions})


history =pd.concat([history[["date","store_nbr","family","sales"]],first_day_history],ignore_index=True)
history =history.sort_values(["store_nbr","family","date"])



all_predictions =list(predictions)

for current_date in pd.date_range("2017-08-17","2017-08-31") :
    current_day =test[
    test["date"] ==current_date].copy()
    previous_day =history[history["date"] ==current_date - pd.Timedelta(days =1)][["store_nbr","family","sales"]].copy()
    current_day =current_day.merge(previous_day,on=["store_nbr","family"],how="left")
    current_day =current_day.rename(columns={"sales": "sales_lag1"})

    seven_days_ago =history[history["date"] == current_date -pd.Timedelta(days=7)][["store_nbr","family","sales"]].copy()
    current_day =current_day.merge(seven_days_ago,on=["store_nbr","family"],how="left")
    current_day =current_day.rename(columns={"sales":"sales_lag7"})

    thirty_days_ago = history[history["date"] == current_date - pd.Timedelta(days=30)][["store_nbr","family","sales"]].copy()
    current_day = current_day.merge(thirty_days_ago,on=["store_nbr","family"],how='left')
    current_day =current_day.rename(columns={"sales":"sales_lag30"})

    rolling7 =history[(history["date"] < current_date)& (history["date"] >= current_date -pd.Timedelta(days=7)
                         )].groupby(["store_nbr","family"]
                      )["sales"].mean().reset_index()
   
    rolling7 = rolling7.rename(
    columns={"sales": "sales_rolling7"}
    )

    current_day = current_day.merge(
    rolling7,
    on=["store_nbr", "family"],
    how="left"
    )

    rolling30 = history[(history["date"] < current_date)&(history["date"] >= current_date - pd.Timedelta(days=30))].groupby(
        ["store_nbr","family"]
    )["sales"].mean().reset_index()
    rolling30 = rolling30.rename(columns={"sales":"sales_rolling30"})
    current_day =current_day.merge(rolling30,on=["store_nbr","family"],how="left")



    model_data =current_day.drop(["id","date"],axis=1)
    
    
    processed =preprocessor.transform(model_data)
    
    predictions =model.predict(processed)
    predictions =predictions.clip(min=0)
    all_predictions.extend(predictions)
    new_history =pd.DataFrame({
        "date":current_date,
        "store_nbr": current_day["store_nbr"].values,
        "family" :current_day["family"].values,
        "sales": predictions
    })
    history=pd.concat([history,new_history],ignore_index=True)



subimission3=pd.DataFrame({
    "id":test["id"].values,
    "sales":all_predictions
})

subimission3.to_csv("subimission3.csv",index=False)
