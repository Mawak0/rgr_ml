import streamlit as st
import pandas as pd
import sys
import os
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.models_loader import ModelsLoader

st.markdown("# Предсказания мошеннических транзакций")

models_loader = ModelsLoader()
models = {
    "Decision Tree": models_loader.load_dtc(),
    "Random Forest": models_loader.load_rfc(),
    "Gradient Boosting": models_loader.load_gbc(),
    "Stacking": models_loader.load_stacking(),
    "CatBoost": models_loader.load_cbc(),
    "FCNN": models_loader.load_fcnn()
}

model = st.selectbox(
    label="Модель",
    options=models.keys(),
    index=None,
    placeholder="Выберите модель",
    key=1
)

st.markdown("## Введите данные о транзакции для получения предсказания")

distance_from_home = st.number_input(
    label="Расстояние от дома (км)",
    min_value=0.0,
    value=None,
    step=0.01,
    format="%.2f",
    placeholder="Введите расстояние от места жительства"
)

distance_from_last_transaction = st.number_input(
    label="Расстояние от последней транзакции (км)",
    min_value=0.0,
    value=None,
    step=0.01,
    format="%.2f",
    placeholder="Введите расстояние от места последней транзакции"
)

ratio_to_median_purchase_price = st.number_input(
    label="Отношение к медианной цене покупки",
    min_value=0.0,
    value=None,
    step=0.01,
    format="%.2f",
    placeholder="Введите отношение цены к медианной"
)

repeat_retailer = st.selectbox(
    label="Повторный ритейлер",
    options=[("Да", 1), ("Нет", 0)],
    index=None,
    format_func=lambda x: x[0],
    placeholder="Транзакция у того же ритейлера?"
)

used_chip = st.selectbox(
    label="Использован чип карты",
    options=[("Да", 1), ("Нет", 0)],
    index=None,
    format_func=lambda x: x[0],
    placeholder="Использовался ли чип карты?"
)

used_pin_number = st.selectbox(
    label="Использован PIN-код",
    options=[("Да", 1), ("Нет", 0)],
    index=None,
    format_func=lambda x: x[0],
    placeholder="Вводился ли PIN-код?"
)

online_order = st.selectbox(
    label="Онлайн-заказ",
    options=[("Да", 1), ("Нет", 0)],
    index=None,
    format_func=lambda x: x[0],
    placeholder="Является ли онлайн-заказом?"
)

if all(x is not None for x in [distance_from_home, distance_from_last_transaction,
                               ratio_to_median_purchase_price, repeat_retailer,
                               used_chip, used_pin_number, online_order]):


    repeat_retailer_val = repeat_retailer[1]
    used_chip_val = used_chip[1]
    used_pin_number_val = used_pin_number[1]
    online_order_val = online_order[1]

    X = np.array([
        distance_from_home,
        distance_from_last_transaction,
        ratio_to_median_purchase_price,
        repeat_retailer_val,
        used_chip_val,
        used_pin_number_val,
        online_order_val
    ])

    if model is not None:
        predict = models[model].predict(X.reshape(1, -1))

        if predict == 0:
            st.markdown(":green[Это не мошенническая транзакция]")
        elif predict == 1:
            st.markdown(":red[Это мошенническая транзакция]")

st.markdown("## Загрузка данных из csv")

model_for_csv = st.selectbox(
    label="Модель",
    options=models.keys(),
    index=None,
    placeholder="Выберите модель",
    key=2
)

try:
    uploaded_file = st.file_uploader(
        label="Загрузить csv файл",
        type="csv",
    )

    if uploaded_file is not None:
        input_df = pd.read_csv(uploaded_file)


        required_columns = [
            'distance_from_home',
            'distance_from_last_transaction',
            'ratio_to_median_purchase_price',
            'repeat_retailer',
            'used_chip',
            'used_pin_number',
            'online_order'
        ]

        if all(col in input_df.columns for col in required_columns):
            if model_for_csv is not None:
                pred = models[model_for_csv].predict(input_df[required_columns])

                output_df = input_df.copy()
                output_df["fraud_prediction"] = pred
                output_df["fraud_prediction"] = output_df["fraud_prediction"].map({
                    0: "Это не мошенничество",
                    1: "Это мошенничество"
                })



                def color_fraud(val):
                    color = 'red' if val == "Это мошенничество" else 'green'
                    return f'color: {color}'


                styled_df = output_df.style.applymap(color_fraud, subset=['fraud_prediction'])

                st.dataframe(styled_df)
        else:
            st.error("Файл не содержит всех необходимых колонок. Требуемые колонки: " + ", ".join(required_columns))
except Exception as e:
    st.error(f"Ошибка при обработке файла: {str(e)}")
