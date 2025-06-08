import streamlit as st
import pandas as pd

@st.cache_data
def show_data():
    original_dataset = pd.read_csv(r"data/card_transdata_orginal.csv")
    processed_dataset = pd.read_csv(r"data/card_transdata_processed.csv", index_col=0)

    dataset_info = """

    В изначальном датасете присутствуют следующие признаки
    - distance_from_home - расстояние от дома, где произошла транзакция.
    - distance_from_last_transaction - расстояние от места последней сделки.
    - ratio_to_median_purchase_price - отношение цены покупки транзакции к медианной цене покупки.
    - repeat_retailer - транзакция совершена у того же ритейлера.
    - used_chip - Проведена ли транзакция с использованием чипа (кредитной карты).
    - used_pin_number - Транзакция совершена с использованием PIN-кода.
    - online_order - Является ли транзакция онлайн-заказом.
    - fraud - Является ли транзакция мошеннической.
    """

    processing_info = """
    Во время обработки датасета признаки типа float были приведены к типу int, 
    так как после запятой у всех признаков шел 0. Даннные масштабированны.

    Также в данных присутствовали пропущенные значения, но их было немного, поэтому такие объекты были удалены. Дубликаты были тоже удалены.
    В свою очередь выбросов было немного, они были удалены методом 3-х сигм. 

    Были построены частотные графики признаков их масштаб был приведен логарифмическому. 
    На наборе данных сильной корреляции между признаками не наблюдалось.
    Была небольшая корреляция между fraud и ratio_to_median_purchase_price.

    """

    st.title("Информация о датасете")

    st.markdown("## Датасет до обработки")
    st.dataframe(original_dataset)

    st.markdown(dataset_info)
    st.markdown(processing_info)

    st.markdown("## Датасет после обработки")
    st.dataframe(processed_dataset)

show_data()
