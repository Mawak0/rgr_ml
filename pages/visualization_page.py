import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import pointbiserialr


@st.cache_data
def visualize():
    FIG_SIZE = (10, 8)
    data = pd.read_csv(r"data/card_transdata_processed.csv", index_col=0)


    st.markdown("# Визуализация зависимостей")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 1. Cравнение распределений")
        fig1, ax1 = plt.subplots(figsize=FIG_SIZE)
        sns.boxplot(x='fraud', y='ratio_to_median_purchase_price', data=data, ax=ax1)
        ax1.set_title('Распределение ratio_to_median_purchase_price для мошеннических и обычных операций')
        ax1.set_xlabel('Мошенническая операция (1) или нет (0)')
        ax1.set_ylabel('Отношение к медианной цене')
        st.pyplot(fig1)

        st.markdown("#### 3. Вероятность мошенничества")
        fig3, ax3 = plt.subplots(figsize=FIG_SIZE)


        bins = np.linspace(data['ratio_to_median_purchase_price'].min(),
                           data['ratio_to_median_purchase_price'].max(),
                           20)
        data['ratio_bin'] = pd.cut(data['ratio_to_median_purchase_price'], bins=bins)
        prob_data = data.groupby('ratio_bin')['fraud'].mean().reset_index()
        prob_data['ratio_mid'] = prob_data['ratio_bin'].apply(lambda x: x.mid)


        sns.lineplot(x='ratio_mid', y='fraud', data=prob_data, ax=ax3, marker='o')
        ax3.set_title('Вероятность мошенничества в зависимости от отношения к медианной цене')
        ax3.set_xlabel('Отношение к медианной цене (середина интервала)')
        ax3.set_ylabel('Вероятность мошенничества')
        ax3.grid(True)
        st.pyplot(fig3)

    with col2:
        st.markdown("#### 2. Наложенные гистограммы")
        fig2, ax2 = plt.subplots(figsize=FIG_SIZE)
        sns.histplot(data=data, x='ratio_to_median_purchase_price', hue='fraud',
                     element='step', stat='density', common_norm=False, bins=50, ax=ax2)
        ax2.set_title('Плотность распределения для разных классов')
        ax2.set_xlabel('Отношение к медианной цене')
        ax2.set_ylabel('Плотность вероятности')
        st.pyplot(fig2)

        st.markdown("#### 4. Точечно-бисериальная корреляция")


        corr, p_value = pointbiserialr(data['ratio_to_median_purchase_price'], data['fraud'])
        corr_matrix = pd.DataFrame(
            np.array([[1, corr], [corr, 1]]),
            columns=['fraud', 'ratio_to_median_purchase_price'],
            index=['fraud', 'ratio_to_median_purchase_price']
        )

        fig4, ax4 = plt.subplots(figsize=FIG_SIZE)

        sns.heatmap(corr_matrix,
                    annot=True,
                    fmt=".2f",
                    cmap='coolwarm',
                    vmin=-1,
                    vmax=1,
                    square=True,
                    cbar_kws={"shrink": 0.8},
                    ax=ax4)


        ax4.set_title(f'Точечно-бисериальная корреляция', pad=20)
        ax4.xaxis.tick_top()
        st.pyplot(fig4)


visualize()
