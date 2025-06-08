import streamlit as st

pages = {
    "Страницы": [
        st.Page("pages/contact_page.py", title="Информация о разработчике", icon="💁‍♂️"),
        st.Page("pages/data_page.py", title="Датасет", icon="🔢"),
        st.Page("pages/visualization_page.py", title="Визуализации", icon="📊"),
        st.Page("pages/predictions_page.py", title="Предсказание", icon="🔮")
    ]
}


pg = st.navigation(pages)
pg.run()
