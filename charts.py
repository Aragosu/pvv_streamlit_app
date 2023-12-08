import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd

#============================= 0. Чтение + небольшая очистка

# удаление ненужных столбцов
full_data = pd.read_csv('full_data.csv').drop(columns=['ID','REG_ADDRESS_PROVINCE','POSTAL_ADDRESS_PROVINCE'])

# маска для имен столбцов на русском
desd = {'clmns': list(full_data.columns),
        'desc':['Возраст',
                'Пол',
                'Образование',
                'Семейное положение',
                'Кол-во детей',
                'Кол-во иждевенцев',
                'Статус работы',
                'Статус пенсии',
                'Регион фактического пребывания',
                'Наличие квартиры',
                'Кол-во автомобилей',
                'Целевая - отклик_число',
                'Отрасль работы',
                'Должность',
                'Направление деятельности',
                'Время работы на текущем месте',
                'Семейный доход',
                'Личный доход',
                'Сумма последнего кредита',
                'Срок кредита',
                'Первоначальный взнос',
                'Кол-во кредитов',
                'Кол-во закрытых кредитов',
                'Целевая - отклик'
               ]}
df_desc = pd.DataFrame(desd)

# разбивка столбцов на категории для виджетов
digit_columns = full_data.select_dtypes(include=[np.int64, np.float64]).columns
specify_columns = ['GEN_INDUSTRY','FACT_ADDRESS_PROVINCE']
alpha_columns = [i for i in full_data.select_dtypes(include=[object]).columns if i not in specify_columns]



#============================= 2. Работа с выбросами
# функция для очистки выбросов (для некоторых столбцов)
def del_out_col(col):
    Q1 = col.quantile(0.25)
    Q3 = col.quantile(0.75)
    IQR = Q3 - Q1
    filtered_col = col[(col >= Q1 - 1.5*IQR) & (col <= Q3 + 1.5*IQR)]
    return filtered_col

outliners_column = ['WORK_TIME']

#============================= 3. Функции для виджетов
def dist_bar(data):
    if data.nunique() > 20:
        nbin = 15
    else:
        nbin = data.nunique()+1
    name = data.name
    name_col = df_desc[df_desc.clmns == name].desc.values
    numbers = list(data)
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=numbers,
                               nbinsx=nbin,
                               name="count",
                               marker_color='#5555ff',
                               texttemplate='%{y}',
                               textfont_size=20,
                               textfont_color='#c8ff00'))
    fig.update_layout(title=f'{name_col[0]}',
                      xaxis_title=f'{name_col[0]}',
                      yaxis_title='Кол-во клиентов')
    return fig

def dist_bar_y(data):
    if data.nunique() > 20:
        nbin = 15
    else:
        nbin = data.nunique()+1
    name = data.name
    name_col = df_desc[df_desc.clmns == name].desc.values
    numbers = list(data)
    fig = go.Figure()
    fig.add_trace(go.Histogram(y=numbers,
                               nbinsx=nbin,
                               name="count",
                               marker_color='#5555ff',
                               texttemplate='%{x}',
                               textfont_size=20,
                               textfont_color='#c8ff00'))
    fig.update_layout(title=f'{name_col[0]}',
                      xaxis_title=f'{name_col[0]}',
                      yaxis_title='Кол-во клиентов',
                      width=500,
                      height=1000
                      )
    return fig

def box_plot(data):
    name = data.name
    x0 = data.values
    fig = go.Figure()
    fig.add_trace(go.Box(x=x0,name=name))
    return fig

def stat_data(data):
    data_dict = {'Минимум':[round(min(data),1)],
                 '25-проц':[round(np.percentile(data, 25),1)],
                 '50-проц(медиана)': [round(data.median(),1)],
                 '75-проц':[round(np.percentile(data, 75),1)],
                 'Максимум':[round(max(data),1)],
                 'Среднее':[round(data.mean(),1)]
                }
    res = pd.DataFrame.from_dict(data_dict)
    return res

def pie_chart(data):
    name = data.name
    name_col = df_desc[df_desc.clmns == name].desc.values
    data = data.value_counts()
    sunflowers_colors = ['#fb5607',
                         '#ffbe0b',
                         '#ff006e',
                         '#8338ec',
                         '#8338ec',
                         '#3a86ff']
    fig = go.Figure()
    fig.add_trace(go.Pie(labels=data.index,
                         values=data.values,
                         marker_colors=sunflowers_colors))
    fig.update_layout(title_text=name_col[0])
    fig.update_traces(hole=.4,
                      textposition='outside',
                      textinfo='percent+label')
    return fig

def corr_bar(data):
    df_corr = data.corr()
    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(
            x = df_corr.columns,
            y = df_corr.index,
            z = np.array(df_corr),
            colorscale='Inferno',
            text=df_corr.values,
            texttemplate='%{text:.2f}'
        )
    )
    return fig

def hist_target(col):
    df = full_data
    name_col = df_desc[df_desc.clmns == col].desc.values
    fig = px.histogram(df, x=col, color="TARGET_text").update_xaxes(categoryorder='total descending')
    fig.update_layout(title=f'Зависимость признака "{name_col[0]}" от целевой',
                          xaxis_title=f'Признак "{name_col[0]}"',
                          yaxis_title='Кол-во клиентов')
    return fig


#===================================== 4. Верстка дэша
st.write('''
# EDA по клиентам банка
''')
st.warning("В данных не были обработаны пропуски, т.к. это не требовалось по заданию")


# оч хотел все по вкладкам)
tab1, tab2, tab3 = st.tabs(['Графики распределения + статистика',
                            'Матрица корреляций',
                            'График зависимости целевой'])
with tab1:
    option = st.selectbox(
        'Графики распределения каждого признака',
        (list(full_data.columns)),
        key = "1")
    if option in digit_columns:
        if option in outliners_column:
            st.info(f"В графикe по {option} не были учтены выбросы")
            st.plotly_chart(dist_bar(del_out_col(full_data[option])), use_container_width=True)
        else:
            st.plotly_chart(dist_bar(full_data[option]), use_container_width=True)
        st.plotly_chart(box_plot(full_data[option]), use_container_width=True)
        st.dataframe(stat_data(full_data[option]))
    elif option in specify_columns:
        st.plotly_chart(dist_bar_y(full_data[option]), use_container_width=True)
    else:
        st.plotly_chart(pie_chart(full_data[option]), use_container_width=True)
with tab2:
    st.info("Матрица корреляции составлена только по числовым признакам")
    st.plotly_chart(corr_bar(full_data[digit_columns]), use_container_width=True)
    st.table(df_desc[df_desc.clmns.isin(digit_columns.values)])
with tab3:
    option2 = st.selectbox(
        'Графики распределения каждого признака',
        (list(full_data.columns)),
        key = "2")
    st.plotly_chart(hist_target(option2), use_container_width=True)



st.markdown('Автор: Панфиленко В.В.')
