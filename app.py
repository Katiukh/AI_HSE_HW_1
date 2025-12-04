# app.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import re 
import seaborn as sns
import phik


# Настройки страницы
st.set_page_config(
    page_title="Предсказание стоимости автомобилей",
    page_icon="🚗",
    layout="wide"
)

# Загрузка pickle файла 
@st.cache_resource
def load_model():
    with open('car_price_model.pkl', 'rb') as f:
        config = pickle.load(f)
    return config

def preprocess_input(data, config):
    """
    Преобразует входные данные (ручной ввод или CSV) 
    в ТОЧНО ТОТ ЖЕ ФОРМАТ, что и при обучении модели
    
    Порядок:
    1. Фиксим max_power, mileage, torque, engine, name, seats
    2. StandardScaler масштабирование вещественных признаков 
    3. OneHot кодирование категориальных признаков
    4. Создание года и пробега в квадрате 
    """
    
    # Извлекаем компоненты из конфига
    scaler = config['scaler']
    onehot_encoder = config['onehot_encoder']
    original_cat_cols = config['original_cat_columns']
    original_num_cols = config['original_num_columns']
    engineered_features = config.get('engineered_features', [])
    final_feature_names = config['final_feature_names']
    medians = config['medians']
    
    # Создаем DataFrame
    if isinstance(data, dict):
        # Ручной ввод - словарь
        df = pd.DataFrame([data])
    else:
        # DataFrame
        df = data.copy()
        
        
    
    
    # Обработка max_power 
    def fix_max_power_single(value):
        if pd.isna(value):
            return np.nan
        
        if isinstance(value, (int, float)):
            return float(value)
        
        if isinstance(value, str):
            try:
                if value.endswith(' bhp'):
                    return float(value[:-4])
                elif value == '0' or value == '0.0':
                    return 0.0
                else:
                    return float(value)
            except:
                return 0.0
        
        return 0.0
    
    if 'max_power' in df.columns:
        df['max_power'] = df['max_power'].apply(fix_max_power_single)
        df['max_power'] = pd.to_numeric(df['max_power'], errors='coerce')
    
    # Обработка mileage 
    def convert_mileage_single(mileage_str, fuel_type):
        """Конвертация mileage из km/kg в km/l"""
        if pd.isna(mileage_str):
            return np.nan
        
        gas_density = {'LPG': 0.54, 'CNG': 0.17}
        
        try:
            parts = str(mileage_str).lower().split()
            mileage_value = float(parts[0])
            
            if len(parts) > 1:
                unit_of_measure = parts[1]
                if unit_of_measure == 'km/kg':
                    if fuel_type in gas_density:
                        return mileage_value * gas_density[fuel_type]
                    else:
                        return mileage_value  
            return mileage_value
        except:
            return np.nan
    
    if 'mileage' in df.columns and 'fuel' in df.columns:
        df['mileage'] = df.apply(lambda row: convert_mileage_single(row["mileage"], row["fuel"]), axis=1)
    
    # Обработка torque (извлечение torque и max_torque_rpm)
    def extract_torque_and_rpm_single(torque_str):
        """Извлечение torque и RPM из строки"""
        if pd.isna(torque_str):
            return np.nan, np.nan
        
        torque_str = str(torque_str).lower()
        
        # Крутящий момент
        torque_value = np.nan
        torque_match = re.search(r'(\d+\.?\d*)\s*(nm|kgm)', torque_str)
        
        if not torque_match:
            torque_match = re.search(r'(\d+\.?\d*)', torque_str)
        
        if torque_match:
            torque_value = float(torque_match.group(1))
            if torque_match.lastindex >= 2 and torque_match.group(2) == 'kgm':
                torque_value = torque_value * 9.80665  # конвертация kgm в Nm
        
        # Обороты (max_torque_rpm)
        rpm_value = np.nan
        rpm_range_match = re.search(r'(\d+)\s*-\s*(\d+)\s*rpm', torque_str)
        if rpm_range_match:
            rpm_value = float(rpm_range_match.group(2).replace(',', ''))
        else:
            rpm_single_match = re.search(r'[@at]\s*(\d+[\d,]*)', torque_str)
            if rpm_single_match:
                rpm_value = float(rpm_single_match.group(1).replace(',', ''))
        
        return torque_value, rpm_value
    
    if 'torque' in df.columns:
        torque_rpm_results = df['torque'].apply(
            lambda x: pd.Series(extract_torque_and_rpm_single(x))
        )
        
        # Если в данных уже есть max_torque_rpm, оставляем его, иначе создаем
        if 'max_torque_rpm' not in df.columns:
            df[['torque', 'max_torque_rpm']] = torque_rpm_results
        else:
            df['torque'] = torque_rpm_results[0]
            df['max_torque_rpm'] = df['max_torque_rpm'].combine_first(torque_rpm_results[1])
    
    #  Обработка engine (удаление 'CC')
    def fix_engine_single(value):
        """Обработка значения engine"""
        if pd.isna(value):
            return np.nan
        
        if isinstance(value, str):
            return float(value.replace(' CC', ''))
        return float(value)
    
    if 'engine' in df.columns:
        df['engine'] = df['engine'].apply(fix_engine_single)
    
    # Обработка name (Оставляем первое слово)
    df['name'] = df['name'].str.split().str[0]
    
    # РАЗДЕЛЕНИЕ НА КАТЕГОРИАЛЬНЫЕ И ЧИСЛОВЫЕ
    # Числовые
    num_cols_to_scale = []
    for col in (original_num_cols):
        if col in df.columns:
            num_cols_to_scale.append(col)
    
    # Заполнение пропусков медианой 
    for col in num_cols_to_scale:
        df[col] = df[col].fillna(medians[col])
    
    # Обработка seats
    df['seats'] = df['seats'].astype(str)
    
    # Категориальные
    cat_cols_to_encode = []
    for col in original_cat_cols:
        if col in df.columns:
            cat_cols_to_encode.append(col)
    
    # Кодирование
    if cat_cols_to_encode:
        X_cat = df[cat_cols_to_encode].copy()
        
        for col in X_cat.columns:
            X_cat[col] = X_cat[col].astype(str)
        
        # OneHot кодирование
        X_cat_encoded = onehot_encoder.transform(X_cat)
        X_cat_df = pd.DataFrame(
            X_cat_encoded,
            columns=onehot_encoder.get_feature_names_out(cat_cols_to_encode)
        )
    else:
        X_cat_df = pd.DataFrame()
    
    # Масштабирование
    
    if num_cols_to_scale:
        X_num = df[num_cols_to_scale].copy()
        
        # Заполняем пропуски медианой 
        X_num = X_num.fillna(X_num.median())
        
        # StandardScaler
        X_num_scaled = scaler.transform(X_num)
        X_num_df = pd.DataFrame(
            X_num_scaled,
            columns=num_cols_to_scale
        )
    else:
        X_num_df = pd.DataFrame()
    
    # Объединение
    X_processed = pd.concat([X_cat_df, X_num_df], axis=1)
    
    # Feature Engineering
    
    if 'km_driven' in X_processed.columns and 'km_driven_squared' in engineered_features:
        X_processed['km_driven_squared'] = X_processed['km_driven'] ** 2
    
    if 'year' in X_processed.columns and 'year_squared' in engineered_features:
        X_processed['year_squared'] = X_processed['year'] ** 2
    
    # Убедимся что все финальные признаки есть
    X_processed = X_processed.reindex(columns=final_feature_names, fill_value=0)
    
    # Отладка 
    print(f"Обработано {X_processed.shape[1]} признаков")
    print(f"Пропущенные признаки: {[col for col in final_feature_names if col not in X_processed.columns]}")
    
    return X_processed

def plot_ridge_coefficients(model, feature_names, top_n=20):
    coefficients = model.coef_
    if len(coefficients.shape) > 1:
        coefficients = coefficients[0]
    
    coef_df = pd.DataFrame({
        'Feature': feature_names,
        'Coef': coefficients,
        'Coef Abs': np.abs(coefficients)
    }).sort_values('Coef Abs', ascending=False).head(top_n)
    
    return coef_df

def main():
    st.title("🚗 Предсказание стоимости автомобилей")
    st.markdown("---")
    
    config = load_model()
    
    st.sidebar.title("Навигация")
    page = st.sidebar.radio(
        "Выберите раздел:",
        ["🏠 Главная", "📝 Ручной ввод", "📁 Загрузить CSV", "📊 Визуализация весов модели", "📈 EDA"] 
    )
    
    if page == "🏠 Главная":
        st.info("**Информация о модели:**")
        st.write(f"Тип модели: {config['model_type']}")
        st.write(f"Alpha: {config['model'].alpha:.2f}")
        st.write(f"Количество признаков после препроцессинга: {len(config['final_feature_names'])}")
        all_features = list(config['original_cat_columns']) + list(config['original_num_columns'])
        features_str = ", ".join(all_features)
        st.write(f"Изначальные признаки: {features_str}")
        st.write("Процедуры препроцессинга, примененные к данным: удаление дубликатов, заполнение пропусков медианой, масштабирование, кодирование категориальных признаков, добавление года и пробега в квадрате")
            
        # Страница ручного ввода
    elif page == "📝 Ручной ввод":
        st.header("📝 Ручной ввод параметров автомобиля")
        
        # Создаем вкладки для разных групп признаков
        tab_num, tab_cat, tab_other = st.tabs(["🔢 Числовые признаки", "🏷️ Категориальные признаки", "📊 Результат"])
        
        with tab_num:
            st.subheader("Числовые характеристики")
            
            col1, col2 = st.columns(2)
            
            with col1:
                year = st.number_input("Год выпуска", 
                                      min_value=1983, max_value=2020, value=2018,
                                      help="Год выпуска автомобиля")
                
                km_driven = st.number_input("Пробег (км)",  value=50000,
                                           help="Общий пробег автомобиля")
                
                engine = st.number_input("Объем двигателя (CC)", value=2000,
                                        help="Объем двигателя в кубических сантиметрах")
                
                max_power = st.text_input("Мощность (bhp)", value=150,
                                         help="Brake Horsepower — это эффективная мощность на маховике с учётом всех потерь на трение внутри двигателя")
            
            with col2:
                mileage = st.text_input("Расход топлива", "15.0 kmpl",
                                       help="Пример: '15.0 kmpl' или '25.0 km/kg'")
                
                torque = st.text_input("Крутящий момент (Нм)", value=200,
                                      help="Крутящий момент в автомобиле — это сила, с которой двигатель передаёт вращение на коленчатый вал")

                
                # Если есть max_torque_rpm в исходных данных
                if 'max_torque_rpm' in config.get('original_num_columns', []):
                    max_torque_rpm = st.number_input("Обороты макс. момента", value=3000, help="Обороты максимального момента — это частота вращения двигателя, при которой достигается максимальный крутящий момент")
        
        with tab_cat:
            st.subheader("Категориальные характеристики")
            
            cat_features = config.get('original_cat_columns', [])
            
            category_options = {
                'fuel': ["Petrol", "Diesel", "CNG", "LPG"],
                'seller_type': ["Individual", "Dealer", "Trustmark Dealer"],
                'transmission': ["Manual", "Automatic"],
                'owner': ["First Owner", "Second Owner", "Third Owner","Fourth & Above Owner", "Test Drive Car"],
                'seats': ['2','3','4','5','6','7','8','9','10','11','12','13','14']
            }
            
            cat_inputs = {}
            
            # Разделяем на две группы
            other_features = [f for f in cat_features if f != 'name']
            
            # Поле для name (если есть)
            if 'name' in cat_features:
                name = st.text_input("Название автомобиля", value="Maruti Swift Dzire VDI")
                cat_inputs['name'] = name
            
            # Остальные признаки
            for feature in other_features:
                if feature in category_options:
                    cat_inputs[feature] = st.selectbox(
                        feature.replace('_', ' ').title(),
                        category_options[feature]
                    )
                else:
                    cat_inputs[feature] = st.text_input(
                        feature.replace('_', ' ').title(),
                        value=""
                    )
        
        with tab_other:
            st.subheader("Предсказание цены")
            
            # Кнопка для предсказания
            if st.button("🎯 Предсказать стоимость", type="primary", use_container_width=True):
                # Собираем все данные
                car_data = {
                    'year': year,
                    'km_driven': km_driven,
                    'engine': engine,
                    'max_power': max_power,
                    'mileage': mileage,
                    'torque': torque,
                }
                
                # Добавляем категориальные признаки
                car_data.update(cat_inputs)
                
                # Добавляем max_torque_rpm если он есть
                if 'max_torque_rpm' in config.get('original_num_columns', []):
                    car_data['max_torque_rpm'] = max_torque_rpm
                
                try:
                    # Предобработка
                    with st.spinner("Обработка данных..."):
                        X_processed = preprocess_input(car_data, config)
                    
                    # Предсказание
                    with st.spinner("Выполнение предсказания..."):
                        model = config['model']
                        price = model.predict(X_processed)[0]
                    
                    # Отображение результата
                    st.success(f"### 🎉 Предсказанная стоимость: **{price:,.0f} руб**")
                    
                    
                
                except Exception as e:
                    st.error(f"Ошибка при предсказании: {str(e)}")
                    st.info("Проверьте, что все поля заполнены корректно")
    
    # Страница загрузки CSV
    elif page == "📁 Загрузить CSV":
        st.header("📁 Загрузка CSV файла")
        
        st.subheader("Необходимые колонки:")
        # Показываем необходимые колонки
        req_cols = list(config.get('original_num_columns', [])) + list(config.get('original_cat_columns', []))
        req_cols.remove('max_torque_rpm')
        for i, col in enumerate(req_cols, 1):
            st.write(f"{i}. `{col}`")
        
        # Загрузка файла
        uploaded_file = st.file_uploader(
            "Выберите CSV файл",
            type=["csv"],
            help="Загрузите файл с данными автомобилей"
        )
        
        if uploaded_file:
            try:
                # Загружаем данные
                df = pd.read_csv(uploaded_file)
                
                if 'selling_price' in df.columns:
                    df = df.drop('selling_price', axis=1)
                    
                st.success(f"✅ Файл загружен успешно")
                st.write(f"**Записей:** {len(df)}")
                st.write(f"**Колонок:** {len(df.columns)}")
                
                # Предпросмотр данных
                with st.expander("📋 Предпросмотр загруженных данных (первые 5 строк)"):
                    st.dataframe(df.head())
                
                # Проверка колонок
                missing_cols = [col for col in req_cols if col not in df.columns]
                
                if missing_cols:
                    st.error(f"❌ Отсутствуют необходимые колонки: {missing_cols}")
                else:
                    # Кнопка для обработки
                    if st.button("🚀 Обработать и предсказать", type="primary"):
                        with st.spinner("Обработка данных..."):
                            try:
                                X_processed = preprocess_input(df, config)
                                
                                with st.spinner("Выполнение предсказаний..."):
                                    model = config['model']
                                    predictions = model.predict(X_processed)
                                
                                # Добавляем предсказания к данным
                                df_result = df.copy()
                                df_result['predicted_price'] = predictions
                                
                                st.success(f"✅ Предсказания выполнены для {len(predictions)} автомобилей")
                                
                                # Визуализация результатов
                                st.subheader("📈 Распределение предсказанных цен")
                                
                                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
                                
                                # Гистограмма
                                ax1.hist(predictions, bins=30, edgecolor='black', alpha=0.7, color='skyblue')
                                ax1.set_xlabel("Цена, руб")
                                ax1.set_ylabel("Количество")
                                ax1.set_title("Распределение цен")
                                ax1.grid(True, alpha=0.3)
                                
                                # Box plot
                                ax2.boxplot(predictions, vert=False)
                                ax2.set_xlabel("Цена, руб")
                                ax2.set_title("Box plot цен")
                                ax2.grid(True, alpha=0.3)
                                
                                st.pyplot(fig)
                                
                                # Скачивание результатов
                                st.subheader("💾 Скачать результаты")
                                
                                csv_result = df_result.to_csv(index=False).encode('utf-8')
                                
                               
                                st.download_button(
                                        label="📥 Скачать CSV с предсказаниями",
                                        data=csv_result,
                                        file_name="car_price_predictions.csv",
                                        mime="text/csv",
                                        use_container_width=True
                                    )
                                
                                
                                # Таблица с результатами
                                with st.expander("📋 Показать все предсказания"):
                                    st.dataframe(df_result)
                                
                            except Exception as e:
                                st.error(f"Ошибка при обработке: {str(e)}")
                                st.info("Проверьте формат данных в CSV файле")
                
            except Exception as e:
                st.error(f"Ошибка при чтении файла: {str(e)}")
    
    # Страница визуализации весов модели
    elif page == "📊 Визуализация весов модели":
        st.header("📊 Визуализация коэффициентов Ridge модели")
        
        model = config['model']
        feature_names = config['final_feature_names']
        
        st.subheader("⚙️ Настройки отображения")
        
        col_set1, col_set2 = st.columns(2)
        
        with col_set1:
            top_n = st.slider("Количество признаков", 5, 56, 20)
        
        with col_set2:
            sort_by = st.selectbox("Сортировка", 
                                  ["По абсолютному значению по убыванию", "По убыванию", "По возрастанию", "По абсолютному значению по возрастанию"])
        
        # Получаем коэффициенты
        coef_df = plot_ridge_coefficients(model, feature_names, top_n=top_n)
        
        if sort_by == "По убыванию":
            coef_df = coef_df.sort_values('Coef', ascending=False)
        elif sort_by == "По возрастанию":
            coef_df = coef_df.sort_values('Coef', ascending=True)
        elif sort_by == "По абсолютному значению по убыванию":
            coef_df = coef_df.sort_values('Coef Abs', ascending=False)
        else:
            coef_df = coef_df.sort_values('Coef Abs', ascending=True)
        
        # Визуализация
        st.subheader("График коэффициентов")
        fig, ax = plt.subplots(figsize=(12, 8))
        colors = ['green' if x > 0 else 'red' for x in coef_df['Coef']]
        bars = ax.barh(coef_df['Feature'], coef_df['Coef'], color=colors, alpha=0.7)
        ax.axvline(x=0, color='black', linewidth=0.5)
        ax.set_xlabel('Значение коэффициента')
        ax.set_title('Коэффициенты Ridge модели')
        ax.invert_yaxis()
        ax.grid(True, axis='x', alpha=0.3)

        st.pyplot(fig)

        
    # Страница EDA графиков
    elif page == "📈 EDA":
        st.header("Основные графики разведочного анализа данных")
        df_EDA = pd.read_csv("df_train.csv")
        num_cols = df_EDA.select_dtypes(include=["int64", "float64"]).columns.to_list()
        cat_cols = df_EDA.select_dtypes(include=["object"]).columns.to_list()
        
        # Создаем вкладки для разных типов графиков
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Распределения", "📈 Зависимости", "📉 Регрессии", "🔥 Корреляция"])
        
        with tab1:
            # В разделе EDA, например в "Распределения":
            st.subheader("Гистограммы распределения числовых переменных")
            
            # Создаем сетку 3x3
            fig, axes = plt.subplots(3, 3, figsize=(15, 12))
            axes = axes.flatten()
            
            for i, column in enumerate(num_cols):
                if i < len(axes):  # Проверка, чтобы не выйти за границы
                    sns.histplot(data=df_EDA, x=column, ax=axes[i], kde=True, bins=10)
                    axes[i].set_title(f'Распределение {column}', fontweight='bold')
                    axes[i].grid(True, alpha=0.3)
            
            
            plt.tight_layout()
            st.pyplot(fig)
        
        with tab2:
            st.subheader("Диаграмма рассеяния")
    
            col1, col2 = st.columns(2)
            
            with col1:
                x_var = st.selectbox("Выберите переменную для оси X:", num_cols)
            
            with col2:
                y_var = st.selectbox("Выберите переменную для оси Y:", 
                                    [col for col in num_cols if col != x_var])
            
            fig, ax = plt.subplots(figsize=(8, 5))
            
            # Рисуем точки
            ax.scatter(df_EDA[x_var], df_EDA[y_var], 
                    alpha=0.5, s=30, color='blue')
            
            ax.set_xlabel(x_var)
            ax.set_ylabel(y_var)
            ax.set_title(f'{x_var} vs {y_var}', fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            st.pyplot(fig, use_container_width=False)
            
            
        
        with tab3:
            st.subheader("Линейные регрессии по категориям")
        
            # Выбираем категориальный признак для разбивки
            cat_for_split = st.selectbox(
                "Выберите категориальный признак для разбивки на графики:",
                cat_cols[1:]
            )
            
            # Выбираем числовой признак для оси X
            num_for_x = st.selectbox(
                "Выберите числовой признак для оси X:",
                [col for col in num_cols if col != 'selling_price']
            )
            

            # Кнопка для построения
            if st.button("📊 Построить графики", type="primary"):
                with st.spinner("Создание графиков..."):
                    # Создаем lmplot
                    g = sns.lmplot(
                        data=df_EDA,
                        x=num_for_x,
                        y="selling_price",
                        col=cat_for_split,
                        hue=cat_for_split,
                        facet_kws={
                            'sharey': True,
                            'sharex': False,
                            'legend_out': True
                        },
                        scatter=True,
                        fit_reg=True,
                        line_kws={'color': 'crimson', 'lw': 2},
                        palette="Purples",
                        height=4,  # высота каждого subplot
                        aspect=1.2  # соотношение сторон
                    )
                    
                    # Настраиваем заголовки
                    g.set_titles("{col_name}")  # название для каждого subplot
                    g.fig.suptitle(
                        f'Зависимость selling_price от {num_for_x} по категориям {cat_for_split}',
                        y=1.05,
                        fontsize=14,
                        fontweight='bold'
                    )
                    
                    # Настройка осей
                    g.set_axis_labels(num_for_x, "selling_price")
                    
                    # Автоматическая подгонка layout
                    plt.tight_layout()
                    
                    # Показываем в Streamlit
                    st.pyplot(g.fig)
                
        with tab4:
            st.subheader("Корреляция")
            
            # Выбор типа корреляции
            corr_type = st.selectbox("Выберите тип:", ["Pearson", "phik"])
            
            if corr_type == "Pearson":
                num_df = df_EDA[num_cols]
                corr_matrix = num_df.corr()
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                        vmin=-1, vmax=1, center=0, ax=ax)
                st.pyplot(fig, use_container_width=False)
            else:
                phik_matrix = df_EDA.phik_matrix()
                fig, ax = plt.subplots(figsize=(12, 10))
                sns.heatmap(phik_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                        vmin=0, vmax=1, ax=ax)
                st.pyplot(fig, use_container_width=False)
                
# Запуск приложения
if __name__ == "__main__":
    main()
    
    