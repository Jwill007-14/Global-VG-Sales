import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import kaggle
from matplotlib import rcParams
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import ConfusionMatrixDisplay
import plotly.express as px
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

df=pd.read_csv(r"C:\Users\James Wilson\OneDrive\Documents\Work\Coding\Jupyter\Lessons\Globalvg\Code\data\cleaned_gvg_data.csv")
# Set page title and icon
st.set_page_config(page_title="Game Class Dataset Explorer", page_icon= '🎮' )

#Step 1- Sidebar navigation

page = st.sidebar.selectbox("Select a Page", ["Home", "Data Overview", "Exploratory Data Analysis","Model Training and Evaluation", "About"])

#Step 2- Adding Content to Pages

if page == "Home":
    st.title("🎧 Global Video Game Sales Dataset Explorer")
    st.subheader('Enjoy this video game sales dataset explorer app')
    st.write("""
    This app provides a platform for exploring some data on Global Video Game Sales as well as the genre and on which platform the game was purchased on.
    Feel free to look through the visual representations provided in the app.
    Use the side bar to navigate through the apps pages.
    """)
    #st.image("", caption='The Starbucks logo')

elif page == "Data Overview":
    st.title("🎧 Data Overview")
    st.subheader("About the Data")
    st.write("""
    This dataset focuses on the selling of video games and the type of game it is.
    There are several ways to catagorize a video game. The most often used would be the genre of the game(). Not every game can be played on every system, so it is also important to know what system the game is being purched on as well. With that having been said lets dive into this data set on video games and hopefully you will come out at the end of this data set having learned something new.
    """)
    #st.image("",caption="Drink sizes at Starbucks") 
    st.subheader("Quick Glance at the Data")
    if st.checkbox("Show DataFrame"):
        st.dataframe(df)
    if st.checkbox("Show Shape of Data"):
        st.write(f"The dataset contains {df.shape[0]} rows and {df.shape[1]} columns.")

elif page == 'Exploratory Data Analysis':
    st.title("🕹️ Exploratory Data Analysis (EDA)")
    st.subheader("Select the type of visualization you'd like to explore:")
    eda_type = st.multiselect("Visualization Options", ['Histograms', 'Box Plots', 'Scatterplots', 'Count Plots'])
#Histograms
    if 'Histograms' in eda_type:
        st.subheader("Histograms - Visualizing Numerical Distributions")
        h_selected_col = st.selectbox("Select a numerical column for the histogram:",['Name','Platform','Year','Genre','Publisher','NA_Sales','EU_Sales','JP_Sales','Other_Sales','Global_Sales'])
        if h_selected_col:
            chart_title = f"Distribution of {h_selected_col}"
        if st.checkbox("Show by Genre"):
            st.write("""
            
            0=Action
            
            1=Sports
            
            2=Misc
            
            3=Role-Playing
            
            4=Shooter
            
            5=Adventure
            
            6=Racing
            
            7=Platform
            
            8=Simulation
            
            9=Fighting
            
            10=Strategy

            11=Puzzle
            """)
            st.plotly_chart(px.histogram(df, x=h_selected_col, color=("Genre"), title=chart_title, barmode='overlay'))
        else:
            st.plotly_chart(px.histogram(df, x=h_selected_col, title=chart_title))
#Boxplots
    if 'Box Plots' in eda_type:
        st.subheader("Box Plots - Visualizing Numerical Distributions")
        h_selected_col = st.selectbox("Select a numerical column for the Box Plot:",['Name','Platform','Year','Genre','Publisher','NA_Sales','EU_Sales','JP_Sales','Other_Sales','Global_Sales'])
        if h_selected_col:
            chart_title = f"Distribution of {h_selected_col}"
        if st.checkbox("Show by Genre"):
            st.plotly_chart(px.box(df, x=h_selected_col, color="Genre", title=chart_title, boxmode='overlay'))
        else:
            st.plotly_chart(px.histogram(df, x=h_selected_col, title=chart_title))
#Scatterplots
    if 'Scatterplots' in eda_type:
        st.subheader("Scatterplots - Visualizing Relationships")
        selected_col_x = st.selectbox("Select x-axis variable:", ['Name','Platform','Year','Genre','Publisher','NA_Sales','EU_Sales','JP_Sales','Other_Sales','Global_Sales'])
        selected_col_y = st.selectbox("Select y-axis variable:", ['Rank','Platform','Year','Genre','Publisher','NA_Sales','EU_Sales','JP_Sales','Other_Sales','Global_Sales'])
        if selected_col_x and selected_col_y:
            chart_title = f"{selected_col_x.title().replace('_', ' ')} vs. {selected_col_y.title().replace('_', ' ')}"
            st.plotly_chart(px.scatter(df, x=selected_col_x, y=selected_col_y, color='Genre', title=chart_title))
#Countplots
    if 'Count Plots' in eda_type:
        st.subheader("Count Plots - Visualizing Categorical Distributions")
        selected_col = st.selectbox("Select a categorical variable:", ['Name','Platform','Year','Genre','Publisher','NA_Sales','EU_Sales','JP_Sales','Other_Sales','Global_Sales'])
        if selected_col:
            chart_title = f'Distribution of {selected_col.title()}'
            st.plotly_chart(px.histogram(df, x=selected_col, color='Genre', title=chart_title))


# Model Training and Evaluation Page
elif page == "Model Training and Evaluation":
    st.title("🛠️ Model Training and Evaluation")
    # Sidebar for model selection
    st.sidebar.subheader("Choose a Machine Learning Model")
    model_option = st.sidebar.selectbox("Select a model", ["K-Nearest Neighbors", "Logistic Regression", "Random Forest"])

    # Prepare the data
    X=df.drop(columns= ['Name','Platform','Genre','Publisher','Year']).values
    y=df['Genre']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, stratify=y)

    # Scale the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Initialize the selected model
    if model_option == "K-Nearest Neighbors":
        k = st.sidebar.slider("Select the number of neighbors (k)", min_value=1, max_value=20, value=3)
        model = KNeighborsClassifier(n_neighbors=k)
    elif model_option == "Logistic Regression":
        model = LogisticRegression()
    else:
        model = RandomForestClassifier()

    # Train the model on the scaled data
    model.fit(X_train_scaled, y_train)

    # Display training and test accuracy
    st.write(f"**Model Selected: {model_option}**")
    st.write(f"Training Accuracy: {model.score(X_train_scaled, y_train):.2f}")
    st.write(f"Test Accuracy: {model.score(X_test_scaled, y_test):.2f}")

    # Display confusion matrix
    st.subheader("Confusion Matrix")
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay.from_estimator(model, X_test_scaled, y_test, ax=ax, cmap='Blues')
    st.pyplot(fig)

elif page == "About":
    st.title("🎧 About")
    st.write("""This is an example of a multi-page Streamlit app using a Global vif=deo game sales dataset.
    The app showcases knowledge on how each page should be edited to help show case the dataset""")