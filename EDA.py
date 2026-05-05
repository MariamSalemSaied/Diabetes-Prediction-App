import streamlit as st
import pandas as pd
import numpy as np
import io
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def show():
    st.title("Exploratory Data Analysis")
    # Load the og dataset
    df_full = pd.read_csv('diabetes_binary.csv')

    #droping duplicates if any
    duplicates = df_full.duplicated().sum()
    if duplicates > 0:
        df_full = df_full.drop_duplicates()
    
    #splitting the data into train and test sets with stratification
    train_df, test_df = train_test_split(
    df_full,
    test_size=0.20,
    stratify=df_full['Diabetes_binary'],
    random_state=42
)
    #the splitted df 
    df = train_df.copy()

    #printing data
    st.markdown("Dataset")
    st.dataframe(df)


    #Dataset explanation 
    df_des = pd.DataFrame({"Feature": df.columns, "Description": ["Target feature","High blood pressure ","High cholesterol","Cholesterol check in last 5 years","Body Mass Index","Smoked at least 100 cigarettes in life","Ever had a stroke","Coronary heart disease or attack","Physical activity in past 30 days","Fruits consumed daily","Vegetables consumed daily","Heavy alcohol consumption ","healthcare coverage","Couldn't see doctor due to cost","General health (1=Excellent → 5=Poor)","Days of poor mental health (0–30)","Days of poor physical health (0–30)","Difficulty walking ","0=Female, 1=Male","Age category (1=18-24 → 13=80+)","Education level (1=None → 6=College graduate)","Income level (1=Less than $10,000 → 8=$75,000 or more)"]})
    st.header("Features Explanation")
    st.dataframe(df_des)

    st.markdown('''The features in this dataset fall into three categories:
    Binary features (0/1): HighBP, HighChol, CholCheck, Smoker, Stroke, HeartDiseaseorAttack, PhysActivity, Fruits, Veggies, HvyAlcoholConsump, AnyHealthcare, NoDocbcCost, DiffWalk, Sex
    Ordinal features: GenHlth (1–5), Age (1–13), Education (1–6), Income (1–8)
    Numerical features: BMI, MentHlth, PhysHlth ''')


    #data description
    st.header("Data description")
    st.write(df.describe())


    #info about the data
    st.header("Data information") 
    buffer = io.StringIO()
    df.info(buf=buffer)
    s = buffer.getvalue()
    st.text(s)


    #value counts for each column
    st.header("Values count for each column")
    for column in df.columns:
        st.write(f"Value counts for {column}:")
        st.write(df[column].value_counts())


    #target variable distribution
    st.header("Target variable distribution")
    fig, ax = plt.subplots(figsize=(3, 3))
    counts = df['Diabetes_binary'].value_counts()
    labels = {0.0: 'No Diabetes', 1.0: 'Diabetes'}
    wedges, texts, autotexts = ax.pie(
        counts.values,
        labels=[f"{labels[k]}\n({v:,})" for k, v in counts.items()],
        autopct='%1.1f%%',
        startangle=90,
        textprops={'fontsize': 8},
        wedgeprops=dict(edgecolor='white', linewidth=2),
        colors=['#540863', '#92487A']
    )
    #ax.set_title('Target Variable Distribution', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.figure(figsize=(4, 3))
    st.pyplot(fig)


    #outliers detection using boxplots
    st.header("Outliers detection using boxplots")  
    numerical_features = ['BMI', 'MentHlth', 'PhysHlth']
    for feature in numerical_features:
        fig, ax = plt.subplots(figsize=(4, 3))
        sns.boxplot(x=df[feature], ax=ax, color='#92487A')
        ax.set_title(f'Boxplot of {feature}', fontsize=12, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)
    

    #outliers handling
    cap_val = df['BMI'].quantile(0.99)
    df['BMI'] = df['BMI'].clip(upper=cap_val)

    st.markdown('''Based on the boxplots above, all numerical features were examined:

BMI — showed extreme values on the higher end → capped at the 99th percentile
MentHlth and PhysHlth — right-skewed but high values reflect genuine health conditions and were left unchanged as removing them would lose real patient information''')

    #correlation heatmap
    st.header("Correlation heatmap")    
    plt.figure(figsize=(12, 10))
    sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="PiYG", linewidths=0.5)
    plt.title('Correlation Heatmap', fontsize=14, fontweight='bold')    
    plt.tight_layout()
    st.pyplot(plt)


    #Features vs Target — Distribution by Class
    st.header("Features vs Target — Distribution by Class")
    all_cols = [c for c in df.columns if c != 'Diabetes_binary']

    fig, axes = plt.subplots(4, 6, figsize=(28, 20))
    axes = axes.ravel()

    for i, col in enumerate(all_cols):
        ct = df.groupby('Diabetes_binary')[col].mean()
        axes[i].bar(
            ['No Diabetes', 'Diabetes'],
            ct.values,
            color=['#540863', '#92487A'],
            edgecolor='black', width=0.5
        )
        axes[i].set_title(col, fontsize=10, fontweight='bold')
        axes[i].set_ylabel('Mean Value')

    for j in range(len(all_cols), len(axes)):
        axes[j].set_visible(False)

    plt.suptitle('Feature Mean Values by Diabetes Class',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig)

    #quick observations 
    st.header("Quick observations")
    st.markdown('''BMI shows a clear difference between diabetic and non-diabetic groups
GenHlth is strongly associated with diabetes — poorer health ratings appear more frequently in diabetic patients
Age shows a strong positive trend — older age groups have higher diabetes rates
HighBP and HighChol are notably more frequent in diabetic patients
PhysActivity, Fruits, and Veggies show lower rates among diabetic patients
Income and Education show inverse relationships — higher levels associate with lower diabetes rates
MentHlth and PhysHlth are moderately correlated with each other, suggesting some overlap in information
HvyAlcoholConsump and CholCheck show the weakest association with the target''')


    #key insights
    st.header("Key insights")
    st.markdown('''The dataset is perfectly balanced — 50% diabetic and 50% non-diabetic. No class weighting or resampling is needed.

All features are already numeric — no OneHotEncoding required. Only StandardScaler was applied in the preprocessing step.

BMI showed extreme outliers and was capped at the 99th percentile. MentHlth and PhysHlth were left unchanged as their high values reflect genuine health conditions.

GenHlth, BMI, Age, HighBP, and DiffWalk show the strongest correlations with the target variable.

MentHlth and PhysHlth are moderately correlated with each other — potential candidates for dimensionality reduction.

No NaN values were found. Duplicate rows were identified and removed to ensure data quality.

Several binary features show low association with the target — CholCheck, Fruits, Veggies, and Sex are candidates for removal in the Feature Selection step.''')

#_____________________________________________________________________________________________________________________________________
    #preprocessing-scaling
    X_train = df.drop('Diabetes_binary', axis=1)
    y_train = df['Diabetes_binary'].astype(int)

    X_test = test_df.drop('Diabetes_binary', axis=1)
    y_test = test_df['Diabetes_binary'].astype(int)

    scaler = StandardScaler()                            # z = (x - mu)/ sigma
    X_train_processed = scaler.fit_transform(X_train)
    X_test_processed  = scaler.transform(X_test)

    print(f"X_train shape: {X_train_processed.shape}")
    print(f"X_test shape:  {X_test_processed.shape}")

    X_train_processed = pd.DataFrame(
    scaler.fit_transform(X_train),
    columns=X_train.columns
)

    X_test_processed = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns
    )
#______________________________________________________________________________________________________________________________________
    

    
    
