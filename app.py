import streamlit as st
import numpy as np
import pandas as pd
import pickle


#DATA PREPROCESSING
def preprocessing(user_input):
   
    if user_input is None or user_input.empty:
        st.write("ERROR: User input is empty or None!")

    cat_features = user_input.select_dtypes(include=['object'])
    num_features = user_input.select_dtypes(exclude=['object'])

    cat_features = pd.get_dummies(cat_features, columns=cat_features.columns, dummy_na=True, dtype='float64')

    test = pd.concat([num_features, cat_features], axis=1)

    test = test.loc[:, ~test.columns.duplicated()]

    return test
    


#MODEL LOADING AND CLASSIFICATION
def model_classification(test):
    pickle_in = open("xgboost_model.pkl", "rb")
    Classifier = pickle.load(pickle_in)

    expected_features = Classifier.get_booster().feature_names
    test = test.reindex(columns=expected_features, fill_value=0)

    result = Classifier.predict(test)

    return result





#Sidebar
st.sidebar.markdown(
    """
    <style>
        div[role="radiogroup"] > label {
            font-size: 24px !important;   
            margin-bottom: 15px !important; 
        }
    </style>
    """, 
    unsafe_allow_html=True
)
st.sidebar.header("Binary Mushroom Classification")
page = st.sidebar.radio("",["Home","Mushroom Classifier"])



#HOME PAGE
if(page == "Home"):
    
    st.markdown("<h2 style='font-weight: bold;'>BINARY MUSHROOM CLASSIFICATION</h1>", unsafe_allow_html=True)
    image_path = "./Background.jpg"
    st.image(image_path, width=600)
    st.subheader("Project Overview")
    st.write("""
    This project showcases a machine learning model trained to identify whether a mushroom is edible or poisonous based on its physical characteristics. 
    The model utilizes **31M+ mushroom dataset from Kaggle**, which includes key features like color, size, and shape, to make accurate predictions. 
    Trained using **XGBoost**, the model achieves an impressive **98.4% accuracy**.

    As a new learner in **ML/AI**, I built this project to explore how machine learning can be applied to real-world problems.
    """)

    # Key Features section
    st.subheader("Key Features:")
    st.markdown("""
    - **Predicts Edibility/Poisonous:** this tool determines if a mushroom is edible or poisonous based on its physical features.
    - **User-Friendly Interface:** The tool is easy to use, even for beginners interested in machine learning.
    """)

    
    st.subheader("Model Training")
    st.write("""
    The model has been trained using [dataset](https://www.kaggle.com/competitions/playground-series-s4e8/data) from Kaggle. After preprocessing the data, the **XGBoost** algorithm was used to train the model, which achieved an accuracy of **98.4%**.
    
    Model Training - [Github Repository](https://github.com/RadhapyariDevi/Binary-Mushroom-Classification/blob/main/mushroom%20_classification_model.ipynb)
    """)

   
    st.subheader("How to Use:")
    st.write("""
    Try out different features and see how the model makes its predictions. 
    <span style="color:red; font-weight:bold;">This is designed for educational purposes only, it should not be used for real-life consumption of mushrooms!</span>  
    """)






#mUSHROOM CLASSIFIER PAGE
if (page == "Mushroom Classifier"):
    st.header("🔍 Enter Mushroom Features")

    
    #FEATURES TO BE COLLECTED FOR CLASSIFICATION
    cap_shape_mapping = {
        'Flat (f)': 'f',
        'Convex (x)': 'x',
        'Conical (p)': 'p',
        'Bell (b)': 'b',
        'Umbonate (o)': 'o',
        'conical (c)': 'c',
        'Sunken (s)': 's',
        'Other': 'other',
        'None': 'nan'
    }

    cap_surface_mapping = {
        'Smooth (s)': 's',
        'Homogeneous (h)': 'h',
        'Scaly (y)': 'y',
        'Fibrous (l)': 'l',
        'tomentose-fuzzy (t)': 't',
        'Wrinkled (e)': 'e',
        'Glistening (g)': 'g',
        'None': 'nan',
        'Dimpled (d)': 'd',
        'Sticky (i)': 'i',
        'Woolly (w)': 'w',
        'Dry (k)': 'k',
        'Other (other)': 'other',
        'Flaky (f)': 'f',
        'Not applicable (n)': 'n',
    }

    cap_color_mapping = {
        'Unknown (u)': 'u',
        'Orange (o)': 'o',
        'Brown (b)': 'b',
        'Green (g)': 'g',
        'White (w)': 'w',
        'Brownish (n)': 'n',
        'Gray (e)': 'e',
        'Yellow (y)': 'y',
        'Red (r)': 'r',
        'Pink (p)': 'p',
        'Black (k)': 'k',
        'Light Brown (l)': 'l',
        'Other (other)': 'other',
        'None': 'nan',
    }

    does_bruise_or_bleed_mapping = {
        'No (f)': 'f',
        'Yes (t)': 't',
        'Other': 'other',
        'None': 'nan',
    }

    gill_attachment_mapping = {
        'Attached (a)': 'a',
        'Free (x)': 'x',
        'Sinuate (s)': 's',
        'Descending (d)': 'd',
        'Emarginate (e)': 'e',
        'None': 'nan',
        'Forked (f)': 'f',
        'Palmate (p)': 'p',
        'Other': 'other',
        'Cleft (c)': 'c',
    }

    gill_spacing_mapping = {
        'Close (c)': 'c',
        'None': 'nan',
        'Distant (d)': 'd',
        'Forked (f)': 'f',
        'Other ': 'other',
    }

    gill_color_mapping = {
        'White (w)': 'w',
        'Brown (n)': 'n',
        'Gray (g)': 'g',
        'Black (k)': 'k',
        'Yellow (y)': 'y',
        'Buff (f)': 'f',
        'Pink (p)': 'p',
        'Orange (o)': 'o',
        'Blue (b)': 'b',
        'Purple (u)': 'u',
        'Green(e)': 'e',
        'Red (r)': 'r',
        'Other ': 'other',
        'None': 'nan',
    }

    stem_surface_mapping = {
        'Scaly (y)': 'y',
        'Smooth (s)': 's',
        'Fibrillose (t)': 't',
        'Glabrous (g)': 'g',
        'Hollow (h)': 'h',
        'Wrinkled (k)': 'k',
        'Interwoven (i)': 'i',
        'Flaky (f)': 'f',
        'Other (other)': 'other',
        'Missing (nan)': 'nan',
    }

    stem_color_mapping = {
        'White (w)': 'w',
        'Orange (o)': 'o',
        'Brown (n)': 'n',
        'Yellow (y)': 'y',
        'Green (e)': 'e',
        'Purple (u)': 'u',
        'Pink (p)': 'p',
        'Black (f)': 'f',
        'Gray (g)': 'g',
        'Red (r)': 'r',
        'Light Brown (l)': 'l',
        'Blue (b)': 'b',
        'Other': 'other',
        'None': 'nan',
    }

    has_ring_mapping = {
        'No (f)': 'f',
        'Yes (t)': 't',
        'Other (other)': 'other',
        'Missing (nan)': 'nan',
    }

    ring_type_mapping = {
        'Flared (f)': 'f',
        'Zone (z)': 'z',
        'Escutcheon (e)': 'e',
        'Missing (nan)': 'nan',
        'Pendant (p)': 'p',
        'Ladder (l)': 'l',
        'Grooved (g)': 'g',
        'Ring (r)': 'r',
        'Membranous (m)': 'm',
        'Other (other)': 'other',
        'Torn (t)': 't',
    }

    habitat_mapping = {
        'Woods (d)': 'd',
        'Leaves (l)': 'l',
        'Grasses (g)': 'g',
        'Hills (h)': 'h',
        'Pastures (p)': 'p',
        'Meadows (m)': 'm',
        'Urban (u)': 'u',
        'Waste (w)': 'w',
        'Other': 'other',
        'None': 'nan',
    }

    season_mapping = {
        'Autumn (a)': 'a',
        'Winter (w)': 'w',
        'Summer (u)': 'u',
        'Spring (s)': 's',
    }

    #INPUTS FROM USER
    with st.form("mushroom_form"): 
        cap_diameter = st.number_input("Cap Diameter (cm)",value=None, step=None, format="%.2f")
        stem_height = st.number_input("Stem Height (cm)",value=None, step=None, format="%.2f")
        stem_width = st.number_input("Stem Width (cm)",value=None, step=None, format="%.2f")
        cap_shape = st.selectbox("Cap Shape", list(cap_shape_mapping.keys()))
        cap_surface = st.selectbox("Cap Surface", list(cap_surface_mapping.keys()))
        cap_color = st.selectbox("Cap Color", list(cap_color_mapping.keys()))
        does_bruise_or_bleed = st.selectbox("Does Bruise or Bleed", list(does_bruise_or_bleed_mapping.keys()))
        gill_attachment = st.selectbox("Gill Attachment", list(gill_attachment_mapping.keys()))
        gill_spacing = st.selectbox("Gill Spacing", list(gill_spacing_mapping.keys()))
        gill_color = st.selectbox("Gill Color", list(gill_color_mapping.keys()))
        stem_surface = st.selectbox("Stem Surface", list(stem_surface_mapping.keys()))
        stem_color = st.selectbox("Stem Color", list(stem_color_mapping.keys()))
        has_ring = st.selectbox("Has Ring", list(has_ring_mapping.keys()))
        ring_type = st.selectbox("Ring Type", list(ring_type_mapping.keys()))
        habitat = st.selectbox("Habitat", list(habitat_mapping.keys()))
        season = st.selectbox("Season", list(season_mapping.keys()))

    
        submit = st.form_submit_button("Classify Mushroom")

    #CLASSIFICATION BEGINS..
    if submit:
        # Creating DataFrame
        user_input = pd.DataFrame({
            'cap-diameter': [cap_diameter],
            'stem-height': [stem_height],
            'stem-width': [stem_width],
            'cap-shape': [cap_shape_mapping[cap_shape]],
            'cap-surface': [cap_surface_mapping[cap_surface]],
            'cap-color': [cap_color_mapping[cap_color]],
            'does-bruise-or-bleed': [does_bruise_or_bleed_mapping[does_bruise_or_bleed]],
            'gill-attachment': [gill_attachment_mapping[gill_attachment]],
            'gill-spacing': [gill_spacing_mapping[gill_spacing]],
            'gill-color': [gill_color_mapping[gill_color]],
            'stem-surface': [stem_surface_mapping[stem_surface]],
            'stem-color': [stem_color_mapping[stem_color]],
            'has-ring': [has_ring_mapping[has_ring]],
            'ring-type': [ring_type_mapping[ring_type]],
            'habitat': [habitat_mapping[habitat]],
            'season': [season_mapping[season]]
        })

        #PASSING USER INPUT FOR PREPROCESSING
        test = preprocessing(user_input)
        
        #CLASSIFICATION
        result = model_classification(test)
        
        # FINAL RESULT
        if result[0] == 1:
            st.markdown("""
                <div style="background-color: #ffcccb; padding: 15px; border-radius: 10px; font-size: 25px; color: red;">
                    ❌ <b>The mushroom is Poisonous!</b>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
                <div style="background-color: #d4edda; padding: 15px; border-radius: 10px; font-size: 25px; color: green;">
                    ✅ <b>The mushroom is Edible!</b>
                </div>
            """, unsafe_allow_html=True)
        
        st.warning('This model is for educational project only and should not be used for real-life consumption decisions.', icon="⚠️")

        
        


    
