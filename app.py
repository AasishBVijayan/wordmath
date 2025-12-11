import streamlit as st
import gensim.downloader as api
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.decomposition import PCA

# --- 1. CONFIG & CACHING ---
st.set_page_config(page_title="WordMath: Vector Algebra", layout="wide")

st.title("🧮 WordMath: The Algebra of Language")
st.markdown("""
This app demonstrates **Word2Vec** operations. 
Models learn that *King* is to *Man* as *Queen* is to *Woman*.
We use **GloVe-50** (50-dimensional vectors) for this demo.
""")

# Load the model only once (cached) to save memory/time
@st.cache_resource
def load_model():
    with st.spinner("Downloading language model (approx 60MB)... this happens only once!"):
        model = api.load("glove-wiki-gigaword-50")
    return model

try:
    model = load_model()
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# --- 2. SIDEBAR: THE CALCULATOR ---
st.sidebar.header("🧮 Vector Calculator")
st.sidebar.write("Formula: **Positive - Negative = Result**")

# Inputs
pos1 = st.sidebar.text_input("Positive Word 1 (+)", "king").lower().strip()
pos2 = st.sidebar.text_input("Positive Word 2 (+)", "woman").lower().strip()
neg1 = st.sidebar.text_input("Negative Word (-)", "man").lower().strip()

solve_btn = st.sidebar.button("Calculate Analogy")

# --- 3. MAIN LOGIC ---
col1, col2 = st.columns([1, 2])

if solve_btn:
    # Validation: Check if words exist in the model's dictionary
    words = [pos1, pos2, neg1]
    missing = [w for w in words if w not in model]
    
    if missing:
        st.error(f"The following words are not in the vocabulary: {', '.join(missing)}")
    else:
        # Perform Vector Arithmetic
        try:
            # result = king - man + woman
            result_list = model.most_similar(positive=[pos1, pos2], negative=[neg1], topn=1)
            target_word, confidence = result_list[0]
            
            # --- DISPLAY RESULTS ---
            with col1:
                st.subheader("Result")
                st.metric("Output Word", target_word.title())
                st.metric("Confidence", f"{confidence:.2%}")
                
                st.markdown(f"""
                ### The Equation:
                $${pos1} - {neg1} + {pos2} \\approx \\mathbf{{{target_word}}}$$
                """)

            # --- DISPLAY VISUALIZATION ---
            with col2:
                st.subheader("2D Vector Space Projection")
                
                # 1. Get vectors for all 4 words
                plot_words = [pos1, neg1, pos2, target_word]
                vectors = np.array([model[w] for w in plot_words])
                
                # 2. Reduce dimensions from 50 to 2 using PCA
                pca = PCA(n_components=2)
                vectors_2d = pca.fit_transform(vectors)
                
                # 3. Create DataFrame for Plotly
                df = pd.DataFrame(vectors_2d, columns=['x', 'y'])
                df['word'] = plot_words
                df['type'] = ['Input', 'Input', 'Input', 'Result']
                
                # 4. Plot
                fig = px.scatter(df, x='x', y='y', text='word', color='type', 
                                 size=[10, 10, 10, 15], 
                                 title="Word Relationships in 2D",
                                 template="plotly_white")
                
                fig.update_traces(textposition='top center')
                fig.update_layout(showlegend=True)
                
                # Add arrows to show the relationship (Man -> King) vs (Woman -> Queen)
                # Note: We use annotations for arrows
                fig.add_annotation(x=df.iloc[1]['x'], y=df.iloc[1]['y'], # Man
                                   ax=df.iloc[0]['x'], ay=df.iloc[0]['y'], # King
                                   xref='x', yref='y', axref='x', ayref='y',
                                   arrowhead=2, arrowsize=1, arrowwidth=2, arrowcolor='blue')
                                   
                fig.add_annotation(x=df.iloc[2]['x'], y=df.iloc[2]['y'], # Woman
                                   ax=df.iloc[3]['x'], ay=df.iloc[3]['y'], # Result
                                   xref='x', yref='y', axref='x', ayref='y',
                                   arrowhead=2, arrowsize=1, arrowwidth=2, arrowcolor='red')

                st.plotly_chart(fig, use_container_width=True)
                
                st.caption("The blue arrow represents the 'Royal' transformation. The red arrow applies that same transformation to 'Woman'.")

        except Exception as e:
            st.error(f"Calculation Error: {e}")

else:
    with col1:
        st.info("👈 Enter words in the sidebar to solve an analogy!")