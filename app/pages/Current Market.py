import streamlit as st

############## page setup #########

st.set_page_config(
    layout="wide", 
    page_icon="📈", 
    page_title="Current Market",
    initial_sidebar_state="expanded"
)

############## body ###############

st.title("Current Market")


st.components.v1.iframe(src="https://public.tableau.com/app/profile/jan.gfeller/viz/EDA_17341824351580/Treeboard", width=None, height=None, scrolling=False)






st.image("current.png")


#st.components.v1.iframe(src="https://public.tableau.com/app/profile/jan.gfeller/viz/EDA_17341824351580/Treeboard", width=None, height=None, scrolling=False)