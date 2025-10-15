import gspread as gs
import plotly as pl
from google.oauth2.service_account import Credentials
import streamlit as st
import pandas as pd
from urllib.parse import urlparse, parse_qs

#Google sheet setup
creds_info = st.secrets
#print(creds_info)
scopes = ["https://www.googleapis.com/auth/spreadsheets.readonly"]
creds = Credentials.from_service_account_info(creds_info, scopes=scopes)
client = gs.authorize(creds)

# Test connection
#st.write("Connected sheets:", [s.title for s in client.openall()])

# Open sheets
sheet = client.open_by_url("https://docs.google.com/spreadsheets/d/1XEIVMPSS69BHDBGPw8fKkuqze5iqXSP7JFfwdGQhSHk/edit#gid=406100282").worksheet("Individual")

#Load into dataframe
data = pd.DataFrame(sheet.get_all_records())

#Display first rows
#st.dataframe(data.head())


#Streamlit UI
st.set_page_config(layout="wide")
st.title("Form Submission Report Viewer")
st.title("Hello Streamlit!")
st.write("If you see this, Streamlit is working.")

#Get UUI from query params

query_params = st.query_params
uuid_param = query_params.get("uuid", [None])[0]

# Optional: let user input manually if query param not provided
uuid_input = st.text_input("Enter your report UUID:", value=uuid_param or "")

if uuid_input:
    user_data = data[data["UUID"] == uuid_input]
    if not user_data.empty:
        st.success("✅ Report found")
        # TODO: Replace with your designed report
        st.dataframe(user_data)
    else:
        st.error("No report found for this ID.")
        

# Backend logic to determine user trait scores (1 trait = 4 Questions) order of traits: Purposeful, playful, adventurous, adaptable, curious, charitable, engaged, ethical

def compute_trait_scores(df):
    # Define the question ranges for each trait (1-indexed)
    trait_ranges = {
        "Purposeful": range(3, 7),
        "Playful": range(7, 11),
        "Adventurous": range(11, 15),
        "Adaptable": range(15, 19),
        "Curious": range(19, 22),
        "Charitable": range(22, 26),
        "Engaged": range(26, 30),
        "Ethical": range(30, 34),
    }

    df_traits = df.copy()
    #Fixing UUID
    df_traits.columns = df_traits.columns.str.strip()



    for trait, col_range in trait_ranges.items():
            cols = df.columns[list(col_range[0] - 1 + i for i in range(len(col_range)))]
            
            # Convert columns to numeric, coerce errors to NaN
            df_traits[cols] = df_traits[cols].apply(pd.to_numeric, errors='coerce')
            
            # Compute mean ignoring NaN
            df_traits[trait] = df_traits[cols].mean(axis=1)

    id_cols = df.columns[:2]  # Timestamp, Name
    trait_cols = list(trait_ranges.keys())
    return df_traits[list(id_cols) + trait_cols]


# --- Process and display data ---
if uuid_input:
    user_data = data[data["UUID"] == uuid_input]

    if not user_data.empty:
        st.success("✅ Report found for this UUID")

        # Compute aggregated scores for all users
        df_traits = compute_trait_scores(data)

        # Display all computed trait scores
        st.write("### Full Aggregated Table (for testing)")
        st.dataframe(df_traits)

        # Show only this user’s trait averages
        user_traits = df_traits[df_traits["UUID"] == uuid_input]
        st.write("### Individual User Trait Scores")
        st.dataframe(user_traits)

        # Print raw values (optional, for debugging)
        st.write("### Debug Output (Raw Values)")
        st.write(user_traits.to_dict(orient="records")[0])

    else:
        st.error("No report found for this ID.")
else:
    st.info("Enter or pass your UUID in the URL to test the aggregation.")

# Backend logic to determine users strength and growth traits by aggregating trait scores and comparing  --> not sure what to do if there are ties

# Link peer with individual through name match

# Generate welcome

welcome = st.title("Welcome"+"to the"+"Wisdom Playbook")

# Generate congratulation message

def congrats_message():
    "Congratulations!\nYou've taken first steps towards reflecting on your own wisdom. Your self-assessment shows your areas of strength are:"

# Generate overview graph

# Generate trait detailed graphs --> create funciton as the only thing changing will be the trait/results

# Generate summary statement
