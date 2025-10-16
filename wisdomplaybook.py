import gspread as gs
import plotly as pl
from google.oauth2.service_account import Credentials
import streamlit as st
import pandas as pd
from urllib.parse import urlparse, parse_qs
import textwrap
import streamlit.components.v1 as components


#Google sheet setup
creds_info = st.secrets
#print(creds_info)
scopes = ["https://www.googleapis.com/auth/spreadsheets.readonly"]
creds = Credentials.from_service_account_info(creds_info, scopes=scopes)
client = gs.authorize(creds)

# Open sheets
sheet = client.open_by_url("https://docs.google.com/spreadsheets/d/1XEIVMPSS69BHDBGPw8fKkuqze5iqXSP7JFfwdGQhSHk/edit#gid=406100282").worksheet("Individual")

#Load into dataframe
data = pd.DataFrame(sheet.get_all_records())

#Streamlit UI
st.set_page_config(layout="wide")

#Get UUI from query params

query_params = st.query_params
uuid_param = query_params.get("uuid", [None])[0]

# Optional: let user input manually if query param not provided
uuid_input = st.text_input("Enter your report code:", value=uuid_param or "")

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
            df_traits[trait] = df_traits[cols].mean(axis=1).round(1)

    id_cols = ["Timestamp", "What is your first name?", "UUID"]
    id_cols = [col for col in id_cols if col in df_traits.columns]  # safeguard
    trait_cols = list(trait_ranges.keys())
    return df_traits[list(id_cols) + trait_cols]

# Compute aggregated scores for all users and get df_traits
df_traits = compute_trait_scores(data)

# Backend logic to determine users strength and growth traits by aggregating trait scores and comparing  --> not sure what to do if there are ties
def determine_strength_growth(user_row, trait_cols, top_n=3):
    """
    Determine the top and bottom traits for a given user.

    Parameters:
        user_row: pandas Series representing a user's trait scores
        trait_cols: list of trait column names
        top_n: number of traits to include for strengths and growth areas

    Returns:
        strengths (list): top N trait names
        growth (list): bottom N trait names
    """
    # Extract only the numeric trait values
    traits = user_row[trait_cols].astype(float)

    # Sort traits descending for strengths, ascending for growth
    sorted_traits = traits.sort_values(ascending=False)

    # Get top N strengths (including ties)
    strengths_cutoff = sorted_traits.iloc[top_n - 1]
    strengths = sorted_traits[sorted_traits >= strengths_cutoff].index.tolist()

    # Get bottom N growth traits (including ties)
    growth_cutoff = sorted_traits.iloc[-top_n]
    growth = sorted_traits[sorted_traits <= growth_cutoff].index.tolist()

    return strengths, growth

#Defining the traits
trait_cols = ["Purposeful", "Playful", "Adventurous", "Adaptable",
              "Curious", "Charitable", "Engaged", "Ethical"]

# --- Show only after UUID is entered ---
if uuid_input:
    user_data = data[data["UUID"] == uuid_input]

    if not user_data.empty:
        # Compute trait scores for all users
        df_traits = compute_trait_scores(data)

        # Filter to the current user's trait scores
        user_traits = df_traits[df_traits["UUID"] == uuid_input]

        if not user_traits.empty:

            #load styles.css
            def load_css(file_name: str):
                with open(file_name) as f:
                    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

            # Call this at the top of your Streamlit app
            load_css("styles.css")

            user_row = user_traits.iloc[0]

            # Compute strengths and growth only for this user
            strengths, growth = determine_strength_growth(user_row, trait_cols)

            # Display only this user’s traits
            st.write("### Your Trait Scores")
            st.dataframe(user_traits[trait_cols].T.rename(columns={user_traits.index[0]: "Score"}))

            st.write("### Your Strengths and Growth Areas")
            st.write(f"**Top Strengths:** {', '.join(strengths)}")
            st.write(f"**Growth Opportunities:** {', '.join(growth)}")

            #Get user's name
            user_name = user_data["What is your first name?"].iloc[0]

            def display_dynamic_message(user_name, strengths, growth):
                # format lists
                strengths_str = ", ".join(strengths)
                growth_str = ", ".join(growth)

                message_html = f"""
                <div class="welcome-card">
                    <h2>Welcome, {user_name}, to the Wisdom Playbook 🧭</h2>

                    <div class="congrats-card">
                        <h1>🎉 Congratulations, {user_name}!</h1>
                        <p>You’ve taken the first steps toward reflecting on your own wisdom.</p>
                        <p>Your <strong>strength traits</strong> are: <span class="strengths">{strengths_str}</span>.</p>
                        <p>This means you excel at applying these strengths in daily life.</p>
                        <p>Your <strong>growth traits</strong> are: <span class="growth">{growth_str}</span>.</p>
                        <p>These are the areas with the most potential for reflection and development.</p>
                    </div>
                </div>
                """

                #Remove leading spaces for HTML render
                components.html(message_html, height=400)

                # display HTML
                #st.markdown(message_html, unsafe_allow_html=True)

            # Call function to display message
            display_dynamic_message(user_name, strengths, growth)

        else:
            st.error("No trait data found for this UUID.")
    else:
        st.error("No report found for this UUID.")
else:
    st.info("Enter or pass your UUID in the URL to view your report.")


# Link peer with individual through name match



# Generate congratulation message

def congrats_message():
    "Congratulations!\nYou've taken first steps towards reflecting on your own wisdom. Your self-assessment shows your areas of strength are:"

# Generate overview graph

# Generate trait detailed graphs --> create funciton as the only thing changing will be the trait/results

# Generate summary statement
