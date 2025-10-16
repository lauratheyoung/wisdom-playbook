import streamlit as st
import pandas as pd
import numpy as np
from typing import List, Tuple, Optional, Dict
from google.oauth2.service_account import Credentials
import gspread
import plotly.graph_objects as go
import streamlit.components.v1 as components

# -------------------------
# Config / Constants
# -------------------------
SPREADSHEET_URL = "https://docs.google.com/spreadsheets/d/1XEIVMPSS69BHDBGPw8fKkuqze5iqXSP7JFfwdGQhSHk"
INDIV_SHEET_NAME = "Individual"
PEER_SHEET_NAME = "Peer Review"
TRAITS = ["Purposeful", "Playful", "Adventurous", "Adaptable",
          "Curious", "Charitable", "Engaged", "Ethical"]

# Map trait -> list of question column headers (prefer explicit names)
# Replace these keys with the exact header strings from your sheet.
TRAIT_QUESTION_COLUMNS: Dict[str, List[str]] = {
    "Purposeful": ["Q3", "Q4", "Q5", "Q6"],
    "Playful": ["Q7", "Q8", "Q9", "Q10"],
    "Adventurous": ["Q11", "Q12", "Q13", "Q14"],
    "Adaptable": ["Q15", "Q16", "Q17", "Q18"],
    "Curious": ["Q19", "Q20", "Q21"],
    "Charitable": ["Q22", "Q23", "Q24", "Q25"],
    "Engaged": ["Q26", "Q27", "Q28", "Q29"],
    "Ethical": ["Q30", "Q31", "Q32", "Q33"],
}
# If you must use index-based ranges, create the mapping once here instead.

# -------------------------
# Google client & fetch (cached)
# -------------------------
@st.cache_resource
def get_gspread_client(_creds_info: dict):
    scopes = ["https://www.googleapis.com/auth/spreadsheets.readonly"]
    creds = Credentials.from_service_account_info(_creds_info, scopes=scopes)
    return gspread.authorize(creds)

@st.cache_data(ttl=300)  # cache sheet content for 5min (adjust TTL as needed)
def fetch_sheet_df(_client: gspread.client.Client, sheet_name: str) -> pd.DataFrame:
    """
    Fetch worksheet by name from the same spreadsheet URL and return DataFrame.
    """
    sh = _client.open_by_url(SPREADSHEET_URL)
    try:
        ws = sh.worksheet(sheet_name)
    except Exception as e:
        raise RuntimeError(f"Could not open worksheet '{sheet_name}': {e}")
    df = pd.DataFrame(ws.get_all_records())
    # normalize column names (strip)
    df.columns = df.columns.str.strip()
    return df

# -------------------------
# Data processing functions
# -------------------------
def ensure_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """Convert columns to numeric once; return df copy with conversions."""
    df = df.copy()
    df[cols] = df[cols].apply(pd.to_numeric, errors="coerce")
    return df

def compute_trait_scores_from_questions(df: pd.DataFrame,
                                        mapping: Dict[str, List[str]]) -> pd.DataFrame:
    """
    Given a mapping trait -> list of question column names,
    compute mean per trait and return DataFrame with ID cols + trait columns.
    """
    df = df.copy()
    # Flatten all question columns needed
    all_q_cols = [c for cols in mapping.values() for c in cols]
    missing = [c for c in all_q_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing expected question columns: {missing}")

    df = ensure_numeric(df, all_q_cols)

    for trait, cols in mapping.items():
        df[trait] = df[cols].mean(axis=1).round(1)

    # pick ID cols if they exist
    id_cols = [c for c in ["Timestamp", "What is your first name?", "What is your last name?", "UUID"] if c in df.columns]
    return df[id_cols + list(mapping.keys())].copy()

def determine_strength_growth(user_row: pd.Series, trait_cols: List[str], top_n: int = 3
                              ) -> Tuple[List[str], List[str]]:
    """
    Return strengths (top_n, inclusive of ties) and growth (bottom_n, inclusive of ties).
    Deterministic tie handling: sort by value desc then by trait name.
    """
    traits = user_row[trait_cols].astype(float).copy()
    # sort descending, tie-breaker alphabetical for determinism
    sorted_desc = traits.sort_values(ascending=False).sort_index()  # stable deterministic
    sorted_desc = sorted_desc.sort_values(ascending=False)  # ensures primary sort by value
    # Strengths (inclusive ties)
    if len(sorted_desc) == 0:
        return [], []
    strengths_cutoff = sorted_desc.iloc[top_n - 1] if top_n <= len(sorted_desc) else sorted_desc.iloc[-1]
    strengths = sorted_desc[sorted_desc >= strengths_cutoff].index.tolist()

    # Growths
    sorted_asc = traits.sort_values(ascending=True).sort_index()
    sorted_asc = sorted_asc.sort_values(ascending=True)
    growth_cutoff = sorted_asc.iloc[top_n - 1] if top_n <= len(sorted_asc) else sorted_asc.iloc[-1]
    growth = sorted_asc[sorted_asc <= growth_cutoff].index.tolist()

    return strengths, growth

def aggregate_peer_means(peer_df: pd.DataFrame, trait_cols: List[str], name_col: str="Full Name") -> pd.DataFrame:
    """
    Return DataFrame indexed by reviewed person's Full Name with mean trait scores from peers.
    """
    # ensure numeric
    peer_df = ensure_numeric(peer_df, trait_cols)
    # groupby Full Name and take mean
    grouped = peer_df.groupby(name_col, as_index=True)[trait_cols].mean().round(1)
    return grouped

def compute_consistency(user_row: pd.Series, peer_mean: Optional[pd.Series], trait_cols: List[str],
                        tolerance: float = 1.0) -> Tuple[float, List[str], List[str]]:
    if peer_mean is None:
        return 0.0, [], trait_cols.copy()
    diffs = (user_row[trait_cols].astype(float) - peer_mean[trait_cols].astype(float)).abs()
    consistent = diffs[diffs <= tolerance].index.tolist()
    inconsistent = diffs[diffs > tolerance].index.tolist()
    consistency_pct = round(len(consistent) / len(trait_cols) * 100, 1)
    return consistency_pct, consistent, inconsistent

# -------------------------
# UI helpers (plot + HTML)
# -------------------------
def plot_trait_comparison(user_row: pd.Series, peer_mean_scores: Optional[pd.Series], trait_cols: List[str]):
    self_scores = [(user_row[trait] / 6) * 100 for trait in trait_cols]
    peer_scores = [(peer_mean_scores[trait] / 6) * 100 if peer_mean_scores is not None else 0 for trait in trait_cols]
    delta_scores = [round(peer - self_, 1) for self_, peer in zip(self_scores, peer_scores)]

    fig = go.Figure()
    fig.add_trace(go.Bar(y=trait_cols, x=self_scores, name="Self Assessment", orientation="h",
                         text=[f"{s}%" for s in self_scores], textposition="outside"))
    fig.add_trace(go.Bar(y=trait_cols, x=peer_scores, name="Peer Review", orientation="h",
                         text=[f"{p}%" for p in peer_scores], textposition="outside"))
    for i, trait in enumerate(trait_cols):
        fig.add_annotation(x=max(self_scores[i], peer_scores[i]) + 5, y=trait,
                           text=f"Δ {delta_scores[i]}", showarrow=False, xanchor='left')
    fig.update_layout(barmode='group', title='Self vs Peer Trait Assessment',
                      xaxis=dict(title='Score (%)', range=[0, 100]),
                      yaxis=dict(title='Trait'),
                      height=50 * len(trait_cols) + 150,
                      margin=dict(l=150, r=50, t=60, b=50))
    return fig

def load_css_once(path="styles.css"):
    try:
        css = open(path).read()
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        # silently ignore if missing, or log
        pass

# -------------------------
# Streamlit app wiring
# -------------------------
def main():
    st.set_page_config(layout="wide")
    load_css_once("styles.css")

    # Use the helper function to get your gspread client
    client = get_gspread_client(st.secrets)

    # fetch dataframes (cached)
    try:
        data = fetch_sheet_df(client, INDIV_SHEET_NAME)
        peerdata = fetch_sheet_df(client, PEER_SHEET_NAME)
    except Exception as e:
        st.error(f"Error fetching sheets: {e}")
        st.stop()

    # Compute trait score frames
    try:
        df_traits = compute_trait_scores_from_questions(data, TRAIT_QUESTION_COLUMNS)
        df_peer_traits = compute_trait_scores_from_questions(peerdata, TRAIT_QUESTION_COLUMNS)
    except KeyError as e:
        st.error(f"Sheet format problem: {e}")
        st.stop()

    # add fullname columns for matching (strip whitespace)
    if "What is your first name?" in data.columns and "What is your last name?" in data.columns:
        data["Full Name"] = data["What is your first name?"].str.strip() + " " + data["What is your last name?"].str.strip()
    if "Who are you peer reviewing? (First and Last Name)" in peerdata.columns:
        peerdata["Full Name"] = peerdata["Who are you peer reviewing? (First and Last Name)"].str.strip()

    # UI: get uuid param or input
    query_params = st.experimental_get_query_params()
    uuid_param = query_params.get("uuid", [None])[0]
    uuid_input = st.text_input("Enter your report code:", value=uuid_param or "")

    if not uuid_input:
        st.info("Enter or pass your report code in the URL to view your report.")
        return

    # case-insensitive and trimmed lookup
    uuid_input_stripped = uuid_input.strip()
    user_rows = data[data.get("UUID", "").astype(str).str.strip() == uuid_input_stripped]

    if user_rows.empty:
        st.error("No report found for this report code.")
        return

    # Get the user's trait row (from df_traits)
    user_traits = df_traits[df_traits.get("UUID", "").astype(str).str.strip() == uuid_input_stripped]
    if user_traits.empty:
        st.error("No trait data found for this report code.")
        return

    user_row = user_traits.iloc[0]

    # strengths & growth
    strengths, growth = determine_strength_growth(user_row, TRAITS, top_n=3)

    # peers: aggregate means by Full Name and find matching mean for this user
    peer_means_df = aggregate_peer_means(peerdata, TRAITS, name_col="Full Name")
    # Find peer mean for this user's full name
    full_name = data.loc[user_rows.index[0], "Full Name"] if "Full Name" in data.columns else None
    peer_mean_scores = peer_means_df.loc[full_name] if full_name in peer_means_df.index else None

    consistency_pct, consistent_traits, inconsistent_traits = compute_consistency(user_row, peer_mean_scores, TRAITS)

    # show a simple dataframe of the user's scores
    st.dataframe(user_traits[TRAITS].T.rename(columns={user_traits.index[0]: "Individual Score"}))

    # display message (HTML component)
    user_name = user_rows["What is your first name?"].iloc[0] if "What is your first name?" in user_rows.columns else "Participant"
    strengths_str = ", ".join(strengths)
    growth_str = ", ".join(growth)
    peer_html = ""
    if peer_mean_scores is not None:
        peer_html = f"""
        <div class="peer-card">
            <p>Peer consistency: <strong>{consistency_pct}%</strong> ({', '.join(consistent_traits)})</p>
            <p>Inconsistencies: {', '.join(inconsistent_traits)}</p>
        </div>
        """

    html = f"""
    <div class="welcome-card">
      <h2>Welcome, {user_name}, to the Wisdom Playbook 🧭</h2>
      <div class="congrats-card">
        <h1>🎉 Congratulations, {user_name}!</h1>
        <p>Your <strong>strength traits</strong> are: <span class="strengths">{strengths_str}</span>.</p>
        <p>Your <strong>growth traits</strong> are: <span class="growth">{growth_str}</span>.</p>
      </div>
      {peer_html}
    </div>
    """
    components.html(html, height=400)

    # plot chart
    fig = plot_trait_comparison(user_row, peer_mean_scores, TRAITS)
    st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()


# import gspread as gs
# import plotly as pl
# import plotly.graph_objects as go
# from google.oauth2.service_account import Credentials
# import streamlit as st
# import pandas as pd
# from urllib.parse import urlparse, parse_qs
# import streamlit.components.v1 as components


# #Google sheet setup
# creds_info = st.secrets
# #print(creds_info)
# scopes = ["https://www.googleapis.com/auth/spreadsheets.readonly"]
# creds = Credentials.from_service_account_info(creds_info, scopes=scopes)
# client = gs.authorize(creds)

# # Open sheets
# sheet = client.open_by_url("https://docs.google.com/spreadsheets/d/1XEIVMPSS69BHDBGPw8fKkuqze5iqXSP7JFfwdGQhSHk/edit#gid=406100282").worksheet("Individual")
# peersheet = client.open_by_url("https://docs.google.com/spreadsheets/d/1XEIVMPSS69BHDBGPw8fKkuqze5iqXSP7JFfwdGQhSHk/edit?gid=1274798178#gid=1274798178").worksheet("Peer Review")

# #Load into dataframe
# data = pd.DataFrame(sheet.get_all_records())
# #Create peer dataframe
# peerdata = pd.DataFrame(peersheet.get_all_records())

# #Streamlit UI
# st.set_page_config(layout="wide")

# #Get UUI from query params

# query_params = st.query_params
# uuid_param = query_params.get("uuid", [None])[0]

# # Optional: let user input manually if query param not provided
# uuid_input = st.text_input("Enter your report code:", value=uuid_param or "")

# # Backend logic to determine user trait scores (1 trait = 4 Questions) order of traits: Purposeful, playful, adventurous, adaptable, curious, charitable, engaged, ethical

# def compute_trait_scores(df):
#     # Define the question ranges for each trait (1-indexed)
#     trait_ranges = {
#         "Purposeful": range(3, 7),
#         "Playful": range(7, 11),
#         "Adventurous": range(11, 15),
#         "Adaptable": range(15, 19),
#         "Curious": range(19, 22),
#         "Charitable": range(22, 26),
#         "Engaged": range(26, 30),
#         "Ethical": range(30, 34),
#     }

#     df_traits = df.copy()
#     #Fixing UUID
#     df_traits.columns = df_traits.columns.str.strip()

#     for trait, col_range in trait_ranges.items():
#             cols = df.columns[list(col_range[0] - 1 + i for i in range(len(col_range)))]
            
#             # Convert columns to numeric, coerce errors to NaN
#             df_traits[cols] = df_traits[cols].apply(pd.to_numeric, errors='coerce')
            
#             # Compute mean ignoring NaN
#             df_traits[trait] = df_traits[cols].mean(axis=1).round(1)

#     id_cols = ["Timestamp", "What is your first name?", "UUID"]
#     id_cols = [col for col in id_cols if col in df_traits.columns]  # safeguard
#     trait_cols = list(trait_ranges.keys())
#     return df_traits[list(id_cols) + trait_cols]

# # Compute aggregated scores for all users and get df_traits
# df_traits = compute_trait_scores(data)
# # Compute for peer data
# df_peer_traits = compute_trait_scores(peerdata)


# # Backend logic to determine users strength and growth traits by aggregating trait scores and comparing  --> not sure what to do if there are ties
# def determine_strength_growth(user_row, trait_cols, top_n=3):
#     """
#     Determine the top and bottom traits for a given user.

#     Parameters:
#         user_row: pandas Series representing a user's trait scores
#         trait_cols: list of trait column names
#         top_n: number of traits to include for strengths and growth areas

#     Returns:
#         strengths (list): top N trait names
#         growth (list): bottom N trait names
#     """
#     # Extract only the numeric trait values
#     traits = user_row[trait_cols].astype(float)

#     # Sort traits descending for strengths, ascending for growth
#     sorted_traits = traits.sort_values(ascending=False)

#     # Get top N strengths (including ties)
#     strengths_cutoff = sorted_traits.iloc[top_n - 1]
#     strengths = sorted_traits[sorted_traits >= strengths_cutoff].index.tolist()

#     # Get bottom N growth traits (including ties)
#     growth_cutoff = sorted_traits.iloc[-top_n]
#     growth = sorted_traits[sorted_traits <= growth_cutoff].index.tolist()

#     return strengths, growth

# #Defining the traits
# trait_cols = ["Purposeful", "Playful", "Adventurous", "Adaptable",
#               "Curious", "Charitable", "Engaged", "Ethical"]

# # --- Show only after UUID is entered ---
# if uuid_input:
#     user_data = data[data["UUID"] == uuid_input]

#     if not user_data.empty:

#         # Compute strengths/growth for peer assessments
#         df_peer_traits["Peer_Strengths"] = ""
#         df_peer_traits["Peer_Growth"] = ""
#         for i, row in df_peer_traits.iterrows():
#             s, g = determine_strength_growth(row, trait_cols)
#             df_peer_traits.at[i, "Peer_Strengths"] = ", ".join(s)
#             df_peer_traits.at[i, "Peer_Growth"] = ", ".join(g)

#         # Merge self and peer data via FullName
#         df_traits["Full Name"] = data["What is your first name?"].str.strip() + " " + data["What is your last name?"].str.strip()
#         df_peer_traits["Full Name"] = peerdata["Who are you peer reviewing? (First and Last Name)"].str.strip()
        
#         #st.write(df_peer_traits)


#         # Filter to the current user's trait scores
#         user_traits = df_traits[df_traits["UUID"] == uuid_input]

#         if not user_traits.empty:

#             #load styles.css
#             def load_css(file_name: str):
#                 with open(file_name) as f:
#                     st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

#             # Call this at the top of your Streamlit app
#             load_css("styles.css")

#             user_row = user_traits.iloc[0]

#             # Compute strengths and growth only for this user
#             strengths, growth = determine_strength_growth(user_row, trait_cols)

#             # Display only this user’s traits & rename mean to score
#             st.dataframe(user_traits[trait_cols].T.rename(columns={user_traits.index[0]: "Individual Score"}))

#             #Get user's name
#             user_name = user_data["What is your first name?"].iloc[0]

#             # Filter all peer rows for this user
#             peer_rows = df_peer_traits[df_peer_traits["Full Name"] == user_row["Full Name"]]

#             if not peer_rows.empty:
#                 # Aggregate by mean across all peer reviews for each trait
#                 peer_mean_scores = peer_rows[trait_cols].astype(float).mean()
#             else:
#                 peer_mean_scores = None

#             #Compute consistency using mean peer scores
#             def compute_consistency(user_row, peer_mean_scores, trait_cols, tolerance=1.0):
#                 consistent_traits = []
#                 inconsistent_traits = []

#                 if peer_mean_scores is None:
#                     return 0, consistent_traits, inconsistent_traits

#                 for trait in trait_cols:
#                     self_score = float(user_row[trait])
#                     peer_score = float(peer_mean_scores[trait])
#                     if abs(self_score - peer_score) <= tolerance:
#                         consistent_traits.append(trait)
#                     else:
#                         inconsistent_traits.append(trait)

#                 consistency_pct = round(len(consistent_traits) / len(trait_cols) * 100, 1)
#                 return consistency_pct, consistent_traits, inconsistent_traits

#             consistency_pct, consistent_traits, inconsistent_traits = compute_consistency(user_row, peer_mean_scores, trait_cols)

#             def display_dynamic_message(user_name, strengths, growth, 
#                             peer_strengths=None, peer_growth=None, 
#                             consistency_pct=None, consistent_traits=None, inconsistent_traits=None):
#                 # format lists
#                 strengths_str = ", ".join(strengths)
#                 growth_str = ", ".join(growth)

#                 message_html = f"""
#                 <div class="welcome-card">
#                     <h2>Welcome, {user_name}, to the Wisdom Playbook 🧭</h2>

#                     <div class="congrats-card">
#                         <h1>🎉 Congratulations, {user_name}!</h1>
#                         <p>You’ve taken the first steps toward reflecting on your own wisdom.</p>
#                         <p>Your <strong>strength traits</strong> are: <span class="strengths">{strengths_str}</span>.</p>
#                         <p>This means you excel at applying these strengths in daily life.</p>
#                         <p>Your <strong>growth traits</strong> are: <span class="growth">{growth_str}</span>.</p>
#                         <p>These are the areas with the most potential for reflection and development.</p>
#                     </div>
#                 """

#                 # Add peer feedback section only if peer reviews exist
#                 if peer_strengths and peer_growth and consistency_pct is not None:
#                     peer_strengths_str = ", ".join(peer_strengths)
#                     peer_growth_str = ", ".join(peer_growth)
#                     consistent_str = ", ".join(consistent_traits)
#                     inconsistent_str = ", ".join(inconsistent_traits)

#                     message_html += f"""
#                     <div class="peer-card">
#                         <p>There was consistency in how you assessed yourself and how your friends perceived you for <strong>{consistency_pct}%</strong> of wisdom statements ({consistent_str}).</p>
#                         <p>You may want to reflect on the inconsistencies between your own assessment and those of your friends, in particular, these wisdom statements: {inconsistent_str}.</p>
#                     </div>
#                     """

#                 # Close main welcome card div
#                 message_html += "</div>"

#                 # Display HTML
#                 components.html(message_html, height=500)

#             # Call function to display message
#             display_dynamic_message(user_name, strengths, growth, s, g, consistency_pct, consistent_traits, inconsistent_traits)

#             def plot_trait_comparison(user_row, peer_mean_scores, trait_cols):
#                 """
#                 user_row: pandas Series with individual user's trait scores
#                 peer_mean_scores: pandas Series with aggregated peer scores
#                 trait_cols: list of traits in the order to plot

#                 """
#                 # Extract and normalize scores (convert 0–6 scale to %)
#                 self_scores = [(user_row[trait] / 6) * 100 for trait in trait_cols]
#                 peer_scores = [(peer_mean_scores[trait] / 6) * 100 if peer_mean_scores is not None else 0 for trait in trait_cols]

#                 # Compute delta (in %)
#                 delta_scores = [round(peer - self_, 1) for self_, peer in zip(self_scores, peer_scores)]


#                 # Build horizontal bar chart
#                 fig = go.Figure()

#                 # Individual self-assessment bars
#                 fig.add_trace(go.Bar(
#                     y=trait_cols,
#                     x=self_scores,
#                     name='Self Assessment',
#                     orientation='h',
#                     marker_color='steelblue',
#                     text=[f"{s}%" for s in self_scores],
#                     textposition='outside'
#                 ))

#                 # Peer assessment bars
#                 fig.add_trace(go.Bar(
#                     y=trait_cols,
#                     x=peer_scores,
#                     name='Peer Review',
#                     orientation='h',
#                     marker_color='darkorange',
#                     text=[f"{p}%" for p in peer_scores],
#                     textposition='outside'
#                 ))

#                 # Add delta annotation
#                 for i, trait in enumerate(trait_cols):
#                     fig.add_annotation(
#                         x=max(self_scores[i], peer_scores[i]) + 5,  # offset a little
#                         y=trait,
#                         text=f"Δ {delta_scores[i]}",
#                         showarrow=False,
#                         font=dict(color='black', size=12),
#                         xanchor='left',
#                         yanchor='middle'
#                     )

#                 # Layout
#                 fig.update_layout(
#                 barmode='group',
#                 title='Self vs Peer Trait Assessment',
#                 xaxis=dict(title='Score (%)', range=[0, 100]),
#                 yaxis=dict(title='Trait'),
#                 height=50*len(trait_cols) + 100,
#                 margin=dict(l=150, r=50, t=50, b=50)
#                 )

#                 return fig

                
                
#             # Example call after computing user_row and peer_mean_scores
#             fig = plot_trait_comparison(user_row, peer_mean_scores, trait_cols)
#             st.plotly_chart(fig, use_container_width=True)



#         else:
#             st.error("No trait data found for this report code.")
#     else:
#         st.error("No report found for this report code.")
# else:
#     st.info("Enter or pass your report code in the URL to view your report.")


# # Link peer with individual through name match

# # Create full name columns
# data["FullName"] = data["What is your first name?"].str.strip() + " " + data["What is your last name?"].str.strip()
# peerdata["FullName"] = peerdata["Who are you peer reviewing? (First and Last Name)"].str.strip()

# # Generate overview graph

# # Generate trait detailed graphs --> create funciton as the only thing changing will be the trait/results

# # Generate summary statement
