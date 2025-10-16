import gspread as gs
import plotly.graph_objects as go
from google.oauth2.service_account import Credentials
import streamlit as st
import pandas as pd
import streamlit.components.v1 as components

# Set-up
creds_info = st.secrets
scopes = ["https://www.googleapis.com/auth/spreadsheets.readonly"]
creds = Credentials.from_service_account_info(creds_info, scopes=scopes)
client = gs.authorize(creds)

# Open sheets
spreadsheet = client.open_by_url("https://docs.google.com/spreadsheets/d/1XEIVMPSS69BHDBGPw8fKkuqze5iqXSP7JFfwdGQhSHk/edit#gid=406100282")
sheet = spreadsheet.worksheet("Individual")
peersheet = spreadsheet.worksheet("Peer Review")

#Load into dataframe
data = pd.DataFrame(sheet.get_all_records())
peerdata = pd.DataFrame(peersheet.get_all_records())

#Streamlit UI
st.set_page_config(layout="wide")
query_params = st.query_params
uuid_param = query_params.get("uuid", [None])[0]
uuid_input = st.text_input("Enter your report code:", value=uuid_param or "")

#Initialising constant variables

# List of traits in order
TRAIT_COLS = ["Purposeful", "Playful", "Adventurous", "Adaptable",
              "Curious", "Charitable", "Engaged", "Ethical"]

# Column ranges for each trait (0-indexed for pandas)
TRAIT_RANGES = {
    "Purposeful": range(2, 5),
    "Playful": range(6, 9),
    "Adventurous": range(10, 13),
    "Adaptable": range(14, 17),
    "Curious": range(18, 20),
    "Charitable": range(21, 24),
    "Engaged": range(25, 28),
    "Ethical": range(29, 32),
}

# Backend logic to determine user trait scores (1 trait = 4 Questions) order of traits: Purposeful, playful, adventurous, adaptable, curious, charitable, engaged, ethical

def compute_trait_scores(df):
    df_traits = df.copy()
    df_traits.columns = df_traits.columns.str.strip()
    
    # Convert all relevant columns to numeric at once
    numeric_cols = sum([list(r) for r in TRAIT_RANGES.values()], [])
    df_traits.iloc[:, numeric_cols] = df_traits.iloc[:, numeric_cols].apply(pd.to_numeric, errors='coerce')

    for trait, col_range in TRAIT_RANGES.items():
        cols = df_traits.columns[list(col_range)]  # get actual column names
        # Convert to numeric safely (turn invalid entries into NaN)
        df_traits[cols] = df_traits[cols].apply(pd.to_numeric, errors='coerce')
        # Compute mean ignoring NaN
        df_traits[trait] = df_traits[cols].mean(axis=1).round(1)

    id_cols = ["Timestamp", "What is your first name?", "UUID"]
    id_cols = [col for col in id_cols if col in df_traits.columns]
    return df_traits[id_cols + TRAIT_COLS]


# Compute aggregated scores for all users and get df_traits
df_traits = compute_trait_scores(data)
# Compute for peer data
df_peer_traits = compute_trait_scores(peerdata)


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

# --- Helper functions ---
def load_css(file_name: str):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

def compute_peer_strengths(df, trait_cols):
    strengths_list, growth_list = [], []
    for _, row in df.iterrows():
        s, g = determine_strength_growth(row, trait_cols)
        strengths_list.append(", ".join(s))
        growth_list.append(", ".join(g))
    df["Peer_Strengths"] = strengths_list
    df["Peer_Growth"] = growth_list
    return df

def get_user_peer_feedback(peer_rows, trait_cols):
    if peer_rows.empty:
        return None, None, None
    peer_mean_scores = peer_rows[trait_cols].astype(float).mean()
    strengths, growth = determine_strength_growth(peer_mean_scores, trait_cols)
    return strengths, growth, peer_mean_scores

def compute_consistency(user_row, peer_mean_scores, trait_cols, tolerance=1.0):
    consistent, inconsistent = [], []
    if peer_mean_scores is None:
        return 0, consistent, inconsistent
    for trait in trait_cols:
        if abs(float(user_row[trait]) - float(peer_mean_scores[trait])) <= tolerance:
            consistent.append(trait)
        else:
            inconsistent.append(trait)
    pct = round(len(consistent) / len(trait_cols) * 100, 1)
    return pct, consistent, inconsistent

def display_dynamic_message(user_name, strengths, growth, 
                            peer_strengths=None, peer_growth=None, 
                            consistency_pct=None, consistent_traits=None, inconsistent_traits=None):
    strengths_str = ", ".join(strengths)
    growth_str = ", ".join(growth)
    message_html = f"""
    <div class="welcome-card">
        <h2>Welcome, {user_name}, to the Wisdom Playbook 🧭</h2>
        <div class="congrats-card">
            <h1>🎉 Congratulations, {user_name}!</h1>
            <p>Your <strong>strength traits</strong> are: <span class="strengths">{strengths_str}</span>.</p>
            <p>Your <strong>growth traits</strong> are: <span class="growth">{growth_str}</span>.</p>
        </div>
    """
    if peer_strengths and peer_growth and consistency_pct is not None:
        message_html += f"""
        <div class="peer-card">
            <p>Consistency with peers: <strong>{consistency_pct}%</strong> ({', '.join(consistent_traits)})</p>
            <p>Inconsistencies: {', '.join(inconsistent_traits)}</p>
        </div>
        """
    message_html += "</div>"
    components.html(message_html, height=500)

def plot_trait_comparison(user_row, peer_mean_scores, trait_cols):
    """
    user_row: pandas Series with individual user's trait scores
    peer_mean_scores: pandas Series with aggregated peer scores
    trait_cols: list of traits in the order to plot

    """
    # Extract and normalize scores (convert 0–6 scale to %) and round to 1 decimal
    self_scores = [round((user_row[trait] / 6) * 100, 1) for trait in trait_cols]
    peer_scores = [round((peer_mean_scores[trait] / 6) * 100, 1) if peer_mean_scores is not None else 0 for trait in trait_cols]

    # Compute delta (in %)
    delta_scores = [round(peer - self_, 1) for self_, peer in zip(self_scores, peer_scores)]

    # Build horizontal bar chart
    fig = go.Figure()

    # Individual self-assessment bars
    fig.add_trace(go.Bar(
        y=trait_cols,
        x=self_scores,
        name='Self Assessment',
        orientation='h',
        marker_color='steelblue',
        text=[f"{s}%" for s in self_scores],
        textposition='outside'
    ))

    # Peer assessment bars
    fig.add_trace(go.Bar(
        y=trait_cols,
        x=peer_scores,
        name='Peer Review',
        orientation='h',
        marker_color='darkorange',
        text=[f"{p}%" for p in peer_scores],
        textposition='outside'
    ))

    # Add delta annotation
    for i, trait in enumerate(trait_cols):
        fig.add_annotation(
            x=max(self_scores[i], peer_scores[i]) + 5,  # offset a little
            y=trait,
            text=f"Δ {delta_scores[i]}",
            showarrow=False,
            font=dict(color='black', size=12),
            xanchor='left',
            yanchor='middle'
        )

    fig.update_layout(
        barmode='group',
        title='Self vs Peer Wisdom Traits Assessment',
        xaxis=dict(title='Score (%)', range=[0, 100]),
        height=50*len(trait_cols) + 100,
        margin=dict(l=150, r=50, t=50, b=100),  # increase bottom margin for legend
        legend=dict(
            orientation='h',
            y=-0.2,  # position below the x-axis
            x=0.1,
            xanchor='left',
            yanchor='top'
        )
    )
    return fig



# --- Main logic ---
if not uuid_input:
    st.info("Enter or pass your report code in the URL to view your report.")
    st.stop()

user_data = data[data["UUID"] == uuid_input]
if user_data.empty:
    st.error("No report found for this report code.")
    st.stop()

df_traits["Full Name"] = data["What is your first name?"].str.strip() + " " + data["What is your last name?"].str.strip()
df_peer_traits["Full Name"] = peerdata["Who are you peer reviewing? (First and Last Name)"].str.strip()

user_traits = df_traits[df_traits["UUID"] == uuid_input]
if user_traits.empty:
    st.error("No trait data found for this report code.")
    st.stop()

# Load CSS
load_css("styles.css")

# Compute user's strengths/growth
user_row = user_traits.iloc[0]
strengths, growth = determine_strength_growth(user_row, TRAIT_COLS)
st.dataframe(user_traits[TRAIT_COLS].T.rename(columns={user_traits.index[0]: "Individual Score"}))

# User's name
user_name = user_data["What is your first name?"].iloc[0]

# Peer computations
df_peer_traits = compute_peer_strengths(df_peer_traits, TRAIT_COLS)
peer_rows = df_peer_traits[df_peer_traits["Full Name"] == user_row["Full Name"]]
peer_strengths, peer_growth, peer_mean_scores = get_user_peer_feedback(peer_rows, TRAIT_COLS)
consistency_pct, consistent_traits, inconsistent_traits = compute_consistency(user_row, peer_mean_scores, TRAIT_COLS)

# Display message
display_dynamic_message(
    user_name, strengths, growth, 
    peer_strengths, peer_growth, 
    consistency_pct, consistent_traits, inconsistent_traits
)

fig = plot_trait_comparison(user_row, peer_mean_scores, TRAIT_COLS)
st.plotly_chart(fig, use_container_width=True)



# # --- Show only after UUID is entered ---
# if uuid_input:
#     user_data = data[data["UUID"] == uuid_input]

#     if not user_data.empty:

#         def compute_peer_strengths(df):
#             strengths_list = []
#             growth_list = []

#             for _, row in df.iterrows():  # still small, okay; can optimize later with numpy if huge
#                 s, g = determine_strength_growth(row, TRAIT_COLS)
#                 strengths_list.append(", ".join(s))
#                 growth_list.append(", ".join(g))
            
#             df["Peer_Strengths"] = strengths_list
#             df["Peer_Growth"] = growth_list
#             return df

#         df_peer_traits = compute_peer_strengths(df_peer_traits)


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
#             strengths, growth = determine_strength_growth(user_row, TRAIT_COLS)

#             # Display only this user’s traits & rename mean to score
#             st.dataframe(user_traits[TRAIT_COLS].T.rename(columns={user_traits.index[0]: "Individual Score"}))

#             #Get user's name
#             user_name = user_data["What is your first name?"].iloc[0]

#             # Filter all peer rows for this user
#             peer_rows = df_peer_traits[df_peer_traits["Full Name"] == user_row["Full Name"]]

#             if not peer_rows.empty:
#                 # Aggregate by mean across all peer reviews for each trait
#                 peer_mean_scores = peer_rows[TRAIT_COLS].astype(float).mean()
#             else:
#                 peer_mean_scores = None

#             # --- Compute peer strengths and growth for all peer rows ---
#             def compute_peer_strengths(df, trait_cols):
#                 """
#                 Compute peer strengths and growth for each peer review row.
#                 Returns df with new columns 'Peer_Strengths' and 'Peer_Growth'
#                 """
#                 strengths_list = []
#                 growth_list = []

#                 for _, row in df.iterrows():
#                     s, g = determine_strength_growth(row, trait_cols)
#                     strengths_list.append(", ".join(s))
#                     growth_list.append(", ".join(g))

#                 df["Peer_Strengths"] = strengths_list
#                 df["Peer_Growth"] = growth_list
#                 return df


#             # --- Compute aggregated peer strengths/growth for a specific user ---
#             def get_user_peer_feedback(peer_rows, trait_cols):
#                 """
#                 Aggregate peer rows for a single user to get average peer scores
#                 and determine strengths/growth traits from the average.
#                 """
#                 if peer_rows.empty:
#                     return None, None, None  # no peers

#                 peer_mean_scores = peer_rows[trait_cols].astype(float).mean()
#                 strengths, growth = determine_strength_growth(peer_mean_scores, trait_cols)
#                 return strengths, growth, peer_mean_scores


#             # --- Compute consistency between self and peer ---
#             def compute_consistency(user_row, peer_mean_scores, trait_cols, tolerance=1.0):
#                 """
#                 Returns consistency percentage and lists of consistent and inconsistent traits.
#                 """
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


#             # --- Display message ---
#             def display_dynamic_message(user_name, strengths, growth, 
#                                         peer_strengths=None, peer_growth=None, 
#                                         consistency_pct=None, consistent_traits=None, inconsistent_traits=None):
#                 strengths_str = ", ".join(strengths)
#                 growth_str = ", ".join(growth)

#                 message_html = f"""
#                 <div class="welcome-card">
#                     <h2>Welcome, {user_name}, to the Wisdom Playbook 🧭</h2>

#                     <div class="congrats-card">
#                         <h1>🎉 Congratulations, {user_name}!</h1>
#                         <p>You’ve taken the first steps toward reflecting on your own wisdom.</p>
#                         <p>Your <strong>strength traits</strong> are: <span class="strengths">{strengths_str}</span>.</p>
#                         <p>Your <strong>growth traits</strong> are: <span class="growth">{growth_str}</span>.</p>
#                     </div>
#                 """

#                 if peer_strengths and peer_growth and consistency_pct is not None:
#                     peer_strengths_str = ", ".join(peer_strengths)
#                     peer_growth_str = ", ".join(peer_growth)
#                     consistent_str = ", ".join(consistent_traits)
#                     inconsistent_str = ", ".join(inconsistent_traits)

#                     message_html += f"""
#                     <div class="peer-card">
#                         <p>There was consistency in how you assessed yourself and how your friends perceived you for <strong>{consistency_pct}%</strong> of wisdom statements ({consistent_str}).</p>
#                         <p>You may want to reflect on the inconsistencies: {inconsistent_str}.</p>
#                     </div>
#                     """

#                 message_html += "</div>"
#                 components.html(message_html, height=500)


#             # 1. Compute peer strengths/growth for all peers once
#             df_peer_traits = compute_peer_strengths(df_peer_traits, TRAIT_COLS)

#             # 2. Filter the current user's trait row
#             user_row = df_traits[df_traits["UUID"] == uuid_input].iloc[0]
#             user_name = user_row["What is your first name?"]

#             # 3. Compute user's strengths/growth
#             strengths, growth = determine_strength_growth(user_row, TRAIT_COLS)

#             # 4. Filter peer rows for this user and compute aggregated peer feedback
#             peer_rows = df_peer_traits[df_peer_traits["Full Name"] == user_row["Full Name"]]
#             peer_strengths, peer_growth, peer_mean_scores = get_user_peer_feedback(peer_rows, TRAIT_COLS)

#             # 5. Compute consistency
#             consistency_pct, consistent_traits, inconsistent_traits = compute_consistency(user_row, peer_mean_scores, TRAIT_COLS)

#             # 6. Display message
#             display_dynamic_message(
#                 user_name,
#                 strengths,
#                 growth,
#                 peer_strengths=peer_strengths,
#                 peer_growth=peer_growth,
#                 consistency_pct=consistency_pct,
#                 consistent_traits=consistent_traits,
#                 inconsistent_traits=inconsistent_traits
#             )


#             def plot_trait_comparison(user_row, peer_mean_scores, trait_cols):
#                 """
#                 user_row: pandas Series with individual user's trait scores
#                 peer_mean_scores: pandas Series with aggregated peer scores
#                 trait_cols: list of traits in the order to plot

#                 """
#                 # Extract and normalize scores (convert 0–6 scale to %) and round to 1 decimal
#                 self_scores = [round((user_row[trait] / 6) * 100, 1) for trait in trait_cols]
#                 peer_scores = [round((peer_mean_scores[trait] / 6) * 100, 1) if peer_mean_scores is not None else 0 for trait in trait_cols]

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

#                 fig.update_layout(
#                     barmode='group',
#                     title='Self vs Peer Wisdom Traits Assessment',
#                     xaxis=dict(title='Score (%)', range=[0, 100]),
#                     height=50*len(trait_cols) + 100,
#                     margin=dict(l=150, r=50, t=50, b=100),  # increase bottom margin for legend
#                     legend=dict(
#                         orientation='h',
#                         y=-0.2,  # position below the x-axis
#                         x=0.1,
#                         xanchor='left',
#                         yanchor='top'
#                     )
#                 )
#                 return fig
                
#             # Example call after computing user_row and peer_mean_scores
#             fig = plot_trait_comparison(user_row, peer_mean_scores, TRAIT_COLS)
#             st.plotly_chart(fig, use_container_width=True)

#         else:
#             st.error("No trait data found for this report code.")
#     else:
#         st.error("No report found for this report code.")
# else:
#     st.info("Enter or pass your report code in the URL to view your report.")


# Link peer with individual through name match

# Create full name columns
data["FullName"] = data["What is your first name?"].str.strip() + " " + data["What is your last name?"].str.strip()
peerdata["FullName"] = peerdata["Who are you peer reviewing? (First and Last Name)"].str.strip()

# Generate overview graph

# Generate trait detailed graphs --> create funciton as the only thing changing will be the trait/results

# Generate summary statement
